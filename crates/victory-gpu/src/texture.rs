//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// Original packed 16-bit format (86 or 115) when the host texture was
    /// widened to R8G8B8A8_UNORM, so detiled 2-byte texels expand into the
    /// 4-byte staging buffer.
    widened_from: Option<u32>,
    last: Mutex<Last>,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after,
/// with room to read the memory into.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    read: Vec<u8>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// State recorded when a host copy or resolve writes a texture.
///
/// While the texture's console memory stays untouched after this point,
/// [`upload`] keeps the host-written contents instead of overwriting them with
/// stale title memory. Once the title writes the memory again, the entry is
/// cleared and detiling resumes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HostWritten {
    /// Write-watched memory (`MEM_WRITE_WATCH`): journal asking number at the
    /// host copy.
    Watched { start: usize, end: usize, at: u64 },
    /// Unwatched memory (e.g. physical-page views): content hash at the host
    /// copy, or `None` if the memory was not committed at copy time.
    Hashed {
        memory: usize,
        size: usize,
        hash: Option<blake3::Hash>,
    },
    /// Non-title-written texture or no detileable console layout: stays
    /// host-written until forgotten.
    Sticky,
}

/// A GPU copy or resolve recorded on a command list, committed to
/// [`HOST_WRITTEN`] only once that list actually executes.
pub(crate) struct PendingHostWrite {
    key: usize,
    span: Option<(usize, usize)>,
    alive: Arc<AtomicBool>,
}

#[derive(Default)]
struct HostWrites {
    entries: HashMap<usize, HostWritten>,
    /// Liveness token per resource with a pending or committed host write.
    /// Flipped to `false` by [`forget`] so a list submitted after the resource
    /// was released neither reads freed memory nor poisons a reused address.
    alive: HashMap<usize, Arc<AtomicBool>>,
}

/// Textures the host GPU wrote by a copy or resolve: their contents live on
/// the host until the title writes their console memory again.
static HOST_WRITTEN: Mutex<Option<HostWrites>> = Mutex::new(None);

/// Console memory pointer and layout byte size for a title-written, detileable
/// texture. Returns `None` for render targets, depth/UAV textures, buffers, or
/// unsupported layouts.
fn texture_span(resource: This) -> Option<(*const u8, usize)> {
    {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = map.as_ref().and_then(|m| m.get(&(resource as usize))) {
            return Some((s.memory, s.layout.size as usize));
        }
    }
    if !crate::resource::host_resource(resource).is_some_and(title_written) {
        return None;
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let tex = crate::resource::texture_layout(&desc)?;
    let layout = &tex.planes[0].layout;
    if layout.desc.swizzle.xor() || layout.size == 0 {
        return None;
    }
    usize::try_from(layout.size).ok().map(|size| (memory, size))
}

/// Hashes `[memory, memory + size)` when committed, reading through the kernel
/// so concurrently unmapped pages do not fault.
fn hash_unwatched(memory: *const u8, size: usize) -> Option<blake3::Hash> {
    if !committed(memory, size) {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    let mut chunk = vec![0u8; size.min(64 << 10)];
    let mut offset = 0;
    while offset < size {
        let n = (size - offset).min(chunk.len());
        // SAFETY: `offset < size`; `copy_mapped` reads via the kernel, zeroing
        // any page unmapped concurrently.
        let src = unsafe { memory.add(offset) };
        crate::host::copy_mapped(chunk.as_mut_ptr(), src, n, false);
        hasher.update(&chunk[..n]);
        offset += n;
    }
    Some(hasher.finalize())
}

/// Records that a host copy or resolve on `list` writes `resource`.
///
/// Marks `resource` used on `list` immediately so a later [`upload`] in the
/// same command list will not overwrite the copy, but defers updating the
/// global [`HOST_WRITTEN`] map until `list` actually executes so discarded or
/// unsubmitted command lists do not suppress later title uploads.
pub(crate) fn host_written(list: &HostList, resource: This) {
    let key = resource as usize;
    list.first_use(key);
    let span = texture_span(resource).map(|(memory, size)| (memory as usize, size));
    let alive = {
        let mut guard = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .get_or_insert_with(HostWrites::default)
            .alive
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicBool::new(true)))
            .clone()
    };
    list.defer_host_write(PendingHostWrite { key, span, alive });
}

/// Commits a recorded host copy or resolve after its command list has run.
pub(crate) fn commit_host_write(pending: &PendingHostWrite) {
    if !pending.alive.load(Ordering::Acquire) {
        return;
    }
    let state = match pending.span {
        Some((start, size)) => {
            if let Some(end) = start.checked_add(size) {
                if let Some((at, _)) = crate::watch::writes(start, end, false) {
                    HostWritten::Watched { start, end, at }
                } else {
                    HostWritten::Hashed {
                        memory: start,
                        size,
                        hash: hash_unwatched(start as *const u8, size),
                    }
                }
            } else {
                HostWritten::Sticky
            }
        }
        None => HostWritten::Sticky,
    };
    let mut guard = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if pending.alive.load(Ordering::Acquire) {
        guard
            .get_or_insert_with(HostWrites::default)
            .entries
            .insert(pending.key, state);
    }
}

/// Whether `resource` still holds host-written contents (no title write to its
/// console memory since the copy). Removes the entry once title memory changes.
fn still_host_written(resource: This) -> bool {
    let key = resource as usize;
    let state = {
        let guard = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        let Some(&state) = guard.as_ref().and_then(|hw| hw.entries.get(&key)) else {
            return false;
        };
        state
    };
    let unchanged = match state {
        HostWritten::Sticky => true,
        HostWritten::Watched { start, end, at } => match crate::watch::writes(start, end, false) {
            Some((_, written)) => written.iter().all(|&(_, w)| w <= at),
            None => true,
        },
        HostWritten::Hashed { memory, size, hash } => {
            let Some(current) = hash_unwatched(memory as *const u8, size) else {
                return true;
            };
            hash == Some(current)
        }
    };
    if !unchanged {
        let mut guard = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(hw) = guard.as_mut()
            && hw.entries.get(&key) == Some(&state)
        {
            hw.entries.remove(&key);
        }
    }
    unchanged
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut guard = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(hw) = guard.as_mut() {
        hw.entries.remove(&resource);
        if let Some(alive) = hw.alive.remove(&resource) {
            alive.store(false, Ordering::Release);
        }
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    if !list.first_use(resource as usize) || still_host_written(resource) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = footprint_len(footprint, staging.rows[i], staging.size);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

/// GetCopyableFootprints may end the buffer at the last row's active bytes.
fn footprint_len(
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    rows: u32,
    buffer_size: u64,
) -> u64 {
    let f = footprint.Footprint;
    let pitched = u64::from(f.RowPitch) * u64::from(rows) * u64::from(f.Depth);
    pitched.min(buffer_size - footprint.Offset)
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let size = self.layout.size as usize;
        if size == 0 {
            return 0;
        }
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, start + size, false) {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            self.detile(&last.read);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        last.read.resize(size, 0);
        match read_resident(self.memory, &mut last.read) {
            Residency::Unmapped => {
                last.hash = None;
                #[cfg(feature = "diagnostics")]
                tracing::debug!(
                    memory = ?self.memory,
                    size,
                    desc = ?self.layout.desc,
                    "a sampled texture's memory is not committed"
                );
                0
            }
            Residency::Full => {
                let hash = blake3::hash(&last.read);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
                if last.hash == Some(hash) {
                    return size;
                }
                last.hash = Some(hash);
                self.detile(&last.read);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                size
            }
            Residency::Partial(intervals) => {
                let (hash, resident_bytes) = hash_intervals(&last.read, &intervals);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHashed, resident_bytes as u64);
                if last.hash == Some(hash) {
                    return resident_bytes;
                }
                zero_gaps(&mut last.read, &intervals);
                last.hash = Some(hash);
                self.detile(&last.read);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, resident_bytes as u64);
                resident_bytes
            }
        }
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        let mut packed = Vec::new();
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = footprint_len(footprint, rows, self.size) as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) = detile_subresource(
                &self.layout,
                tiled,
                mip,
                slice,
                linear,
                pitch,
                rows as usize,
                f.Depth as usize,
                self.widened_from,
                &mut packed,
            ) {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Which byte intervals of a title texture are currently mapped and readable.
#[derive(Debug, PartialEq, Eq)]
enum Residency {
    Unmapped,
    Full,
    Partial(Vec<(usize, usize)>),
}

/// Copies committed, readable regions of `[src, src + dst.len())` into `dst`
/// without `VirtualQuery` overhead on the fully mapped fast path.
fn read_resident(src: *const u8, dst: &mut [u8]) -> Residency {
    if dst.is_empty() || crate::host::read_into(src, dst) {
        return Residency::Full;
    }
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    const PAGE_SIZE: usize = 4096;

    let base = src as usize;
    let end = base.saturating_add(dst.len());
    let mut p = base;
    let mut intervals: Vec<(usize, usize)> = Vec::new();
    let push_interval = |intervals: &mut Vec<(usize, usize)>, start: usize, stop: usize| {
        if start >= stop {
            return;
        }
        if let Some(last) = intervals.last_mut()
            && last.1 == start
        {
            last.1 = stop;
        } else {
            intervals.push((start, stop));
        }
    };

    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            break;
        }
        let region_end = (info.BaseAddress as usize)
            .saturating_add(info.RegionSize)
            .min(end);
        let step = region_end.saturating_sub(p);
        if step == 0 {
            break;
        }
        let next_p = p.saturating_add(step);
        if info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0 {
            p = next_p;
            continue;
        }
        let offset = p - base;
        let region_len = region_end - p;
        if crate::host::read_into(p as *const u8, &mut dst[offset..offset + region_len]) {
            push_interval(&mut intervals, offset, offset + region_len);
        } else {
            // Concurrent unmap inside a committed region: fall back to 4 KiB pages.
            let mut q = p;
            while q < region_end {
                let next_page = (q & !(PAGE_SIZE - 1)).saturating_add(PAGE_SIZE);
                let page_end = next_page.min(region_end);
                let page_len = page_end.saturating_sub(q);
                if page_len == 0 {
                    break;
                }
                let page_offset = q - base;
                if crate::host::read_into(
                    q as *const u8,
                    &mut dst[page_offset..page_offset + page_len],
                ) {
                    push_interval(&mut intervals, page_offset, page_offset + page_len);
                }
                q = q.saturating_add(page_len);
            }
        }
        p = next_p;
    }

    if intervals.is_empty() {
        Residency::Unmapped
    } else if intervals.as_slice() == [(0, dst.len())] {
        Residency::Full
    } else {
        Residency::Partial(intervals)
    }
}

/// Hashes only the resident byte intervals (and their bounds) and returns `(hash, resident_bytes)`.
fn hash_intervals(data: &[u8], intervals: &[(usize, usize)]) -> (blake3::Hash, usize) {
    let mut hasher = blake3::Hasher::new();
    let mut resident_bytes = 0usize;
    for &(start, end) in intervals {
        hasher.update(&(start as u64).to_le_bytes());
        hasher.update(&(end as u64).to_le_bytes());
        hasher.update(&data[start..end]);
        resident_bytes += end - start;
    }
    (hasher.finalize(), resident_bytes)
}

/// Zeroes unmapped gaps between sorted `intervals` in `data` so evicted tiles leave no stale bytes.
fn zero_gaps(data: &mut [u8], intervals: &[(usize, usize)]) {
    let mut cursor = 0usize;
    for &(start, end) in intervals {
        if cursor < start {
            data[cursor..start].fill(0);
        }
        cursor = end;
    }
    if cursor < data.len() {
        data[cursor..].fill(0);
    }
}

#[allow(clippy::too_many_arguments)]
fn detile_subresource(
    layout: &victory_formats::texture::Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    linear: &mut [u8],
    row_pitch: usize,
    rows: usize,
    depth: usize,
    widened_from: Option<u32>,
    packed: &mut Vec<u8>,
) -> Result<(), victory_formats::texture::Error> {
    let Some(packed_format) = widened_from else {
        return layout.detile(tiled, mip, slice, linear, row_pitch, row_pitch * rows);
    };
    let width = layout
        .mips
        .get(mip as usize)
        .map_or(0, |m| m.width as usize);
    let packed_pitch = width * 2;
    let packed_plane = packed_pitch * rows;
    packed.resize(packed_plane * depth, 0);
    layout.detile(tiled, mip, slice, packed, packed_pitch, packed_plane)?;
    crate::resource::expand_packed16_rows(
        packed_format,
        width,
        rows * depth,
        packed,
        packed_pitch,
        linear,
        row_pitch,
    );
    Ok(())
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let title_format = crate::resource::pc_format(desc.format);
    let widened_from = (d.Format.0 as u32 == crate::resource::RGBA8_UNORM
        && crate::resource::PACKED_16.contains(&title_format))
    .then_some(title_format);
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        widened_from,
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualAlloc, VirtualFree,
    };

    /// RAII guard for a `VirtualAlloc(..., MEM_RESERVE, ...)` address range.
    struct VirtualRegion {
        ptr: *mut u8,
        len: usize,
    }

    impl VirtualRegion {
        fn reserve(len: usize) -> Self {
            // SAFETY: reserves uncommitted virtual address space of `len` bytes, released on drop.
            let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS) }
                .cast::<u8>();
            assert!(!ptr.is_null(), "VirtualAlloc MEM_RESERVE failed");
            Self { ptr, len }
        }

        fn commit(&self, offset: usize, len: usize) {
            assert!(offset.checked_add(len).is_some_and(|end| end <= self.len));
            // SAFETY: commits pages within our reserved virtual address range.
            let committed = unsafe {
                VirtualAlloc(self.ptr.add(offset).cast(), len, MEM_COMMIT, PAGE_READWRITE)
            };
            assert!(!committed.is_null(), "VirtualAlloc MEM_COMMIT failed");
        }

        fn decommit(&self, offset: usize, len: usize) {
            assert!(offset.checked_add(len).is_some_and(|end| end <= self.len));
            // SAFETY: decommits pages within our reserved virtual address range.
            let ok = unsafe { VirtualFree(self.ptr.add(offset).cast(), len, MEM_DECOMMIT) };
            assert_ne!(ok, 0, "VirtualFree MEM_DECOMMIT failed");
        }

        /// Writes `bytes` at `offset`; caller must ensure `[offset..offset + bytes.len())` is committed.
        unsafe fn write_slice(&self, offset: usize, bytes: &[u8]) {
            assert!(
                offset
                    .checked_add(bytes.len())
                    .is_some_and(|end| end <= self.len)
            );
            // SAFETY: caller guarantees the destination range is committed and writable.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len());
            }
        }
    }

    impl Drop for VirtualRegion {
        fn drop(&mut self) {
            // SAFETY: `self.ptr` was allocated with `VirtualAlloc(..., MEM_RESERVE, ...)` and not yet freed.
            unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        }
    }

    /// RAII guard releasing a console COM resource so `STAGING` entries are cleaned up on drop.
    struct ResourceGuard(This);

    impl Drop for ResourceGuard {
        fn drop(&mut self) {
            type Release = extern "system" fn(This) -> u32;
            crate::object::slot::<Release>(self.0, 0x10)(self.0);
        }
    }

    #[test]
    fn partial_residency_reads_only_mapped_regions_and_detects_new_mips() {
        const TILE: usize = 64 * 1024;
        const TOTAL: usize = 256 * 1024;
        let region = VirtualRegion::reserve(TOTAL);
        let mut dst = vec![0xAAu8; TOTAL];

        assert_eq!(read_resident(region.ptr, &mut dst), Residency::Unmapped);

        region.commit(0, TILE);
        let tile0: Vec<u8> = (0..TILE)
            .map(|i| ((i * 37 + 11) & 0xff) as u8 | 1)
            .collect();
        // SAFETY: `[0..TILE)` was committed above.
        unsafe { region.write_slice(0, &tile0) };

        assert_eq!(
            read_resident(region.ptr, &mut dst),
            Residency::Partial(vec![(0, TILE)])
        );
        assert_eq!(dst[..TILE], tile0[..]);
        let (hash0, bytes0) = hash_intervals(&dst, &[(0, TILE)]);
        assert_eq!(bytes0, TILE);

        // Mutating unmapped tail bytes in `dst` must not affect the resident hash.
        dst[TILE..].fill(0x55);
        assert_eq!(
            read_resident(region.ptr, &mut dst),
            Residency::Partial(vec![(0, TILE)])
        );
        let (hash0_repeat, bytes0_repeat) = hash_intervals(&dst, &[(0, TILE)]);
        assert_eq!((hash0_repeat, bytes0_repeat), (hash0, TILE));

        // Commit a disjoint second 64 KiB tile at `[128 KiB..192 KiB)`.
        region.commit(2 * TILE, TILE);
        let tile2: Vec<u8> = (0..TILE)
            .map(|i| ((i * 53 + 29) & 0xff) as u8 | 1)
            .collect();
        // SAFETY: `[2 * TILE..3 * TILE)` was committed above.
        unsafe { region.write_slice(2 * TILE, &tile2) };

        let res = read_resident(region.ptr, &mut dst);
        assert_eq!(
            res,
            Residency::Partial(vec![(0, TILE), (2 * TILE, 3 * TILE)])
        );
        assert_eq!(dst[..TILE], tile0[..]);
        assert_eq!(dst[2 * TILE..3 * TILE], tile2[..]);
        let (hash_two, bytes_two) = hash_intervals(&dst, &[(0, TILE), (2 * TILE, 3 * TILE)]);
        assert_eq!(bytes_two, 2 * TILE);
        assert_ne!(hash_two, hash0);

        // Decommit the second tile while leaving junk across unmapped gaps; `zero_gaps` must clear it.
        region.decommit(2 * TILE, TILE);
        dst[TILE..].fill(0xCC);
        let res_after_evict = read_resident(region.ptr, &mut dst);
        assert_eq!(res_after_evict, Residency::Partial(vec![(0, TILE)]));
        let Residency::Partial(intervals) = res_after_evict else {
            unreachable!();
        };
        let (hash_after_evict, bytes_after_evict) = hash_intervals(&dst, &intervals);
        assert_eq!((hash_after_evict, bytes_after_evict), (hash0, TILE));
        zero_gaps(&mut dst, &intervals);
        assert_eq!(dst[..TILE], tile0[..]);
        assert!(dst[TILE..].iter().all(|&b| b == 0));

        // Committing the remaining gaps yields `Residency::Full`.
        region.commit(TILE, TILE);
        region.commit(2 * TILE, 2 * TILE);
        assert_eq!(read_resident(region.ptr, &mut dst), Residency::Full);
    }

    fn readback_mip(
        host: &crate::host::Host,
        resource: This,
        host_tex: &ID3D12Resource,
        subresource: u32,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let row_pitch = width * 4;
        assert_eq!(row_pitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT, 0);
        let byte_len = row_pitch as usize * height as usize;

        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: byte_len as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a D3D12 readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();

        let back = host.new_list().unwrap();
        crate::list::ensure_state(&back, resource, 0x800);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: width,
                        Height: height,
                        Depth: 1,
                        RowPitch: row_pitch,
                    },
                },
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(host_tex) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource,
            },
        };
        // SAFETY: `back.list` is open and both resources are live.
        unsafe { back.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        back.close();
        host.execute(&[&back]);

        let mut mapped = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: byte_len,
        };
        // SAFETY: `host.execute` waited for GPU completion and the buffer spans `byte_len` bytes.
        unsafe {
            readback.Map(0, Some(&range), Some(&mut mapped)).unwrap();
            let out = std::slice::from_raw_parts(mapped.cast::<u8>(), byte_len).to_vec();
            readback.Unmap(0, None);
            out
        }
    }

    #[test]
    fn partially_resident_mip_tail_uploads_to_host_texture() {
        let Some(host) = crate::host::host() else {
            return;
        };
        use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};

        let tex = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            width: 256,
            height: 256,
            depth_or_array: 1,
            mips: 3,
            format: DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let layout = &tex.planes[0].layout;
        assert_eq!(layout.first_tail_mip, 2);
        assert_eq!(layout.mips[2].offset, 0);
        assert!(layout.mips[2].tail.is_some());
        assert_eq!(layout.mips[1].offset, 65536);
        assert_eq!(layout.mips[0].offset, 131072);
        assert_eq!(layout.size, 393216);

        let region = VirtualRegion::reserve(layout.size as usize);
        region.commit(0, 65536);

        let expected_mip2: Vec<u8> = (0..64 * 64 * 4)
            .map(|i| (((i * 13 + 7) % 251) as u8) | 1)
            .collect();
        let mut tiled = vec![0u8; layout.size as usize];
        layout
            .tile(&mut tiled, 2, 0, &expected_mip2, 256, 64 * 256)
            .unwrap();
        assert!(tiled[65536..].iter().all(|&b| b == 0));
        // SAFETY: tile 0 `[0..65536)` was committed above.
        unsafe { region.write_slice(0, &tiled[..65536]) };
        assert!(!committed(region.ptr, layout.size as usize));

        const _: () =
            assert!(size_of::<D3D12_RESOURCE_DESC>() == size_of::<crate::resource::ResourceDesc>());
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 3,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT(0x109),
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut resource = std::ptr::null();
        let hr = crate::resource::create_at_address(
            std::ptr::null(),
            region.ptr as u64,
            (&raw const desc).cast(),
            0,
            std::ptr::null(),
            &crate::resource::PAGEABLE_IIDS[0],
            &mut resource,
        );
        assert_eq!(hr, crate::object::S_OK);
        let resource_guard = ResourceGuard(resource);
        let host_tex = crate::resource::host_resource(resource_guard.0).unwrap();

        let list = host.new_list().unwrap();
        upload(&list, resource_guard.0, host_tex);
        list.close();
        host.execute(&[&list]);

        assert_eq!(
            readback_mip(host, resource_guard.0, host_tex, 2, 64, 64),
            expected_mip2
        );
        assert!(
            readback_mip(host, resource_guard.0, host_tex, 1, 128, 128)
                .iter()
                .all(|&b| b == 0)
        );

        // Stream in mip 1 while leaving mip 0 `[131072..393216)` unmapped.
        region.commit(65536, 65536);
        let expected_mip1: Vec<u8> = (0..128 * 128 * 4)
            .map(|i| (((i * 29 + 19) % 241) as u8) | 1)
            .collect();
        tiled.fill(0);
        layout
            .tile(&mut tiled, 1, 0, &expected_mip1, 512, 128 * 512)
            .unwrap();
        // SAFETY: tile 1 `[65536..131072)` was committed above.
        unsafe { region.write_slice(65536, &tiled[65536..131072]) };
        assert!(!committed(region.ptr, layout.size as usize));

        list.reset();
        upload(&list, resource_guard.0, host_tex);
        list.close();
        host.execute(&[&list]);

        assert_eq!(
            readback_mip(host, resource_guard.0, host_tex, 1, 128, 128),
            expected_mip1
        );
        assert_eq!(
            readback_mip(host, resource_guard.0, host_tex, 2, 64, 64),
            expected_mip2
        );

        // Evict mip 1: `zero_gaps` must clear stale cached bytes so host mip 1 becomes zeroed.
        region.decommit(65536, 65536);
        list.reset();
        upload(&list, resource_guard.0, host_tex);
        list.close();
        host.execute(&[&list]);

        assert!(
            readback_mip(host, resource_guard.0, host_tex, 1, 128, 128)
                .iter()
                .all(|&b| b == 0)
        );
        assert_eq!(
            readback_mip(host, resource_guard.0, host_tex, 2, 64, 64),
            expected_mip2
        );
    }

    #[test]
    fn upload_slice_stops_at_the_end_of_the_buffer() {
        // 10x6 RGBA8 with two mips. Mip 1 has three 20-byte rows at pitch 256.
        let mip = |offset, width, height| D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: offset,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Width: width,
                Height: height,
                Depth: 1,
                RowPitch: 256,
                ..Default::default()
            },
        };
        let total = 1536 + 2 * 256 + 20;
        assert_eq!(footprint_len(&mip(0, 10, 6), 6, total), 1536);
        assert_eq!(footprint_len(&mip(1536, 5, 3), 3, total), 532);
    }

    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};

    #[test]
    fn detile_widened_packed16_expands_into_rgba8_staging() {
        for format in crate::resource::PACKED_16 {
            let desc = TextureDesc {
                dimension: Dimension::Tex2d,
                width: 17,
                height: 9,
                depth_or_array: 1,
                mips: 2,
                format,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            };
            let tex = Texture::new(&desc).unwrap();
            let layout = &tex.planes[0].layout;
            let mut tiled = vec![0u8; layout.size as usize];
            let mut expected_mips = Vec::new();

            for mip in 0..2u32 {
                let m = layout.mips[mip as usize];
                let row_bytes = m.width as usize * 2;
                let mut src_mip = vec![0u8; row_bytes * m.height as usize];
                for y in 0..m.height as usize {
                    for x in 0..m.width as usize {
                        let px = ((mip as u16 + 1) * 0x1111)
                            .wrapping_add((y as u16) * 0x0103 + x as u16);
                        let at = y * row_bytes + x * 2;
                        src_mip[at..at + 2].copy_from_slice(&px.to_le_bytes());
                    }
                }
                layout
                    .tile(&mut tiled, mip, 0, &src_mip, row_bytes, src_mip.len())
                    .unwrap();
                expected_mips.push(src_mip);
            }

            let mut packed = Vec::new();
            for (mip, src_mip) in expected_mips.iter().enumerate() {
                let m = layout.mips[mip];
                let width = m.width as usize;
                let rows = m.height as usize;
                let row_pitch = (width * 4).next_multiple_of(256);
                let exact_len = row_pitch * (rows - 1) + width * 4;
                let mut staging = vec![0x5au8; exact_len];
                detile_subresource(
                    layout,
                    &tiled,
                    mip as u32,
                    0,
                    &mut staging,
                    row_pitch,
                    rows,
                    1,
                    Some(format),
                    &mut packed,
                )
                .unwrap();
                for y in 0..rows {
                    let mut expected = vec![0u8; width * 4];
                    crate::resource::expand_packed16(
                        format,
                        &src_mip[y * width * 2..][..width * 2],
                        &mut expected,
                    );
                    assert_eq!(
                        &staging[y * row_pitch..][..width * 4],
                        &expected[..],
                        "format {format} mip {mip} row {y}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod host_written_tests {
    use super::*;
    use crate::object::{HRESULT, S_OK, guid, slot};
    use crate::resource::{
        buffer_bytes, host_resource, host_transition, test_buffer, test_texture_at, texture_layout,
        texture_memory,
    };

    type Release = extern "system" fn(This) -> u32;

    const RGBA8: u32 = 28;
    const W: u32 = 4;
    const H: u32 = 4;
    const PITCH: u32 = 256;

    #[repr(C)]
    struct Location {
        resource: This,
        kind: u32,
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    }

    fn rows_footprint() -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(RGBA8 as i32),
                Width: W,
                Height: H,
                Depth: 1,
                RowPitch: PITCH,
            },
        }
    }

    fn pattern(seed: u8) -> Vec<u8> {
        let mut linear = vec![0u8; (H * PITCH) as usize];
        for row in 0..H as usize {
            for col in 0..(W as usize * 4) {
                linear[row * PITCH as usize + col] = seed.wrapping_add((row * 17 + col * 3) as u8);
            }
        }
        linear
    }

    fn write_tiled(resource: This, mem: *mut u8, size: usize, linear: &[u8]) {
        let (_, desc) = texture_memory(resource).expect("texture memory");
        let layout = texture_layout(&desc).expect("texture layout").planes[0]
            .layout
            .clone();
        assert_eq!(layout.size as usize, size);
        let mut tiled = vec![0u8; size];
        layout
            .tile(
                &mut tiled,
                0,
                0,
                linear,
                PITCH as usize,
                (H * PITCH) as usize,
            )
            .expect("tile");
        // SAFETY: `mem` points to at least `size` writable bytes owned by the test.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), mem, size) };
    }

    fn record_buffer_to_texture(list: This, dst: This, linear: &[u8]) -> This {
        let src = test_buffer(linear.len() as u64);
        let (data, _) = buffer_bytes(src).expect("buffer bytes");
        // SAFETY: `src` is a committed buffer of `linear.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(linear.as_ptr(), data, linear.len()) };

        let rows = rows_footprint();
        let to = Location {
            resource: dst,
            kind: 0,
            footprint: rows,
        };
        let from = Location {
            resource: src,
            kind: 1,
            footprint: rows,
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, std::ptr::null());
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        src
    }

    fn copy_buffer_to_texture(queue: This, allocator: This, list: This, dst: This, linear: &[u8]) {
        let src = record_buffer_to_texture(list, dst, linear);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_eq!(
            slot::<extern "system" fn(This, This, This) -> HRESULT>(list, 0x50)(
                list,
                allocator,
                std::ptr::null(),
            ),
            S_OK
        );
        assert_eq!(slot::<Release>(src, 0x10)(src), 0);
    }

    fn run_upload(host: &crate::host::Host, resource: This) {
        let list = host.new_list().expect("new host list");
        let tex = host_resource(resource).expect("host texture");
        upload(&list, resource, tex);
        list.close();
        host.execute(&[&list]);
    }

    fn read_back(host: &crate::host::Host, resource: This) -> Vec<u8> {
        use windows::core::Interface;
        let bytes = (H * PITCH) as u64;
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .expect("readback buffer");
        }
        let readback = readback.expect("readback");
        let host_image = host_resource(resource).expect("host texture");
        let back = host.new_list().expect("readback list");
        const COPY_SOURCE: u32 = 0x800;
        let before = host_transition(resource, COPY_SOURCE).map(|(_, b)| b);
        let rows = rows_footprint();
        let out_to = Location {
            resource: readback.as_raw().cast_const().cast(),
            kind: 1,
            footprint: rows,
        };
        let out_from = Location {
            resource: host_image.as_raw().cast_const().cast(),
            kind: 0,
            footprint: rows,
        };
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            if let Some(before) = before {
                back.list.ResourceBarrier(&[crate::host::transition(
                    host_image,
                    D3D12_RESOURCE_STATES(before as i32),
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            back.list.CopyTextureRegion(
                (&raw const out_to).cast(),
                0,
                0,
                0,
                (&raw const out_from).cast(),
                None,
            );
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (`execute` waits); the buffer has `bytes` bytes.
        unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: bytes as usize,
                    }),
                    Some(&mut mapped),
                )
                .expect("map readback");
            let slice = std::slice::from_raw_parts(mapped.cast::<u8>(), bytes as usize).to_vec();
            readback.Unmap(0, None);
            slice
        }
    }

    fn assert_rows_eq(actual: &[u8], expected: &[u8], label: &str) {
        for row in 0..H as usize {
            let at = row * PITCH as usize;
            let row_bytes = W as usize * 4;
            assert_eq!(
                &actual[at..at + row_bytes],
                &expected[at..at + row_bytes],
                "{label}: row {row}"
            );
        }
    }

    fn make_queue_and_list() -> (This, This, This) {
        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) =
            (std::ptr::null(), std::ptr::null(), std::ptr::null());
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(
                std::ptr::null(),
                queue_desc.as_ptr().cast(),
                &queue_iid,
                &mut queue,
            ),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(std::ptr::null(), 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(
                std::ptr::null(),
                0,
                0,
                allocator,
                std::ptr::null(),
                &list_iid,
                &mut list,
            ),
            S_OK
        );
        (queue, allocator, list)
    }

    /// A write-watched texture that receives a GPU copy keeps the copied
    /// pixels while untouched, then resumes detiling once the title writes
    /// its memory again (issue #758).
    #[test]
    fn host_written_watched_texture_uploads_after_title_write() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some(host) = host() else {
            return;
        };
        let alloc_size = 64 << 10;
        // SAFETY: fresh write-watched allocation, freed after the resource is released.
        let mem = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                alloc_size,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!mem.is_null());

        let image = test_texture_at(mem as u64, u64::from(W), H, RGBA8, 0);
        let (_, desc) = texture_memory(image).expect("texture memory");
        let layout_size = texture_layout(&desc).expect("layout").planes[0].layout.size as usize;
        assert!(layout_size <= alloc_size);

        let (queue, allocator, list) = make_queue_and_list();

        // Title writes pattern A *before* the GPU copy: must not falsely
        // invalidate the GPU copy that comes next.
        let pat_a = pattern(0x10);
        let pat_b = pattern(0x50);
        let pat_c = pattern(0x90);
        let pat_d = pattern(0xc0);
        let pat_e = pattern(0xe0);

        write_tiled(image, mem, layout_size, &pat_a);
        copy_buffer_to_texture(queue, allocator, list, image, &pat_b);

        // No title write since the GPU copy: upload must keep pattern B.
        run_upload(host, image);
        assert_rows_eq(
            &read_back(host, image),
            &pat_b,
            "keeps GPU copy before write",
        );

        // Title writes pattern C into the texture's memory: upload must detile C.
        write_tiled(image, mem, layout_size, &pat_c);
        run_upload(host, image);
        assert_rows_eq(&read_back(host, image), &pat_c, "uploads after title write");

        // Second GPU copy -> keep D -> later title write -> upload E (with
        // Staging already allocated).
        copy_buffer_to_texture(queue, allocator, list, image, &pat_d);
        run_upload(host, image);
        assert_rows_eq(&read_back(host, image), &pat_d, "keeps second GPU copy");

        write_tiled(image, mem, layout_size, &pat_e);
        run_upload(host, image);
        assert_rows_eq(
            &read_back(host, image),
            &pat_e,
            "uploads after second title write",
        );

        assert_eq!(slot::<Release>(image, 0x10)(image), 0);
        assert_eq!(slot::<Release>(list, 0x10)(list), 0);
        assert_eq!(slot::<Release>(allocator, 0x10)(allocator), 0);
        assert_eq!(slot::<Release>(queue, 0x10)(queue), 0);
        // SAFETY: `image` is released; freeing our allocation.
        unsafe { VirtualFree(mem.cast(), 0, MEM_RELEASE) };
    }

    /// An unwatched texture (physical-page view / content-hashed path) also
    /// keeps GPU-copied pixels until its console memory changes, then uploads
    /// the new contents (issue #758).
    #[test]
    fn host_written_unwatched_texture_uploads_after_title_write() {
        let Some(host) = host() else {
            return;
        };
        let mut backing = vec![0u8; 64 << 10];
        let mem = backing.as_mut_ptr();
        let image = test_texture_at(mem as u64, u64::from(W), H, RGBA8, 0);
        let (_, desc) = texture_memory(image).expect("texture memory");
        let layout_size = texture_layout(&desc).expect("layout").planes[0].layout.size as usize;
        assert!(layout_size <= backing.len());

        let (queue, allocator, list) = make_queue_and_list();
        let pat_a = pattern(0x21);
        let pat_b = pattern(0x61);
        let pat_c = pattern(0xa1);

        // Initial CPU upload creates Staging, then GPU copy overwrites host texture.
        write_tiled(image, mem, layout_size, &pat_a);
        run_upload(host, image);
        assert_rows_eq(&read_back(host, image), &pat_a, "initial CPU upload");

        copy_buffer_to_texture(queue, allocator, list, image, &pat_b);
        run_upload(host, image);
        assert_rows_eq(
            &read_back(host, image),
            &pat_b,
            "keeps GPU copy while unchanged",
        );

        // Later title write changes content hash: upload must detile pattern C.
        write_tiled(image, mem, layout_size, &pat_c);
        run_upload(host, image);
        assert_rows_eq(
            &read_back(host, image),
            &pat_c,
            "uploads after unwatched write",
        );

        assert_eq!(slot::<Release>(image, 0x10)(image), 0);
        assert_eq!(slot::<Release>(list, 0x10)(list), 0);
        assert_eq!(slot::<Release>(allocator, 0x10)(allocator), 0);
        assert_eq!(slot::<Release>(queue, 0x10)(queue), 0);
    }

    /// A recorded buffer-to-texture copy that is discarded or reset without
    /// ever executing must not mark the texture host-written or suppress
    /// uploading new title pixels (both unwatched and write-watched memory).
    #[test]
    fn unsubmitted_texture_copy_does_not_suppress_title_upload() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some(host) = host() else {
            return;
        };
        for watched in [false, true] {
            let mut backing = vec![0u8; 64 << 10];
            let mem = if watched {
                // SAFETY: fresh write-watched allocation, freed after release.
                let p = unsafe {
                    VirtualAlloc(
                        std::ptr::null(),
                        64 << 10,
                        MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                        PAGE_READWRITE,
                    )
                }
                .cast::<u8>();
                assert!(!p.is_null());
                p
            } else {
                backing.as_mut_ptr()
            };
            let image = test_texture_at(mem as u64, u64::from(W), H, RGBA8, 0);
            let (_, desc) = texture_memory(image).expect("texture memory");
            let layout_size = texture_layout(&desc).expect("layout").planes[0].layout.size as usize;
            assert!(layout_size <= 64 << 10);

            let old = pattern(0x10);
            let title = pattern(0x50);
            let unused_copy = pattern(0x90);
            let title_after_reset = pattern(0xd0);

            write_tiled(image, mem, layout_size, &old);
            run_upload(host, image);

            // Title writes new pixels, then records and discards a copy list
            // without submitting it.
            write_tiled(image, mem, layout_size, &title);
            let (queue, allocator, list) = make_queue_and_list();
            let src = record_buffer_to_texture(list, image, &unused_copy);
            assert_eq!(slot::<Release>(list, 0x10)(list), 0);
            assert_eq!(slot::<Release>(src, 0x10)(src), 0);
            assert_eq!(slot::<Release>(allocator, 0x10)(allocator), 0);
            assert_eq!(slot::<Release>(queue, 0x10)(queue), 0);

            run_upload(host, image);
            assert_rows_eq(
                &read_back(host, image),
                &title,
                if watched {
                    "discarded copy (watched)"
                } else {
                    "discarded copy (unwatched)"
                },
            );

            // Recording and resetting a list without submitting also must not
            // mark the texture host-written.
            write_tiled(image, mem, layout_size, &title_after_reset);
            let (queue, allocator, list) = make_queue_and_list();
            let src = record_buffer_to_texture(list, image, &unused_copy);
            assert_eq!(
                slot::<extern "system" fn(This, This, This) -> HRESULT>(list, 0x50)(
                    list,
                    allocator,
                    std::ptr::null(),
                ),
                S_OK
            );
            run_upload(host, image);
            assert_rows_eq(
                &read_back(host, image),
                &title_after_reset,
                if watched {
                    "reset copy (watched)"
                } else {
                    "reset copy (unwatched)"
                },
            );

            assert_eq!(slot::<Release>(src, 0x10)(src), 0);
            assert_eq!(slot::<Release>(list, 0x10)(list), 0);
            assert_eq!(slot::<Release>(allocator, 0x10)(allocator), 0);
            assert_eq!(slot::<Release>(queue, 0x10)(queue), 0);
            assert_eq!(slot::<Release>(image, 0x10)(image), 0);
            if watched {
                // SAFETY: `image` is released; freeing our allocation.
                unsafe { VirtualFree(mem.cast(), 0, MEM_RELEASE) };
            }
        }
    }

    /// A texture written by a GPU copy and then sampled in the same command
    /// list must keep the GPU-copied pixels instead of overwriting them with a
    /// staging upload, and still resume uploading after a later title write.
    #[test]
    fn same_list_upload_after_copy_keeps_gpu_copy() {
        let Some(host) = host() else {
            return;
        };
        let mut backing = vec![0u8; 64 << 10];
        let mem = backing.as_mut_ptr();
        let image = test_texture_at(mem as u64, u64::from(W), H, RGBA8, 0);
        let (_, desc) = texture_memory(image).expect("texture memory");
        let layout_size = texture_layout(&desc).expect("layout").planes[0].layout.size as usize;

        let initial = pattern(0x10);
        let copied = pattern(0x70);
        let later_title = pattern(0xb0);
        write_tiled(image, mem, layout_size, &initial);
        run_upload(host, image);

        let (queue, allocator, list) = make_queue_and_list();
        let src = test_buffer(copied.len() as u64);
        let (data, _) = buffer_bytes(src).expect("buffer bytes");
        // SAFETY: `src` is a committed buffer of `copied.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(copied.as_ptr(), data, copied.len()) };

        let rows = rows_footprint();
        let to = Location {
            resource: image,
            kind: 0,
            footprint: rows,
        };
        let from = Location {
            resource: src,
            kind: 1,
            footprint: rows,
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, std::ptr::null());
        // Simulate sampling the texture later in the same command list.
        let host_list = crate::list::host_list(list).expect("host list");
        let host_tex = host_resource(image).expect("host texture");
        upload(host_list, image, host_tex);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert_rows_eq(
            &read_back(host, image),
            &copied,
            "same-list copy then upload",
        );

        write_tiled(image, mem, layout_size, &later_title);
        run_upload(host, image);
        assert_rows_eq(&read_back(host, image), &later_title, "later title write");

        assert_eq!(slot::<Release>(src, 0x10)(src), 0);
        assert_eq!(slot::<Release>(list, 0x10)(list), 0);
        assert_eq!(slot::<Release>(allocator, 0x10)(allocator), 0);
        assert_eq!(slot::<Release>(queue, 0x10)(queue), 0);
        assert_eq!(slot::<Release>(image, 0x10)(image), 0);
    }

    /// Releasing a texture after recording a copy but before submitting its
    /// list must not leave a stale `HOST_WRITTEN` entry behind.
    #[test]
    fn released_texture_pending_host_write_is_ignored_on_submit() {
        let Some(_) = host() else {
            return;
        };
        let mut backing = vec![0u8; 64 << 10];
        let image = test_texture_at(backing.as_mut_ptr() as u64, u64::from(W), H, RGBA8, 0);
        let (queue, allocator, list) = make_queue_and_list();
        let src = record_buffer_to_texture(list, image, &pattern(0x33));
        // Release the title resource before executing the list (the list still
        // keeps the underlying host ID3D12Resource alive via `states`).
        assert_eq!(slot::<Release>(image, 0x10)(image), 0);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);
        assert!(!still_host_written(image));

        assert_eq!(slot::<Release>(src, 0x10)(src), 0);
        assert_eq!(slot::<Release>(list, 0x10)(list), 0);
        assert_eq!(slot::<Release>(allocator, 0x10)(allocator), 0);
        assert_eq!(slot::<Release>(queue, 0x10)(queue), 0);
    }

    /// Non-title-written textures (such as render targets) stay sticky in
    /// `HOST_WRITTEN` once submitted without hashing or write-watch queries,
    /// and clear on release.
    #[test]
    fn host_written_render_target_stays_sticky_until_forgotten() {
        let Some(host) = host() else {
            return;
        };
        let mut backing = vec![0u8; 64 << 10];
        const ALLOW_RENDER_TARGET: u32 = 1;
        let rt = test_texture_at(
            backing.as_mut_ptr() as u64,
            u64::from(W),
            H,
            RGBA8,
            ALLOW_RENDER_TARGET,
        );
        let list = host.new_list().expect("new host list");
        host_written(&list, rt);
        assert!(!still_host_written(rt), "not committed until submitted");
        list.close();
        host.execute(&[&list]);
        assert!(still_host_written(rt));
        backing[0] ^= 0xff;
        assert!(still_host_written(rt));
        assert_eq!(slot::<Release>(rt, 0x10)(rt), 0);
        assert!(!still_host_written(rt));
    }
}
