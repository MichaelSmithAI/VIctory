//! Opt-in, list-local descriptor admission diagnostics. Does not change allocation.
use std::collections::HashMap;
use std::sync::{OnceLock, atomic::{AtomicU64, Ordering}};

type Key = (u64, u64); // Exactly State::fresh_tables' existing cache key.

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_DESCRIPTOR_ADMISSION").is_some())
}

#[derive(Default)]
pub(super) struct Census {
    id: u64,
    active: bool,
    new: Vec<(Key, u32)>,
    hits: Vec<Key>,
    pending: HashMap<Key, u32>,
    accepted: u64,
    rejected: u64,
    new_slots: u64,
    rejected_origin_slots: u64,
    recovered_slots: u64,
    pending_slots: u64,
    peak_pending_slots: u64,
    failed_requests: u64,
    failed_request_slots: u64,
    failed_with_prior_pending: u64,
    failed_widths: [u64; 8],
}

impl Census {
    pub(super) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let mut census = Self::default();
        census.id = NEXT.fetch_add(1, Ordering::Relaxed);
        census
    }

    pub(super) fn begin(&mut self) {
        debug_assert!(!self.active);
        self.active = true;
        self.new.clear();
        self.hits.clear();
    }

    pub(super) fn reserve(&mut self, key: Key, width: u32) {
        if self.active { self.new.push((key, width)); }
    }

    pub(super) fn hit(&mut self, key: Key) {
        if self.active && self.pending.contains_key(&key) { self.hits.push(key); }
    }

    pub(super) fn miss(&mut self, width: u32) {
        if !self.active { return; }
        self.failed_requests += 1;
        self.failed_request_slots += u64::from(width);
        if self.pending_slots != 0 { self.failed_with_prior_pending += 1; }
        let bin = match width {
            0..=1 => 0, 2..=4 => 1, 5..=16 => 2, 17..=64 => 3,
            65..=256 => 4, 257..=1024 => 5, 1025..=4096 => 6, _ => 7,
        };
        self.failed_widths[bin] += 1;
    }

    pub(super) fn finish(&mut self, accepted: bool) {
        debug_assert!(self.active);
        self.active = false;
        self.new_slots += self.new.iter().map(|&(_, n)| u64::from(n)).sum::<u64>();
        if accepted {
            self.accepted += 1;
            for key in self.hits.drain(..) {
                if let Some(n) = self.pending.remove(&key) {
                    self.recovered_slots += u64::from(n);
                    self.pending_slots -= u64::from(n);
                }
            }
        } else {
            self.rejected += 1;
            for (key, n) in self.new.drain(..) {
                // New allocations are inserted in fresh_tables once per key.
                debug_assert!(!self.pending.contains_key(&key));
                self.pending.insert(key, n);
                self.rejected_origin_slots += u64::from(n);
                self.pending_slots += u64::from(n);
            }
            self.peak_pending_slots = self.peak_pending_slots.max(self.pending_slots);
        }
        self.new.clear();
        self.hits.clear();
        debug_assert_eq!(self.rejected_origin_slots, self.recovered_slots + self.pending_slots);
        if !accepted && self.rejected.is_power_of_two() { self.report("pressure"); }
    }

    fn report(&self, phase: &str) {
        tracing::info!(
            phase, list_recording = self.id, present = crate::queue::presented(),
            accepted_binds = self.accepted, rejected_binds = self.rejected,
            new_slots = self.new_slots, rejected_origin_slots = self.rejected_origin_slots,
            recovered_slots = self.recovered_slots, pending_slots = self.pending_slots,
            peak_pending_slots = self.peak_pending_slots,
            failed_requests = self.failed_requests, failed_request_slots = self.failed_request_slots,
            failed_with_prior_pending = self.failed_with_prior_pending,
            failed_widths = ?self.failed_widths, active_bind = self.active,
            "descriptor admission census"
        );
    }
}

impl Drop for Census {
    fn drop(&mut self) {
        if self.new_slots != 0 || self.rejected != 0 || self.active {
            self.report("recording_end");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejection_then_reuse_counts_once() {
        let mut c = Census::default();
        c.begin(); c.reserve((1, 2), 4); c.miss(3); c.finish(false);
        assert_eq!((c.new_slots, c.pending_slots, c.recovered_slots), (4, 4, 0));
        c.begin(); c.hit((1, 2)); c.miss(7); c.finish(false);
        assert_eq!((c.pending_slots, c.recovered_slots, c.failed_with_prior_pending), (4, 0, 1));
        c.begin(); c.hit((1, 2)); c.hit((1, 2)); c.finish(true);
        assert_eq!((c.pending_slots, c.recovered_slots, c.peak_pending_slots), (0, 4, 4));
    }
    #[test]
    fn distinct_records_and_accepted_new_slots() {
        let mut c = Census::default();
        c.begin(); c.reserve((1, 1), 3); c.finish(true);
        c.begin(); c.reserve((1, 2), 2); c.reserve((1, 3), 1); c.miss(17); c.finish(false);
        c.begin(); c.hit((1, 2)); c.reserve((2, 3), 5); c.finish(true);
        assert_eq!((c.new_slots, c.rejected_origin_slots, c.recovered_slots, c.pending_slots), (11, 3, 2, 1));
        assert_eq!(c.failed_widths[3], 1);
    }
}
