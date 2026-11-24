#!/usr/bin/env python3
"""Count captured graphics operations targeting selected formats in the same run."""
import argparse
from collections import Counter
from pathlib import Path
import re

OP = re.compile(
    r'\bframe (\d+) op: pipeline 0x[0-9a-fA-F]+ topology \d+ '
    r'targets \[(.*?)\] depth 0x[0-9a-fA-F]+ -> (.*?) \|\| (.*)$'
)
TARGET = re.compile(r'"(0x[0-9a-fA-F]+)"')
INDIRECT = re.compile(r'^indirect kind (\d+)\b')
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')

SAFE_OUTCOMES = {'ran'} | {
    f'DROPPED: {reason}' for reason in (
        'rectangle topology',
        'pipeline filtered',
        'unsupported indirect kind',
        'wrong host pipeline kind',
        'index buffer outside title buffers',
        'no fresh descriptor table slots',
        'no bound view heap for bounded root SRV',
        'no bounded root SRV descriptor slots',
        'arguments outside title buffers',
        'count outside title buffers',
        'no host signature',
        'not bounded (no clamp, or arguments not 4-byte aligned)',
    )
}


def tally(log, capture, frame, formats):
    names = re.compile(rf'^frame-{frame}-\d+-(0x[0-9a-fA-F]+)-\d+x\d+-f(\d+)\.png$')
    targets = set()
    for path in capture.iterdir():
        match = names.fullmatch(path.name)
        if match and int(match[2]) in formats:
            targets.add(int(match[1], 16))
    if not targets:
        raise ValueError('no matching render target filenames for this frame')
    counts = Counter()
    for line in log:
        match = OP.search(ANSI.sub('', line.rstrip()))
        if not match or int(match[1]) != frame:
            continue
        if targets.isdisjoint(int(pointer, 16) for pointer in TARGET.findall(match[2])):
            continue
        kind = INDIRECT.match(match[4])
        category = f'indirect kind {kind[1]}' if kind else match[4].split()[0]
        outcome = match[3]
        if outcome not in SAFE_OUTCOMES:
            outcome = 'DROPPED: pipeline or other cause'
        counts[category, outcome] += 1
    if not counts:
        raise ValueError('no matching frame operations; verify trace logging and same-run inputs')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('capture', type=Path, help='same-run local directory of target filenames')
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--format', type=int, action='append', required=True)
    args = parser.parse_args()
    try:
        with args.log.open(encoding='utf-8', errors='replace') as log:
            counts = tally(log, args.capture, args.frame, set(args.format))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print('operation\toutcome\tcount')
    for (category, outcome), count in sorted(counts.items()):
        print(f'{category}\t{outcome}\t{count}')


if __name__ == '__main__':
    main()
