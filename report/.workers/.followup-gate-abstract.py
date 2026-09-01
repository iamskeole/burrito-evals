#!/usr/bin/env python3
"""Phase 1 QA gate: verify the Abstract edit preserved wording exactly and stayed in-bounds."""
import re, sys, difflib

R = '/home/p/code/local/burrito-evals/report/'

def norm(s):
    return re.sub(r'\s+', ' ', s).strip()

def extract(path):
    lines = open(path).read().split('\n')
    start = None
    for i, l in enumerate(lines):
        if l.startswith('## Abstract'):
            start = i
            break
    if start is None:
        return None, None, None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith('## '):
            end = j
            break
    return start, end, '\n'.join(lines[start:end])

# (a) extract after
after_start, after_end, after_sec = extract(R + 'plots.md')
if after_sec is None:
    print('GATE FAIL: Abstract heading not found in edited file')
    sys.exit(1)
open(R + '.followup-abstract.after.txt', 'w').write(after_sec + '\n')

# (b) whitespace-normalized before/after must be byte-identical
before_norm = norm(open(R + '.followup-abstract.before.txt').read())
after_norm = norm(after_sec)
if before_norm == after_norm:
    print('CHECK b OK: whitespace-normalized before/after abstract identical')
else:
    # compact word-level diff for the failure note
    b, a = before_norm.split(), after_norm.split()
    sm = difflib.SequenceMatcher(None, b, a)
    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            diffs.append(f'{tag} before[{i1}:{i2}]="{" ".join(b[i1:i2])[:40]}" after[{j1}:{j2}]="{" ".join(a[j1:j2])[:40]}"')
    print('GATE FAIL (b): wording drift — N difference locations:', len(diffs))
    for d in diffs[:6]:
        print('  ', d)
    sys.exit(2)

# (c) containment: only the Abstract section may differ
bl = open(R + '.followup-plots.before.md').read().split('\n')
al = open(R + 'plots.md').read().split('\n')
bstart, bend, _ = extract(R + '.followup-plots.before.md')
if bl[:bstart] != al[:bstart]:
    print('GATE FAIL (c): lines before Abstract section changed')
    sys.exit(3)
if bstart < len(bl) and bl[bend:] != al[after_end:]:
    print('GATE FAIL (c): lines after Abstract section changed')
    sys.exit(3)
if bstart != after_start:
    print(f'NOTE: Abstract heading moved from line {bstart+1} to {after_start+1} (should not happen)')

# paragraph count of the new abstract body (exclude heading + immediate blank)
body = after_sec.split('\n')
paras = [p for p in (x.strip() for x in body[1:]) if p]
print(f'CHECK c OK: containment verified; abstract now has {len(paras)} non-empty paragraph blocks (was 3)')
