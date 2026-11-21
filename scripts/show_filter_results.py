"""
Show filter results vs paper (Table 1) in a formatted table.

Run:
    python scripts/show_filter_results.py
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loading.loader import HUTBLoader
from src.preprocessing.filter import SentenceFilter
from collections import Counter

# ── Load ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ud_hindi')
loader   = HUTBLoader(DATA_DIR)
raw      = loader.load_train()
sentences = [s['tokens'] for s in raw]

# ── Filter ────────────────────────────────────────────────────────────
filt = SentenceFilter()
filtered, _ = filt.filter_with_rejection_tracking(sentences)

# ── Classify ──────────────────────────────────────────────────────────
counts = Counter()
for sent in filtered:
    root = filt.get_root(sent)
    if not root:
        counts['unknown'] += 1
        continue

    prev      = filt.preverbal_dependents(sent)
    subj      = next((t for t in prev if t['deprel'] in filt.subject_deps), None)
    obj       = next((t for t in prev if t['deprel'] in filt.object_deps), None)
    iobj_prev = next((t for t in prev if t['deprel'] in filt.indirect_object_deps), None)

    root_deps = [t for t in sent if t['head'] == root['id']]
    iobj_any  = any(t['deprel'] in (filt.indirect_object_deps | {'obl'})
                    for t in root_deps)

    if subj and obj:
        if obj['id'] < subj['id']:
            counts['O-fronted'] += 1
            if iobj_any:
                counts['DO-fronted'] += 1
        else:
            counts['Canonical'] += 1
    elif subj and iobj_prev:
        if iobj_prev['id'] < subj['id']:
            counts['O-fronted'] += 1
            counts['IO-fronted'] += 1
        else:
            counts['Canonical'] += 1
    elif subj:
        counts['Canonical'] += 1
    else:
        counts['unknown'] += 1

# ── Paper targets ─────────────────────────────────────────────────────
PAPER = {
    'Canonical' : (1763, 88.33),
    'O-fronted' : ( 233, 11.67),
    'DO-fronted': ( 133,  6.67),
    'IO-fronted': ( 101,  5.06),
}
total = len(filtered)

# ── Print rejection breakdown ─────────────────────────────────────────
print()
print('REJECTION BREAKDOWN')
print('─' * 35)
for reason, count in filt.stats['rejected'].items():
    if count > 0:
        print(f'  {reason:<22}: {count:,}')
print(f'  {"passed":<22}: {total:,}')
print(f'  {"raw total":<22}: {len(sentences):,}')

# ── Print comparison table ────────────────────────────────────────────
print()
print('WORD ORDER CATEGORIES — OURS vs PAPER (Table 1)')
w = [12, 7, 8, 7, 8, 6]
sep = '├' + '┼'.join('─'*(c+2) for c in w) + '┤'
top = '┌' + '┬'.join('─'*(c+2) for c in w) + '┐'
bot = '└' + '┴'.join('─'*(c+2) for c in w) + '┘'

def row(vals):
    return '│' + '│'.join(f' {str(v):<{w[i]}} ' for i,v in enumerate(vals)) + '│'

print(top)
print(row(['Category', 'Ours', '%', 'Paper', 'Paper%', 'Diff']))
print(sep)

for cat, (pn, pp) in PAPER.items():
    n   = counts.get(cat, 0)
    pct = n / total * 100 if total else 0
    diff = n - pn
    diff_str = f'{diff:+d}'
    print(row([cat, f'{n:,}', f'{pct:.2f}%', f'{pn:,}', f'{pp:.2f}%', diff_str]))
    print(sep)

print(row(['TOTAL', f'{total:,}', '', '1,996', '', f'{total-1996:+d}']))
print(bot)
print()
