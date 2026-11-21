"""
One-off data fix: relabel stale order_type on reference (is_original=1) rows.

A full-dataset scan comparing stored order_type against the CURRENT
VariantGenerator.classify_order() (run on the stored deprel_sequence) found
19 reference sentences stuck at 'Canonical' that should be:
  - 'IO-fronted' (18 cases): iobj precedes nsubj in the sequence
  - 'DO-fronted' (1 case):   obj precedes nsubj in the sequence
All 76,545 generated-variant rows already matched current logic (0 mismatches);
only original/reference rows were affected. Same root cause as the
2026-08-02 Other->Canonical fix and the 2026-07-11 O-fronted->DO-fronted
fix: stale labels from an older version of classify_order(), never
regenerated in the CSV.

Patches (backup suffix .bak_pre_2026-08-02fix2):
  data/processed/variants/variants.csv
  data/processed/features/features.csv
  models/adaptive_lstm/final_var_score.csv
"""
import os
import sys
import shutil
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.variant_generation.generator import VariantGenerator

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = [
    os.path.join(BASE, 'data', 'processed', 'variants', 'variants.csv'),
    os.path.join(BASE, 'data', 'processed', 'features', 'features.csv'),
    os.path.join(BASE, 'models', 'adaptive_lstm', 'final_var_score.csv'),
]

BACKUP_SUFFIX = '.bak_pre_2026-08-02fix2'

g = VariantGenerator()

def recompute(seq):
    toks = seq.split() if isinstance(seq, str) else []
    return g.classify_order(toks)

total_fixed = 0

for path in TARGETS:
    if not os.path.exists(path):
        print(f'skip (not found): {path}')
        continue

    backup_path = path + BACKUP_SUFFIX
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        print(f'backed up -> {backup_path}')
    else:
        print(f'backup already exists, skipping backup: {backup_path}')

    df = pd.read_csv(path)
    recomputed = df['deprel_sequence'].apply(recompute)
    mask = (recomputed != df['order_type']) & (df['is_original'] == 1)
    n = mask.sum()
    df.loc[mask, 'order_type'] = recomputed[mask]

    df.to_csv(path, index=False)
    print(f'{path}: relabeled {n} row(s)')
    total_fixed = max(total_fixed, n)

print(f'\nDone. {total_fixed} reference sentence(s) relabeled per file.')
