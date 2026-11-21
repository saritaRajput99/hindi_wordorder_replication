"""
One-off data fix: relabel stale order_type='Other' rows.

These 5 rows (all is_original=1, i.e. the corpus reference sentence itself)
were written by an older version of VariantGenerator.classify_order() before
it was fixed. Running the CURRENT classify_order() on their stored
deprel_sequence shows all 5 are actually 'Canonical' (verified separately).

The paper defines only Canonical / O-fronted (DO-fronted / IO-fronted) —
there is no "Other" category, so no live code should ever emit it under
normal data; these were pure staleness.

Patches (each with a .bak_pre_2026-08-02fix backup, same convention as the
2026-07-11 O-fronted -> DO-fronted fix):
  data/processed/variants/variants.csv
  data/processed/features/features.csv
  models/adaptive_lstm/final_var_score.csv
"""
import os
import shutil
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = [
    os.path.join(BASE, 'data', 'processed', 'variants', 'variants.csv'),
    os.path.join(BASE, 'data', 'processed', 'features', 'features.csv'),
    os.path.join(BASE, 'models', 'adaptive_lstm', 'final_var_score.csv'),
]

BACKUP_SUFFIX = '.bak_pre_2026-08-02fix'

BAD_VARIANT_IDS = {
    'train-s208_var_000',
    'train-s421_var_000',
    'train-s889_var_000',
    'train-s1198_var_000',
    'train-s2292_var_000',
}

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
    id_col = 'variant_id' if 'variant_id' in df.columns else 'sent_id'

    mask = df[id_col].isin(BAD_VARIANT_IDS) & (df['order_type'] == 'Other')
    n = mask.sum()
    df.loc[mask, 'order_type'] = 'Canonical'

    df.to_csv(path, index=False)
    print(f'{path}: relabeled {n} row(s) Other -> Canonical')

print('\nDone.')
