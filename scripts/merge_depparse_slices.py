"""
Merge the 6 parallel Dep-parser slices (kaggle_depparse_only.py run 6 times
with SLICE_INDEX=0..5 across 2 Kaggle accounts) into one depparse_surprisals_new.json
— the same file merge_three_checkpoints.py expects.

Verifies full coverage (no gaps) and checks for any variant_id appearing in
more than one slice with conflicting values (shouldn't happen by construction
— slicing is done on TEXT, and all variant_ids sharing a text stay in the same
slice — but checked anyway since a correctness bug here would be easy to miss).

Download all 6 depparse_slice{N}_surprisals_new.json files to
models/dependency_parser/ before running this.

Run:
    python scripts/merge_depparse_slices.py
Then:
    python scripts/merge_three_checkpoints.py
    python scripts/assemble_features.py
"""

import json
from pathlib import Path

NUM_SLICES = 6
FEAT_CSV = 'data/processed/features/features.csv'
OUTPUT = 'models/dependency_parser/depparse_surprisals_new.json'


def main():
    import pandas as pd
    df = pd.read_csv(FEAT_CSV)
    expected_vids = set(df['variant_id'].astype(str))
    print(f'Expected variant_ids (from {FEAT_CSV}): {len(expected_vids):,}')

    merged = {}
    conflicts = []
    seen_in = {}   # vid -> which slice first set it

    for i in range(NUM_SLICES):
        path = f'models/dependency_parser/depparse_slice{i}_surprisals_new.json'
        if not Path(path).exists():
            print(f'MISSING: {path} — not all 6 slices downloaded yet!')
            continue
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        real = sum(1 for v in data.values() if v is not None)
        print(f'slice {i}: {len(data):,} variant_ids ({real:,} real) from {path}')

        for vid, val in data.items():
            if vid in merged and merged[vid] != val:
                conflicts.append((vid, seen_in[vid], merged[vid], i, val))
            else:
                merged[vid] = val
                seen_in[vid] = i

    print(f'\nTotal merged variant_ids: {len(merged):,}')

    if conflicts:
        print(f'\n!!! {len(conflicts)} CONFLICTS — same variant_id, different values across slices !!!')
        for vid, s1, v1, s2, v2 in conflicts[:10]:
            print(f'  {vid}: slice {s1}={v1}  vs  slice {s2}={v2}')
    else:
        print('No conflicts — no variant_id disagreed across slices.')

    missing = expected_vids - set(merged.keys())
    extra = set(merged.keys()) - expected_vids
    if missing:
        print(f'\n!!! GAP: {len(missing)} variant_ids from features.csv are missing from all 6 slices !!!')
        print(f'  Example missing: {list(missing)[:5]}')
    else:
        print('No gaps — every variant_id in features.csv is covered by the 6 slices.')
    if extra:
        print(f'\n(Note: {len(extra)} variant_ids in the merged output are not in features.csv — harmless, ignored downstream.)')

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False)
    real_total = sum(1 for v in merged.values() if v is not None)
    print(f'\nSaved: {OUTPUT}')
    print(f'Real scores: {real_total:,}/{len(merged):,}')


if __name__ == '__main__':
    main()
