"""
Merge the 3 separate per-feature outputs from the parallel-Kaggle-notebook
split (kaggle_pcfg_only.py, kaggle_modelblocks_only.py, kaggle_depparse_only.py)
into one results/syntactic_surprisal.json, in the same
{variant_id: {surprisal_pcfg, surprisal_modelblocks, surprisal_depparse}}
schema assemble_features.py already merges from.

Download each notebook's Cell 8 output and place here before running (one
model/ subfolder per feature, same convention as models/adaptive_lstm,
models/xglm, models/gemma-3b):
  pcfg_surprisals_new.json        -> models/pcfg_parser/pcfg_surprisals_new.json
  modelblocks_surprisals_new.json -> models/modelblocks/modelblocks_surprisals_new.json
  depparse_surprisals_new.json    -> models/dependency_parser/depparse_surprisals_new.json
                                      (produced by merge_depparse_slices.py from the 6 slices)

Run:
    python scripts/merge_three_checkpoints.py
Then:
    python scripts/assemble_features.py
"""

import json
from pathlib import Path

SOURCES = {
    'surprisal_pcfg': 'models/pcfg_parser/pcfg_surprisals_new.json',
    'surprisal_modelblocks': 'models/modelblocks/modelblocks_surprisals_new.json',
    'surprisal_depparse': 'models/dependency_parser/depparse_surprisals_new.json',
}
OUTPUT = 'results/syntactic_surprisal.json'


def main():
    merged = {}
    for col, path in SOURCES.items():
        if not Path(path).exists():
            print(f'SKIP {col}: {path} not found')
            continue
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        real = sum(1 for v in data.values() if v is not None)
        print(f'{col}: loaded {len(data):,} entries from {path} ({real:,} real)')
        for vid, val in data.items():
            merged.setdefault(str(vid), {})[col] = val

    for vid in merged:
        for col in SOURCES:
            merged[vid].setdefault(col, None)

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False)
    print(f'\nSaved: {OUTPUT}')
    print(f'Total variant_ids: {len(merged):,}')
    for col in SOURCES:
        real = sum(1 for v in merged.values() if v.get(col) is not None)
        print(f'  {col}: {real:,}/{len(merged):,} real')


if __name__ == '__main__':
    main()
