"""
Assemble the final features.csv from everything scored so far:
  base      : output/variants_final/variants_scored.csv
              (reference_id, variant_id, is_original, is_flip_pair,
               deprel_sequence, order_type, dl, text, surprisal_trigram
               [SRILM-trained], is_score)
  + surprisal_lstm   <- models/adaptive_lstm/lstm_plain_surprisals.json
  + surprisal_xglm   <- models/xglm/xglm_surprisals.json
  + surprisal_gemma  <- models/gemma-3b/gemma3_4b_surprisals.json
  + surprisal_pcfg        <- results/syntactic_surprisal.json
  + surprisal_modelblocks <- results/syntactic_surprisal.json
  + surprisal_depparse    <- results/syntactic_surprisal.json

The last 3 all come from one file, results/syntactic_surprisal.json, produced
by EITHER scripts/score_syntactic_surprisal.py (slow — hits the shared public
HF Space) OR scripts/kaggle_syntactic_surprisal.py (fast — self-hosts the same
Space's own code on Kaggle; download its kaggle_working output and place it at
results/syntactic_surprisal.json). Per Ranjan's instruction, PCFG surprisal
here means THIS tool's version, not the older local Berkeley-Parser-only
pipeline in scripts/build_pcfg_berkeley.py / models/pcfg/pcfg_surprisals.json
— that one is a different implementation and is NOT merged here.

Run:
    python scripts/assemble_features.py
"""

import json
import pandas as pd
from pathlib import Path

BASE_CSV   = 'output/variants_final/variants_scored.csv'
OUTPUT_CSV = 'data/processed/features/features.csv'

SOURCES = {
    'surprisal_lstm' : 'models/adaptive_lstm/lstm_plain_surprisals.json',
    'surprisal_xglm' : 'models/xglm/xglm_surprisals.json',
    'surprisal_gemma': 'models/gemma-3b/gemma3_4b_surprisals.json',
}

SYNTACTIC_JSON = 'results/syntactic_surprisal.json'
SYNTACTIC_COLS = ['surprisal_pcfg', 'surprisal_modelblocks', 'surprisal_depparse']


def main():
    df = pd.read_csv(BASE_CSV)
    print(f'Base: {len(df):,} rows, {len(df.columns)} columns: {list(df.columns)}')

    for col, path in SOURCES.items():
        if not Path(path).exists():
            print(f'  SKIP {col}: {path} not found')
            continue
        with open(path, encoding='utf-8') as f:
            scores = json.load(f)
        df[col] = df['variant_id'].map(scores)
        n_missing = df[col].isna().sum()
        print(f'  Merged {col} from {path}: '
              f'{len(df) - n_missing:,}/{len(df):,} matched'
              + (f'  ({n_missing} missing)' if n_missing else ''))

    if Path(SYNTACTIC_JSON).exists():
        with open(SYNTACTIC_JSON, encoding='utf-8') as f:
            syn = json.load(f)
        # keys are variant_id as str (from the Kaggle/HF-Space scorers); match
        # df['variant_id'] (likely int/str) by casting to str on both sides.
        syn = {str(k): v for k, v in syn.items()}
        for col in SYNTACTIC_COLS:
            df[col] = df['variant_id'].astype(str).map(lambda v: syn.get(v, {}).get(col))
            n_missing = df[col].isna().sum()
            print(f'  Merged {col} from {SYNTACTIC_JSON}: '
                  f'{len(df) - n_missing:,}/{len(df):,} matched'
                  + (f'  ({n_missing} missing)' if n_missing else ''))
    else:
        print(f'  SKIP {SYNTACTIC_COLS}: {SYNTACTIC_JSON} not found')

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nSaved: {OUTPUT_CSV}')
    print(f'Final shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')

    print('\nStill missing (add once available):')
    for col in SYNTACTIC_COLS:
        if col not in df.columns:
            print(f'  {col}  <- scripts/kaggle_syntactic_surprisal.py (full run not done yet)')


if __name__ == '__main__':
    main()
