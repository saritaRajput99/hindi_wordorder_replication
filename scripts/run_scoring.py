"""
Run trigram surprisal + IS score on all  variants.
DL is already in variants.csv.

Output: variants.csv updated with columns:
  surprisal_trigram  — trigram LM surprisal (nats, lower = more expected)
  is_score           — information status score (-1, 0, +1)
  dl                 — dependency length (already present)

Run:
    python scripts/run_scoring.py
"""

import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path

from src.data_loading.loader      import HUTBLoader
from src.preprocessing.filter     import SentenceFilter
from src.surprisal_models.surprisal import ARPAModel
from src.feature_extraction.givenness import GivennessScorer

# ── Paths ──────────────────────────────────────────────────────────────────
VARIANTS_CSV    = 'output/variants_final/variants.csv'
VARIANTS_JSONL  = 'output/variants_final/variants_tokens.jsonl'
ARPA_PATH       = 'models/trigram/hindi_trigram_srilm.arpa'   # SRILM-trained, not KenLM
OUTPUT_CSV      = 'output/variants_final/variants_scored.csv'
RESULTS_DIR     = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

print('=' * 60)
print('STEP 1: Load variants')
print('=' * 60)

df = pd.read_csv(VARIANTS_CSV)
print(f'  Loaded: {len(df):,} variants, {df["reference_id"].nunique():,} refs')

# Load tokens from jsonl
print('  Loading tokens from jsonl...')
tokens_map = {}
with open(VARIANTS_JSONL, encoding='utf-8') as f:
    for line in f:
        vid, tjson = line.strip().split('\t', 1)
        tokens_map[vid] = tjson
print(f'  Tokens loaded: {len(tokens_map):,}')

# Attach tokens_json to df
df['tokens_json'] = df['variant_id'].map(tokens_map)
missing = df['tokens_json'].isna().sum()
if missing:
    print(f'  WARNING: {missing} variants have no tokens — filling with empty')
    df['tokens_json'] = df['tokens_json'].fillna('[]')

# ── Reload filtered sentences for IS context ──────────────────────────────
print()
print('=' * 60)
print('STEP 2: Reload filtered sentences (for IS context)')
print('=' * 60)

loader   = HUTBLoader('data/ud_hindi')
raw      = loader.load_train()
orig_sent_ids = [
    raw[i]['sent_id'] if raw[i].get('sent_id') else f'sent_{i:04d}'
    for i in range(len(raw))
]
filt     = SentenceFilter()
filtered, _, filtered_sent_ids = filt.filter_with_rejection_tracking(
    [s['tokens'] for s in raw], sent_ids=orig_sent_ids
)
metadata_df = pd.DataFrame([{'sent_id': sid} for sid in filtered_sent_ids])
print(f'  Filtered sentences: {len(filtered):,}')

# ── STEP 3: Trigram surprisal ─────────────────────────────────────────────
print()
print('=' * 60)
print('STEP 3: Trigram surprisal (ARPAModel)')
print('=' * 60)

arpa = ARPAModel()
arpa.load(ARPA_PATH)

trigram_scores = []
errors = 0
t0 = time.time()

for i, (_, row) in enumerate(df.iterrows()):
    try:
        tokens = json.loads(row['tokens_json'])
        words  = [t['form'] for t in tokens if isinstance(t.get('form'), str) and t['form'].strip()]
        if not words:
            words = ['<unk>']
        s = arpa.sentence_surprisal(words)
    except Exception:
        s = 0.0
        errors += 1
    trigram_scores.append(s)

    if (i + 1) % 10000 == 0:
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed
        eta     = (len(df) - i - 1) / max(rate, 1e-6)
        print(f'  {i+1:>6,}/{len(df):,}  ({(i+1)/len(df)*100:.1f}%)  ETA: {eta/60:.1f} min')

df['surprisal_trigram'] = trigram_scores
print(f'  Done. Errors: {errors}')
print(f'  Corpus mean  : {df[df["is_original"]==1]["surprisal_trigram"].mean():.3f}')
print(f'  Variant mean : {df[df["is_original"]==0]["surprisal_trigram"].mean():.3f}')

# ── STEP 4: IS score ──────────────────────────────────────────────────────
print()
print('=' * 60)
print('STEP 4: Information Status (IS) score')
print('=' * 60)

scorer = GivennessScorer()
df = scorer.add_to_variants(df, filtered, metadata_df, column_name='is_score')

print(f'  IS distribution:')
print(df['is_score'].value_counts().to_string())

# ── STEP 5: Verify DL already present ────────────────────────────────────
print()
print('=' * 60)
print('STEP 5: DL check')
print('=' * 60)
print(f'  DL corpus mean  : {df[df["is_original"]==1]["dl"].mean():.2f}')
print(f'  DL variant mean : {df[df["is_original"]==0]["dl"].mean():.2f}')
print(f'  DLT supported   : {df[df["is_original"]==1]["dl"].mean() < df[df["is_original"]==0]["dl"].mean()}')

# ── STEP 6: Save ─────────────────────────────────────────────────────────
print()
print('=' * 60)
print('STEP 6: Save')
print('=' * 60)

# Drop tokens_json before saving (keep csv lightweight)
save_df = df.drop(columns=['tokens_json'])
save_df.to_csv(OUTPUT_CSV, index=False)
print(f'  Saved: {OUTPUT_CSV}')
print(f'  Columns: {list(save_df.columns)}')
print(f'  Rows: {len(save_df):,}')

# Also save scoring summary
summary = {
    'trigram_corpus_mean' : round(df[df['is_original']==1]['surprisal_trigram'].mean(), 4),
    'trigram_variant_mean': round(df[df['is_original']==0]['surprisal_trigram'].mean(), 4),
    'dl_corpus_mean'      : round(df[df['is_original']==1]['dl'].mean(), 4),
    'dl_variant_mean'     : round(df[df['is_original']==0]['dl'].mean(), 4),
    'is_score_dist'       : df['is_score'].value_counts().to_dict(),
}
import json as _json
with open(RESULTS_DIR / 'scoring_summary.json', 'w', encoding='utf-8') as f:
    _json.dump(summary, f, indent=2, ensure_ascii=False)
print(f'  Saved: results/scoring_summary.json')
print()
print('Done.')
