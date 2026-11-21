"""
Compute IS (Information Status) scores for all 78,865 variants.
Uses src/feature_extraction/givenness.py (paper-aligned implementation).

For each variant:
  - Tokens loaded from variants_tokens.jsonl (surface-order IDs, updated heads)
  - Context = preceding sentence from train.conllu (content words only)
  - IS score = +1 (Given-New), 0 (GG/NN), -1 (New-Given)

Output: models/is_score/is_scores.json  {variant_id → is_score}
"""
import os, sys, json, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

import pandas as pd
from feature_extraction.givenness import GivennessScorer, CONTENT_POS

CONLLU    = os.path.join(BASE, 'data', 'ud_hindi', 'train.conllu')
JSONL     = os.path.join(BASE, 'data', 'processed', 'variants', 'variants_tokens.jsonl')
VARIANTS  = os.path.join(BASE, 'data', 'processed', 'variants', 'variants.csv')
OUT_DIR   = os.path.join(BASE, 'models', 'is_score')
OUT_JSON  = os.path.join(OUT_DIR, 'is_scores.json')

os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 60)
print('IS Score Computation')
print('=' * 60)

# ── 1. Run sanity check ───────────────────────────────────────
print('\n[0] Sanity check ...')
scorer = GivennessScorer()
scorer.verify()

# ── 2. Extract CoNLL-U sentences in order ─────────────────────
print(f'\n[1] Extracting CoNLL-U sentences from train.conllu ...')
t0 = time.time()

conllu_sents = []    # list of (surface_str, [token_dicts])
current = []

def flush(toks):
    if not toks: return
    surface = ' '.join(t['form'] for t in toks)
    conllu_sents.append((surface, toks))

with open(CONLLU, encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line.strip():
            flush(current); current = []; continue
        if line.startswith('#'): continue
        parts = line.split('\t')
        if len(parts) < 10 or '-' in parts[0] or '.' in parts[0]: continue
        try: int(parts[0])
        except: continue
        feats = {}
        if parts[5] != '_':
            for kv in parts[5].split('|'):
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    feats[k] = v
        current.append({
            'id'     : int(parts[0]),
            'form'   : parts[1],
            'lemma'  : parts[2],
            'upostag': parts[3],
            'head'   : int(parts[6]),
            'deprel' : parts[7],
            'feats'  : feats,
            'misc'   : parts[9],
        })
flush(current)

# Build surface → (index, tokens) map
surface_to_idx = {}
for i, (surf, toks) in enumerate(conllu_sents):
    surface_to_idx[surf] = i

print(f'    {len(conllu_sents):,} sentences extracted in {time.time()-t0:.1f}s')

# ── 3. Load variants metadata ──────────────────────────────────
print(f'\n[2] Loading variants.csv ...')
df = pd.read_csv(VARIANTS)
ref_df   = df[df['is_original'] == 1][['reference_id', 'text']].drop_duplicates()
ref_text = dict(zip(ref_df['reference_id'], ref_df['text']))
print(f'    {len(df):,} variants,  {len(ref_text):,} references')

# Match reference texts to CoNLL-U index
matched = sum(1 for t in ref_text.values() if t in surface_to_idx)
print(f'    References matched to CoNLL-U: {matched:,}/{len(ref_text):,}')

# ── 4. Load variants_tokens.jsonl into memory (vid → tokens) ──
print(f'\n[3] Loading variants_tokens.jsonl ...')
t0 = time.time()
vid_to_tokens = {}
with open(JSONL, 'rb') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        vid, tok_json = line.split(b'\t', 1)
        vid = vid.decode('utf-8')
        vid_to_tokens[vid] = json.loads(tok_json)

print(f'    {len(vid_to_tokens):,} variants loaded in {time.time()-t0:.1f}s')

# ── 5. Compute IS scores ───────────────────────────────────────
print(f'\n[4] Computing IS scores ...')
t0 = time.time()
is_scores = {}
errors = 0
no_context = 0
zero_count = 0

all_vids = list(df['variant_id'])
ref_ids  = dict(zip(df['variant_id'], df['reference_id']))

for i, vid in enumerate(all_vids):
    try:
        tokens = vid_to_tokens.get(vid)
        if tokens is None:
            is_scores[vid] = 0; errors += 1; continue

        # Get context (preceding sentence in CoNLL-U)
        ref_surface = ref_text.get(ref_ids.get(vid, ''), '')
        ctx_idx     = surface_to_idx.get(ref_surface, None)
        context_tokens = None
        if ctx_idx is not None and ctx_idx > 0:
            context_tokens = conllu_sents[ctx_idx - 1][1]
        else:
            no_context += 1

        score = scorer.compute_is_score(tokens, context_tokens)
        is_scores[vid] = score
        if score == 0:
            zero_count += 1

    except Exception:
        is_scores[vid] = 0
        errors += 1

    if (i + 1) % 10000 == 0:
        elapsed = time.time() - t0
        eta     = elapsed / (i+1) * (len(all_vids) - i - 1)
        print(f'    {i+1:>6,}/{len(all_vids):,}  ETA {eta/60:.1f} min')

print(f'    Done in {(time.time()-t0)/60:.1f} min')
print(f'    No context (first sentence): {no_context:,}')
print(f'    Errors (set 0):  {errors:,}')

# ── 6. Stats ───────────────────────────────────────────────────
print(f'\n[5] Score distribution:')
from collections import Counter
dist = Counter(is_scores.values())
total = len(is_scores)
for k in [+1, 0, -1]:
    print(f'    {k:+d} : {dist[k]:>6,}  ({100*dist[k]/total:.1f}%)')

# Per order type
df['is_score'] = df['variant_id'].map(is_scores)
print(f'\n    By order_type (mean IS score):')
for ot, grp in df.groupby('order_type'):
    corp = grp[grp['is_original']==1]['is_score'].mean()
    gen  = grp[grp['is_original']==0]['is_score'].mean()
    print(f'    {ot:<14}  corpus={corp:+.3f}  generated={gen:+.3f}')

corp_mean = df[df['is_original']==1]['is_score'].mean()
gen_mean  = df[df['is_original']==0]['is_score'].mean()
print(f'\n    Overall corpus mean  : {corp_mean:+.4f}')
print(f'    Overall variant mean : {gen_mean:+.4f}')
print(f'    Corpus > Variant (expected True): {corp_mean > gen_mean}')
print(f'    Paper coefficient: +0.26 (DO-fronted)')

# ── 7. Save ────────────────────────────────────────────────────
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(is_scores, f)
print(f'\n[6] Saved: {OUT_JSON}')
print(f'    Entries: {len(is_scores):,}')
print('\n' + '='*60)
print('DONE. IS scores ready.')
print('='*60)
