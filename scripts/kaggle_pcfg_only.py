# ═══════════════════════════════════════════════════════════════
# KAGGLE NOTEBOOK — PCFG surprisal ONLY
# One of 3 parallel notebooks (this one, kaggle_modelblocks_only.py,
# kaggle_depparse_only.py) — run all 3 SIMULTANEOUSLY in separate Kaggle
# sessions instead of sequentially in kaggle_syntactic_surprisal.py. PCFG is
# the fastest of the 3 (~0.3s/sent), so this notebook will likely finish
# well before the other two — that's fine, just let it sit done.
#
# Lighter setup than the combined script: PCFG only needs Java + the
# Berkeley Parser jar/grammar — no C++ compile (synproc), no Stanza/torch.
#
# Before running:
#   1. Upload features.csv as a Kaggle Dataset named "hindi-wordorder-feature"
#   2. Enable Internet (Settings -> Internet: On). GPU not needed.
# ═══════════════════════════════════════════════════════════════


# ── CELL 1: Clone (with git-lfs!) — no compile needed for PCFG ──
# !apt-get update -qq && apt-get install -y -qq git-lfs default-jre-headless
# !git lfs install
# !rm -rf /kaggle/working/space_src
# !git clone https://huggingface.co/spaces/Muraliedhar/SyntacticTreeSurprisal /kaggle/working/space_src
# !cd /kaggle/working/space_src && git lfs pull
#
# import os
# ROOT = "/kaggle/working/space_src"
# check_file = f"{ROOT}/external_resources/berkeleyparser/berkeleyParser.jar"
# print(f"{check_file}: {os.path.getsize(check_file):,} bytes "
#       f"({'OK — real file' if os.path.getsize(check_file) > 100_000 else 'STILL A POINTER — git lfs pull failed'})")


# ── CELL 2: Python deps (just flask, for importing flask_server) ─
# !pip install -q flask flask-cors


# ── CELL 3: Import flask_server (PCFG logic lives here) ──────────
import os
import sys
import json
import time
from pathlib import Path

WORKSPACE = "/kaggle/working/space_src/workspace"
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

import flask_server as fs   # noqa: E402

assert Path(fs.JAR).exists(), f"Berkeley Parser jar missing: {fs.JAR}"
grammar = fs.BUILTIN['hdtb']['grammar']
assert Path(grammar).exists(), f"Grammar missing: {grammar}"
print("Berkeley Parser jar + grammar found. Ready.")


# ── CELL 4: Load features.csv, dedupe by text ────────────────────
import pandas as pd

FEAT_CSV = '/kaggle/input/hindi-wordorder-feature/features.csv'
OUT_DIR = Path('/kaggle/working')

df = pd.read_csv(FEAT_CSV)
print(f'Loaded: {len(df):,} variants')

vid_to_text = dict(zip(df['variant_id'], df['text']))
text_to_vids = {}
for vid, text in vid_to_text.items():
    text_to_vids.setdefault(str(text), []).append(vid)
unique_texts = list(text_to_vids.keys())
print(f'Unique sentences to score: {len(unique_texts):,}')


# ── CELL 5: Scoring function (batched — one java call per chunk) ─
def score_pcfg_chunk(texts, chunk_size=300):
    """PCFG surprisal, batched. Returns {text: surprisal_or_None}."""
    out = {}
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        try:
            results = fs._pcfg_surprisal_batch(chunk, 'hdtb')
        except Exception as e:
            print(f'  PCFG chunk {i} failed: {e}')
            results = [None] * len(chunk)
        for t, r in zip(chunk, results):
            out[t] = round(sum(r.values()), 4) if r else None
    return out


# ── CELL 6b: RESUME ACROSS SESSIONS + seed from the old combined
#             checkpoint (pre-3-way-split), if attached ─────────
RESULTS_JSON = str(OUT_DIR / 'pcfg_checkpoint.json')
CHECKPOINT_DATASET = '/kaggle/input/pcfg-checkpoint/pcfg_checkpoint.json'
OLD_COMBINED_CHECKPOINT = '/kaggle/input/syntactic-checkpointer/kaggle_syntactic_surprisal.json'

scores = {}   # text -> surprisal_pcfg
if Path(RESULTS_JSON).exists():
    with open(RESULTS_JSON, encoding='utf-8') as f:
        scores = json.load(f)
    print(f'Resumed (this session): {len(scores):,} sentences already scored')
elif Path(CHECKPOINT_DATASET).exists():
    with open(CHECKPOINT_DATASET, encoding='utf-8') as f:
        scores = json.load(f)
    print(f'Restored own checkpoint from previous session: {len(scores):,} sentences')
else:
    print('No own checkpoint found — starting fresh (will still seed from combined, below).')

if Path(OLD_COMBINED_CHECKPOINT).exists():
    with open(OLD_COMBINED_CHECKPOINT, encoding='utf-8') as f:
        combined = json.load(f)
    seeded = 0
    for text, vals in combined.items():
        if text not in scores and isinstance(vals, dict) and vals.get('surprisal_pcfg') is not None:
            scores[text] = vals['surprisal_pcfg']
            seeded += 1
    print(f'Seeded {seeded:,} sentences from the old combined checkpoint (not redoing those)')
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False)


# ── CELL 7: FULL RUN (resumable, checkpointed every 300 sentences) ─
todo = [t for t in unique_texts if t not in scores]
print(f'To score: {len(todo):,} / {len(unique_texts):,} unique sentences')

CHECKPOINT_EVERY = 300
t0 = time.time()

for batch_start in range(0, len(todo), CHECKPOINT_EVERY):
    batch = todo[batch_start:batch_start + CHECKPOINT_EVERY]
    bn = len(batch)

    bt0 = time.time()
    batch_scores = score_pcfg_chunk(batch, chunk_size=300)
    t_batch = time.time() - bt0

    scores.update(batch_scores)

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False)

    done = batch_start + bn
    elapsed = time.time() - t0
    eta = (len(todo) - done) / (done / elapsed) / 60 if done else 0
    print(f'  {done:,}/{len(todo):,}  batch {t_batch:.1f}s ({t_batch/bn*1000:.0f} ms/sent)  '
          f'ETA {eta:.1f} min  (checkpointed -> {RESULTS_JSON})')

print(f'\nDone: {len(scores):,} unique sentences scored -> {RESULTS_JSON}')


# ── CELL 8: Expand back out to variant_id, verify, download instructions ─
by_vid = {}
for text, vids in text_to_vids.items():
    val = scores.get(text)
    for vid in vids:
        by_vid[str(vid)] = val

PCFG_JSON = str(OUT_DIR / 'pcfg_surprisals_new.json')
with open(PCFG_JSON, 'w', encoding='utf-8') as f:
    json.dump(by_vid, f, ensure_ascii=False)

real = sum(1 for v in by_vid.values() if v is not None)
print(f'surprisal_pcfg: {real:,}/{len(by_vid):,} real scores')
print(f'\nDownload from /kaggle/working/: pcfg_surprisals_new.json')
print('Place at: models/pcfg_parser/pcfg_surprisals_new.json')
print('Once all 3 notebooks (PCFG, Modelblocks, Dep-parser) are done, merge all 3 into one')
print('results/syntactic_surprisal.json locally (see merge_three_checkpoints.py), then run')
print('assemble_features.py as usual.')
