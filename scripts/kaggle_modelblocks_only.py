# ═══════════════════════════════════════════════════════════════
# KAGGLE NOTEBOOK — Modelblocks (synproc) surprisal ONLY
# One of 3 parallel notebooks (this one, kaggle_pcfg_only.py,
# kaggle_depparse_only.py) — run all 3 SIMULTANEOUSLY in separate Kaggle
# sessions. Modelblocks is the middle-speed one (~1.4s/sent with 4 threads).
#
# Needs the C++ compile step (synproc) but NOT Stanza/torch.
#
# Before running:
#   1. Upload features.csv as a Kaggle Dataset named "hindi-wordorder-feature"
#   2. Enable Internet (Settings -> Internet: On). GPU not needed.
# ═══════════════════════════════════════════════════════════════


# ── CELL 1: Clone (git-lfs!) + compile synproc ───────────────────
# !apt-get update -qq && apt-get install -y -qq git-lfs g++ libarmadillo-dev libxml2-dev
# !git lfs install
# !rm -rf /kaggle/working/space_src
# !git clone https://huggingface.co/spaces/Muraliedhar/SyntacticTreeSurprisal /kaggle/working/space_src
# !cd /kaggle/working/space_src && git lfs pull
#
# import os, subprocess
# ROOT = "/kaggle/working/space_src"
# WORKSPACE = f"{ROOT}/workspace"
# os.makedirs(f"{WORKSPACE}/bin", exist_ok=True)
# for SRC in ["synproc", "ccmodel2synproccptmodel"]:
#     cmd = (
#         f'g++ -L/lib -L/lib64 -I/usr/include/libxml2/ '
#         f'-I{ROOT}/resource-incrsem/include -I{ROOT}/resource-incrsem/src '
#         f'-I{ROOT}/resource-rvtl -I{ROOT}/resource-logreg/include '
#         f'-Wall -O3 -DNDEBUG -fpermissive -std=c++17 '
#         f'{ROOT}/resource-incrsem/src/{SRC}.cpp '
#         f'-lm -larmadillo -Wl,-rpath-link=/lib -lpthread '
#         f'-o {WORKSPACE}/bin/{SRC}'
#     )
#     r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
#     print(SRC, "OK" if r.returncode == 0 else f"FAILED:\n{r.stderr[-2000:]}")
#     os.chmod(f"{WORKSPACE}/bin/{SRC}", 0o755)


# ── CELL 2: Python deps (just flask, for importing flask_server) ─
# !pip install -q flask flask-cors


# ── CELL 3: Import flask_server (Modelblocks logic lives here) ──
import os
import sys
import json
import time
from pathlib import Path

WORKSPACE = "/kaggle/working/space_src/workspace"
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

import flask_server as fs   # noqa: E402

assert Path(fs.SYNPROC).exists(), f"synproc binary missing (cell 1 build failed?): {fs.SYNPROC}"
print("synproc binary found. Ready.")


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


# ── CELL 5: Scoring function + parallel (threads — subprocess-bound,
#            releases the GIL, confirmed ~2.7x speedup with 4 threads) ─
from concurrent.futures import ThreadPoolExecutor

N_WORKERS = 4


def score_modelblocks_one(text):
    """Uses _surprisal_for(), NOT parse_one() — parse_one() wastefully
    spawns a Berkeley Parser JVM per sentence even though Modelblocks
    doesn't need a tree."""
    try:
        surp = fs._surprisal_for(text, 'hdtb', True)
        return round(sum(surp.values()), 4) if surp else None
    except Exception:
        return None


def score_many_parallel(score_fn, texts, n_workers=N_WORKERS):
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        return list(exe.map(score_fn, texts))


# ── CELL 6b: RESUME ACROSS SESSIONS + seed from the old combined
#             checkpoint (pre-3-way-split), if attached ─────────
RESULTS_JSON = str(OUT_DIR / 'modelblocks_checkpoint.json')
CHECKPOINT_DATASET = '/kaggle/input/modelblocks-checkpoint/modelblocks_checkpoint.json'
OLD_COMBINED_CHECKPOINT = '/kaggle/input/syntactic-checkpointer/kaggle_syntactic_surprisal.json'

scores = {}   # text -> surprisal_modelblocks
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
        if text not in scores and isinstance(vals, dict) and vals.get('surprisal_modelblocks') is not None:
            scores[text] = vals['surprisal_modelblocks']
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
    vals = score_many_parallel(score_modelblocks_one, batch)
    t_batch = time.time() - bt0

    for text, v in zip(batch, vals):
        scores[text] = v

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

MB_JSON = str(OUT_DIR / 'modelblocks_surprisals_new.json')
with open(MB_JSON, 'w', encoding='utf-8') as f:
    json.dump(by_vid, f, ensure_ascii=False)

real = sum(1 for v in by_vid.values() if v is not None)
print(f'surprisal_modelblocks: {real:,}/{len(by_vid):,} real scores')
print(f'\nDownload from /kaggle/working/: modelblocks_surprisals_new.json')
print('Place at: models/modelblocks/modelblocks_surprisals_new.json')
print('Once all 3 notebooks (PCFG, Modelblocks, Dep-parser) are done, merge all 3 into one')
print('results/syntactic_surprisal.json locally (see merge_three_checkpoints.py), then run')
print('assemble_features.py as usual.')
