# ═══════════════════════════════════════════════════════════════
# KAGGLE NOTEBOOK — PCFG / Modelblocks / Dependency-Parser surprisal
# Run each block as a separate cell in Kaggle. CPU notebook is enough
# (no GPU needed) — but you DO need Internet enabled (to clone the
# HF Space repo and download the Stanza Hindi model).
#
# WHY THIS EXISTS:
#   scripts/score_syntactic_surprisal.py gets these 3 features by making
#   HTTP calls to a free, shared community HF Space
#   (https://huggingface.co/spaces/Muraliedhar/SyntacticTreeSurprisal).
#   A 200-row test took 18.5 min and only scored 58.5% of rows (the
#   shared Space is slow/flaky under its own queue). Extrapolated to
#   78,865 rows that's ~5 days, with gaps.
#
#   That Space's source is fully public (Docker SDK). Its own code is:
#     - Modelblocks surprisal : `synproc` (C++, resource-incrsem) via
#       flask_server.parse_one()['surprisal']  (UNK-signature handling
#       for OOV words baked into flask_server._unk_sentence)
#     - PCFG surprisal        : Berkeley Parser, incremental per-word
#       bits from prefix log-likelihoods, via
#       flask_server._pcfg_surprisal_batch()  (ONE java call scores many
#       sentences' prefixes at once — this is what makes it fast)
#     - Dep-parser surprisal  : Stanza (predicts a UD tree for our
#       REORDERED variant sentences, which have no gold tree of their
#       own) + the vendored arc-eager oracle model "hi_hdtb_modified_v1",
#       via dep_incremental_backend.parse_units()['total_surprisal']
#       (already a per-sentence sum — no "no JVM" here, pure Python+numpy)
#
#   This script clones that repo and IMPORTS its own modules directly
#   (no network/HTTP, no shared queue) instead of reimplementing the
#   surprisal math by hand — lower risk of a subtly-wrong formula.
#
# Before running:
#   1. Upload features.csv as a Kaggle Dataset named "hindi-wordorder-feature"
#   2. Enable Internet in Notebook settings (Settings -> Internet: On)
#   3. GPU not required — CPU notebook is fine
# ═══════════════════════════════════════════════════════════════


# ── CELL 1: Install git-lfs FIRST, then clone (plain `git clone` without
#            git-lfs installed only fetches tiny LFS *pointer* stubs for every
#            .gr/.jar/.synprocmodel/.npz/.conllu/.pkl file — not the real
#            binaries — which is why the first clone attempt was only 2.3 MiB).
# !apt-get update -qq && apt-get install -y -qq git-lfs g++ libarmadillo-dev libxml2-dev default-jre-headless
# !git lfs install
# !rm -rf /kaggle/working/space_src   # delete any earlier pointer-only clone
# !git clone https://huggingface.co/spaces/Muraliedhar/SyntacticTreeSurprisal /kaggle/working/space_src
# !cd /kaggle/working/space_src && git lfs pull
#
# import os, subprocess
# ROOT = "/kaggle/working/space_src"
# WORKSPACE = f"{ROOT}/workspace"
# os.makedirs(f"{WORKSPACE}/bin", exist_ok=True)
# # Sanity check: a real .gr/.jar file is >1MB; a leftover LFS pointer is ~130 bytes.
# check_file = f"{ROOT}/external_resources/berkeleyparser/berkeleyParser.jar"
# print(f"{check_file}: {os.path.getsize(check_file):,} bytes "
#       f"({'OK — real file' if os.path.getsize(check_file) > 100_000 else 'STILL A POINTER — git lfs pull failed'})")
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


# ── CELL 2: Python deps + Stanza Hindi model ─────────────────────
# Do NOT pin numpy/scipy versions here — Kaggle's base image already ships a
# numpy/scipy stack matched to its preinstalled torch. Pinning different
# versions makes pip's resolver backtrack through many candidate versions,
# rebuilding metadata each time (looks like it's "stuck" repeating
# "Installing build dependencies..." — it's actually thrashing, not hung).
# If you hit that: interrupt the cell, then run just this instead:
# !pip install -q flask flask-cors stanza indic_transliteration
#
# import stanza
# stanza.download('hi', verbose=False)


# ── CELL 3: Import the Space's own modules (no HTTP, direct calls) ─
import os
import sys
import json
import time
from pathlib import Path

WORKSPACE = "/kaggle/working/space_src/workspace"
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)
sys.path.insert(0, str(Path(WORKSPACE) / "dep_incremental"))

import flask_server as fs              # noqa: E402  (Modelblocks + PCFG)
import dep_incremental_backend as dep   # noqa: E402  (Dependency-parser)

# Sanity: confirm the Hindi model files this script depends on exist.
assert Path(fs.JAR).exists(), f"Berkeley Parser jar missing: {fs.JAR}"
assert Path(fs.SYNPROC).exists(), f"synproc binary missing (cell 1 build failed?): {fs.SYNPROC}"
dep_model_dir = Path(WORKSPACE) / "dep_incremental" / "models" / "hi_hdtb_modified_v1"
assert (dep_model_dir / "wts_0.npz").exists(), (
    f"dep-parser model missing: {dep_model_dir}. "
    f"Available: {list((Path(WORKSPACE) / 'dep_incremental' / 'models').iterdir())}"
)
print("All model files found. Ready.")


# ── CELL 4: Load features.csv, dedupe by text ────────────────────
import pandas as pd

FEAT_CSV = '/kaggle/input/hindi-wordorder-feature/features.csv'
OUT_DIR = Path('/kaggle/working')

df = pd.read_csv(FEAT_CSV)
print(f'Loaded: {len(df):,} variants')

# NOTE: a scope-reduction filter (score only is_original==1 OR
# order_type=='Canonical', ~51% of rows) was tried and reverted on
# 2026-11-02. Ranjan's instruction is bidirectional — SOV reference ->
# generate OSV variants, AND OSV-type reference -> generate SOV variant —
# so the ~49% of rows that filter would have skipped (canonical-reference ->
# generated non-canonical variants) are the intentional output of the OTHER
# direction, not unused dead weight. The current classifier scripts
# (run_classifier.py etc.) only implement one direction so far; that's a gap
# in the analysis code, not a sign that half the data is disposable. Score
# the FULL set.

vid_to_text = dict(zip(df['variant_id'], df['text']))
# Dedup identical sentence text across variant_ids (canonical order can repeat
# across a few reference sentences' pair sets) — score once, map back to all ids.
text_to_vids = {}
for vid, text in vid_to_text.items():
    text_to_vids.setdefault(str(text), []).append(vid)
unique_texts = list(text_to_vids.keys())
print(f'Unique sentences to score: {len(unique_texts):,}')


# ── CELL 5: Scoring functions (sum per-word bits -> sentence score) ─

def score_pcfg_chunk(texts, chunk_size=300):
    """PCFG surprisal, batched (ONE java call per chunk) via the Space's own
    incremental prefix-likelihood formula. Returns {text: surprisal_or_None}."""
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


def score_modelblocks_one(text):
    """Modelblocks (synproc) surprisal for one sentence -> sum of per-word bits,
    or None if the incremental parse FAILed / produced nothing usable.

    Uses flask_server._surprisal_for(), NOT parse_one(). parse_one() ALWAYS
    spawns a Berkeley Parser JVM to build a tree even when want_pcfg=False —
    that's a ~3-4s JVM cold-start wasted per sentence for a tree we never use.
    _surprisal_for() runs only the synproc pipeline (no Java at all)."""
    try:
        surp = fs._surprisal_for(text, 'hdtb', True)
        return round(sum(surp.values()), 4) if surp else None
    except Exception:
        return None


def score_depparse_one(text):
    """Dependency-parser surprisal for one sentence (Stanza tree + arc-eager
    oracle). dep.parse_units already returns a per-sentence SUM in
    'total_surprisal'. Returns None if Stanza/the model can't parse it."""
    try:
        for r in dep.parse_units(text, 'sentence', 'hi_hdtb_modified_v1'):
            return r.get('total_surprisal')
    except Exception:
        pass
    return None


# Parallel runner for Modelblocks: ThreadPoolExecutor. score_modelblocks_one
# is os.system/subprocess calls to synproc — threads release the GIL while
# waiting on that subprocess work, so this gets real parallelism. Confirmed:
# ~2.7x speedup measured (3.77s/sent -> 1.40s/sent with 4 threads).
from concurrent.futures import ThreadPoolExecutor

N_WORKERS = 4   # matches Kaggle's typical CPU-notebook core count


def score_many_parallel(score_fn, texts, n_workers=N_WORKERS):
    """Run score_fn(text) over `texts` in parallel, preserving order."""
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        return list(exe.map(score_fn, texts))


# Parallel runner for Dep-parser: multiprocessing, NOT threading. Measured:
# 4 threads gave 5.25s/sent, NO better than plain sequential (4.85s/sent) —
# because dep_incremental_backend.py wraps the actual arc-eager parse in a
# threading.Lock() (shared class-level state via ArcEagerState.recover_
# metadata), which fully serializes that step across threads in one process.
# Separate OS processes each get their OWN unlocked copy of that state, so
# multiprocessing genuinely parallelizes it. Warm up the Stanza pipeline in
# THIS (parent) process first, before forking — children then inherit the
# already-loaded ~300MB+ model via copy-on-write instead of each loading
# their own copy (which would cost N_WORKERS x load-time AND x memory).
import multiprocessing as mp

# PyTorch (Stanza's backend) spins up its own internal multi-threaded BLAS
# pool by default. Combined with fork()-based multiprocessing, a parent
# process's extra torch threads at fork time is a known cause of hangs /
# inconsistent state in child processes. We're already parallelizing at the
# PROCESS level (N_WORKERS forked workers), so cap torch to 1 thread each —
# avoids both the fork hazard and CPU oversubscription (4 procs x N torch
# threads would fight over the same 4 cores anyway).
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

print('Warming up Stanza pipeline in the parent process (one-time)...')
t0 = time.time()
_ = score_depparse_one('यह एक परीक्षण वाक्य है')
print(f'  warm-up done in {time.time()-t0:.1f}s')

_MP_CTX = mp.get_context('fork')   # fork, so children inherit the warm cache


def score_depparse_many_parallel(texts, n_workers=N_WORKERS):
    """Dep-parser surprisal over many sentences using separate PROCESSES
    (forked from this already-warmed-up parent), not threads."""
    with _MP_CTX.Pool(processes=n_workers) as pool:
        return pool.map(score_depparse_one, texts)


# ── CELL 6: SMALL TIMED TEST FIRST (matches the HF-Space-test pattern) ─
# Modelblocks + Dep-parser now run across N_WORKERS=4 threads in parallel
# (see score_many_parallel in Cell 5) instead of one sentence at a time.
TEST_LIMIT = 300   # <- raise once this looks right and you have a real ETA

test_texts = unique_texts[:TEST_LIMIT]

print(f'[1/3] PCFG — scoring {len(test_texts)} sentences in chunks of 150 '
      f'(one java call per chunk; first chunk is slower, cold JVM)...')
t0 = time.time()
pcfg_test = score_pcfg_chunk(test_texts, chunk_size=150)
t_pcfg = time.time() - t0
print(f'      done in {t_pcfg:.1f}s')

print(f'[2/3] Modelblocks (synproc) — scoring {len(test_texts)} sentences, '
      f'{N_WORKERS} parallel threads...')
t0 = time.time()
mb_vals = score_many_parallel(score_modelblocks_one, test_texts)
mb_test = dict(zip(test_texts, mb_vals))
t_mb = time.time() - t0
print(f'      done in {t_mb:.1f}s')

print(f'[3/3] Dep-parser (Stanza + arc-eager) — scoring {len(test_texts)} sentences, '
      f'{N_WORKERS} parallel PROCESSES (not threads — see Cell 5 comment)...')
t0 = time.time()
dep_vals = score_depparse_many_parallel(test_texts)
dep_test = dict(zip(test_texts, dep_vals))
t_dep = time.time() - t0
print(f'      done in {t_dep:.1f}s')

n = len(test_texts)
print(f'PCFG        : {n} sents in {t_pcfg:.1f}s  ({t_pcfg/n*1000:.0f} ms/sent)  '
      f'real={sum(1 for v in pcfg_test.values() if v is not None)}/{n}')
print(f'Modelblocks : {n} sents in {t_mb:.1f}s  ({t_mb/n*1000:.0f} ms/sent)  '
      f'real={sum(1 for v in mb_test.values() if v is not None)}/{n}')
print(f'Dep-parser  : {n} sents in {t_dep:.1f}s  ({t_dep/n*1000:.0f} ms/sent)  '
      f'real={sum(1 for v in dep_test.values() if v is not None)}/{n}')

total_unique = len(unique_texts)
for name, t in [('PCFG', t_pcfg), ('Modelblocks', t_mb), ('Dep-parser', t_dep)]:
    eta_min = (t / n) * total_unique / 60
    print(f'  Full-run ETA for {name}: {eta_min:.1f} min  ({total_unique:,} unique sentences)')


# ── CELL 6b: RESUME ACROSS SESSIONS ──────────────────────────────
# /kaggle/working is WIPED at the start of every new session — a checkpoint
# saved mid-run in a previous session is gone unless you downloaded it AND
# re-uploaded it as a Dataset before starting the next session.
#
# End-of-session routine (do this BEFORE your session hits its time limit):
#   1. In the Kaggle UI, open the "/kaggle/working" file pane (or just let the
#      notebook finish this cell block — it auto-saves every 500 sentences).
#   2. Download kaggle_syntactic_surprisal.json.
#   3. Create/update a Kaggle Dataset (e.g. "syntactic-checkpoint") with that
#      file, and "Add Data" -> that dataset in your next session.
#
# Start-of-NEXT-session routine (run this cell every time you restart):
RESULTS_JSON = str(OUT_DIR / 'kaggle_syntactic_surprisal.json')
CHECKPOINT_DATASET = '/kaggle/input/syntactic-checkpointer/kaggle_syntactic_surprisal.json'
if not Path(RESULTS_JSON).exists() and Path(CHECKPOINT_DATASET).exists():
    import shutil
    shutil.copy(CHECKPOINT_DATASET, RESULTS_JSON)
    print(f'Restored checkpoint from previous session: {CHECKPOINT_DATASET}')
elif not Path(CHECKPOINT_DATASET).exists():
    print('No checkpoint dataset found — this looks like the first session, starting fresh.')


# ── CELL 7: FULL RUN (resumable — re-running skips sentences already
#            in the checkpoint file) ─────────────────────────────
scores = {}   # text -> {surprisal_pcfg, surprisal_modelblocks, surprisal_depparse}
if Path(RESULTS_JSON).exists():
    with open(RESULTS_JSON, encoding='utf-8') as f:
        scores = json.load(f)
    print(f'Resumed: {len(scores):,} sentences already scored')

todo = [t for t in unique_texts if t not in scores]
print(f'To score: {len(todo):,} / {len(unique_texts):,} unique sentences')

CHECKPOINT_EVERY = 300   # sentences per batch; checkpointed after each batch.
# 300 doubles as a live version of the Cell 6 test: the first batch's
# per-stage timing below tells you the same thing Cell 6 would, except this
# time the results are real progress, not thrown away — so Cell 6 can be
# skipped and you can come straight here.
t0 = time.time()

for batch_start in range(0, len(todo), CHECKPOINT_EVERY):
    batch = todo[batch_start:batch_start + CHECKPOINT_EVERY]
    bn = len(batch)

    bt0 = time.time()
    pcfg_batch = score_pcfg_chunk(batch, chunk_size=300)
    t_pcfg_b = time.time() - bt0

    bt0 = time.time()
    mb_vals = score_many_parallel(score_modelblocks_one, batch)
    t_mb_b = time.time() - bt0

    bt0 = time.time()
    dep_vals = score_depparse_many_parallel(batch)
    t_dep_b = time.time() - bt0

    for text, mb, dp in zip(batch, mb_vals, dep_vals):
        scores[text] = {
            'surprisal_pcfg': pcfg_batch.get(text),
            'surprisal_modelblocks': mb,
            'surprisal_depparse': dp,
        }

    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False)
    print(f'  batch: PCFG {t_pcfg_b:.1f}s ({t_pcfg_b/bn*1000:.0f} ms/sent)  '
          f'Modelblocks {t_mb_b:.1f}s ({t_mb_b/bn*1000:.0f} ms/sent)  '
          f'Dep-parser {t_dep_b:.1f}s ({t_dep_b/bn*1000:.0f} ms/sent)')
    done = batch_start + len(batch)
    elapsed = time.time() - t0
    eta = (len(todo) - done) / (done / elapsed) / 60 if done else 0
    print(f'  {done:,}/{len(todo):,}  ETA {eta:.1f} min  (checkpointed -> {RESULTS_JSON})')

with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
    json.dump(scores, f, ensure_ascii=False)
print(f'\nDone: {len(scores):,} unique sentences scored -> {RESULTS_JSON}')


# ── CELL 8: Expand back out to variant_id, verify, download instructions ─
by_vid = {}
for text, vids in text_to_vids.items():
    s = scores.get(text, {})
    for vid in vids:
        by_vid[str(vid)] = s

VID_JSON = str(OUT_DIR / 'syntactic_surprisal.json')
with open(VID_JSON, 'w', encoding='utf-8') as f:
    json.dump(by_vid, f, ensure_ascii=False)

for col in ['surprisal_pcfg', 'surprisal_modelblocks', 'surprisal_depparse']:
    real = sum(1 for s in by_vid.values() if s.get(col) is not None)
    print(f'{col}: {real:,}/{len(by_vid):,} real scores')

print(f'\nDownload from /kaggle/working/: syntactic_surprisal.json')
print('Place it locally at: results/syntactic_surprisal.json')
print('Then merge into features.csv the same way assemble_features.py does, e.g.:')
print("  import json, pandas as pd")
print("  scores = json.load(open('results/syntactic_surprisal.json', encoding='utf-8'))")
print("  df['surprisal_pcfg']        = df['variant_id'].map(lambda v: scores.get(str(v), {}).get('surprisal_pcfg'))")
print("  df['surprisal_modelblocks'] = df['variant_id'].map(lambda v: scores.get(str(v), {}).get('surprisal_modelblocks'))")
print("  df['surprisal_depparse']    = df['variant_id'].map(lambda v: scores.get(str(v), {}).get('surprisal_depparse'))")
