# ═══════════════════════════════════════════════════════════════
# KAGGLE NOTEBOOK — Dependency-Parser surprisal ONLY
# One of 3 parallel notebooks (this one, kaggle_pcfg_only.py,
# kaggle_modelblocks_only.py) — run all 3 SIMULTANEOUSLY in separate Kaggle
# sessions. This is the SLOWEST of the 3 (~2.4s/sent with 4 processes) —
# it'll still be running after the other two finish. That's expected and
# fine; it's also why splitting into parallel notebooks helps most here.
#
# Lightest setup of the 3: no C++ compile, no Java — pure Python (Stanza +
# numpy arc-eager oracle).
#
# Before running:
#   1. Upload features.csv as a Kaggle Dataset named "hindi-wordorder-feature"
#   2. Enable Internet (Settings -> Internet: On). GPU not needed.
# ═══════════════════════════════════════════════════════════════


# ── CELL 1: Clone (git-lfs!) — no compile needed for Dep-parser ─
# !apt-get update -qq && apt-get install -y -qq git-lfs
# !git lfs install
# !rm -rf /kaggle/working/space_src
# !git clone https://huggingface.co/spaces/Muraliedhar/SyntacticTreeSurprisal /kaggle/working/space_src
# !cd /kaggle/working/space_src && git lfs pull


# ── CELL 2: Python deps + Stanza Hindi model ──────────────────────
# !pip install -q flask flask-cors stanza indic_transliteration
# import stanza
# stanza.download('hi', verbose=False)


# ── CELL 3: Import dep_incremental_backend ────────────────────────
import os
import sys
import json
import time
from pathlib import Path

WORKSPACE = "/kaggle/working/space_src/workspace"
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)
sys.path.insert(0, str(Path(WORKSPACE) / "dep_incremental"))

import dep_incremental_backend as dep   # noqa: E402

dep_model_dir = Path(WORKSPACE) / "dep_incremental" / "models" / "hi_hdtb_modified_v1"
assert (dep_model_dir / "wts_0.npz").exists(), (
    f"dep-parser model missing: {dep_model_dir}. "
    f"Available: {list((Path(WORKSPACE) / 'dep_incremental' / 'models').iterdir())}"
)
print("Dep-parser model found. Ready.")


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
all_unique_texts = list(text_to_vids.keys())
print(f'Unique sentences total: {len(all_unique_texts):,}')

# ── 6-WAY PARALLEL SPLIT ──────────────────────────────────────────────────
# Each of the 6 sessions (3 per Kaggle account x 2 accounts) runs this SAME
# script with a DIFFERENT SLICE_INDEX below. SET THIS TO A DIFFERENT VALUE
# (0,1,2,3,4,5) IN EACH SESSION — if two sessions ever use the same value,
# that slice gets double-scored (wasted compute, not wrong data) AND one
# other slice gets silently skipped (a real gap in the final merged output).
#
# Correctness note: this slices the FULL, FIXED 78,865-sentence list (same
# in every session, since it's built identically from the same features.csv)
# BEFORE any "already scored" filtering happens. That's deliberate — if we
# instead sliced each session's own already-scored-filtered `todo` list, the
# slice boundaries would shift differently per session (since each session's
# own prior progress differs), causing exactly the overlap/gap risk this is
# designed to avoid. Slicing the fixed full list first, then filtering
# per-session afterward (in Cell 7), keeps the 6 slices deterministic and
# disjoint no matter what each session already has done.
SLICE_INDEX = 0       # <-- CHANGE THIS: 0, 1, 2, 3, 4, or 5 — one unique value per session
NUM_SLICES  = 6

unique_texts = all_unique_texts[SLICE_INDEX::NUM_SLICES]

print(f'\n{"="*60}')
print(f'  SLICE {SLICE_INDEX} of {NUM_SLICES}  ->  {len(unique_texts):,} sentences assigned to THIS session')
print(f'{"="*60}')
print(f'First 2 in this slice: {unique_texts[:2]}')
print(f'Last 2 in this slice:  {unique_texts[-2:]}')
print(f'Sanity check — total across all 6 slices should sum to {len(all_unique_texts):,}: '
      f'verify each session shows a DIFFERENT SLICE_INDEX and the slice sizes are all close to '
      f'{len(all_unique_texts)//NUM_SLICES:,} before you start Cell 7 in each.')


# ── CELL 5: Scoring function + parallel (PROCESSES, not threads — the
#            Space's own code wraps the arc-eager parse in a
#            threading.Lock(), so threads give ZERO speedup (confirmed
#            empirically). Separate processes each get their own unlocked
#            copy of that state. Warm up Stanza in the parent BEFORE
#            forking so workers inherit the loaded model via COW. ────
import multiprocessing as mp

N_WORKERS = 4


def score_depparse_one(text):
    try:
        for r in dep.parse_units(text, 'sentence', 'hi_hdtb_modified_v1'):
            return r.get('total_surprisal')
    except Exception:
        pass
    return None


try:
    import torch
    torch.set_num_threads(1)   # avoid fork + torch-internal-thread-pool hazard
except ImportError:
    pass

print('Warming up Stanza pipeline in the parent process (one-time)...')
t0 = time.time()
_ = score_depparse_one('यह एक परीक्षण वाक्य है')
print(f'  warm-up done in {time.time()-t0:.1f}s')

_MP_CTX = mp.get_context('fork')

# PERSISTENT pool — created ONCE here, reused across every batch in Cell 7,
# instead of forking a fresh pool per 300-sentence batch. Saves the
# fork/teardown cost on every batch (small per-batch, adds up over ~170+
# batches in a 50k-sentence run). maxtasksperchild=100 still periodically
# recycles each worker after 100 tasks — a long-lived worker accumulating
# memory over tens of thousands of Stanza calls is a real risk; this caps it
# without going back to a fresh pool every single batch.
_PERSISTENT_POOL = _MP_CTX.Pool(processes=N_WORKERS, maxtasksperchild=100)


def score_depparse_many_parallel(texts):
    return _PERSISTENT_POOL.map(score_depparse_one, texts)


# ── CELL 5c: GPU QUICK TEST — SEQUENTIAL ONLY, no multiprocessing ─
# ONLY run this cell if you turned on GPU in Notebook Settings. Do NOT use
# score_depparse_many_parallel (Cell 7's multiprocessing) with GPU on — fork()
# + CUDA is unsafe and commonly crashes ("Cannot re-initialize CUDA in forked
# subprocess"). This is a safe, sequential, single-process test to measure
# raw per-sentence GPU latency before deciding anything further.
#
# import torch
# print('CUDA available:', torch.cuda.is_available())
# print('Device Stanza will use:', 'cuda' if torch.cuda.is_available() else 'cpu (GPU not detected!)')
#
# GPU_TEST_LIMIT = 100
# gpu_test_texts = unique_texts[:GPU_TEST_LIMIT]
# t0 = time.time()
# gpu_test_vals = [score_depparse_one(t) for t in gpu_test_texts]
# t_gpu = time.time() - t0
# n = len(gpu_test_texts)
# real = sum(1 for v in gpu_test_vals if v is not None)
# print(f'Sequential GPU: {n} sents in {t_gpu:.1f}s  ({t_gpu/n*1000:.0f} ms/sent)  real={real}/{n}')
# print(f'Compare to sequential CPU baseline: 4850 ms/sent (measured earlier, no GPU, no parallelism)')
# print(f'Full-run ETA at this rate: {t_gpu/n*78865/60:.1f} min')


# ── CELL 6b: RESUME ACROSS SESSIONS + seed from the old combined
#             checkpoint (pre-3-way-split), if attached ─────────
# Filenames include SLICE_INDEX so the 6 parallel sessions never collide.
RESULTS_JSON = str(OUT_DIR / f'depparse_slice{SLICE_INDEX}_checkpoint.json')
CHECKPOINT_DATASET = f'/kaggle/input/depparse-checkpoint/depparse_slice{SLICE_INDEX}_checkpoint.json'
OLD_COMBINED_CHECKPOINT = '/kaggle/input/syntactic-checkpointer/kaggle_syntactic_surprisal.json'

in_this_slice = set(unique_texts)   # restrict everything below to THIS session's slice only

scores = {}   # text -> surprisal_depparse (only for texts in this slice)
if Path(RESULTS_JSON).exists():
    with open(RESULTS_JSON, encoding='utf-8') as f:
        scores = json.load(f)
    scores = {t: v for t, v in scores.items() if t in in_this_slice}
    print(f'Resumed (this session): {len(scores):,} sentences already scored (in this slice)')
elif Path(CHECKPOINT_DATASET).exists():
    with open(CHECKPOINT_DATASET, encoding='utf-8') as f:
        scores = json.load(f)
    scores = {t: v for t, v in scores.items() if t in in_this_slice}
    print(f'Restored own checkpoint from previous session: {len(scores):,} sentences (in this slice)')
else:
    print('No own checkpoint found — starting fresh (will still seed from combined, below).')

if Path(OLD_COMBINED_CHECKPOINT).exists():
      with open(OLD_COMBINED_CHECKPOINT, encoding='utf-8') as f:
          combined = json.load(f)
      seeded = 0
      for text, vals in combined.items():
          if text not in in_this_slice or text in scores:
              continue
          if isinstance(vals, dict):
              val = vals.get('surprisal_depparse')
          else:
              val = vals   # flat schema — the value IS the depparse score directly
          if val is not None:
              scores[text] = val
              seeded += 1
      print(f'Seeded {seeded:,} sentences from the old combined checkpoint')
      with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
          json.dump(scores, f, ensure_ascii=False)

# ── CELL 7: FULL RUN (resumable, checkpointed every 100 sentences) ─
# RETRY fix: scores.get(t) is None catches BOTH "never attempted" (t not in
# scores) AND "attempted but failed" (scores[t] is explicitly None) — a
# sentence that failed once is no longer permanently stuck; it gets retried
# on this run. Bounded naturally: this is one pass through `todo`, not a
# while-loop, so a sentence that keeps failing just costs one more attempt
# per run, not an infinite retry loop.
todo = [t for t in unique_texts if scores.get(t) is None]
print(f'To score (incl. retrying previous failures): {len(todo):,} / {len(unique_texts):,} unique sentences')

CHECKPOINT_EVERY = 100   # smaller than before — less lost work per crash
t0 = time.time()

for batch_start in range(0, len(todo), CHECKPOINT_EVERY):
    batch = todo[batch_start:batch_start + CHECKPOINT_EVERY]
    bn = len(batch)

    bt0 = time.time()
    vals = score_depparse_many_parallel(batch)   # uses the PERSISTENT pool now
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
_PERSISTENT_POOL.close()
_PERSISTENT_POOL.join()


# ── CELL 8: Expand back out to variant_id, verify, download instructions ─
# Only emit variant_ids whose TEXT is in THIS slice — NOT all of
# text_to_vids. Writing a None for every sentence outside this slice would
# let a later-merged file silently overwrite another slice's real values
# with None during the final 6-way merge.
by_vid = {}
for text in unique_texts:   # this slice only
    val = scores.get(text)
    for vid in text_to_vids[text]:
        by_vid[str(vid)] = val

DEP_JSON = str(OUT_DIR / f'depparse_slice{SLICE_INDEX}_surprisals_new.json')
with open(DEP_JSON, 'w', encoding='utf-8') as f:
    json.dump(by_vid, f, ensure_ascii=False)

real = sum(1 for v in by_vid.values() if v is not None)
print(f'surprisal_depparse (slice {SLICE_INDEX}): {real:,}/{len(by_vid):,} real scores')
print(f'\nDownload from /kaggle/working/: depparse_slice{SLICE_INDEX}_surprisals_new.json')
print('Place at: models/dependency_parser/depparse_slice{}_surprisals_new.json'.format(SLICE_INDEX))
print('Once ALL 6 slices are done (all in models/dependency_parser/), run')
print('scripts/merge_depparse_slices.py (verifies full coverage,')
print('no gaps/overlaps), then merge_three_checkpoints.py, then assemble_features.py as usual.')
