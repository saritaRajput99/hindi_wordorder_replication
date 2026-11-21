# ═══════════════════════════════════════════════════════════════
# KAGGLE NOTEBOOK — Train the Hindi trigram LM with real SRILM
# Run each block as a separate cell in Kaggle.
#
# WHY THIS EXISTS:
#   models/trigram/hindi_trigram.arpa (currently used by run_scoring.py)
#   was built with KenLM, not SRILM — the original paper
#   (Ranjan & van Schijndel 2024) used SRILM with Good-Turing
#   discounting on a 1M-sentence corpus. This notebook trains the
#   real thing from scratch so results are directly comparable to
#   the paper, not just "assumed equivalent to KenLM."
#
# Before running, upload TWO Kaggle datasets:
#   1. "hindi-sentences" -> just hindi_sentences.txt
#        (extract it from C:\Users\dell\Downloads\archive.zip —
#         that's the already-built ~948,581-sentence Hindi Wikipedia
#         corpus, same one the adaptive LSTM trained on. No need to
#         re-download/re-parse the Wikipedia XML dump.)
#   2. "srilm-bin" -> ngram-count and ngram
#        (from C:\Users\dell\Downloads\SRILM-Lab\ — just these two
#         binaries, nothing else from that folder is needed)
# Attach both datasets to this notebook (Add Data). No GPU needed —
# trigram counting is CPU-only and fast (minutes).
# ═══════════════════════════════════════════════════════════════


# ── CELL 1: Stage the SRILM binaries and check they actually run ──
import os, shutil, stat

SRILM_IN  = '/kaggle/input/srilm-bin'
SRILM_DIR = '/kaggle/working/srilm'
os.makedirs(SRILM_DIR, exist_ok=True)

for name in ['ngram-count', 'ngram']:
    src = f'{SRILM_IN}/{name}'
    dst = f'{SRILM_DIR}/{name}'
    shutil.copy(src, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

# Sanity check: binary was built outside Kaggle's image, so there's a
# real chance of a glibc/arch mismatch. This surfaces it immediately
# instead of failing silently later.
print('--- file type ---')
os.system(f'file {SRILM_DIR}/ngram-count')
print('--- dynamic deps (look for "not found") ---')
os.system(f'ldd {SRILM_DIR}/ngram-count')
print('--- version/help probe ---')
os.system(f'{SRILM_DIR}/ngram-count -help 2>&1 | head -20')


# ── CELL 2: Point at the corpus (already extracted, just loading) ──
import os

corpus_path = '/kaggle/input/hindi-sentences/hindi_sentences.txt'
assert os.path.exists(corpus_path), f'Not found: {corpus_path} — check the dataset mount name'

n_lines = sum(1 for _ in open(corpus_path, encoding='utf-8'))
print(f'Corpus: {corpus_path}')
print(f'Sentences: {n_lines:,}')
with open(corpus_path, encoding='utf-8') as f:
    for _ in range(3):
        print(' ', next(f).strip())


# ── CELL 3: Train the trigram LM with real SRILM ───────────────────
# Default ngram-count smoothing (no -kndiscount/-wbdiscount flag) is
# Good-Turing — matches "the original paper used SRILM with
# Good-Turing discounting." -unk ensures an <unk> entry is written,
# which our local ARPAModel reader (src/surprisal_models/surprisal.py)
# expects.
import os

out_arpa = '/kaggle/working/hindi_trigram_srilm.arpa'

cmd = (
    f'{SRILM_DIR}/ngram-count '
    f'-text {corpus_path} '
    f'-order 3 '
    f'-unk '
    f'-lm {out_arpa}'
)
print(cmd)
ret = os.system(cmd)
print('exit code:', ret)
print('output size (MB):', os.path.getsize(out_arpa) / 1e6 if os.path.exists(out_arpa) else 'MISSING')


# ── CELL 4: Quick sanity check on the produced ARPA file ───────────
with open('/kaggle/working/hindi_trigram_srilm.arpa', encoding='utf-8', errors='replace') as f:
    head = [next(f) for _ in range(15)]
print(''.join(head))
# Expect: \data\, ngram 1=.../2=.../3=... counts, then \1-grams:,
# \2-grams:, \3-grams: sections — standard ARPA format, the exact
# format src/surprisal_models/surprisal.py's ARPAModel.load() already
# reads. No code changes needed on the local/Windows side.


# ── CELL 5: Done — download from the Kaggle Output panel ──────────
print('Download hindi_trigram_srilm.arpa from the Kaggle Output panel.')
print('Then locally:')
print('  hindi_wordorder_replication/models/trigram/hindi_trigram_srilm.arpa')
print('  from src.surprisal_models.surprisal import ARPAModel')
print("  arpa = ARPAModel(); arpa.load('models/trigram/hindi_trigram_srilm.arpa')")
print('  # score variants.csv -> surprisal_trigram_srilm, compare to the')
print('  # existing KenLM-based surprisal_trigram column')


# ── FALLBACK: if the uploaded ngram-count binary won't run on Kaggle ──
# (glibc/arch mismatch is common when a binary is compiled on a
# different Linux image than Kaggle's). Options, in order of effort:
#   1. Try the other SRILM-Lab zip timestamps you have locally
#      (20261005.../20261008...) — one may be a different build.
#   2. Build SRILM from source on Kaggle: free registration for the
#      source tarball at
#      https://www.speech.sri.com/projects/srilm/download.html,
#      upload that tarball as a dataset, `make` on Kaggle (SRILM's
#      own Makefile; ~10-15 min build, no GPU needed).
