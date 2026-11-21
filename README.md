# Hindi Non-Canonical Word Order: LLM vs. LSTM Surprisal

Replication of Ranjan & van Schijndel (2024, CogSci), *"Does Dependency
Locality Predict Non-canonical Word Order in Hindi?"*, on UD_Hindi-HDTB,
extended by replacing the original paper's LSTM surprisal feature
with surprisal from two pretrained LLMs (XGLM-1.7B, Gemma-3).

This README is written so that anyone — not just someone already familiar
with the project — can clone/copy the repo and get it running. It is split
into two tiers:

- **Tier 1** — verify the paper's reported numbers from already-computed
  data. Takes a few minutes, no GPU needed. Do this first.
- **Tier 2** — regenerate everything from raw text (treebank → trained
  models → surprisal scores). Takes hours, needs a GPU/Java/Colab/Kaggle.
  Only needed if you don't trust the cached numbers and want to rebuild
  them yourself.

Everything below in Tier 1 was actually run end-to-end while writing this
README, so the commands, file paths, and output numbers are copy-pasted
from a real run, not assumed.

## Live demo notebook )

`notebooks/live_demo_mtp1_mtp2.ipynb` runs the entire project end to end,
cell by cell, either **locally** (no GPU, no internet) **or in Google Colab**
(the notebook's Setup cells auto-detect Colab, mount Drive, and install any
missing packages — see the notebook's own Setup section for what to upload).
MTP-2's project (`hindi_dialog_generation`) is merged in as a subfolder of
this checkout (`hindi_wordorder_replication/hindi_dialog_generation/`), so
**only this one folder needs to be uploaded/synced to Drive** for Colab —
no separate MTP-2 upload required.

**Scope split** (as scoped with the supervisor): **MTP-1 is the replication
only** (Parts 1–4: load the treebank, filter it, generate variants, extract
the paper's original 5 features, reproduce the paper's reported accuracy).
**Everything beyond that is MTP-2** (Parts 5–10): swapping the LSTM
surprisal feature for XGLM-1.7B/Gemma-3, the 3 new tool-based syntactic
features (PCFG/Modelblocks/Dep-parser), the global bidirectional
cross-check, and the dialogue-generation pilot.

1. **[MTP-1]** Reloads the raw treebank and reruns filtering + variant
   generation live (a correctness check against the cached feature table),
   then reproduces the paper's reported accuracy with the original 5-feature
   LSTM classifier — this is the end of MTP-1,
2. **[MTP-2]** The LSTM→XGLM/Gemma-3 surprisal swap, the 3 new tool-based
   syntactic features, and a global bidirectional cross-check,
3. **[MTP-2]** **Merges in the dialogue-generation pilot** — reads
   `hindi_dialog_generation`'s own generated dialog variants + XGLM
   surprisal scores (read-only; nothing is written there) and scores them
   with this project's own `src/ranking/` classifier,
4. Prints a final summary across every stage, MTP-1 through MTP-2.

Every number in it is computed fresh by the notebook itself — the **only**
hardcoded values anywhere are Ranjan & van Schijndel (2024)'s own published
results, used purely as the external comparison baseline in Part 4 (flagged
explicitly in that cell). Runs in well under a minute; open it in Jupyter and
"Restart & Run All".

## What's in this repository

## What's in this repository

```

 hindi_wordorder_replication/   # the data pipeline + all analysis code (start here)
├── data/                      # treebank, variants, extracted features
├── models/                    # trained model weights + cached surprisal scores (JSON)
├── scripts/                   # runnable entry-point scripts (Tier 1 & Tier 2)
├── src/                       # library code imported by scripts/
├── notebooks/                 # Colab notebook: downloads HDTB + Wikipedia data,
                            #   trains the adaptive LSTM (Tier 2)
├── results/                   # CSV/HTML outputs — created/updated by the scripts below
└── requirements.txt
                     # the paper draft (paper.tex + custom.bib + figures/)
### Pipeline at a glance

```
RAW DATA                         PROCESSING                              FEATURES              RESULTS
─────────                        ──────────                              ────────              ───────
UD_Hindi-HDTB (GitHub)    ┐
                          ├──▶ [Step 1] loader.py reads it        ┐
Hindi Wikipedia dump      ┘                                       │
(dumps.wikimedia.org)           [Step 2] run_filter_and_generate.py
                                 (filter.py → generator.py →      │
                                  grammar_filter.py)              │
                                       │                          │
                                       ▼                          │
                         output/variants_final/variants.csv       │
                                       │ (copy by hand)            │
                                       ▼                          │
                     data/processed/variants/variants.csv  ◀──────┘
                                       │
            ┌──────────────┬──────────┼──────────┬──────────────┬─────────────────┐
            ▼              ▼          ▼          ▼              ▼                 ▼
      [Step 3]        [Step 4]   [Step 5]   [Step 6]        [Step 7]          [Step 10]
    LSTM (Colab)    PCFG(local) Trigram+Wiki IS score     XGLM/Gemma      PCFG/Modelblocks/
  lstm_model.pt   pcfg_surprisals  (Kaggle+     is_score    (Kaggle GPU)    Dep-parser (Kaggle,
  +surprisals.json   .json      local KenLM)   .json      xglm/gemma      3 parallel notebooks)
                                 trigram_        .json      _surprisals     syntactic_
                                 surprisals.json            .json          surprisal.json
            │              │          │          │              │                 │
            └──────────────┴──────────┴──────────┴──────────────┴─────────────────┘
                                       │
                                       ▼
                    [Step 11] assemble_features.py merges all of the above
                   with output/variants_final/variants_scored.csv as the base
                                       │
                                       ▼
                data/processed/features/features.csv  (9 feature columns)
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 ▼                     ▼                     ▼
        Tier 1 scripts          "2b" scripts            generate_report.py /
   (compare_with_paper.py,  (joachims_transformation_   05_make_llm_figures.py
    run_llm_comparison.py,   general.py, run_classifier.py,
    error_analysis_          error_analysis_lstm_vs_
    llm_generalized.py)      gemma.py)
                 │                     │                     │
                 ▼                     ▼                     ▼
              results/*.csv                           *.pdf,*.png
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer (developed/tested with 3.14.6)
- ~4 GB free disk space if you want the full repo including trained-model
  weights (see "Large files" below); only a few MB if you only run Tier 1

## 1. Setup (one-time)

```bash
cd hindi_wordorder_replication

# create a virtual environment
python -m venv venv

# activate it
venv\Scripts\activate          # Windows (cmd.exe)
.\venv\Scripts\Activate.ps1    # Windows (PowerShell)
source venv/bin/activate       # macOS / Linux

# install dependencies
pip install -r requirements.txt
```

`requirements.txt` only lists what Tier 1 needs (pandas, numpy, scipy,
scikit-learn, statsmodels, matplotlib, seaborn). Tier 2's heavier,
optional dependencies (torch, transformers, etc.) are commented out in
that same file — install them only if you're doing Tier 2 steps that
need them.

## 2. Verify the paper's results (Tier 1)

Run these three scripts **in this order**, from inside
`hindi_wordorder_replication/`. Each one is described below exactly as
observed when run, including which files it creates or updates.

### Step 1 — `python scripts/compare_with_paper.py`

Reads the precomputed `data/processed/features/features.csv` plus the
cached PCFG/IS-score JSON files and prints a side-by-side comparison of
our regression coefficients against the original paper's, for both the
DO-fronted and IO-fronted constructions. **Writes no files** — console
output only.

Actual output (abridged):

```
ACCURACY SUMMARY
  Construction      Our Refs  Paper Refs   Our Acc  Paper Acc    Diff
  -----------------------------------------------------------------
  DO-fronted             156         133    78.60%      80.5%    -1.9%
  IO-fronted              91         101    92.61%      90.0%    +2.6%
```

### Step 2 — `python scripts/run_llm_comparison.py`

Reproduces the LLM-extension comparison table: single-feature ablations
plus full 5-feature models for LSTM / XGLM-1.7B / Gemma-3.

**Creates/overwrites:**
- `results/llm_comparison.csv` — per-model accuracy comparison table
- `results/llm_coefficients.csv` — regression coefficients for every model variant

(If either file is open/locked elsewhere, the script falls back to
writing `results/llm_comparison_new.csv` / `results/llm_coefficients_new.csv`
instead, so it never crashes on a locked file.)

Actual accuracy numbers from this run:

| | DO-fronted | IO-fronted |
|---|---|---|
| LSTM (paper's setup) | 78.6% | 92.6% |
| LSTM → XGLM-1.7B | 90.9% | 94.2% |
| LSTM → Gemma-3 | 81.3% | 93.1% |

### Step 3 — `python scripts/error_analysis_llm_generalized.py`

Reproduces the dependency-length/OOV-rate stratified error analysis
(which sentences the LLMs get right that the LSTM gets wrong, and why).

**Creates/overwrites:**
- `results/error_analysis_llm_generalized.csv`
- `results/error_analysis_llm_generalized.html`

Actual output (abridged):

```
[XGLM-1.7B]  N=247
  both_correct                :  168  (68.0%)
  lstm_wrong_xglm_right       :   41  (16.6%)
  both_wrong                  :   25  (10.1%)
  lstm_right_xglm_wrong       :   13  (5.3%)

Saved CSV:  results/error_analysis_llm_generalized.csv
```

### Summary: what gets written where (Tier 1)

| Script | File(s) created/updated |
|---|---|
| `compare_with_paper.py` | none (console only) |
| `run_llm_comparison.py` | `results/llm_comparison.csv`, `results/llm_coefficients.csv` |
| `error_analysis_llm_generalized.py` | `results/error_analysis_llm_generalized.csv`, `.html` |

If your numbers differ from the table above, something upstream changed —
re-check against `results/llm_comparison.csv` and
`results/llm_coefficients.csv`, which hold these same values directly.

### Summary: what gets written where (Tier 2, including Kaggle)

| Script | Where it runs | File(s) created/updated |
|---|---|---|
| `run_filter_and_generate.py` | locally | `output/variants_final/{variants.csv, variants_tokens.jsonl, variants.conllu}`, `results/filter_comparison.csv`, `results/rejected_sentences.csv`, `results/variant_summary.csv` |
| `kaggle_srilm_trigram.py` | Kaggle (CPU) | `hindi_trigram_srilm.arpa` → copy to `models/trigram/` |
| `notebooks/MTP_prject_folder_structure.ipynb` | Colab | `models/adaptive_lstm/lstm_model.pt`, `vocab.json` |
| `run_scoring.py` | locally | `output/variants_final/variants_scored.csv` (adds `surprisal_trigram`, `is_score`) |
| `kaggle_plain_lstm_scoring.py` | Kaggle (GPU optional) | `lstm_plain_surprisals.json` → copy to `models/adaptive_lstm/` |
| `kaggle_llm_scoring.py` | Kaggle (T4 GPU) | `xglm_surprisals.json` → copy to `models/xglm/`; `gemma3_4b_surprisals.json` → copy to `models/gemma-3b/` |
| `kaggle_syntactic_surprisal.py` (or the `kaggle_*_only.py` split + `merge_depparse_slices.py` + `merge_three_checkpoints.py`) | Kaggle (CPU) | `results/syntactic_surprisal.json` (`surprisal_pcfg`, `surprisal_modelblocks`, `surprisal_depparse`) |
| `assemble_features.py` | locally | `data/processed/features/features.csv` (rebuilds the shared Tier 1 input, merging every file above) |

Superseded, not part of the current pipeline (kept only for history,
not used by `assemble_features.py` or any Tier 1 script): `build_pcfg_berkeley.py`
(old whole-sentence Berkeley-only PCFG), `build_trigram_filtered.py` (old
KenLM trigram, replaced by the SRILM version above), `compute_is_scores.py`
(IS score is now computed inline by `run_scoring.py`), and
`run_analysis_no_pcfg.py` (old features.csv builder from the retired
`O-fronted`-era data path).

Note: a handful of older result files (`variant_comparison.csv`,
`ref_sentences_comparison.csv`, `regression_results_full.csv`,
`regression_results_with_pcfg.csv`, and others) predate a data fix that
retired a stale `O-fronted` category (see "A note on data history"
below) and no longer match what any current script produces. They are
**not present in this copy of the repo** (removed, not archived, despite
an earlier version of this README saying they'd been moved to a
`results/_archive_stale_o_fronted/` folder — that folder does not exist
here) — if you need them for historical comparison, you'd need an older
snapshot of the project.

## 3. Full reproduction from raw text (Tier 2, optional)

Only needed if you want to regenerate the cached scores above from
scratch instead of trusting the committed ones. This is slow and needs
extra tooling (GPU, Java, or a Colab/Kaggle account). Rough pipeline
order, run from `hindi_wordorder_replication/`:

1. **Get the treebank.** Download `hi_hdtb-ud-{train,dev,test}.conllu`
   from `https://github.com/UniversalDependencies/UD_Hindi-HDTB` into
   `data/ud_hindi/` (openly licensed, no restrictions).
2. **Filter + generate variants.**
   `python scripts/run_filter_and_generate.py` — loads the treebank,
   filters to sentences with well-defined subjects/objects, and
   generates non-canonical word-order variants via
   `src/variant_generation/generator.py`'s `classify_order()`
   (Canonical / DO-fronted / IO-fronted). Writes `variants.csv`,
   `variants_tokens.jsonl`, `variants.conllu` to `output/variants_final/`
   — every downstream step in this list reads directly from that folder,
   no copy step needed.
3. **Trigram LM (SRILM).** The paper trained its trigram baseline with
   SRILM (Good-Turing discounting), not KenLM — `kaggle_srilm_trigram.py`
   is a Kaggle notebook that trains the real SRILM version from scratch
   on the ~948,581-sentence Hindi Wikipedia corpus (same corpus the LSTM
   trains on), using the `ngram-count`/`ngram` SRILM binaries uploaded as
   a Kaggle dataset. Download its output and place it at
   `models/trigram/hindi_trigram_srilm.arpa`. (Skip this if that file
   already exists — it's ~370 MB and already committed in this repo.)
4. **LSTM training.** Run
   `notebooks/MTP_prject_folder_structure.ipynb` on Google Colab — it
   installs its own dependencies (`torch`, `transformers`, `nltk`,
   `spacy`, `benepar`, `conllu`, `kenlm`, etc., see the `!pip install`
   cells near the top) and trains a 2-layer/650-hidden LSTM from scratch
   on 948,581 Hindi Wikipedia sentences, 40 epochs, CPU. This is the
   slowest step — budget real time for it. (Skip if `models/adaptive_lstm/
   lstm_model.pt` + `vocab.json` already exist — they're committed.)
5. **Trigram surprisal + IS score.** `python scripts/run_scoring.py` —
   reads `output/variants_final/variants.csv`, computes `surprisal_trigram`
   (via the SRILM ARPA model from step 3) and `is_score` (givenness, via
   `src/feature_extraction/givenness.py`), and writes
   `output/variants_final/variants_scored.csv` — this is the base file
   `assemble_features.py` (step 9) builds on.
6. **Plain-LSTM surprisal (Kaggle flow).** `scripts/kaggle_plain_lstm_scoring.py`
   — paste its `# ── CELL N` blocks into a Kaggle notebook. Per the
   professor's instruction, this uses the already-trained LSTM's plain
   forward pass (no per-sentence adaptation step, unlike the paper's
   "adaptive" LSTM) — same model weights, no retraining. Upload a Kaggle
   dataset with `lstm_model.pt`, `vocab.json`, `variants.csv`,
   `variants_tokens.jsonl`; GPU optional (plain forward pass is cheap).
   Download the output and place it at
   `models/adaptive_lstm/lstm_plain_surprisals.json`.
7. **XGLM-1.7B / Gemma-3 surprisal scoring (Kaggle flow).**
   `scripts/kaggle_llm_scoring.py` — paste its `# ── CELL N` blocks into
   a Kaggle Notebook. Full steps:
   1. Upload `data/processed/features/features.csv` to Kaggle as a
      Dataset named `hindi-features` (so it lands at
      `/kaggle/input/hindi-features/features.csv`, the path the script
      expects).
   2. Create a new Kaggle Notebook, attach that dataset, then in
      **Notebook settings** turn on **GPU (T4 x1)** and **Internet**.
   3. For Gemma-3 only: accept the license at
      `huggingface.co/google/gemma-3-4b-pt`, then add your HuggingFace
      token under **Add-ons → Secrets** with the name `HF_TOKEN`.
   4. Run the cells top to bottom. Expect ~30 min for XGLM-1.7B and
      ~45 min for Gemma-3-4B on a T4. Scoring checkpoints every 500 rows,
      so a cell can resume if interrupted.
   5. Download `xglm_surprisals.json` and `gemma3_4b_surprisals.json`
      from `/kaggle/working/` and place them at `models/xglm/
      xglm_surprisals.json` and **`models/gemma-3b/gemma3_4b_surprisals.json`**
      — note the script's own final print statement says
      `models/gemma3/...`, which is wrong; `run_llm_comparison.py`
      actually reads from `models/gemma-3b/`, so use that path.

      Locally (no Kaggle), `src/surprisal_models/xglm_surprisal.py` and
      `gemma_surprisal.py` do the same scoring if you have GPU access —
      install the optional `torch`/`transformers`/`accelerate`/
      `bitsandbytes` block in `requirements.txt` first.
8. **PCFG / Modelblocks / Dep-parser surprisal (the 3 tool-based syntactic
   features).** Per the professor's instruction, PCFG surprisal for this
   project must come from the same hosted tool as the Modelblocks and
   Dep-parser features — **not** from the older, separate
   `scripts/build_pcfg_berkeley.py` pipeline (a different, whole-sentence
   5-fold-CV custom-grammar methodology; `assemble_features.py` never
   merges its output). Two ways to get these 3 scores, both producing the
   same `{variant_id: {surprisal_pcfg, surprisal_modelblocks,
   surprisal_depparse}}` JSON schema:
   - **Combined (simpler, slower):** `scripts/kaggle_syntactic_surprisal.py`
     — one Kaggle notebook (CPU-only, Internet required) that self-hosts
     the open-source HF Space `Muraliedhar/SyntacticTreeSurprisal`
     (Berkeley Parser for PCFG, `synproc`/C++ for Modelblocks, Stanza +
     a vendored arc-eager oracle for Dep-parser) and scores all 3
     features sequentially. Download its output and place it at
     `results/syntactic_surprisal.json` directly.
   - **Split (faster, parallel):** run `kaggle_pcfg_only.py`,
     `kaggle_modelblocks_only.py`, and `kaggle_depparse_only.py`
     simultaneously in 3 separate Kaggle sessions (PCFG is fastest,
     Dep-parser slowest — split Dep-parser further across up to 6
     parallel sessions/accounts using its `SLICE_INDEX=0..5` setting if
     you need more speed). Then:
     1. `python scripts/merge_depparse_slices.py` — merges the 6
        downloaded `depparse_slice{N}_surprisals_new.json` files (place
        them in `models/dependency_parser/` first) into one
        `depparse_surprisals_new.json`. Skip this if you ran the
        Dep-parser notebook as a single (non-sliced) session.
     2. `python scripts/merge_three_checkpoints.py` — merges the PCFG
        (`models/pcfg_parser/pcfg_surprisals_new.json`), Modelblocks
        (`models/modelblocks/modelblocks_surprisals_new.json`), and
        Dep-parser outputs into one `results/syntactic_surprisal.json`.
   - A slow, no-Kaggle fallback also exists (`scripts/score_syntactic_surprisal.py`,
     hits the same HF Space's shared public queue over HTTP) but took
     ~5 days projected for the full 78,865-row corpus due to queue
     flakiness — the Kaggle routes above exist specifically to avoid that.
9. **Rebuild `features.csv`.** `python scripts/assemble_features.py` —
   merges the base `output/variants_final/variants_scored.csv` (step 5)
   with `lstm_plain_surprisals.json` (step 6), `xglm_surprisals.json` /
   `gemma3_4b_surprisals.json` (step 7), and `results/syntactic_surprisal.json`
   (step 8) into `data/processed/features/features.csv`. It skips any
   source file that isn't present yet and prints exactly what it merged
   and what's still missing, so you can run it incrementally as each
   Kaggle step finishes rather than waiting for all of them. **Every
   Tier 1 script reads `features.csv`, so this step has to run before
   step 10.**

   (The older `scripts/run_analysis_no_pcfg.py` also writes
   `features.csv`, but from the retired `O-fronted`-era data path — it
   is not part of the current pipeline; use `assemble_features.py`.)
10. Re-run the Tier 1 scripts — they'll pick up your freshly regenerated
    JSON caches and `features.csv` automatically instead of the
    committed ones.

## A note on data history

Earlier in this project, `generator.py`'s `classify_order()` was fixed
(three separate times) after the sentence-label scan found stale
`order_type` values left over from an older version of that function.
The fixes relabeled some sentences between Canonical/DO-fronted/IO-fronted
and **retired a stale `O-fronted` label that the current `classify_order()`
no longer produces at all**. `scripts/fix_stale_canonical_labels.py` and
`fix_stale_other_labels.py` are the one-off scripts that applied those
fixes to `variants.csv`/`features.csv`/`final_var_score.csv` — they don't
need to be re-run; the `.bak_pre_*` files next to those three CSVs are
the pre-fix backups, kept for audit.

A consequence: several older result files in `results/` were generated
*before* this fix and still contain `O-fronted` rows/columns with real
(pre-fix) counts — `variant_comparison.csv`, `ref_sentences_comparison.csv`,
`regression_results_full.csv`, `regression_results_with_pcfg.csv`, and a
couple of others — these are **not included in this copy of the repo**.
**No script can regenerate their `O-fronted` numbers** — that's not a
bug, it's because the category was deliberately retired; re-running any
current script today will correctly show `0` sentences in that category.

If you compare against an older copy of the paper or results,
DO-fronted/IO-fronted reference counts (104→156, 73→91) and all
downstream accuracy numbers will look different — the current numbers
(Tier 1's table above) are the corrected ones.

## Large files not strictly required for Tier 1

This project is large (~4 GB excluding the Python environment), mostly
because of trained-model artifacts. These files account for most of it
and are **only needed for Tier 2**, not for verifying the paper's numbers:

| File | Size | What it is |
|---|---|---|
| `models/trigram/hindi_trigram_srilm.arpa` | 372 MB | the trigram LM actually used by `run_scoring.py` (SRILM-trained, matches the paper's methodology) |
| `models/trigram/hindi_trigram.arpa` | 974 MB | older KenLM-trained trigram LM — superseded by the SRILM one above, kept for history |
| `models/trigram/hindi_trigram.binary` | 385 MB | binary form of that older KenLM LM |
| `models/adaptive_lstm/lstm_model.pt` | 306 MB | trained LSTM weights |

Tier 1 only needs the small, already-computed per-sentence surprisal
scores (`models/*/*.json`, a few MB each), which are already in the repo.

## Data-licensing note

`hindi_wordorder_replication/data/emille_corpus/` contains a copy of a
small subset of the EMILLE corpus (the parallel/translated section,
~6,884 sentences). The **original** Ranjan & van Schijndel paper trained
its trigram baseline on EMILLE — but **this repo's code does not use
it**: none of the scripts in `scripts/`/`src/` reference it, and the
trigram LM actually shipped here (`models/trigram/hindi_trigram.arpa`)
was built from a Hindi Wikipedia dump instead, because this EMILLE subset
is too small (~145x short of the 1M sentences needed). The raw EMILLE
files are only present from early data exploration (see
`notebooks/MTP_prject_folder_structure.ipynb`, cell documenting "DATA
SOURCE DECISION"). EMILLE is licensed by the University of Lancaster —
check that you're allowed to redistribute/receive it before sharing this
folder further, or simply delete `data/emille_corpus/` since nothing in
the pipeline depends on it. UD_Hindi-HDTB, by contrast, is openly
licensed and is re-downloaded by the pipeline rather than committed to
the repo (see Tier 2, step 1).



## Troubleshooting

- **`python` not found / wrong version** — make sure Python 3.10+ is on
  your PATH (`python --version`), and that you've activated the venv
  before running `pip install` or any script.
- **PowerShell blocks `Activate.ps1`** — run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in
  that terminal session, then re-run the activate command.
- **`FileNotFoundError` for a `data/` or `models/` path** — Tier 1 scripts
  must be run from inside `hindi_wordorder_replication/`, not the repo
  root.
- **Results differ from the numbers in this README** — check
  `results/llm_comparison.csv` and `results/llm_coefficients.csv` (the
  same scripts write these files), and confirm you're reading the cached
  JSON surprisal files in `models/*/*.json` rather than ones you
  regenerated via Tier 2.

## Author

Sarita Rani — `m25ai2076@iitj.ac.in`
