"""
DEMO SCRIPT 4 — Analysis & P-values
=====================================
Shows the logistic regression results, coefficients, p-values,
and classifier accuracy — the core statistical analysis.

Run:
    cd D:\MTP_WO\hindi_wordorder_replication
    python scripts/04_show_analysis_pvalues.py
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import pandas as pd
import numpy as np

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ─── Explain the analysis ────────────────────────────────────────────────────
print()
print("=" * 65)
print("  ANALYSIS APPROACH  (Ranjan & van Schijndel 2024)")
print("=" * 65)
print("""
  Question: What predicts Hindi word order — DO/IO fronting?

  Model: Logistic Regression with pairwise ranking transform.
         10-fold cross-validation.

  Predictors (features):
    1. Surprisal (trigram LM)   — how unexpected is the word order?
    2. Surprisal (PCFG)         — structural surprisal
    3. Surprisal (adaptive LSTM)— neural surprisal
    4. Dependency Length (DL)   — syntactic complexity
    5. Information Status (IS)  — is the fronted NP 'given' (discourse-old)?

  Hypothesis:
    Non-canonical orders (DO-fronted, IO-fronted) are predicted by:
    → Lower surprisal (more expected in context)
    → Shorter dependency length
    → Higher givenness of the fronted element
""")

# ─── Load precomputed classifier results ─────────────────────────────────────
print("=" * 65)
print("  CLASSIFIER ACCURACY RESULTS")
print("=" * 65)

clf_path = os.path.join(RESULTS_DIR, 'classifier_results.csv')
df_clf   = pd.read_csv(clf_path)
print()
print(df_clf.to_string(index=False))

# ─── Table 2/3: Coefficients and p-values ────────────────────────────────────
print()
print("=" * 65)
print("  TABLE 2 — DO-fronted  (DOSV corpus vs SDOV canonical variants)")
print("  Paper N = 1,663")
print("=" * 65)

PAPER_TABLE2 = {
    'surprisal_trigram'       : {'coef': -2.08, 'se': 0.245, 'z': -8.48, 'sig': '***', 'p': '<0.001'},
    'surprisal_adaptive_lstm' : {'coef': -1.65, 'se': 0.240, 'z': -6.86, 'sig': '***', 'p': '<0.001'},
    'dl'                      : {'coef': -0.58, 'se': 0.091, 'z': -6.35, 'sig': '***', 'p': '<0.001'},
    'is_score'                : {'coef': +0.26, 'se': 0.055, 'z': +4.68, 'sig': '***', 'p': '<0.001'},
    'surprisal_pcfg'          : {'coef': +0.52, 'se': 0.135, 'z': +3.88, 'sig': '***', 'p': '<0.001'},
}

print()
print(f"  {'Feature':<28} {'Coef':>7} {'SE':>7} {'z':>7} {'p':>8}  Sig")
print("  " + "-" * 65)
for feat, vals in PAPER_TABLE2.items():
    print(f"  {feat:<28} {vals['coef']:>7.2f} {vals['se']:>7.3f} "
          f"{vals['z']:>7.2f} {vals['p']:>8}  {vals['sig']}")

print()
print("  Interpretation:")
print("  - surprisal_trigram coef = -2.08 (***)")
print("    → When the corpus sentence has LOWER surprisal than the variant,")
print("      the corpus order is strongly preferred.  Most powerful predictor.")
print()
print("  - dl coef = -0.58 (***)")
print("    → Corpus sentences also have shorter dependency lengths.")
print("      DLT is supported for DO-fronted constructions.")
print()
print("  - is_score coef = +0.26 (***)")
print("    → Fronted DOs tend to be discourse-given (information status).")
print("      Given elements are placed earlier (Given-Before-New principle).")

# ─── Table 3 ─────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  TABLE 3 — IO-fronted  (IOSV corpus vs SIOV canonical variants)")
print("  Paper N = 1,353")
print("=" * 65)

PAPER_TABLE3 = {
    'surprisal_trigram'       : {'coef': -3.05, 'se': 0.382, 'z': -7.976, 'sig': '***', 'p': '<0.001'},
    'surprisal_adaptive_lstm' : {'coef': -2.83, 'se': 0.365, 'z': -7.780, 'sig': '***', 'p': '<0.001'},
    'dl'                      : {'coef': +0.15, 'se': 0.142, 'z': +1.045, 'sig': 'n.s.', 'p': '0.296'},
    'is_score'                : {'coef': +0.13, 'se': 0.074, 'z': +1.767, 'sig': 'n.s.', 'p': '0.077'},
    'surprisal_pcfg'          : {'coef': -0.89, 'se': 0.238, 'z': -3.725, 'sig': '***', 'p': '<0.001'},
}

print()
print(f"  {'Feature':<28} {'Coef':>7} {'SE':>7} {'z':>7} {'p':>8}  Sig")
print("  " + "-" * 65)
for feat, vals in PAPER_TABLE3.items():
    print(f"  {feat:<28} {vals['coef']:>7.2f} {vals['se']:>7.3f} "
          f"{vals['z']:>7.2f} {vals['p']:>8}  {vals['sig']}")

print()
print("  Key contrast with Table 2:")
print("  - dl coef = +0.15 (n.s., p=0.296)")
print("    → DL does NOT significantly predict IO-fronting.")
print("      IO-fronted sentences are NOT shorter on average.")
print()
print("  - Surprisal is still the dominant predictor (both ***).")
print("    IO-fronting is driven by contextual expectation, not length.")

# ─── Table 4: Accuracy by predictor ──────────────────────────────────────────
print()
print("=" * 65)
print("  TABLE 4 — Accuracy by Predictor and Order Type")
print("=" * 65)

t4_path = os.path.join(RESULTS_DIR, 'tables', 'table4_accuracy_by_predictor.csv')
if os.path.exists(t4_path):
    df_t4 = pd.read_csv(t4_path)
    print()
    print(df_t4.to_string(index=False))
else:
    print("  (table4 file not found — showing paper targets)")
    print()
    print(f"  {'Order Type':<16} {'All Features':>14} {'Surprisal only':>16} {'DL only':>9}")
    print("  " + "-" * 58)
    print(f"  {'DO-fronted':<16} {'87.5%':>14} {'84.2%':>16} {'61.3%':>9}")
    print(f"  {'IO-fronted':<16} {'82.3%':>14} {'80.1%':>16} {'52.4%':>9}")

# ─── DO-fronted key result ────────────────────────────────────────────────────
print()
print("=" * 65)
print("  KEY RESULT — DO-fronted Classifier Accuracy")
print("=" * 65)

meta_path = os.path.join(RESULTS_DIR, 'tables', 'table2_do_fronted_metadata.json')
with open(meta_path) as f:
    do_meta = json.load(f)

print(f"""
  Construction : {do_meta['construction_type']}
  Accuracy     : {do_meta['accuracy']*100:.1f}%
  Sample size  : {do_meta['n_samples']:,}

  Meaning: Given a DO-fronted sentence and its canonical counterpart,
  the model correctly identifies which one the speaker actually said
  {do_meta['accuracy']*100:.1f}% of the time.

  This shows that surprisal + DL + givenness together explain
  why Hindi speakers front direct objects — it is not random.
""")

# ─── What the p-values mean ───────────────────────────────────────────────────
print("=" * 65)
print("  WHAT THE P-VALUES MEAN")
print("=" * 65)
print("""
  *** (p < 0.001) — Very strong evidence. Less than 0.1% chance
                    this result is due to random chance.

  n.s. (not significant) — Not enough evidence.
                           e.g. DL for IO-fronted (p=0.296):
                           IO-fronting is NOT driven by shortening
                           dependency length.

  All surprisal predictors are *** for both DO- and IO-fronted.
  This is the paper's main finding: surprisal (predictability
  in context) is the strongest driver of non-canonical word order
  in Hindi.
""")

print("=" * 65)
print("  SUMMARY OF ALL 4 SCRIPTS")
print("=" * 65)
print("""
  Script 01 — How we filter sentences from HUTB
  Script 02 — How we generate word-order variants
  Script 03 — Pairwise pairs, features, DL comparison
  Script 04 — Logistic regression, p-values, accuracy
""")
