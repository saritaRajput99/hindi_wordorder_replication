"""
DEMO SCRIPT 3 — Pairwise Pairs & Features
==========================================
Shows how we build pairwise (reference vs variant) pairs,
what features each pair has, and the dependency length comparison.

Run:
    cd D:\MTP_WO\hindi_wordorder_replication
    python scripts/03_show_pairs_and_features.py
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np

DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# ─── Load features CSV ───────────────────────────────────────────────────────
print()
print("=" * 62)
print("  STEP 1 — FEATURES DATASET")
print("=" * 62)

feat_path = os.path.join(DATA_DIR, 'processed', 'features', 'features.csv')
df = pd.read_csv(feat_path)

print(f"\n  Loaded: {len(df):,} rows")
print(f"\n  Columns in features.csv:")
for col in df.columns:
    print(f"    {col}")

print(f"\n  5 features used in the model:")
print(f"    dl                    — Dependency Length")
print(f"    surprisal_trigram     — Trigram language model surprisal")
print(f"    surprisal_pcfg        — PCFG parser surprisal")
print(f"    surprisal_adaptive_lstm — Neural LM surprisal")
print(f"    is_score              — Information Status (givenness) score")

# ─── Show sample rows ─────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  STEP 2 — SAMPLE ROWS (original corpus sentences)")
print("=" * 62)
print()

orig = df[df['is_original'] == 1].head(8)
print(f"  {'ref_id':<12} {'order':<14} {'dl':>5} {'surp_tri':>10} {'is_score':>9}")
print("  " + "-" * 56)
for _, row in orig.iterrows():
    print(f"  {row['reference_id']:<12} {row['order_type']:<14} "
          f"{row['dl']:>5.0f} {row['surprisal_trigram']:>10.2f} {row['is_score']:>9.4f}")

# ─── Show variants of one sentence ───────────────────────────────────────────
print()
print("=" * 62)
print("  STEP 3 — ORIGINAL vs VARIANTS for one sentence")
print("=" * 62)
print()

# Pick a reference sentence that has both Canonical and O-fronted variants
for ref_id, group in df.groupby('reference_id'):
    order_types = group['order_type'].unique()
    if 'Canonical' in order_types and 'O-fronted' in order_types and len(group) >= 3:
        demo_group = group
        break

orig_row = demo_group[demo_group['is_original'] == 1].iloc[0]
print(f"  Reference sentence: {ref_id}")
print(f"  {orig_row['text']}")
print()
print(f"  {'is_orig':<8} {'order_type':<14} {'dl':>5} {'surp_tri':>10} {'is_score':>9}  text")
print("  " + "-" * 85)

for _, row in demo_group.sort_values('is_original', ascending=False).head(8).iterrows():
    orig_marker = '<-- ORIGINAL' if row['is_original'] else ''
    print(f"  {int(row['is_original']):<8} {row['order_type']:<14} "
          f"{row['dl']:>5.0f} {row['surprisal_trigram']:>10.2f} "
          f"{row['is_score']:>9.4f}  {row['text'][:35]} {orig_marker}")

# ─── Pairwise transform explanation ──────────────────────────────────────────
print()
print("=" * 62)
print("  STEP 4 — PAIRWISE RANKING TRANSFORM  (Joachims 2002)")
print("=" * 62)
print("""
  Problem: Class imbalance — 2,320 corpus sentences vs ~88,000 variants.
  A naive classifier would just always predict 'variant'.

  Solution: Pairwise transform.
    For each (reference r, variant v) pair:
      Positive: X = features(r) - features(v)   label = 1
      Negative: X = features(v) - features(r)   label = 0

  Result: Perfectly balanced dataset (50% / 50%).
  The classifier learns: does the corpus sentence have
  LOWER surprisal, LOWER DL, HIGHER givenness than the variant?
""")

# Show one concrete pair
ref_row = demo_group[demo_group['is_original'] == 1].iloc[0]
var_row = demo_group[demo_group['is_original'] == 0].iloc[0]

feature_cols = ['dl', 'surprisal_trigram', 'surprisal_pcfg', 'is_score']
print(f"  Concrete example:")
print(f"    Reference : {ref_row['text'][:55]}")
print(f"    Variant   : {var_row['text'][:55]}")
print()
print(f"  {'Feature':<24} {'Reference':>12} {'Variant':>10} {'Diff (r-v)':>12}")
print("  " + "-" * 62)
for feat in feature_cols:
    r_val = ref_row[feat]
    v_val = var_row[feat]
    diff  = r_val - v_val
    print(f"  {feat:<24} {r_val:>12.4f} {v_val:>10.4f} {diff:>12.4f}")

print()
print(f"  Label = 1 (reference preferred over this variant)")

# ─── DL comparison ────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  STEP 5 — DEPENDENCY LENGTH TEST  (key DLT result)")
print("=" * 62)

orig_dl  = df[df['is_original'] == 1]['dl'].mean()
other_dl = df[df['is_original'] == 0]['dl'].mean()

print(f"\n  Mean DL — corpus sentences          : {orig_dl:.2f}")
print(f"  Mean DL — generated variants        : {other_dl:.2f}")
print(f"  Difference  (variant - corpus)      : {other_dl - orig_dl:+.2f}")
print()
if orig_dl < other_dl:
    print("  RESULT: Corpus DL < Variant DL")
    print("          DLT is supported — Hindi speakers prefer shorter dependency lengths.")
else:
    print("  RESULT: DL alone does not fully explain word order.")
    print("          Paper shows surprisal + givenness are also needed.")

# ─── Per order-type DL ────────────────────────────────────────────────────────
print()
print("  Mean DL by order type (corpus sentences only):")
orig_only = df[df['is_original'] == 1]
for order, grp in orig_only.groupby('order_type'):
    print(f"    {order:<14} : {grp['dl'].mean():.2f}  (n={len(grp)})")

# ─── Table 1 ─────────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  STEP 6 — TABLE 1: Order Distribution  (paper replication)")
print("=" * 62)
t1_path = os.path.join(RESULTS_DIR, 'tables', 'table1_order_distribution.csv')
df_t1   = pd.read_csv(t1_path)
print()
print(df_t1.to_string(index=False))

print()
print("  Done. Run script 04 next to see analysis / p-values.")
print()
