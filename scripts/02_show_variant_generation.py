"""
DEMO SCRIPT 2 — Variant Generation
=====================================
Shows how we generate all word-order permutations for each sentence.
Picks real sentences from HUTB and shows original + every generated variant.

Run:
    cd D:\MTP_WO\hindi_wordorder_replication
    python scripts/02_show_variant_generation.py
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loading.loader          import HUTBLoader
from preprocessing.filter         import SentenceFilter
from variant_generation.generator import VariantGenerator

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ─── Load and filter ─────────────────────────────────────────────────────────
loader = HUTBLoader(os.path.join(DATA_DIR, 'ud_hindi'))
raw    = loader.load_train()

fltr = SentenceFilter()
filtered, _ = fltr.filter_with_rejection_tracking([s['tokens'] for s in raw[:500]])
meta = fltr.create_metadata(filtered)

gen = VariantGenerator(max_variants=99)

# ─── How variant generation works ────────────────────────────────────────────
print()
print("=" * 62)
print("  HOW VARIANT GENERATION WORKS")
print("=" * 62)
print("""
  For each filtered sentence:

  1. Find ALL preverbal direct dependents of the root verb.
     Each moves as a full subtree (NP modifiers travel with head).

  2. Generate ALL permutations of those constituents.
     Cap at 99 variants per sentence (paper cutoff).

       2 args  → 2!  =   2 variants
       3 args  → 3!  =   6 variants
       4 args  → 4!  =  24 variants
       5 args  → 5! = 120 → capped at 99

  3. Repair dependency head pointers after permutation.
  4. Compute dependency length (DL) for each variant.
  5. Classify order type:
       Canonical   = Subject before Object (SOV)
       O-fronted   = Object before Subject, no IO
       DO-fronted  = Object before Subject, IO present
       IO-fronted  = Indirect Object before Subject
""")

# ─── Example 1: 2-argument sentence (SOV) ────────────────────────────────────
print("=" * 62)
print("  EXAMPLE 1 — 2-argument sentence (nsubj + obj)")
print("  2! = 2 variants")
print("=" * 62)

demo2 = None
for i, (sent, row) in enumerate(zip(filtered, meta.itertuples())):
    prevs = fltr.preverbal_dependents(sent)
    deprels = [t['deprel'] for t in prevs]
    if len(prevs) == 2 and 'nsubj' in deprels and 'obj' in deprels:
        demo2 = sent; meta2 = row; break

if demo2:
    root  = fltr.get_root(demo2)
    prevs = fltr.preverbal_dependents(demo2)
    print(f"\n  Original sentence:")
    print(f"    {meta2.text}")
    print(f"\n  ROOT verb : '{root['form']}'  (position {root['id']})")
    print(f"\n  Preverbal arguments (these get permuted):")
    for t in prevs:
        print(f"    pos={t['id']}  '{t['form']}'  deprel={t['deprel']}")

    variants = gen.generate_variants(demo2)
    print(f"\n  Generated {len(variants)} variants:")
    print(f"  {'#':<4} {'Order Type':<14} {'DL':<5} {'Deprel Sequence':<20}  Text")
    print("  " + "-" * 70)
    for i, v in enumerate(variants):
        text = ' '.join(t['form'] for t in v['tokens'])
        seq  = ' → '.join(v['deprel_seq'])
        orig = '  ← ORIGINAL' if v['is_original'] else ''
        print(f"  {i+1:<4} {v['order_type']:<14} {v['dl']:<5} {seq:<20}  {text}{orig}")

# ─── Example 2: 3-argument sentence (with IO) ────────────────────────────────
print()
print("=" * 62)
print("  EXAMPLE 2 — 3-argument sentence (nsubj + iobj + obj)")
print("  3! = 6 variants")
print("=" * 62)

demo3 = None
for sent, row in zip(filtered, meta.itertuples()):
    prevs   = fltr.preverbal_dependents(sent)
    deprels = [t['deprel'] for t in prevs]
    if len(prevs) == 3 and 'iobj' in deprels:
        demo3 = sent; meta3 = row; break

if demo3:
    root  = fltr.get_root(demo3)
    prevs = fltr.preverbal_dependents(demo3)
    print(f"\n  Original sentence:")
    print(f"    {meta3.text}")
    print(f"\n  ROOT verb : '{root['form']}'  (position {root['id']})")
    print(f"\n  Preverbal arguments (these get permuted):")
    for t in prevs:
        print(f"    pos={t['id']}  '{t['form']}'  deprel={t['deprel']}")

    variants = gen.generate_variants(demo3)
    print(f"\n  Generated {len(variants)} variants:")
    print(f"  {'#':<4} {'Order Type':<14} {'DL':<5} {'Deprel Sequence':<30}  Text")
    print("  " + "-" * 80)
    for i, v in enumerate(variants):
        text = ' '.join(t['form'] for t in v['tokens'])
        seq  = ' → '.join(v['deprel_seq'])
        orig = '  ← ORIGINAL' if v['is_original'] else ''
        print(f"  {i+1:<4} {v['order_type']:<14} {v['dl']:<5} {seq:<30}  {text}{orig}")
else:
    print("\n  No 3-arg IO sentence found in this 500-sentence sample.")

# ─── Summary stats ────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  FULL DATASET STATS  (from precomputed variants.csv)")
print("=" * 62)

import pandas as pd
variants_path = os.path.join(DATA_DIR, 'processed', 'variants', 'variants.csv')
df = pd.read_csv(variants_path)

n_ref   = df['reference_id'].nunique()
n_total = len(df)
n_orig  = df['is_original'].sum()

print(f"\n  Reference sentences (passed filter) : {n_ref:,}")
print(f"  Paper target                        : 1,996")
print(f"  Total variants generated            : {n_total:,}")
print(f"  Paper target                        : ~72,833")
print(f"  Average variants per sentence       : {n_total/n_ref:.1f}")
print()
print("  Order type distribution:")
for order, count in df['order_type'].value_counts().items():
    pct = count / n_total * 100
    bar = '#' * int(pct / 3)
    print(f"    {order:<14} : {count:>7,}  ({pct:5.1f}%)  {bar}")

print()
print("  Done. Run script 03 next to see pairwise pairs & features.")
print()
