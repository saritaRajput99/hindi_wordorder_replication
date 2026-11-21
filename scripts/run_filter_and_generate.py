"""
Step 1: Run SentenceFilter on UD_Hindi-HDTB train.conllu
Step 2: Save filter stats and order-type comparison vs paper (Table 1)
Step 3: Run VariantGenerator with grammar filter
Step 4: Save variants to output/variants_final/

Paper reference (Ranjan & van Schijndel 2024, Table 1):
  Canonical  : 1763  (88.33%)
  O-fronted  :  233  (11.67%)  <- total non-canonical
  DO-fronted :  133  ( 6.67%)  <- subset: obj<subj + iobj/obl present
  IO-fronted :  101  ( 5.06%)  <- subset: iobj<subj
  TOTAL      : 1996
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from collections import Counter
from pathlib import Path

from src.data_loading.loader import HUTBLoader
from src.preprocessing.filter import SentenceFilter
from src.variant_generation.generator import VariantGenerator
from src.variant_generation.grammar_filter import GrammarFilter

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR   = 'data/ud_hindi'
OUTPUT_DIR = 'output/variants_final'
RESULTS_DIR = 'results'

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# STEP 1: LOAD
# ══════════════════════════════════════════════════════════════════════
print('=' * 60)
print('STEP 1: Loading UD_Hindi-HDTB train.conllu')
print('=' * 60)

loader = HUTBLoader(DATA_DIR)
raw = loader.load_train()
sentences = [s['tokens'] for s in raw]
orig_sent_ids = [
    raw[i]['sent_id'] if raw[i].get('sent_id') else f'sent_{i:04d}'
    for i in range(len(raw))
]
print(f'  Raw sentences: {len(sentences)}')

# ══════════════════════════════════════════════════════════════════════
# STEP 2: FILTER
# ══════════════════════════════════════════════════════════════════════
print()
print('=' * 60)
print('STEP 2: Applying SentenceFilter (paper criteria)')
print('=' * 60)

filt = SentenceFilter()
filtered, rejected_df, filtered_sent_ids = filt.filter_with_rejection_tracking(
    sentences, sent_ids=orig_sent_ids
)

print(f'  Passed : {len(filtered)}  (paper: 1996)')
print()
print('  Rejection breakdown:')
for reason, count in filt.stats['rejected'].items():
    if count > 0:
        print(f'    {reason:<22}: {count}')

# ── Order classification (paper-aligned) ──
# Paper structure: O-fronted = total non-canonical (obj before subj)
#   DO-fronted = subset of O-fronted where iobj/obl also present
#   IO-fronted = subset where iobj appears before subj
counts = Counter()
for sent in filtered:
    root = filt.get_root(sent)
    if not root:
        counts['unknown'] += 1
        continue

    prev      = filt.preverbal_dependents(sent)
    subj      = next((t for t in prev if filt.norm_dep(t, root['id']) in filt.subject_deps), None)
    obj       = next((t for t in prev if filt.norm_dep(t, root['id']) in filt.object_deps), None)
    iobj_prev = next((t for t in prev if filt.norm_dep(t, root['id']) in filt.indirect_object_deps), None)

    # iobj OR obl as direct root dep (HDTB uses obl for many indirect objects)
    root_deps = [t for t in sent if t['head'] == root['id']]
    iobj_any  = any(t['deprel'] in (filt.indirect_object_deps | {'obl'})
                    for t in root_deps)

    if subj and obj:
        if obj['id'] < subj['id']:
            counts['O-fronted'] += 1          # total non-canonical (paper Table 1 row)
            if iobj_any:
                counts['DO-fronted'] += 1     # subset: has iobj/obl
        else:
            counts['Canonical'] += 1
    elif subj and iobj_prev:
        if iobj_prev['id'] < subj['id']:
            counts['O-fronted'] += 1          # IO-fronted also non-canonical
            counts['IO-fronted'] += 1
        else:
            counts['Canonical'] += 1
    elif subj:
        counts['Canonical'] += 1
    else:
        counts['unknown'] += 1

total = len(filtered)
non_canonical = counts['O-fronted']  # O-fronted IS the total non-canonical

PAPER = {
    'Canonical' : (1763, 88.33),
    'O-fronted' : ( 233, 11.67),
    'DO-fronted': ( 133,  6.67),
    'IO-fronted': ( 101,  5.06),
}

print()
print('  Order-type comparison vs paper (Table 1):')
print(f'  {"Category":<16} {"Ours":>6} {"Ours%":>7}  {"Paper":>6} {"Paper%":>7}  {"Diff":>6}')
print('  ' + '-' * 58)
for cat, (paper_n, paper_pct) in PAPER.items():
    n   = counts.get(cat, 0)
    pct = n / total * 100
    lbl = cat + (' (total non-canon)' if cat == 'O-fronted' else '')
    print(f'  {cat:<16} {n:>6}  {pct:>6.2f}%  {paper_n:>6}  {paper_pct:>6.2f}%  {n-paper_n:>+6}')
print('  ' + '-' * 58)
print(f'  {"TOTAL":<16} {total:>6}            {1996:>6}            {total-1996:>+6}')
print(f'  {"Non-canonical":<16} {non_canonical:>6}             {233:>6}            {non_canonical-233:>+6}')

# ── Save filter results ──
filter_rows = []
for cat, (paper_n, paper_pct) in PAPER.items():
    n = counts.get(cat, 0)
    filter_rows.append({
        'category'   : cat,
        'ours'       : n,
        'ours_pct'   : round(n / total * 100, 2),
        'paper'      : paper_n,
        'paper_pct'  : paper_pct,
        'diff'       : n - paper_n,
    })
filter_rows.append({
    'category': 'TOTAL',
    'ours': total, 'ours_pct': 100.0,
    'paper': 1996, 'paper_pct': 100.0,
    'diff': total - 1996,
})
filter_rows.append({
    'category': 'Non-canonical',
    'ours': non_canonical, 'ours_pct': round(non_canonical/total*100,2),
    'paper': 233, 'paper_pct': 11.67,
    'diff': non_canonical - 233,
})

filter_df = pd.DataFrame(filter_rows)
filter_csv = Path(RESULTS_DIR) / 'filter_comparison.csv'
filter_df.to_csv(filter_csv, index=False)
print(f'\n  Saved: {filter_csv}')

rejected_df.to_csv(Path(RESULTS_DIR) / 'rejected_sentences.csv', index=False)
print(f'  Saved: {Path(RESULTS_DIR) / "rejected_sentences.csv"}')

# ══════════════════════════════════════════════════════════════════════
# STEP 3: GRAMMAR FILTER (transition bigrams from filtered corpus)
# ══════════════════════════════════════════════════════════════════════
print()
print('=' * 60)
print('STEP 3: Building grammar filter (transition bigrams)')
print('=' * 60)

gf = GrammarFilter()
gf.build_from_corpus(filtered)
print(f'  Attested transition pairs: {len(gf.attested)}')

# ══════════════════════════════════════════════════════════════════════
# STEP 4: GENERATE VARIANTS
# ══════════════════════════════════════════════════════════════════════
print()
print('=' * 60)
print('STEP 4: Generating variants (max 99 per sentence)')
print('=' * 60)

metadata_df = pd.DataFrame([{'sent_id': sid} for sid in filtered_sent_ids])

gen = VariantGenerator(max_variants=99, grammar_filter=gf)
df  = gen.generate_for_all(filtered, metadata_df, output_dir=OUTPUT_DIR)

# ── Variant comparison vs paper ──
# Paper Table 1 does NOT pool all variants and bucket each by its own
# absolute order_type. Its 72,833 total is the FULL unrestricted pool
# (flip and non-flip variants alike); the four category counts in Table 1
# are the FLIP-MATCHED subset only — which is why those four counts sum to
# only ~half of 72,833, not the whole thing (verified: 2834+1663+1353+30851
# = 36,701). generator.py now tags every variant with is_flip_pair instead
# of discarding non-flip ones, so we reproduce that exact structure: full
# pool kept intact (paper-matching total), flip-matched subset identified
# via is_flip_pair == 1, broken down by the reference's own category.
ref     = df[df['is_original'] == 1]
gen_var = df[df['is_original'] == 0]

ref_order_type = ref.set_index('reference_id')['order_type']
gen_var = gen_var.copy()
gen_var['ref_order_type'] = gen_var['reference_id'].map(ref_order_type)

flip = gen_var[gen_var['is_flip_pair'] == 1]
flip_counts = {
    'Canonical' : int((flip['ref_order_type'] == 'Canonical').sum()),
    'DO-fronted': int((flip['ref_order_type'] == 'DO-fronted').sum()),
    'IO-fronted': int((flip['ref_order_type'] == 'IO-fronted').sum()),
}
print(f'\n  Flip-matched variants (is_flip_pair==1): {len(flip):,} / {len(gen_var):,} '
      f'({len(flip)/len(gen_var)*100:.1f}% — expect ~50%, matching the paper\'s own split)')

print()
print('  Variant order-type comparison vs paper (ref-category-matched):')
PAPER_VAR = {
    'Canonical' : (30851, 42.36),
    'DO-fronted': ( 1663,  2.28),
    'IO-fronted': ( 1353,  1.86),
}
print(f'  {"Category":<16} {"Ours":>7} {"Ours%":>7}  {"Paper":>7} {"Paper%":>7}  {"Diff":>7}')
print('  ' + '-' * 62)
for cat, (paper_n, paper_pct) in PAPER_VAR.items():
    n   = flip_counts[cat]
    pct = n / len(gen_var) * 100 if len(gen_var) else 0
    print(f'  {cat:<16} {n:>7}  {pct:>6.2f}%  {paper_n:>7}  {paper_pct:>6.2f}%  {n-paper_n:>+7}')
print('  ' + '-' * 62)
print(f'  {"Ref total":<16} {len(ref):>7}            {1996:>7}            {len(ref)-1996:>+7}')
print(f'  {"Var total (all)":<16} {len(gen_var):>7}           {72833:>7}           {len(gen_var)-72833:>+7}')

# Save variant summary (ref-category-matched flip counts, per paper Table 1)
var_summary = Path(RESULTS_DIR) / 'variant_summary.csv'
summary_rows = []
for cat in ['Canonical', 'DO-fronted', 'IO-fronted']:
    r = ref[ref['order_type'] == cat].shape[0]
    summary_rows.append({'order_type': cat, 'ref_count': r, 'var_count': flip_counts[cat]})
pd.DataFrame(summary_rows).to_csv(var_summary, index=False)
print(f'\n  Saved: {var_summary}')
print(f'  Variants CSV: {OUTPUT_DIR}/variants.csv')
print()
print('Done.')
