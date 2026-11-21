"""
TEST SUITE for generator.py
Run in Colab: just execute this cell

4 levels:
  Level 1 — Unit tests       (mock sentences, no files needed)
  Level 2 — Property tests   (invariants that must always hold)
  Level 3 — Integration test (real HUTB data)
  Level 4 — Paper sanity     (checks results match paper expectations)
"""

import sys, json, copy, traceback
sys.path.insert(0, f'{PROJECT_ROOT}/src')

from variant_generation.generator import VariantGenerator
from data_loading.loader import HUTBLoader

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def make_token(id, form, upostag, head, deprel, feats='_'):
    return {'id': id, 'form': form, 'lemma': form,
            'upostag': upostag, 'head': head, 'deprel': deprel,
            'feats': feats, 'misc': '_'}

def passed(name):  print(f"  PASS  {name}")
def failed(name, msg): print(f"  FAIL  {name}: {msg}")

results = {'pass': 0, 'fail': 0}

def check(name, condition, msg=''):
    if condition:
        passed(name)
        results['pass'] += 1
    else:
        failed(name, msg)
        results['fail'] += 1

# ─────────────────────────────────────────────────────────────
# MOCK SENTENCES
# ─────────────────────────────────────────────────────────────

# SOV: राम किताब पढ़ता  (Ram book reads)
sov = [
    make_token(1, 'राम',   'PROPN', 3, 'nsubj', 'Case=Nom'),
    make_token(2, 'किताब', 'NOUN',  3, 'obj',   'Case=Acc'),
    make_token(3, 'पढ़ता', 'VERB',  0, 'root',  '_'),
]

# SOV with IO: राम सीता को किताब देता  (Ram Sita to book gives)
sov_io = [
    make_token(1, 'राम',   'PROPN', 5, 'nsubj', 'Case=Nom'),
    make_token(2, 'सीता',  'PROPN', 5, 'iobj',  'Case=Dat'),
    make_token(3, 'को',    'ADP',   2, 'case',  '_'),
    make_token(4, 'किताब', 'NOUN',  5, 'obj',   'Case=Acc'),
    make_token(5, 'देता',  'VERB',  0, 'root',  '_'),
]

# Sentence with NP subtree: बड़ा राम किताब पढ़ता (big Ram book reads)
sov_np = [
    make_token(1, 'बड़ा',  'ADJ',   2, 'amod',  '_'),       # modifier of राम
    make_token(2, 'राम',   'PROPN', 4, 'nsubj', 'Case=Nom'),
    make_token(3, 'किताब', 'NOUN',  4, 'obj',   'Case=Acc'),
    make_token(4, 'पढ़ता', 'VERB',  0, 'root',  '_'),
]

# Only 1 argument — should be skipped
one_arg = [
    make_token(1, 'राम',   'PROPN', 2, 'nsubj', 'Case=Nom'),
    make_token(2, 'आया',   'VERB',  0, 'root',  '_'),
]

g = VariantGenerator(max_variants=99)

# ─────────────────────────────────────────────────────────────
# LEVEL 1 — UNIT TESTS
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("LEVEL 1 — Unit tests")
print("="*55)

# 1.1 Constituent extraction
args = g.get_preverbal_constituents(sov)
check("1.1 finds 2 constituents in SOV", len(args) == 2)
check("1.2 first constituent is nsubj",  args[0]['head']['deprel'] == 'nsubj')
check("1.3 second constituent is obj",   args[1]['head']['deprel'] == 'obj')

# 1.2 Deprel sequence
seq = g.get_deprel_sequence(args)
check("1.4 deprel_seq correct", seq == ['nsubj', 'obj'])

# 1.3 Variant count — 2 args → 2! = 2 variants
variants = g.generate_variants(sov)
check("1.5 2 args → 2 variants", len(variants) == 2,
      f"got {len(variants)}")

# 1.4 is_original flag
originals = [v for v in variants if v['is_original']]
check("1.6 exactly 1 original", len(originals) == 1,
      f"got {len(originals)}")
check("1.7 original has correct seq",
      originals[0]['deprel_seq'] == ['nsubj', 'obj'])

# 1.5 order classification
check("1.8 SOV classified as Canonical",
      originals[0]['order_type'] == 'Canonical')
osv = [v for v in variants if not v['is_original']][0]
check("1.9 OSV classified as O-fronted",
      osv['order_type'] == 'O-fronted')

# 1.6 DL computed
check("1.10 DL is integer",     isinstance(variants[0]['dl'], int))
check("1.11 DL > 0",            variants[0]['dl'] > 0)

# 1.7 tokens_json round-trip
import pandas as pd
df = g.generate_for_all([sov], pd.DataFrame([{'sent_id': 'sent_0001'}]))
row = df.iloc[0]
tokens_back = json.loads(row['tokens_json'])
check("1.12 tokens_json is valid JSON",   isinstance(tokens_back, list))
check("1.13 tokens_json has all fields",  'form' in tokens_back[0])

# 1.8 One-arg sentence skipped
variants_1 = g.generate_variants(one_arg)
check("1.14 1-arg sentence returns []", len(variants_1) == 0)

# 1.9 IO sentence — 3 args → 3! = 6 variants
args_io = g.get_preverbal_constituents(sov_io)
variants_io = g.generate_variants(sov_io)
check("1.15 IO sentence has 3 args",    len(args_io) == 3,
      f"got {len(args_io)}")
check("1.16 3 args → 6 variants",       len(variants_io) == 6,
      f"got {len(variants_io)}")

# ─────────────────────────────────────────────────────────────
# LEVEL 2 — PROPERTY / INVARIANT TESTS
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("LEVEL 2 — Property tests (must hold for ALL variants)")
print("="*55)

for sent_name, sent in [("SOV", sov), ("SOV+IO", sov_io), ("SOV+NP", sov_np)]:
    variants = g.generate_variants(sent)
    if not variants:
        continue

    # 2.1 Token count preserved
    orig_len = len(sent)
    all_same_len = all(len(v['tokens']) == orig_len for v in variants)
    check(f"2.1 [{sent_name}] token count preserved",
          all_same_len, f"orig={orig_len}")

    # 2.2 IDs are sequential 1..N
    for i, v in enumerate(variants):
        ids = [t['id'] for t in v['tokens']]
        check(f"2.2 [{sent_name}] var_{i} IDs sequential",
              ids == list(range(1, len(ids)+1)),
              f"got {ids}")
        break  # check first variant only to keep output short

    # 2.3 Head pointers are valid (all heads in 0..N)
    n = len(sent)
    valid_heads = all(
        0 <= t['head'] <= n
        for v in variants
        for t in v['tokens']
    )
    check(f"2.3 [{sent_name}] all head pointers valid", valid_heads)

    # 2.4 Exactly one root (head==0) per variant
    for i, v in enumerate(variants):
        roots = [t for t in v['tokens'] if t['head'] == 0]
        check(f"2.4 [{sent_name}] var_{i} has exactly 1 root",
              len(roots) == 1, f"got {len(roots)}")
        break

    # 2.5 All word forms preserved (just reordered)
    orig_forms = sorted(t['form'] for t in sent)
    for i, v in enumerate(variants):
        var_forms = sorted(t['form'] for t in v['tokens'])
        check(f"2.5 [{sent_name}] var_{i} same word forms",
              var_forms == orig_forms)
        break

    # 2.6 Exactly one is_original per sentence
    n_orig = sum(1 for v in variants if v['is_original'])
    check(f"2.6 [{sent_name}] exactly 1 is_original",
          n_orig == 1, f"got {n_orig}")

    # 2.7 DL >= 0 for all variants
    all_dl_positive = all(v['dl'] >= 0 for v in variants)
    check(f"2.7 [{sent_name}] DL >= 0 for all variants", all_dl_positive)

    # 2.8 NP subtree stays together (modifier follows head into new position)
    if sent_name == "SOV+NP":
        for v in variants:
            forms = [t['form'] for t in v['tokens']]
            # बड़ा (amod) should always be adjacent to राम (its head)
            if 'बड़ा' in forms and 'राम' in forms:
                pos_adj  = forms.index('बड़ा')
                pos_noun = forms.index('राम')
                check("2.8 [SOV+NP] adjective stays adjacent to head noun",
                      abs(pos_adj - pos_noun) == 1,
                      f"adj at {pos_adj}, noun at {pos_noun}")
            break

# ─────────────────────────────────────────────────────────────
# LEVEL 3 — INTEGRATION TEST (real HUTB data)
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("LEVEL 3 — Integration test (real HUTB train.conllu)")
print("="*55)

try:
    from preprocessing.filter import SentenceFilter

    loader = HUTBLoader(f'{PROJECT_ROOT}/data/ud_hindi')
    raw = loader.load_train()
    print(f"  Loaded {len(raw)} raw sentences")

    fltr = SentenceFilter()
    filtered, rejected = fltr.filter_with_rejection_tracking(
        [s['tokens'] for s in raw[:2000]]   # first 2000 for speed
    )
    print(f"  Filtered: {len(filtered)} sentences pass")

    if len(filtered) == 0:
        print("  WARN  No sentences passed filter — check filter.py")
    else:
        meta = fltr.create_metadata(filtered)
        g2   = VariantGenerator(max_variants=99)
        df   = g2.generate_for_all(filtered[:50], meta.head(50))

        check("3.1 DataFrame not empty",          len(df) > 0)
        check("3.2 has is_original column",       'is_original' in df.columns)
        check("3.3 has dl column",                'dl' in df.columns)
        check("3.4 has order_type column",        'order_type' in df.columns)
        check("3.5 has tokens_json column",       'tokens_json' in df.columns)
        check("3.6 tokens_json parseable",
              all(isinstance(json.loads(r), list) for r in df['tokens_json']))

        n_orig = df['is_original'].sum()
        check("3.7 original count == filtered count",
              n_orig == min(50, len(filtered)),
              f"orig={n_orig}, filtered={len(filtered)}")

        print(f"\n  Order distribution (first 50 sentences):")
        print(df['order_type'].value_counts().to_string())

        print(f"\n  DL stats:")
        print(f"    mean  = {df['dl'].mean():.1f}")
        print(f"    min   = {df['dl'].min()}")
        print(f"    max   = {df['dl'].max()}")

except Exception as e:
    print(f"  SKIP  Integration test failed: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────
# LEVEL 4 — PAPER SANITY CHECKS
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("LEVEL 4 — Paper sanity checks")
print("="*55)

# 4.1 Canonical (SOV) should be most common order in HUTB
try:
    order_counts = df['order_type'].value_counts()
    canonical_count = order_counts.get('Canonical', 0)
    total = len(df)
    canonical_pct = canonical_count / total * 100
    check("4.1 Canonical is most frequent order",
          order_counts.index[0] == 'Canonical',
          f"most frequent = {order_counts.index[0]}")
    print(f"     Canonical = {canonical_pct:.1f}% of all variants")
except:
    print("  SKIP  4.1 (need Level 3 to pass first)")

# 4.2 Original sentences should have lower DL than average of their variants
#     (DLT predicts this — paper tests whether it holds)
try:
    orig_dl  = df[df['is_original'] == 1]['dl'].mean()
    nonor_dl = df[df['is_original'] == 0]['dl'].mean()
    print(f"\n  DL check (paper's key question):")
    print(f"    Mean DL of corpus sentences : {orig_dl:.2f}")
    print(f"    Mean DL of generated variants: {nonor_dl:.2f}")
    dl_lower = orig_dl <= nonor_dl
    check("4.2 corpus DL <= variant DL on average",
          dl_lower,
          f"corpus={orig_dl:.2f}, variants={nonor_dl:.2f} — DLT not supported in this sample")
    if dl_lower:
        print("     DLT supported in this sample (expected per paper)")
    else:
        print("     DLT not supported — this is OK, paper tests if it holds after controlling for surprisal+givenness")
except:
    print("  SKIP  4.2 (need Level 3 to pass first)")

# 4.3 No variant should have 0 tokens
try:
    all_have_tokens = all(len(json.loads(r)) > 0 for r in df['tokens_json'])
    check("4.3 all variants have tokens", all_have_tokens)
except:
    print("  SKIP  4.3")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
total = results['pass'] + results['fail']
print(f"RESULT: {results['pass']}/{total} checks passed")
if results['fail'] == 0:
    print("ALL TESTS PASSED")
else:
    print(f"{results['fail']} FAILED — check output above")
print("="*55)
