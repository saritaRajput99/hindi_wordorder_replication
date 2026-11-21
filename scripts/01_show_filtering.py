"""
DEMO SCRIPT 1 — FINAL STRICT VERSION (CORRECTED)
==============================================
Fix: Proper object detection (no over-counting)

Run:
    python scripts/01_show_filtering.py
"""

import collections
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loading.loader import HUTBLoader
from preprocessing.filter import SentenceFilter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# --------------------------------------------------
# STRICT NORMALIZATION (FINAL FIX)
# --------------------------------------------------
def norm_dep(t, root_id):
    dep = t['deprel']
    feats = t.get('feats', {})

    # extract case
    if isinstance(feats, dict):
        case = feats.get('Case')
    else:
        case = 'Acc' if 'Case=Acc' in str(feats) else None

    # STRICT condition (VERY IMPORTANT)
    if (
        dep == 'obl'
        and case == 'Acc'
        and t['head'] == root_id   # core argument
        and t['id'] < root_id      # preverbal
    ):
        return 'obj'

    return dep


# --------------------------------------------------
# FINAL CLASSIFICATION
# --------------------------------------------------
def classify_paper_order(sentence):

    root = next((t for t in sentence if t['head'] == 0), None)
    if not root:
        return "Other"

    root_id = root['id']

    preverbal = [
        t for t in sentence
        if t['head'] == root_id and t['id'] < root_id
    ]

    # roles
    subjects = [t for t in preverbal if norm_dep(t, root_id) in ['nsubj','nsubj:pass']]
    objects  = [t for t in preverbal if norm_dep(t, root_id) == 'obj']
    iobjects = [t for t in preverbal if norm_dep(t, root_id) == 'iobj']

    if not subjects:
        return "Other"

    subj = subjects[0]

    # ---- Direct Object ----
    if objects:
        obj = objects[0]
        if obj['id'] < subj['id']:
            return "DO-fronted"
        else:
            return "Canonical"

    # ---- Indirect Object ----
    if iobjects:
        iobj = iobjects[0]
        if iobj['id'] < subj['id']:
            return "IO-fronted"
        else:
            return "Canonical"

    return "Canonical"


# --------------------------------------------------
# STEP 1 — LOAD DATA
# --------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1 — LOAD RAW DATA")
print("=" * 60)

loader = HUTBLoader(os.path.join(DATA_DIR, 'ud_hindi'))
raw = loader.load_train()

print(f"\nTotal sentences in dataset: {len(raw)}")


# --------------------------------------------------
# STEP 2 — APPLY FILTER
# --------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 — APPLY FILTER")
print("=" * 60)

fltr = SentenceFilter()
sample = [s['tokens'] for s in raw[:500]]

filtered, rejected = fltr.filter_with_rejection_tracking(sample)

print(f"\nInput sentences  : {len(sample)}")
print(f"Passed filter    : {len(filtered)}")
print(f"Rejected         : {len(rejected)}")


# --------------------------------------------------
# STEP 3 — CLASSIFICATION
# --------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 — PAPER WORD ORDER CLASSIFICATION")
print("=" * 60)

orders = [classify_paper_order(sent) for sent in filtered]


# --------------------------------------------------
# STEP 4 — FINAL COUNTS
# --------------------------------------------------
print("\nFINAL WORD ORDER STATS:\n")

vc = collections.Counter(orders)
total = len(orders)
22
categories = ["Canonical", "DO-fronted", "IO-fronted"]

for cat in categories:
    count = vc.get(cat, 0)
    pct = (count / total * 100) if total > 0 else 0
    bar = "#" * int(pct / 3)
    print(f"{cat:<12}: {count:>4} ({pct:5.1f}%) {bar}")

print("\nTOTAL SENTENCES ANALYZED:", total)


# --------------------------------------------------
# STEP 5 — SAMPLE OUTPUT
# --------------------------------------------------
print("\nSample classified sentences:\n")

for i, sent in enumerate(filtered[:10]):
    text = ' '.join(t['form'] for t in sent)
    print(f"[{orders[i]:<12}] {text[:70]}")


print("\nDone ")