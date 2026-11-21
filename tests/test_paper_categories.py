"""
Test paper category detection (O-fronted, DO-fronted, IO-fronted, Canonical).
Uses current API: loader.load_train() + SentenceFilter.
Compares counts against paper Table 1 targets.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loading.loader import HUTBLoader
from src.preprocessing.filter import SentenceFilter

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'data', 'ud_hindi')

# Paper Table 1 reference counts
PAPER_COUNTS = {
    'DO-fronted': 133,
    'IO-fronted': 101,
}

def main():
    print("=" * 60)
    print("PAPER CATEGORY DETECTION TEST")
    print("=" * 60)

    loader    = HUTBLoader(DATA_PATH)
    sentences = loader.load_train()
    print(f"\nTotal sentences: {len(sentences)}")

    sf       = SentenceFilter()
    filtered = sf.filter(sentences)
    print(f"Sentences passing filter: {len(filtered)}")

    # ── Count by order_type ───────────────────────────────────────
    from collections import Counter
    counts = Counter(s.get('order_type', 'Unknown') for s in filtered)
    total  = len(filtered)

    print("\n" + "=" * 60)
    print("PAPER CATEGORY STATISTICS")
    print("=" * 60)
    for cat in ['O-fronted', 'DO-fronted', 'IO-fronted', 'Canonical', 'Other']:
        cnt = counts.get(cat, 0)
        pct = 100 * cnt / total if total else 0
        paper = PAPER_COUNTS.get(cat)
        note  = f"  (paper: {paper})" if paper else ''
        print(f"  {cat:<16} {cnt:>5}  ({pct:.2f}%){note}")
    print(f"  {'Total':<16} {total:>5}")

    # ── Annotation gap check ──────────────────────────────────────
    for cat in ['DO-fronted', 'IO-fronted']:
        ours  = counts.get(cat, 0)
        paper = PAPER_COUNTS[cat]
        diff  = paper - ours
        if diff > 0:
            print(f"\n  NOTE: {cat} — {diff} fewer than paper ({ours} vs {paper})")
            print(f"        Root cause: HUTB k4→obl annotation gap (iobj missing in HDTB)")

    # ── Sample sentences per category ─────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE SENTENCES BY CATEGORY")
    print("=" * 60)

    by_cat = {cat: [s for s in filtered if s.get('order_type') == cat]
              for cat in ['DO-fronted', 'IO-fronted', 'O-fronted', 'Canonical']}

    for cat, sents in by_cat.items():
        if sents:
            print(f"\n{cat} (showing 2 of {len(sents)}):")
            for sent in sents[:2]:
                text = sent.get('text') or ' '.join(t['form'] for t in sent['tokens'])
                print(f"  {text[:80]}")
                subj = next((t['id'] for t in sent['tokens'] if t['deprel'] == 'nsubj'), None)
                obj  = next((t['id'] for t in sent['tokens'] if t['deprel'] == 'obj'), None)
                iobj = next((t['id'] for t in sent['tokens'] if t['deprel'] == 'iobj'), None)
                print(f"    nsubj_id={subj}  obj_id={obj}  iobj_id={iobj}")

    print("\nDone.")

if __name__ == '__main__':
    main()
