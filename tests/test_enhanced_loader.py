"""
Test HUTBLoader and SentenceFilter — verifies loading, filtering, and basic statistics.
Uses current API: loader.load_train() + SentenceFilter.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loading.loader import HUTBLoader
from src.preprocessing.filter import SentenceFilter
import pandas as pd
from collections import Counter

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, 'data', 'ud_hindi')
OUT_DIR   = os.path.join(BASE, 'data', 'processed', 'variants')

def main():
    print("=" * 60)
    print("LOADER + FILTER TEST")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────
    loader    = HUTBLoader(DATA_PATH)
    sentences = loader.load_train()
    print(f"\nTotal sentences loaded: {len(sentences)}")

    # ── Filter ────────────────────────────────────────────────────
    sf       = SentenceFilter()
    filtered = sf.filter(sentences)
    print(f"Sentences passing filter: {len(filtered)}")

    # ── Order-type breakdown ──────────────────────────────────────
    order_types = Counter(s.get('order_type', 'Unknown') for s in filtered)
    print("\nWord order type distribution:")
    for ot, cnt in order_types.most_common():
        pct = 100 * cnt / len(filtered) if filtered else 0
        print(f"  {ot:<16} {cnt:>5}  ({pct:.1f}%)")

    # ── Extract each category ─────────────────────────────────────
    categories = {
        'O-fronted' : [s for s in filtered if s.get('order_type') == 'O-fronted'],
        'DO-fronted': [s for s in filtered if s.get('order_type') == 'DO-fronted'],
        'IO-fronted': [s for s in filtered if s.get('order_type') == 'IO-fronted'],
        'Canonical' : [s for s in filtered if s.get('order_type') == 'Canonical'],
    }

    print("\nCategory counts:")
    for cat, sents in categories.items():
        print(f"  {cat:<16} {len(sents):>5}")

    # ── Sample sentences ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE SENTENCES BY CATEGORY")
    print("=" * 60)
    for cat, sents in categories.items():
        if sents:
            print(f"\n{cat}:")
            for sent in sents[:2]:
                text = sent.get('text') or ' '.join(t['form'] for t in sent['tokens'])
                print(f"  {text[:80]}")

    # ── Save each category ────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    for cat, sents in categories.items():
        if sents:
            fname = os.path.join(OUT_DIR, cat.lower().replace('-', '_') + '_ref.conllu')
            loader.save_conllu(sents, fname)

    # ── Export metadata CSV ───────────────────────────────────────
    rows = []
    for s in filtered:
        rows.append({
            'sent_id'   : s.get('sent_id'),
            'text'      : s.get('text'),
            'order_type': s.get('order_type', 'Unknown'),
            'n_tokens'  : len(s['tokens']),
        })
    df = pd.DataFrame(rows)
    meta_path = os.path.join(BASE, 'data', 'processed', 'metadata.csv')
    df.to_csv(meta_path, index=False)
    print(f"\nMetadata saved: {meta_path}  ({len(df)} rows)")
    print("\nDone.")

if __name__ == '__main__':
    main()
