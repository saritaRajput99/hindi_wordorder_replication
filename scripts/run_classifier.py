"""
Quick pairwise classifier using all available surprisal scores.
Merges: features.csv (order_type, dl, surprisal_trigram, surprisal_lstm) + pcfg_surprisals.json
Runs logistic regression (10-fold CV) for DO-fronted and IO-fronted.
"""
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load base df with LSTM + trigram scores ───────────────────────
print("Loading features (order_type, dl, trigram, LSTM)...")
df = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'features', 'features.csv'))
df = df.rename(columns={'variant_id': 'sent_id'})
print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")

# ── Merge PCFG surprisals ─────────────────────────────────────────
print("Merging PCFG surprisals...")
with open(os.path.join(BASE, 'models', 'pcfg', 'pcfg_surprisals.json'), encoding='utf-8') as f:
    psurp = json.load(f)
df['surprisal_pcfg'] = df['sent_id'].map(psurp).fillna(0.0)
print(f"  Mapped: {(df['surprisal_pcfg'] != 0).sum():,} real scores")

print(f"\nFull df shape: {df.shape}")
print(f"Order type counts:\n{df['order_type'].value_counts().to_string()}")

# ── Paper accuracy targets ────────────────────────────────────────
PAPER = {
    'DO': {
        'dl':   60.0,
        'is':   None,
        'pcfg': None,
        'tri':  None,
        'lstm': None,
        'all':  87.5,
    },
    'IO': {
        'dl':   50.0,
        'is':   None,
        'pcfg': None,
        'tri':  None,
        'lstm': None,
        'all':  75.0,
    }
}

# ── Pairwise classifier ───────────────────────────────────────────
def make_pairs(df, noncanon_type, canon_type):
    """
    For each reference sentence of noncanon_type, pair it with its
    canonical variant (canon_type). Return feature matrix and labels.
    Label=1 if row is the reference (non-canonical wins).
    """
    ref   = df[(df['is_original'] == 1) & (df['order_type'] == noncanon_type)].copy()
    pairs = []
    for _, r in ref.iterrows():
        rid = r['reference_id']
        # canonical variants for this sentence
        cands = df[(df['reference_id'] == rid) &
                   (df['is_original'] == 0) &
                   (df['order_type'] == canon_type)]
        if len(cands) == 0:
            continue
        for _, c in cands.iterrows():
            pairs.append({
                'dl_diff':   r['dl']   - c['dl'],
                'tri_diff':  r['surprisal_trigram'] - c['surprisal_trigram'],
                'pcfg_diff': r['surprisal_pcfg']    - c['surprisal_pcfg'],
                'lstm_diff': r['surprisal_lstm'] - c['surprisal_lstm'],
                'label': 1
            })
            # mirror pair (canonical wins = label 0)
            pairs.append({
                'dl_diff':   c['dl']   - r['dl'],
                'tri_diff':  c['surprisal_trigram'] - r['surprisal_trigram'],
                'pcfg_diff': c['surprisal_pcfg']    - r['surprisal_pcfg'],
                'lstm_diff': c['surprisal_lstm'] - r['surprisal_lstm'],
                'label': 0
            })
    return pd.DataFrame(pairs)


def run_cv(pairs_df, feature_cols, n_folds=10):
    X = pairs_df[feature_cols].values
    y = pairs_df['label'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    kf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in kf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(clf.score(X[test_idx], y[test_idx]) * 100)
    return np.mean(accs), np.std(accs)


def run_all_models(pairs_df, label):
    feature_sets = {
        'DL only':       ['dl_diff'],
        'Trigram only':  ['tri_diff'],
        'PCFG only':     ['pcfg_diff'],
        'LSTM only':     ['lstm_diff'],
        'DL + Trigram':  ['dl_diff', 'tri_diff'],
        'DL + PCFG':     ['dl_diff', 'pcfg_diff'],
        'DL + LSTM':     ['dl_diff', 'lstm_diff'],
        'All features':  ['dl_diff', 'tri_diff', 'pcfg_diff', 'lstm_diff'],
    }
    print(f"\n{'='*60}")
    print(f"  {label}  (N pairs = {len(pairs_df)//2:,})")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'Acc%':>7}  {'Std':>6}  {'Paper%':>8}")
    print(f"  {'-'*50}")
    results = {}
    paper_key = 'DO' if 'DO' in label else 'IO'
    for name, feats in feature_sets.items():
        acc, std = run_cv(pairs_df, feats)
        paper = PAPER[paper_key].get(
            'all' if name == 'All features' else
            'dl'  if name == 'DL only' else
            'tri' if name == 'Trigram only' else
            'pcfg' if name == 'PCFG only' else
            'lstm' if name == 'LSTM only' else None,
            None
        )
        paper_str = f"{paper:.1f}%" if paper else "  —"
        print(f"  {name:<22} {acc:>6.2f}%  {std:>5.2f}  {paper_str:>8}")
        results[name] = acc
    return results


# ── DO-fronted vs Canonical ───────────────────────────────────────
print("\nBuilding DO-fronted pairs (DOSV vs SDOV)...")
do_pairs = make_pairs(df, noncanon_type='DO-fronted', canon_type='Canonical')
print(f"  Pairs: {len(do_pairs):,}  (N ref sentences: {len(do_pairs)//2 // max(1, len(do_pairs)//(2*max(1,len(do_pairs)//2)))})")
do_results = run_all_models(do_pairs, "DO-fronted (DOSV) vs Canonical (SDOV)")

# ── IO-fronted vs Canonical ───────────────────────────────────────
print("\nBuilding IO-fronted pairs (IOSV vs SIOV)...")
io_pairs = make_pairs(df, noncanon_type='IO-fronted', canon_type='Canonical')
io_results = run_all_models(io_pairs, "IO-fronted (IOSV) vs Canonical (SIOV)")

# ── Summary table ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SUMMARY — Ours vs Paper")
print(f"{'='*60}")
print(f"  {'Model':<22} {'DO (ours)':>10} {'DO (paper)':>11} {'IO (ours)':>10} {'IO (paper)':>11}")
print(f"  {'-'*66}")
for name in ['DL only', 'Trigram only', 'PCFG only', 'LSTM only', 'All features']:
    do_acc = do_results.get(name, 0)
    io_acc = io_results.get(name, 0)
    do_paper = PAPER['DO'].get('all' if name=='All features' else 'dl' if name=='DL only' else None, None)
    io_paper = PAPER['IO'].get('all' if name=='All features' else 'dl' if name=='DL only' else None, None)
    do_p = f"{do_paper:.1f}%" if do_paper else "—"
    io_p = f"{io_paper:.1f}%" if io_paper else "—"
    print(f"  {name:<22} {do_acc:>9.2f}%  {do_p:>10}  {io_acc:>9.2f}%  {io_p:>10}")

print(f"\nDone.")
