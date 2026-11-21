"""
Run full pairwise analysis: DL + Trigram + Adaptive LSTM (no PCFG, no IS score).
Reads: models/adaptive_lstm/final_var_score.csv  (dl, surprisal_adaptive_lstm)
       models/trigram/trigram_surprisals.json     (surprisal_trigram)
Runs:  DO-fronted, IO-fronted, O-fronted pairwise logistic regression (10-fold CV)
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats

RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Paper reference values ────────────────────────────────────────
PAPER_DO = {'dl': -0.58, 'surprisal_trigram': -2.08,
            'surprisal_pcfg': +0.52, 'surprisal_adaptive_lstm': -1.65,
            'is_score': +0.26, 'N': 1663, 'acc': 0.89}
PAPER_IO = {'dl': +0.15, 'surprisal_trigram': -3.05,
            'surprisal_pcfg': -0.89, 'surprisal_adaptive_lstm': -2.83,
            'is_score': +0.13, 'N': 1353, 'acc': 0.87}

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Load and merge features
# ══════════════════════════════════════════════════════════════════
print('=' * 60)
print('STEP 1: Loading features')
print('=' * 60)

df = pd.read_csv(os.path.join(BASE, 'models', 'adaptive_lstm', 'final_var_score.csv'))
df = df.rename(columns={'sent_id': 'variant_id'})
print(f'  Loaded final_var_score.csv: {len(df):,} rows')
print(f'  Columns: {list(df.columns)}')

# Merge trigram
trig_path = os.path.join(BASE, 'models', 'trigram', 'trigram_surprisals.json')
with open(trig_path, encoding='utf-8') as f:
    trig = json.load(f)
df['surprisal_trigram'] = df['variant_id'].map(trig).fillna(0.0)
mapped = (df['surprisal_trigram'] > 0).sum()
print(f'  Trigram scores mapped: {mapped:,}/{len(df):,}')

# Feature summary
feat_cols = ['dl', 'surprisal_trigram', 'surprisal_adaptive_lstm']
print(f'  Features for analysis: {feat_cols}')
print(f'  (PCFG and IS score excluded — to be added later)')
print()

# Corpus breakdown
corp = df[df['is_original'] == 1]
print('  Corpus reference sentences:')
for ot, cnt in corp['order_type'].value_counts().items():
    print(f'    {ot:<14}: {cnt}')
print()

# Save features.csv
feat_csv = os.path.join(BASE, 'data', 'processed', 'features', 'features.csv')
os.makedirs(os.path.dirname(feat_csv), exist_ok=True)
df.to_csv(feat_csv, index=False)
print(f'  Saved: {feat_csv}')

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Pairwise builder
# ══════════════════════════════════════════════════════════════════

def build_pairwise(df, feat, ref_ot, var_ot='Canonical'):
    ref_ids = df[(df['is_original'] == 1) &
                 (df['order_type'] == ref_ot)]['reference_id'].unique()
    sub = df[df['reference_id'].isin(ref_ids)]
    print(f'  {ref_ot}: {len(ref_ids):,} refs, {len(sub):,} total rows')

    X_list, y_list = [], []
    for _, grp in sub.groupby('reference_id'):
        ref_rows = grp[grp['is_original'] == 1]
        var_rows = grp[(grp['is_original'] == 0) & (grp['order_type'] == var_ot)]
        if len(ref_rows) == 0 or len(var_rows) == 0:
            continue
        rf = ref_rows.iloc[0][feat].values.astype(float)
        for _, vr in var_rows.iterrows():
            vf = vr[feat].values.astype(float)
            if np.any(np.isnan(rf)) or np.any(np.isnan(vf)):
                continue
            X_list.append(rf - vf); y_list.append(1)
            X_list.append(vf - rf); y_list.append(0)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f'    Pairs: {len(X):,}  (paper target for {ref_ot}: {PAPER_DO["N"] if ref_ot=="DO-fronted" else PAPER_IO["N"] if ref_ot=="IO-fronted" else "N/A"})')
    return X, y


def run_cv(X, y, n_folds=10):
    cv   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    clf  = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    scaler = StandardScaler()
    accs, coefs = [], []
    for tr, te in cv.split(X, y):
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
        coefs.append(clf.coef_[0])
    mean_coef = np.mean(coefs, axis=0)
    # t-test for significance
    pvals = []
    for j in range(X.shape[1]):
        col_coefs = [c[j] for c in coefs]
        t, p = stats.ttest_1samp(col_coefs, 0)
        pvals.append(p)
    return np.mean(accs), np.std(accs), mean_coef, pvals


def print_result(ot, acc, std, feat, coef, pval, paper_ref):
    print(f'\n  Results for {ot}:')
    print(f'  Accuracy: {acc*100:.2f}% (±{std*100:.2f})  Paper: {paper_ref.get("acc","--")*100 if isinstance(paper_ref.get("acc"), float) else "--"}%')
    print(f'  {"Feature":<28} {"Coef":>8}  {"Sig":>4}  {"Paper Coef":>10}')
    print(f'  {"-"*58}')
    for i, f in enumerate(feat):
        sig = '***' if pval[i] < 0.001 else '**' if pval[i] < 0.01 else '*' if pval[i] < 0.05 else ''
        pc  = paper_ref.get(f, '--')
        pc_str = f'{pc:+.2f}' if isinstance(pc, float) else str(pc)
        print(f'  {f:<28} {coef[i]:>+8.3f}  {sig:>4}  {pc_str:>10}')

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Run analysis per order type
# ══════════════════════════════════════════════════════════════════
print('=' * 60)
print('STEP 2: Pairwise logistic regression (10-fold CV)')
print('=' * 60)

results = {}

for ot, paper_ref in [('DO-fronted', PAPER_DO), ('IO-fronted', PAPER_IO), ('O-fronted', {})]:
    print(f'\n[{ot}] corpus vs canonical variants:')
    X, y = build_pairwise(df, feat_cols, ref_ot=ot, var_ot='Canonical')
    if len(X) < 20:
        print(f'  Too few pairs ({len(X)}) — skipping.')
        continue
    acc, std, coef, pval = run_cv(X, y)
    print_result(ot, acc, std, feat_cols, coef, pval, paper_ref)
    results[ot] = {'accuracy': acc, 'std': std,
                   'coef': dict(zip(feat_cols, coef.tolist())),
                   'pval': dict(zip(feat_cols, pval))}

# ══════════════════════════════════════════════════════════════════
# STEP 4 — Save results
# ══════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('STEP 3: Saving results')
print('=' * 60)

rows = []
for ot, res in results.items():
    for feat in feat_cols:
        rows.append({
            'order_type'  : ot,
            'feature'     : feat,
            'coef'        : round(res['coef'][feat], 4),
            'pval'        : round(res['pval'][feat], 4),
            'sig'         : '***' if res['pval'][feat] < 0.001 else '**' if res['pval'][feat] < 0.01 else '*' if res['pval'][feat] < 0.05 else '',
            'accuracy'    : round(res['accuracy']*100, 2),
        })

results_df = pd.DataFrame(rows)
out_csv = os.path.join(RESULTS_DIR, 'regression_results_no_pcfg.csv')
results_df.to_csv(out_csv, index=False)
print(f'  Saved: {out_csv}')
print()
print(results_df.to_string(index=False))

print('\n' + '=' * 60)
print('DONE. Features used: DL, Trigram, Adaptive LSTM')
print('PCFG and IS score excluded (to be added in next run)')
print('=' * 60)
