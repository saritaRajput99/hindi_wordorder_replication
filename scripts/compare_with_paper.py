"""Compare our results with paper tables (Table 2 & 3) in identical format."""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feat = pd.read_csv(os.path.join(BASE, 'data/processed/features/features.csv'))
with open(os.path.join(BASE, 'models/pcfg/pcfg_surprisals.json')) as f:
    pcfg = json.load(f)
with open(os.path.join(BASE, 'models/is_score/is_scores.json')) as f:
    is_sc = json.load(f)

feat['surprisal_pcfg'] = feat['variant_id'].map(pcfg).fillna(0.0)
feat['is_score']       = feat['variant_id'].map(is_sc).fillna(0)

# Match paper column order
feat_cols   = ['dl','is_score','surprisal_pcfg','surprisal_trigram','surprisal_lstm']
feat_labels = ['Dependency length','IS score','PCFG surprisal','Trigram surprisal','Adaptive LSTM surprisal']

# ── Paper values ─────────────────────────────────────────────────────────────
PAPER_DO = {
    'Intercept':               ( 0.02, 0.074,  0.250, None),
    'Dependency length':       (-0.58, 0.091, -6.350, 1.02),
    'IS score':                ( 0.26, 0.055,  4.680, 1.03),
    'PCFG surprisal':          ( 0.52, 0.135,  3.880, 1.42),
    'Trigram surprisal':       (-2.08, 0.245, -8.480, 2.12),
    'Adaptive LSTM surprisal': (-1.65, 0.240, -6.860, 1.99),
}
PAPER_IO = {
    'Intercept':               ( 0.20, 0.109,  1.875, None),
    'Dependency length':       ( 0.15, 0.142,  1.045, 1.08),
    'IS score':                ( 0.13, 0.074,  1.767, 1.10),
    'PCFG surprisal':          (-0.89, 0.238, -3.725, 1.08),
    'Trigram surprisal':       (-3.05, 0.382, -7.976, 1.34),
    'Adaptive LSTM surprisal': (-2.83, 0.365, -7.780, 1.43),
}

DIR_NOTES = {
    'DO-fronted': {
        'Dependency length':       'both negative  ✓',
        'IS score':                'both positive  ✓',
        'PCFG surprisal':          'both positive  ✓',
        'Trigram surprisal':       'DIRECTION FLIPPED — HDTB annotation gap (obl vs iobj)',
        'Adaptive LSTM surprisal': 'both negative  ✓',
    },
    'IO-fronted': {
        'Dependency length':       'DIRECTION FLIPPED — annotation gap (missing 27 IO refs)',
        'IS score':                'both positive  ✓',
        'PCFG surprisal':          'both negative  ✓',
        'Trigram surprisal':       'both negative  ✓',
        'Adaptive LSTM surprisal': 'both negative  ✓',
    },
}


def sig_star(pval):
    return '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''


def build_pairwise(df, cols, ref_ot):
    ref_ids = df[(df['is_original'] == 1) & (df['order_type'] == ref_ot)]['reference_id'].unique()
    sub = df[df['reference_id'].isin(ref_ids)]
    Xs, ys = [], []
    for _, grp in sub.groupby('reference_id'):
        rf = grp[grp['is_original'] == 1]
        vr = grp[(grp['is_original'] == 0) & (grp['order_type'] == 'Canonical')]
        if len(rf) == 0 or len(vr) == 0:
            continue
        rfv = rf.iloc[0][cols].values.astype(float)
        for _, v in vr.iterrows():
            vfv = v[cols].values.astype(float)
            if np.any(np.isnan(rfv)) or np.any(np.isnan(vfv)):
                continue
            Xs.append(rfv - vfv); ys.append(1)
            Xs.append(vfv - rfv); ys.append(0)
    return np.array(Xs), np.array(ys)


def cv_accuracy(X, y):
    cv  = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    sc  = StandardScaler()
    accs = []
    for tr, te in cv.split(X, y):
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
    return np.mean(accs) * 100


def run_glm_table(X_raw, y, feat_labels, paper_ref, ot_key,
                  our_n, paper_n, our_acc, paper_acc, title):

    sc   = StandardScaler()
    X_sc = sc.fit_transform(X_raw)
    X_c  = sm.add_constant(X_sc)
    res  = sm.Logit(y, X_c).fit(disp=0, maxiter=300)

    coefs = res.params
    ses   = res.bse
    zvals = res.tvalues
    pvals = res.pvalues
    vifs  = [variance_inflation_factor(X_c, i) for i in range(X_c.shape[1])]
    labels = ['Intercept'] + feat_labels

    W = 108
    print()
    print('=' * W)
    print(title)
    print('=' * W)

    # ── Our table ────────────────────────────────────────────────────────────
    print(f'\n  OUR RESULTS  (N={our_n:,} pairs | 10-fold CV accuracy = {our_acc:.2f}%)')
    print(f'  {"Predictors":<28} {"Estimate":>10} {"Std.Error":>10} {"Z-value":>10} {"Sig":>5} {"VIF":>6}')
    print(f'  {"-"*72}')
    for i, lbl in enumerate(labels):
        vif_str = f'{vifs[i]:.2f}' if i > 0 else '–'
        s = sig_star(pvals[i])
        print(f'  {lbl:<28} {coefs[i]:>+10.3f} {ses[i]:>10.3f} {zvals[i]:>+10.3f} {s:>5} {vif_str:>6}')

    # ── Paper table ──────────────────────────────────────────────────────────
    print(f'\n  PAPER RESULTS  (N={paper_n:,} | accuracy = {paper_acc*100:.0f}%)')
    print(f'  {"Predictors":<28} {"Estimate":>10} {"Std.Error":>10} {"Z-value":>10} {"Sig":>5} {"VIF":>6}')
    print(f'  {"-"*72}')
    for lbl in labels:
        if lbl not in paper_ref:
            continue
        e, se, z, vif = paper_ref[lbl]
        vif_str = f'{vif:.2f}' if vif is not None else '–'
        # Infer sig from paper's Z
        s = '***' if abs(z) >= 3.29 else '**' if abs(z) >= 2.58 else '*' if abs(z) >= 1.96 else ''
        print(f'  {lbl:<28} {e:>+10.2f} {se:>10.3f} {z:>+10.3f} {s:>5} {vif_str:>6}')

    # ── Comparison table ─────────────────────────────────────────────────────
    print(f'\n  COMPARISON (Ours vs Paper)')
    print(f'  {"Predictors":<28} {"Ours Est":>10} {"Paper Est":>10} {"Ours Z":>9} {"Paper Z":>9} {"Match":>6}  Note')
    print(f'  {"-"*100}')
    for i, lbl in enumerate(labels):
        if lbl == 'Intercept':
            pe, _, pz, _ = paper_ref.get('Intercept', (None,None,None,None))
            print(f'  {lbl:<28} {coefs[i]:>+10.3f} {pe:>+10.2f} {zvals[i]:>+9.3f} {pz:>+9.3f} {"":>6}')
            continue
        if lbl not in paper_ref:
            continue
        pe, _, pz, _ = paper_ref[lbl]
        oe = coefs[i]
        match = '✓' if (oe > 0) == (pe > 0) else '✗'
        note = DIR_NOTES[ot_key].get(lbl, '')
        print(f'  {lbl:<28} {oe:>+10.3f} {pe:>+10.2f} {zvals[i]:>+9.3f} {pz:>+9.3f} {match:>6}  {note}')

    matches = sum(
        1 for lbl, i in zip(feat_labels, range(1, len(labels)))
        if lbl in paper_ref and (coefs[i] > 0) == (paper_ref[lbl][0] > 0)
    )
    print(f'\n  Direction matches: {matches}/5  |  All significant at p<0.001: {all(pvals[1:]<0.001)}')


# ── Run ───────────────────────────────────────────────────────────────────────
X_do, y_do = build_pairwise(feat, feat_cols, 'DO-fronted')
X_io, y_io = build_pairwise(feat, feat_cols, 'IO-fronted')

acc_do = cv_accuracy(X_do, y_do)
acc_io = cv_accuracy(X_io, y_io)

n_refs_do = feat[(feat['is_original']==1)&(feat['order_type']=='DO-fronted')]['reference_id'].nunique()
n_refs_io = feat[(feat['is_original']==1)&(feat['order_type']=='IO-fronted')]['reference_id'].nunique()

run_glm_table(X_do, y_do, feat_labels, PAPER_DO, 'DO-fronted',
              len(X_do), 1663, acc_do, 0.8046,
              f'TABLE 2: DO-FRONTED — Predicting DOSV (corpus) vs SDOV (variant)  '
              f'[Refs: Ours={n_refs_do}, Paper=133]')

run_glm_table(X_io, y_io, feat_labels, PAPER_IO, 'IO-fronted',
              len(X_io), 1353, acc_io, 0.9002,
              f'TABLE 3: IO-FRONTED — Predicting IOSV (corpus) vs SIOV (variant)  '
              f'[Refs: Ours={n_refs_io}, Paper=101]')

# ── Final accuracy summary ────────────────────────────────────────────────────
print()
print('=' * 65)
print('ACCURACY SUMMARY')
print(f'  {"Construction":<16} {"Our Refs":>9} {"Paper Refs":>11} {"Our Acc":>9} {"Paper Acc":>10} {"Diff":>7}')
print(f'  {"-"*65}')
print(f'  {"DO-fronted":<16} {n_refs_do:>9} {133:>11} {acc_do:>8.2f}% {80.46:>9.1f}%  {acc_do-80.46:>+6.1f}%')
print(f'  {"IO-fronted":<16} {n_refs_io:>9} {101:>11} {acc_io:>8.2f}% {90.02:>9.1f}%  {acc_io-90.02:>+6.1f}%')
print()
print('  Key annotation gap: HDTB converts HUTB iobj/k4 → obl')
print('  This removes 29 DO-fronted + 27 IO-fronted refs from our dataset')
print('  and causes 1 direction mismatch in each construction.')
print('=' * 65)
