"""
LLM vs LSTM Surprisal Comparison
=================================
Replaces Adaptive LSTM with XGLM-1.7B and Gemma-3-12B, runs pairwise
logistic regression (same 10-fold CV as paper), prints comparison table.

Run from project root:
    cd D:\\MTP_WO\\hindi_wordorder_replication
    python scripts/run_llm_comparison.py

Flags (edit below):
    RUN_XGLM   — score with XGLM-1.7B
    RUN_GEMMA  — score with Gemma-3-12B
    GEMMA_4BIT — use 4-bit quantization for Gemma (needs bitsandbytes)
    GEMMA_4B   — use Gemma-3-4B instead of 12B (faster, less accurate)

If scores already exist in models/xglm/ or models/gemma3/, scoring is skipped.

Output:
    results/llm_comparison.csv   — full result table
    Printed comparison table
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print('WARNING: statsmodels not available — GLM coefficient tables will be skipped.')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pathlib import Path


def _add_col(df, scores, col):
    """Map variant_id → surprisal score into a new DataFrame column."""
    df = df.copy()
    df[col] = df['variant_id'].map(lambda v: scores.get(v, 100.0))
    real = (df[col] < 100.0).sum()
    print(f'  {col}: {real:,} scores loaded')
    return df


# ── CONFIG ────────────────────────────────────────────────────────────────────
RUN_XGLM   = True    # Set False to skip XGLM scoring (use existing JSON)
RUN_GEMMA  = True    # Set False to skip Gemma scoring
GEMMA_4BIT = False   # True = 4-bit quantization (needs bitsandbytes, saves VRAM)
GEMMA_4B   = False   # True = use gemma-3-4b-pt instead of 12b

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_CSV   = os.path.join(BASE, 'data/processed/features/features.csv')
XGLM_DIR   = os.path.join(BASE, 'models/xglm')
GEMMA_DIR  = os.path.join(BASE, 'models/gemma-3b')
OUT_CSV    = os.path.join(BASE, 'results/llm_comparison.csv')

# ── LOAD BASE FEATURES ────────────────────────────────────────────────────────
print('=' * 65)
print('Loading features ...')
print('=' * 65)

df = pd.read_csv(FEAT_CSV)
print(f'  Variants loaded : {len(df):,}')

# surprisal_pcfg, is_score, surprisal_modelblocks, surprisal_depparse all come
# straight from features.csv now — it's the authoritative merged dataset
# (assemble_features.py). Previously this script re-loaded surprisal_pcfg and
# is_score from separate old JSON files (models/pcfg/pcfg_surprisals.json —
# the OLD local-Berkeley-only PCFG we explicitly stopped using — and
# models/is_score/is_scores.json) and OVERWROTE the correct merged columns
# with those stale ones. Don't reintroduce that.
for col in ['surprisal_pcfg', 'is_score', 'surprisal_modelblocks', 'surprisal_depparse']:
    real = df[col].notna().sum()
    print(f'  {col}: {real:,}/{len(df):,} from features.csv')

# ── XGLM SCORING ──────────────────────────────────────────────────────────────
xglm_col = 'surprisal_xglm'

xglm_json = os.path.join(XGLM_DIR, 'xglm_surprisals.json')
if os.path.exists(xglm_json):
    print(f'\nXGLM scores found — loading from {xglm_json}')
    with open(xglm_json, encoding='utf-8') as f:
        xglm_scores = json.load(f)
    df = _add_col(df, xglm_scores, xglm_col)
elif RUN_XGLM:
    print('\nRunning XGLM-1.7B scoring ...')
    from src.surprisal_models.xglm_surprisal import XGLMSurprisal
    xglm = XGLMSurprisal(model_dir=XGLM_DIR)
    xglm_scores = xglm.compute_all(df)
    df = xglm.add_to_df(df, xglm_scores, col=xglm_col)
    xglm.print_stats(df, col=xglm_col)
else:
    print(f'\nXGLM: skipped (RUN_XGLM=False, no existing JSON)')
    df[xglm_col] = np.nan

# ── GEMMA SCORING ─────────────────────────────────────────────────────────────
gemma_col  = 'surprisal_gemma3'
gemma_name = 'google/gemma-3-4b-pt' if GEMMA_4B else 'google/gemma-3-12b-pt'
gemma_tag  = gemma_name.replace('/', '_').replace('-', '_')
# Also accept the simpler filename written by kaggle_llm_scoring.py
gemma_json_simple = os.path.join(GEMMA_DIR, 'gemma3_4b_surprisals.json')
gemma_json = gemma_json_simple if os.path.exists(gemma_json_simple) \
             else os.path.join(GEMMA_DIR, f'{gemma_tag}_surprisals.json')

if os.path.exists(gemma_json):
    print(f'\nGemma scores found — loading from {gemma_json}')
    with open(gemma_json, encoding='utf-8') as f:
        gemma_scores = json.load(f)
    df = _add_col(df, gemma_scores, gemma_col)
elif RUN_GEMMA:
    print(f'\nRunning {gemma_name} scoring ...')
    from src.surprisal_models.gemma_surprisal import GemmaSurprisal
    gemma = GemmaSurprisal(
        model_dir=GEMMA_DIR,
        model_name=gemma_name,
        load_in_4bit=GEMMA_4BIT,
    )
    gemma_scores = gemma.compute_all(df)
    df = gemma.add_to_df(df, gemma_scores, col=gemma_col)
    gemma.print_stats(df, col=gemma_col)
else:
    print(f'\nGemma: skipped (RUN_GEMMA=False, no existing JSON)')
    df[gemma_col] = np.nan


# ── PAIRWISE CLASSIFIER ───────────────────────────────────────────────────────
def build_pairs(df, feat_cols, order_type):
    """Build pairwise difference vectors: corpus - canonical_variant."""
    ref_ids = df[
        (df['is_original'] == 1) &
        (df['order_type'] == order_type)
    ]['reference_id'].unique()

    sub = df[df['reference_id'].isin(ref_ids)]
    Xs, ys = [], []

    for _, grp in sub.groupby('reference_id'):
        rf = grp[grp['is_original'] == 1]
        vr = grp[(grp['is_original'] == 0) & (grp['order_type'] == 'Canonical')]
        if len(rf) == 0 or len(vr) == 0:
            continue
        rfv = rf.iloc[0][feat_cols].values.astype(float)
        if np.any(np.isnan(rfv)):
            continue   # reference row itself missing a feature (e.g. unparsed sentence)
        for _, v in vr.iterrows():
            vfv = v[feat_cols].values.astype(float)
            if np.any(np.isnan(vfv)):
                continue   # variant row missing a feature — skip this pair, not the whole ref
            delta = rfv - vfv
            Xs.append(delta)
            ys.append(1)
            Xs.append(-delta)
            ys.append(0)

    if not Xs:
        return None, None
    return np.array(Xs), np.array(ys)


def run_cv(X, y, n_folds=10, seed=42):
    """10-fold stratified CV, returns mean accuracy."""
    if X is None or len(X) == 0:
        return float('nan')
    cv   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in cv.split(X, y):
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        lr  = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=seed)
        lr.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], lr.predict(Xte)))
    return np.mean(accs) * 100


# ── BUILD COMPARISON TABLE ────────────────────────────────────────────────────
# Feature sets to compare — each row = one experiment
BASE_FEATS  = ['dl', 'is_score', 'surprisal_pcfg', 'surprisal_trigram']
LSTM_FEATS  = BASE_FEATS + ['surprisal_lstm']
XGLM_FEATS  = BASE_FEATS + [xglm_col]
GEMMA_FEATS = BASE_FEATS + [gemma_col]

# Single-predictor experiments for LLM surprisal alone
SINGLE = {
    'Adaptive LSTM alone': ['surprisal_lstm'],
    'XGLM alone':          [xglm_col],
    'Gemma-3 alone':       [gemma_col],
}

ORDER_TYPES = ['DO-fronted', 'IO-fronted']

NEW3_FEATS     = ['surprisal_pcfg', 'surprisal_modelblocks', 'surprisal_depparse']
NON_LLM_ALL7   = LSTM_FEATS + ['surprisal_modelblocks', 'surprisal_depparse']   # no XGLM/Gemma
EVERYTHING     = BASE_FEATS + ['surprisal_lstm', xglm_col, gemma_col,
                                'surprisal_modelblocks', 'surprisal_depparse']

experiments = [
    ('Trigram only',             ['surprisal_trigram']),
    ('DL only',                  ['dl']),
    ('IS score only',            ['is_score']),
    ('Adaptive LSTM only',       ['surprisal_lstm']),
    ('XGLM only',                [xglm_col]),
    ('Gemma-3 only',             [gemma_col]),
    ('PCFG (tool) only',         ['surprisal_pcfg']),
    ('Modelblocks only',         ['surprisal_modelblocks']),
    ('Dep-parser only',          ['surprisal_depparse']),
    ('All 5 (original paper)',   LSTM_FEATS),
    ('LSTM → XGLM replacement',  XGLM_FEATS),
    ('LSTM → Gemma-3 replacement', GEMMA_FEATS),
    ('3 new tool features combined (PCFG+Modelblocks+Dep-parser)', NEW3_FEATS),
    ('All 7 non-LLM (base+LSTM+PCFG+Modelblocks+Dep-parser, no XGLM/Gemma)', NON_LLM_ALL7),
    ('Everything combined (base+LSTM+XGLM+Gemma+Modelblocks+Dep-parser)', EVERYTHING),
]

print('\n' + '=' * 65)
print('Running logistic regression (10-fold CV) ...')
print('=' * 65)

results = []
for label, feats in experiments:
    # Check if all feature columns exist and are non-null
    available = [c for c in feats if c in df.columns and df[c].notna().any()]
    if len(available) < len(feats):
        missing = set(feats) - set(available)
        print(f'  Skipping "{label}" — missing columns: {missing}')
        row = {'Experiment': label}
        for ot in ORDER_TYPES:
            row[ot] = 'n/a'
        results.append(row)
        continue

    print(f'\n  {label}')
    row = {'Experiment': label}
    for ot in ORDER_TYPES:
        X, y = build_pairs(df, feats, ot)
        acc  = run_cv(X, y)
        row[ot] = f'{acc:.1f}%' if not np.isnan(acc) else 'n/a'
        print(f'    {ot:<15}: {row[ot]}')
    results.append(row)

# ── PRINT TABLE ───────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)

print('\n')
print('=' * 65)
print('COMPARISON TABLE — Pairwise Classification Accuracy (10-fold CV)')
print('=' * 65)
col_w = [42, 14, 14]
header = f'{"Experiment":<{col_w[0]}} {"DO-fronted":>{col_w[1]}} {"IO-fronted":>{col_w[2]}}'
print(header)
print('-' * sum(col_w))
for _, r in res_df.iterrows():
    # Add separator before the "All 5" row
    if 'All 5' in r['Experiment']:
        print('-' * sum(col_w))
    print(f'{r["Experiment"]:<{col_w[0]}} '
          f'{str(r["DO-fronted"]):>{col_w[1]}} '
          f'{str(r["IO-fronted"]):>{col_w[2]}}')
print('=' * sum(col_w))

# ── SAVE ──────────────────────────────────────────────────────────────────────
Path(OUT_CSV).parent.mkdir(exist_ok=True)
try:
    res_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'\nSaved: {OUT_CSV}')
except PermissionError:
    alt = OUT_CSV.replace('.csv', '_new.csv')
    res_df.to_csv(alt, index=False, encoding='utf-8-sig')
    print(f'\nSaved (original locked): {alt}')
print('\nInterpretation:')
print('  • "LSTM → XGLM replacement"  : XGLM replaces LSTM, other 4 features unchanged')
print('  • "LSTM → Gemma-3 replacement": Gemma-3 replaces LSTM, other 4 features unchanged')
print('  • Higher accuracy = better surprisal signal for Hindi word order prediction')


# ── GLM COEFFICIENT TABLES (paper format) ─────────────────────────────────────
if not HAS_STATSMODELS:
    print('\nGLM coefficient tables skipped (statsmodels not available).')
    print('Main accuracy comparison table above is complete.')
    raise SystemExit(0)

def sig_star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


COEF_ROWS = []   # accumulates rows for CSV export


def glm_table(df, feat_cols, feat_labels, order_type, model_name, cv_acc):
    ref_ids = df[(df['is_original'] == 1) & (df['order_type'] == order_type)]['reference_id'].unique()
    sub = df[df['reference_id'].isin(ref_ids)]
    Xs, ys = [], []
    for _, grp in sub.groupby('reference_id'):
        rf = grp[grp['is_original'] == 1]
        vr = grp[(grp['is_original'] == 0) & (grp['order_type'] == 'Canonical')]
        if len(rf) == 0 or len(vr) == 0:
            continue
        rfv = rf.iloc[0][feat_cols].values.astype(float)
        for _, v in vr.iterrows():
            vfv = v[feat_cols].values.astype(float)
            if np.any(np.isnan(rfv)) or np.any(np.isnan(vfv)):
                continue
            Xs.append(rfv - vfv); ys.append(1)
            Xs.append(vfv - rfv); ys.append(0)

    X_raw = np.array(Xs)
    y     = np.array(ys)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_raw)
    X_c   = sm.add_constant(X_sc)
    res   = sm.Logit(y, X_c).fit(disp=0, maxiter=300)

    coefs  = res.params
    ses    = res.bse
    zvals  = res.tvalues
    pvals  = res.pvalues
    vifs   = [variance_inflation_factor(X_c, i) for i in range(X_c.shape[1])]
    labels = ['Intercept'] + feat_labels

    W = 80
    print()
    print('=' * W)
    print(f'  {order_type} — {model_name}  (N={len(X_raw):,} pairs | CV acc = {cv_acc:.1f}%)')
    print('=' * W)
    print(f'  {"Predictors":<30} {"Estimate":>10} {"Std.Error":>10} {"Z-value":>10} {"Sig":>5} {"VIF":>6}')
    print(f'  {"-"*74}')
    for i, lbl in enumerate(labels):
        vif_str = f'{vifs[i]:.4f}' if i > 0 else ''
        s = sig_star(pvals[i])
        print(f'  {lbl:<30} {coefs[i]:>+10.3f} {ses[i]:>10.3f} {zvals[i]:>+10.3f} {s:>5} {vif_str:>6}')
        COEF_ROWS.append({
            'Construction':  order_type,
            'Model':         model_name,
            'CV_Accuracy':   round(cv_acc, 1),
            'Predictor':     lbl,
            'Estimate':      round(coefs[i], 4),
            'Std_Error':     round(ses[i], 4),
            'Z_value':       round(zvals[i], 4),
            'P_value':       round(pvals[i], 6),
            'Sig':           s,
            'VIF':           round(vifs[i], 4) if i > 0 else '',
        })
    print()


# Accuracy values from the CV run just above (results list), not hardcoded —
# previously this used stale numbers hardcoded from an earlier run, which
# silently drifted out of sync with the live data (e.g. after order_type
# relabeling changed the DO/IO-fronted reference counts).
def _acc_lookup(label):
    row = next(r for r in results if r['Experiment'] == label)
    return {ot: (float(row[ot].rstrip('%')) if row[ot] != 'n/a' else float('nan'))
            for ot in ORDER_TYPES}

ORIG_ACC    = _acc_lookup('All 5 (original paper)')
XGLM_ACC    = _acc_lookup('LSTM → XGLM replacement')
GEMM_ACC    = _acc_lookup('LSTM → Gemma-3 replacement')
NONLLM_ACC  = _acc_lookup('All 7 non-LLM (base+LSTM+PCFG+Modelblocks+Dep-parser, no XGLM/Gemma)')
EVERY_ACC   = _acc_lookup('Everything combined (base+LSTM+XGLM+Gemma+Modelblocks+Dep-parser)')

BASE_FEATS  = ['dl', 'is_score', 'surprisal_pcfg', 'surprisal_trigram']
BASE_LABELS = ['Dependency length', 'IS score', 'PCFG surprisal', 'Trigram surprisal']

print('\n\n' + '#' * 80)
print('#  GLM COEFFICIENT TABLES — paper format (standardised predictors)')
print('#' * 80)

for ot in ['DO-fronted', 'IO-fronted']:
    # Original 5-feature (LSTM)
    glm_table(df,
              BASE_FEATS + ['surprisal_lstm'],
              BASE_LABELS + ['Adaptive LSTM surprisal'],
              ot, 'All-5 original (LSTM)', ORIG_ACC[ot])

    # LSTM → XGLM
    if df[xglm_col].notna().any():
        glm_table(df,
                  BASE_FEATS + [xglm_col],
                  BASE_LABELS + ['XGLM surprisal'],
                  ot, 'LSTM → XGLM replacement', XGLM_ACC[ot])

    # LSTM → Gemma-3
    if df[gemma_col].notna().any():
        glm_table(df,
                  BASE_FEATS + [gemma_col],
                  BASE_LABELS + ['Gemma-3 surprisal'],
                  ot, 'LSTM → Gemma-3 replacement', GEMM_ACC[ot])

    # All 7 non-LLM: base + LSTM + Modelblocks + Dep-parser (no XGLM/Gemma)
    glm_table(df,
              BASE_FEATS + ['surprisal_lstm', 'surprisal_modelblocks', 'surprisal_depparse'],
              BASE_LABELS + ['Adaptive LSTM surprisal', 'Modelblocks surprisal', 'Dep-parser surprisal'],
              ot, 'All 7 non-LLM', NONLLM_ACC[ot])

    # Everything combined: base + LSTM + XGLM + Gemma + Modelblocks + Dep-parser
    if df[xglm_col].notna().any() and df[gemma_col].notna().any():
        glm_table(df,
                  BASE_FEATS + ['surprisal_lstm', xglm_col, gemma_col,
                                 'surprisal_modelblocks', 'surprisal_depparse'],
                  BASE_LABELS + ['Adaptive LSTM surprisal', 'XGLM surprisal', 'Gemma-3 surprisal',
                                  'Modelblocks surprisal', 'Dep-parser surprisal'],
                  ot, 'Everything combined', EVERY_ACC[ot])

# ── SAVE COEFFICIENT CSV ───────────────────────────────────────────────────────
COEF_CSV = os.path.join(BASE, 'results/llm_coefficients.csv')
coef_df = pd.DataFrame(COEF_ROWS)
try:
    coef_df.to_csv(COEF_CSV, index=False, encoding='utf-8-sig')
    print(f'\nCoefficients saved: {COEF_CSV}')
except PermissionError:
    alt = COEF_CSV.replace('.csv', '_new.csv')
    coef_df.to_csv(alt, index=False, encoding='utf-8-sig')
    print(f'\nCoefficients saved (original locked): {alt}')
