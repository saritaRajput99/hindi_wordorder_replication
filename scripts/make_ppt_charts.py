# -*- coding: utf-8 -*-
"""
Standalone script: generates the 3 result charts as PNG files you can copy
straight into PowerPoint (Insert > Pictures > This Device, or just copy the
file and paste onto a slide).

Run from anywhere:
    python scripts/make_ppt_charts.py          (from inside hindi_wordorder_replication/)
or:
    python E:\\hindi_wordorder_replication\\scripts\\make_ppt_charts.py

Reads only already-computed project files (features.csv, vocab.json) -- no
GPU, no internet, runs in a few seconds. Output folder:
    results/figures/ppt/
        chart1_llm_vs_lstm.png
        chart2_dl_oov_stratified.png
        chart3_pipeline_synthesis.png
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_CSV = os.path.join(BASE, 'data', 'processed', 'features', 'features.csv')
VOCAB_JSON   = os.path.join(BASE, 'models', 'adaptive_lstm', 'vocab.json')
OUT_DIR      = os.path.join(BASE, 'results', 'figures', 'ppt')
os.makedirs(OUT_DIR, exist_ok=True)

# Dataviz-validated categorical palette (blue / orange / aqua / purple)
C_LSTM, C_XGLM, C_GEMMA, C_NEW = '#2a78d6', '#eb6834', '#1baf7a', '#9b59b6'
INK, GRIDLINE = '#0b0b0b', '#e1e0d9'

def style_axes(ax):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)

def bar_labels(ax, bars, fmt='{:.1f}'):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords='offset points', xytext=(0, 3),
                    ha='center', va='bottom', fontsize=9, color=INK)

print('Loading features.csv ...')
df = pd.read_csv(FEATURES_CSV)

def build_pairs(df, feat_cols, order_type):
    ref_ids = df[(df['is_original'] == 1) & (df['order_type'] == order_type)]['reference_id'].unique()
    sub = df[df['reference_id'].isin(ref_ids)]
    Xs, ys = [], []
    for _, grp in sub.groupby('reference_id'):
        rf = grp[grp['is_original'] == 1]
        vr = grp[(grp['is_original'] == 0) & (grp['order_type'] == 'Canonical')]
        if len(rf) == 0 or len(vr) == 0:
            continue
        rfv = rf.iloc[0][feat_cols].values.astype(float)
        if np.any(np.isnan(rfv)):
            continue
        for _, v in vr.iterrows():
            vfv = v[feat_cols].values.astype(float)
            if np.any(np.isnan(vfv)):
                continue
            delta = rfv - vfv
            Xs.append(delta); ys.append(1)
            Xs.append(-delta); ys.append(0)
    if not Xs:
        return None, None
    return np.array(Xs), np.array(ys)

def run_cv(X, y, n_folds=10, seed=42):
    if X is None or len(X) == 0:
        return float('nan')
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=seed)
        lr.fit(Xtr, y[tr])
        accs.append(accuracy_score(y[te], lr.predict(Xte)))
    return np.mean(accs) * 100

BASE_FEATS  = ['dl', 'is_score', 'surprisal_pcfg', 'surprisal_trigram']
LSTM_FEATS  = BASE_FEATS + ['surprisal_lstm']
XGLM_FEATS  = BASE_FEATS + ['surprisal_xglm']
GEMMA_FEATS = BASE_FEATS + ['surprisal_gemma']
NON_LLM_ALL7 = LSTM_FEATS + ['surprisal_modelblocks', 'surprisal_depparse']
EVERYTHING   = BASE_FEATS + ['surprisal_lstm', 'surprisal_xglm', 'surprisal_gemma',
                              'surprisal_modelblocks', 'surprisal_depparse']

print('Running classifiers (live, ~10-20s) ...')
results = {}
for label, feats in [('All-5 (LSTM)', LSTM_FEATS), ('XGLM', XGLM_FEATS), ('Gemma', GEMMA_FEATS),
                     ('Everything', EVERYTHING)]:
    row = {}
    for ot in ['DO-fronted', 'IO-fronted']:
        X, y = build_pairs(df, feats, ot)
        row[ot] = run_cv(X, y)
    results[label] = row
    print(f'  {label:<14} DO={row["DO-fronted"]:.1f}%  IO={row["IO-fronted"]:.1f}%')

# ═══════════════════════════════════════════════════════════════════════
# CHART 1 — Why LLMs beat the LSTM
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2)
width = 0.26
series = [('LSTM (paper\'s setup)', 'All-5 (LSTM)', C_LSTM),
          ('LSTM -> XGLM-1.7B', 'XGLM', C_XGLM),
          ('LSTM -> Gemma-3', 'Gemma', C_GEMMA)]
for i, (label, key, color) in enumerate(series):
    vals = [results[key]['DO-fronted'], results[key]['IO-fronted']]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, zorder=3)
    bar_labels(ax, bars)
ax.set_xticks(x); ax.set_xticklabels(['DO-fronted', 'IO-fronted'], fontsize=11)
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 100)
ax.set_title('Why LLMs Beat the LSTM', fontsize=13, fontweight='bold')
style_axes(ax)
ax.legend(fontsize=9, frameon=False, loc='lower right')
plt.tight_layout()
p1 = os.path.join(OUT_DIR, 'chart1_llm_vs_lstm.png')
fig.savefig(p1, dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved ->', p1)

# ═══════════════════════════════════════════════════════════════════════
# CHART 2 — Stratified by dependency length and OOV rate
# ═══════════════════════════════════════════════════════════════════════
with open(VOCAB_JSON, encoding='utf-8') as f:
    w2i = json.load(f)['w2i']

records = []
for order_type in ['DO-fronted', 'IO-fronted']:
    corpus_rows = df[(df['is_original'] == 1) & (df['order_type'] == order_type)]
    for _, corp in corpus_rows.iterrows():
        cands = df[(df['reference_id'] == corp['reference_id']) &
                   (df['is_original'] == 0) & (df['order_type'] == 'Canonical')]
        if len(cands) == 0:
            continue
        lstm_correct = corp['surprisal_lstm'] < cands['surprisal_lstm'].mean()
        xglm_correct = corp['surprisal_xglm'] < cands['surprisal_xglm'].mean()
        words = str(corp['text']).split()
        oov = sum(1 for w in words if w not in w2i) / len(words) if words else np.nan
        records.append({'dl': corp['dl'], 'oov_rate': oov,
                         'lstm_correct': lstm_correct, 'xglm_correct': xglm_correct})
err_df = pd.DataFrame(records)

def acc(mask, col):
    return 100 * err_df.loc[mask, col].mean()

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4))
w = 0.32
x = np.arange(2)
lstm_dl  = [acc(err_df['dl'] > 50, 'lstm_correct'), acc(err_df['dl'] <= 50, 'lstm_correct')]
xglm_dl  = [acc(err_df['dl'] > 50, 'xglm_correct'), acc(err_df['dl'] <= 50, 'xglm_correct')]
lstm_oov = [acc(err_df['oov_rate'] > 0.1, 'lstm_correct'), acc(err_df['oov_rate'] <= 0.1, 'lstm_correct')]
xglm_oov = [acc(err_df['oov_rate'] > 0.1, 'xglm_correct'), acc(err_df['oov_rate'] <= 0.1, 'xglm_correct')]

b1 = axL.bar(x - w/2, lstm_dl, w, label='LSTM', color=C_LSTM, zorder=3)
b2 = axL.bar(x + w/2, xglm_dl, w, label='XGLM-1.7B', color=C_XGLM, zorder=3)
axL.set_xticks(x); axL.set_xticklabels(['DL > 50', 'DL <= 50'])
axL.set_ylabel('Accuracy (%)'); axL.set_ylim(0, 100); axL.set_title('By dependency length')
style_axes(axL); bar_labels(axL, b1); bar_labels(axL, b2)

b3 = axR.bar(x - w/2, lstm_oov, w, label='LSTM', color=C_LSTM, zorder=3)
b4 = axR.bar(x + w/2, xglm_oov, w, label='XGLM-1.7B', color=C_XGLM, zorder=3)
axR.set_xticks(x); axR.set_xticklabels(['OOV > 10%', 'OOV <= 10%'])
axR.set_ylim(0, 100); axR.set_title('By out-of-vocabulary rate')
style_axes(axR); bar_labels(axR, b3); bar_labels(axR, b4)
axR.legend(fontsize=9, frameon=False, loc='lower right')

fig.suptitle('Stratified Accuracy: Why the LLM Advantage Grows', fontsize=13, fontweight='bold')
plt.tight_layout()
p2 = os.path.join(OUT_DIR, 'chart2_dl_oov_stratified.png')
fig.savefig(p2, dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved ->', p2)

# ═══════════════════════════════════════════════════════════════════════
# CHART 3 — Pipeline synthesis (MTP-1 -> MTP-2)
# ═══════════════════════════════════════════════════════════════════════
avg_all5 = (results['All-5 (LSTM)']['DO-fronted'] + results['All-5 (LSTM)']['IO-fronted']) / 2
avg_xglm = (results['XGLM']['DO-fronted'] + results['XGLM']['IO-fronted']) / 2
avg_every = (results['Everything']['DO-fronted'] + results['Everything']['IO-fronted']) / 2
# Global bidirectional + MTP-2 tie-back: pre-verified fixed points (see
# scripts/joachims_transformation_general.py and the live_demo notebook's own
# fresh run) -- not recomputed here to keep this script fast and dependency-light.
bidirectional_acc = 95.6
mtp2_tieback_acc = 98.0

labels = ['MTP-1\nAll-5 (LSTM)', 'MTP-1\nLSTM->XGLM', 'MTP-1\nEverything\ncombined',
          'MTP-1\nglobal\nbidirectional', 'MTP-2\nGemma dialog\ntie-back']
values = [avg_all5, avg_xglm, avg_every, bidirectional_acc, mtp2_tieback_acc]

fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(range(len(labels)), values, color=[C_LSTM, C_XGLM, C_NEW, C_NEW, C_GEMMA], width=0.6, zorder=3)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Accuracy (%, avg DO/IO where applicable)')
ax.set_ylim(0, 100)
ax.set_title('MTP-1 -> MTP-2: Every Stage', fontsize=13, fontweight='bold')
style_axes(ax); bar_labels(ax, bars)
plt.tight_layout()
p3 = os.path.join(OUT_DIR, 'chart3_pipeline_synthesis.png')
fig.savefig(p3, dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved ->', p3)

print(f'\nAll 3 charts saved to: {OUT_DIR}')
print('Open that folder and copy/paste each PNG directly onto a PowerPoint slide.')
