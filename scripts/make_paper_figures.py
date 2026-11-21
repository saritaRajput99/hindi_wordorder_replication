"""
Regenerate the 3 figures used in the ACL paper draft, from the Tier-1 result
CSVs (results/llm_comparison.csv, results/llm_coefficients.csv,
results/error_analysis_llm_generalized.csv) — all produced by the earlier
Tier-1 scripts, no retraining/rescoring involved.

Run after the 4 Tier-1 scripts (needs their output CSVs to already exist):
    python scripts/make_paper_figures.py

Writes (PDF for LaTeX + PNG for quick viewing), matching the filenames
referenced in acl_paper/paper.tex's \\includegraphics commands:
    results/figures/accuracy_by_construction.{pdf,png}
    results/figures/coefficient_zstat.{pdf,png}
    results/figures/why_llm_wins_stratified.{pdf,png}
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Validated 3-series categorical palette (fixed order, all-pairs CVD-safe) ──
C_LSTM  = '#2a78d6'   # slot 1 — blue   — Adaptive LSTM
C_XGLM  = '#eb6834'   # slot 2 — orange — XGLM-1.7B
C_GEMMA = '#1baf7a'   # slot 3 — aqua   — Gemma-3
INK        = '#0b0b0b'
INK_MUTED  = '#898781'
GRIDLINE   = '#e1e0d9'
BASELINE   = '#c3c2b7'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10.5,
    'axes.edgecolor': BASELINE,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': INK_MUTED,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)

def bar_labels(ax, bars, fmt='{:.1f}'):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords='offset points', xytext=(0, 3),
                    ha='center', va='bottom', fontsize=8.5, color=INK)

def save(fig, name):
    for ext in ('pdf', 'png'):
        path = os.path.join(FIG_DIR, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=200 if ext == 'png' else None)
        print(f'  Saved: results/figures/{name}.{ext}')

# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — accuracy_by_construction  (Table "llm": bottom 3 rows)
# ═══════════════════════════════════════════════════════════════════════
print('=' * 60)
print('Figure 1: accuracy_by_construction')
print('=' * 60)

cmp_df = pd.read_csv(os.path.join(BASE, 'results', 'llm_comparison.csv'))
cmp_df = cmp_df.set_index('Experiment')
to_pct = lambda s: float(str(s).replace('%', ''))

rows = {
    'Adaptive LSTM\n(paper\'s setup)': 'All 5 (original paper)',
    'LSTM $\\to$ XGLM-1.7B':           'LSTM → XGLM replacement',
    'LSTM $\\to$ Gemma-3':             'LSTM → Gemma-3 replacement',
}
models  = list(rows.keys())
do_vals = [to_pct(cmp_df.loc[r, 'DO-fronted']) for r in rows.values()]
io_vals = [to_pct(cmp_df.loc[r, 'IO-fronted']) for r in rows.values()]

fig, ax = plt.subplots(figsize=(6, 3.4))
x = np.arange(2)
width = 0.26
colors = [C_LSTM, C_XGLM, C_GEMMA]
for i, model in enumerate(models):
    vals = [do_vals[i], io_vals[i]]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=model, color=colors[i],
                  edgecolor='white', linewidth=0.5)
    bar_labels(ax, bars)

ax.set_xticks(x)
ax.set_xticklabels(['DO-fronted', 'IO-fronted'])
ax.set_ylabel('Classification accuracy (%)')
ax.set_ylim(0, 100)
style_axes(ax)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8.5)
fig.tight_layout()
save(fig, 'accuracy_by_construction')
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — coefficient_zstat  (Table "coef")
# ═══════════════════════════════════════════════════════════════════════
print('=' * 60)
print('Figure 2: coefficient_zstat')
print('=' * 60)

coef_df = pd.read_csv(os.path.join(BASE, 'results', 'llm_coefficients.csv'))
surprisal_rows = coef_df[coef_df['Predictor'].str.contains('surprisal', case=False) &
                          coef_df['Predictor'].str.contains('LSTM|XGLM|Gemma', case=False)]

model_label = {
    'All-5 original (LSTM)':        'Adaptive LSTM',
    'LSTM → XGLM replacement':      'XGLM-1.7B',
    'LSTM → Gemma-3 replacement':   'Gemma-3',
}
model_color = {'Adaptive LSTM': C_LSTM, 'XGLM-1.7B': C_XGLM, 'Gemma-3': C_GEMMA}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.5, 3.4))
constructions = ['DO-fronted', 'IO-fronted']
x = np.arange(len(constructions))
width = 0.26

for i, (model_key, label) in enumerate(model_label.items()):
    sub = surprisal_rows[surprisal_rows['Model'] == model_key].set_index('Construction')
    est  = [abs(sub.loc[c, 'Estimate']) for c in constructions]
    zval = [abs(sub.loc[c, 'Z_value']) for c in constructions]
    offs = (i - 1) * width
    b1 = axL.bar(x + offs, est,  width, label=label, color=model_color[label], edgecolor='white', linewidth=0.5)
    b2 = axR.bar(x + offs, zval, width, label=label, color=model_color[label], edgecolor='white', linewidth=0.5)
    bar_labels(axL, b1)
    bar_labels(axR, b2)

for ax, title, ylabel in [(axL, '(a) Coefficient magnitude', '|Estimate|'),
                           (axR, '(b) $|Z|$-statistic', '|Z|')]:
    ax.set_xticks(x)
    ax.set_xticklabels(constructions)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    style_axes(ax)

axR.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
fig.tight_layout()
save(fig, 'coefficient_zstat')
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — why_llm_wins_stratified  (Table "why")
# ═══════════════════════════════════════════════════════════════════════
print('=' * 60)
print('Figure 3: why_llm_wins_stratified')
print('=' * 60)

err_df = pd.read_csv(os.path.join(BASE, 'results', 'error_analysis_llm_generalized.csv'))

def acc(mask, col):
    sub = err_df[mask]
    return 100.0 * sub[col].mean(), len(sub)

dl_long,  n_dl_long  = acc(err_df['dl'] > 50, 'lstm_correct')
dl_short, n_dl_short = acc(err_df['dl'] <= 50, 'lstm_correct')
xg_long,  _          = acc(err_df['dl'] > 50, 'xglm_correct')
xg_short, _          = acc(err_df['dl'] <= 50, 'xglm_correct')

oov_hi,  n_oov_hi  = acc(err_df['oov_rate'] > 0.1, 'lstm_correct')
oov_lo,  n_oov_lo  = acc(err_df['oov_rate'] <= 0.1, 'lstm_correct')
xg_oov_hi, _       = acc(err_df['oov_rate'] > 0.1, 'xglm_correct')
xg_oov_lo, _       = acc(err_df['oov_rate'] <= 0.1, 'xglm_correct')

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.5, 3.4))
width = 0.32

labels_dl = [f'DL > 50\n(n={n_dl_long})', f'DL $\\leq$ 50\n(n={n_dl_short})']
x = np.arange(2)
b1 = axL.bar(x - width/2, [dl_long, dl_short], width, label='Adaptive LSTM', color=C_LSTM, edgecolor='white', linewidth=0.5)
b2 = axL.bar(x + width/2, [xg_long, xg_short], width, label='XGLM-1.7B',    color=C_XGLM, edgecolor='white', linewidth=0.5)
bar_labels(axL, b1); bar_labels(axL, b2)
axL.set_xticks(x); axL.set_xticklabels(labels_dl)
axL.set_title('(a) By dependency length', fontsize=10)
axL.set_ylabel('Accuracy (%)')

labels_oov = [f'OOV > 0.1\n(n={n_oov_hi})', f'OOV $\\leq$ 0.1\n(n={n_oov_lo})']
b3 = axR.bar(x - width/2, [oov_hi, oov_lo], width, label='Adaptive LSTM', color=C_LSTM, edgecolor='white', linewidth=0.5)
b4 = axR.bar(x + width/2, [xg_oov_hi, xg_oov_lo], width, label='XGLM-1.7B', color=C_XGLM, edgecolor='white', linewidth=0.5)
bar_labels(axR, b3); bar_labels(axR, b4)
axR.set_xticks(x); axR.set_xticklabels(labels_oov)
axR.set_title('(b) By OOV rate', fontsize=10)

for ax in (axL, axR):
    ax.set_ylim(0, 100)
    style_axes(ax)

axR.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.5)
fig.tight_layout()
save(fig, 'why_llm_wins_stratified')
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# Table — reference sentences & variants per construction (paper Table 1)
# ═══════════════════════════════════════════════════════════════════════
print('=' * 60)
print('Table: reference sentences & variants per construction type')
print('=' * 60)

feat_for_table = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'features', 'features.csv'))
refs     = feat_for_table[feat_for_table['is_original'] == 1].groupby('order_type')['reference_id'].nunique()
variants = feat_for_table[feat_for_table['is_original'] == 0].groupby('order_type').size()
table1 = pd.DataFrame({'reference_sentences': refs, 'variants': variants}).reindex(
    ['Canonical', 'DO-fronted', 'IO-fronted'])
table1_path = os.path.join(BASE, 'results', 'table_data_summary.csv')
table1.to_csv(table1_path)
print(table1)
print(f'  Saved: results/table_data_summary.csv')

print()
print('Done. All figures/tables written to results/figures/ and results/')
