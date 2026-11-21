"""
DEMO SCRIPT 5 — Regenerate the paper's LLM-comparison figures
=====================================================================


Run (from hindi_wordorder_replication/):
    python scripts/05_make_llm_figures.py

Each figure pops up in a window (close it to move to the next one) AND
is saved to acl_paper/figures/
"""

import os
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS   = os.path.join(BASE, 'results')
FIG_DIR   = os.path.join(os.path.dirname(BASE), 'acl_paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID   = '#0b0b0b', '#898781', '#e1e0d9'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def save_and_show(fig, name):
    pdf = os.path.join(FIG_DIR, f'{name}.pdf')
    png = os.path.join(FIG_DIR, f'{name}.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, bbox_inches='tight', dpi=200)
    print(f"  saved -> {pdf}")
    print(f"  saved -> {png}")
    plt.show()


def pct(s):
    return float(str(s).strip().rstrip('%'))


# ─────────────────────────────────────────────────────────────────────────
# FIGURE 1 — accuracy_by_construction
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  FIGURE 1 — Accuracy by construction (LSTM vs XGLM vs Gemma-3)")
print("=" * 62)

cmp_df = pd.read_csv(os.path.join(RESULTS, 'llm_comparison.csv'))
cmp_df = cmp_df.set_index('Experiment')

rows = {
    'Adaptive LSTM': 'All 5 (original paper)',
    'XGLM-1.7B':     'LSTM → XGLM replacement',
    'Gemma-3':       'LSTM → Gemma-3 replacement',
}
models, do_vals, io_vals = [], [], []
for label, row_name in rows.items():
    models.append(label)
    do_vals.append(pct(cmp_df.loc[row_name, 'DO-fronted']))
    io_vals.append(pct(cmp_df.loc[row_name, 'IO-fronted']))
    print(f"  {label:<14} DO={do_vals[-1]:>5.1f}%   IO={io_vals[-1]:>5.1f}%")

colors = [BLUE, ORANGE, AQUA]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
x = range(2)
width = 0.25
for i, (label, color) in enumerate(zip(models, colors)):
    offs = [x[0] + (i - 1) * width, x[1] + (i - 1) * width]
    vals = [do_vals[i], io_vals[i]]
    bars = ax.bar(offs, vals, width=width * 0.9, color=color, label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.1f}',
                 ha='center', va='bottom', fontsize=8, color=INK)
ax.set_xticks(list(x))
ax.set_xticklabels(['DO-fronted', 'IO-fronted'])
ax.set_ylabel('Classification accuracy (%)')
ax.set_ylim(0, 105)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc='lower right', fontsize=9)
fig.tight_layout()
save_and_show(fig, 'accuracy_by_construction')

# ─────────────────────────────────────────────────────────────────────────
# FIGURE 2 — coefficient_zstat
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  FIGURE 2 — Surprisal coefficient & |Z| by construction/model")
print("=" * 62)

coef_df = pd.read_csv(os.path.join(RESULTS, 'llm_coefficients.csv'))
pred_map = {
    'Adaptive LSTM': 'Adaptive LSTM surprisal',
    'XGLM-1.7B':     'XGLM surprisal',
    'Gemma-3':       'Gemma-3 surprisal',
}
constructions = ['DO-fronted', 'IO-fronted']

coefs = {m: [] for m in pred_map}
zstats = {m: [] for m in pred_map}
for cons in constructions:
    for label, pred_name in pred_map.items():
        row = coef_df[(coef_df['Construction'] == cons) & (coef_df['Predictor'] == pred_name)]
        est = row['Estimate'].iloc[0]
        z = abs(row['Z_value'].iloc[0])
        coefs[label].append(est)
        zstats[label].append(z)
        print(f"  {cons:<12} {label:<14} est={est:>7.2f}  |Z|={z:>6.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
x = range(2)
width = 0.25
for i, (label, color) in enumerate(zip(pred_map, colors)):
    offs = [x[0] + (i - 1) * width, x[1] + (i - 1) * width]
    ax1.bar(offs, coefs[label], width=width * 0.9, color=color, label=label)
    ax2.bar(offs, zstats[label], width=width * 0.9, color=color, label=label)

for ax, title, ylabel in [(ax1, '(a) Coefficient', 'Estimate'),
                           (ax2, '(b) |Z|-statistic', '|Z|')]:
    ax.set_xticks(list(x))
    ax.set_xticklabels(constructions)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, linewidth=0.8)
ax1.legend(frameon=False, loc='lower left', fontsize=8)
fig.tight_layout()
save_and_show(fig, 'coefficient_zstat')

# ─────────────────────────────────────────────────────────────────────────
# FIGURE 3 — why_llm_wins_stratified
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  FIGURE 3 — Why LLMs win: accuracy stratified by DL / OOV rate")
print("=" * 62)

err = pd.read_csv(os.path.join(RESULTS, 'error_analysis_llm_generalized.csv'))
for col in ('lstm_correct', 'xglm_correct'):
    if err[col].dtype == object:
        err[col] = err[col].astype(str).str.strip().str.lower() == 'true'

print(f"  pooled n = {len(err)} (DO-fronted + IO-fronted)")

def acc(mask, col):
    return err.loc[mask, col].mean() * 100

dl_hi, dl_lo = err['dl'] > 50, err['dl'] <= 50
oov_hi, oov_lo = err['oov_rate'] > 0.1, err['oov_rate'] <= 0.1

dl_labels = [f'DL>50 (n={dl_hi.sum()})', f'DL≤50 (n={dl_lo.sum()})']
oov_labels = [f'OOV>0.1 (n={oov_hi.sum()})', f'OOV≤0.1 (n={oov_lo.sum()})']

dl_lstm  = [acc(dl_hi, 'lstm_correct'), acc(dl_lo, 'lstm_correct')]
dl_xglm  = [acc(dl_hi, 'xglm_correct'), acc(dl_lo, 'xglm_correct')]
oov_lstm = [acc(oov_hi, 'lstm_correct'), acc(oov_lo, 'lstm_correct')]
oov_xglm = [acc(oov_hi, 'xglm_correct'), acc(oov_lo, 'xglm_correct')]

print(f"  DL>50   : LSTM={dl_lstm[0]:.1f}%  XGLM={dl_xglm[0]:.1f}%  gap={dl_xglm[0]-dl_lstm[0]:+.1f}")
print(f"  DL<=50  : LSTM={dl_lstm[1]:.1f}%  XGLM={dl_xglm[1]:.1f}%  gap={dl_xglm[1]-dl_lstm[1]:+.1f}")
print(f"  OOV>0.1 : LSTM={oov_lstm[0]:.1f}%  XGLM={oov_xglm[0]:.1f}%  gap={oov_xglm[0]-oov_lstm[0]:+.1f}")
print(f"  OOV<=0.1: LSTM={oov_lstm[1]:.1f}%  XGLM={oov_xglm[1]:.1f}%  gap={oov_xglm[1]-oov_lstm[1]:+.1f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4))
width = 0.3
x = range(2)

for ax, labels, lstm_vals, xglm_vals, title in [
    (ax1, dl_labels, dl_lstm, dl_xglm, '(a) By dependency length'),
    (ax2, oov_labels, oov_lstm, oov_xglm, '(b) By OOV rate'),
]:
    b1 = ax.bar([i - width / 2 for i in x], lstm_vals, width=width * 0.95, color=BLUE, label='Adaptive LSTM')
    b2 = ax.bar([i + width / 2 for i in x], xglm_vals, width=width * 0.95, color=ORANGE, label='XGLM-1.7B')
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.1f}',
                     ha='center', va='bottom', fontsize=8, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Per-sentence accuracy (%)')
    ax.set_ylim(0, 105)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
ax1.legend(frameon=False, loc='lower right', fontsize=8)
fig.tight_layout()
save_and_show(fig, 'why_llm_wins_stratified')

print("\nDone. All three figures regenerated into acl_paper/figures/.")
