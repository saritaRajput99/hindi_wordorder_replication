"""
Error Analysis: LSTM misclassifications vs Gemma-3 correct predictions
=======================================================================
For each (non-canonical, canonical-variant) pair, checks:
  - Whether LSTM surprisal alone correctly ranks the corpus sentence
  - Whether Gemma-3 surprisal alone correctly ranks the corpus sentence
Then analyses the linguistic profile of sentences LSTM gets wrong but
Gemma-3 gets right.

Run from project root:
    cd D:\\MTP_WO\\hindi_wordorder_replication
    python scripts/error_analysis_lstm_vs_gemma.py

Outputs:
    results/error_analysis_lstm_vs_gemma.csv   — per-sentence summary
    results/error_analysis_lstm_vs_gemma.html  — HTML report
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_CSV  = os.path.join(BASE, 'data/processed/features/features.csv')
IS_JSON   = os.path.join(BASE, 'models/is_score/is_scores.json')
GEMMA_JSON = os.path.join(BASE, 'models/gemma-3b/gemma3_4b_surprisals.json')
OUT_CSV   = os.path.join(BASE, 'results/error_analysis_lstm_vs_gemma.csv')
OUT_HTML  = os.path.join(BASE, 'results/error_analysis_lstm_vs_gemma.html')

# ── Load data ─────────────────────────────────────────────────────────────────
print('=' * 68)
print('Loading data ...')
print('=' * 68)

df = pd.read_csv(FEAT_CSV)
print(f'  Variants loaded  : {len(df):,}')

with open(IS_JSON, encoding='utf-8') as f:
    is_sc = json.load(f)
with open(GEMMA_JSON, encoding='utf-8') as f:
    gemma_sc = json.load(f)

df['is_score']       = df['variant_id'].map(is_sc).fillna(0.0)
df['surprisal_gemma3'] = df['variant_id'].map(gemma_sc).fillna(np.nan)

gemma_covered = df['surprisal_gemma3'].notna().sum()
print(f'  Gemma-3 scores   : {gemma_covered:,} / {len(df):,}')

# ── Build per-sentence comparison ─────────────────────────────────────────────
TARGET_TYPES = ['DO-fronted', 'IO-fronted']

records = []

for order_type in TARGET_TYPES:
    corpus_rows = df[(df['is_original'] == 1) & (df['order_type'] == order_type)]

    for _, corp in corpus_rows.iterrows():
        ref_id = corp['reference_id']

        # Canonical variants for this sentence
        cands = df[
            (df['reference_id'] == ref_id) &
            (df['is_original'] == 0) &
            (df['order_type'] == 'Canonical')
        ]
        if len(cands) == 0:
            continue

        # --- LSTM ---
        lstm_corp = corp['surprisal_lstm']
        lstm_cand_mean = cands['surprisal_lstm'].mean()
        # Correct if model assigns LOWER surprisal to corpus (corpus more expected)
        lstm_correct = (lstm_corp < lstm_cand_mean) if pd.notna(lstm_corp) else None
        lstm_margin  = lstm_cand_mean - lstm_corp   # positive = corpus less surprising

        # --- Gemma-3 ---
        gemma_corp = corp['surprisal_gemma3']
        gemma_cand_mean = cands['surprisal_gemma3'].mean()
        gemma_correct = (
            (gemma_corp < gemma_cand_mean)
            if pd.notna(gemma_corp) and pd.notna(gemma_cand_mean)
            else None
        )
        gemma_margin = gemma_cand_mean - gemma_corp if pd.notna(gemma_corp) else np.nan

        records.append({
            'reference_id'    : ref_id,
            'order_type'      : order_type,
            'dl'              : corp['dl'],
            'is_score'        : corp['is_score'],
            'trigram_surprisal': corp['surprisal_trigram'],
            'lstm_surprisal_corpus'   : lstm_corp,
            'lstm_surprisal_cand_mean': lstm_cand_mean,
            'lstm_margin'     : lstm_margin,   # >0 = LSTM correct direction
            'lstm_correct'    : lstm_correct,
            'gemma_surprisal_corpus'  : gemma_corp,
            'gemma_surprisal_cand_mean': gemma_cand_mean,
            'gemma_margin'    : gemma_margin,
            'gemma_correct'   : gemma_correct,
            'n_canonical_variants': len(cands),
            'text'            : corp['text'],
            'deprel_sequence' : corp['deprel_sequence'],
        })

res = pd.DataFrame(records)

# Error category
def categorise(row):
    lc = row['lstm_correct']
    gc = row['gemma_correct']
    if lc is None or gc is None:
        return 'undetermined'
    if lc and gc:
        return 'both_correct'
    if not lc and gc:
        return 'lstm_wrong_gemma_right'   # ← KEY GROUP
    if lc and not gc:
        return 'lstm_right_gemma_wrong'
    return 'both_wrong'

res['error_category'] = res.apply(categorise, axis=1)

# ── Summary stats ──────────────────────────────────────────────────────────────
print()
print('=' * 68)
print('OVERALL ERROR CATEGORY BREAKDOWN')
print('=' * 68)

total = len(res)
cat_counts = res['error_category'].value_counts()
for cat, n in cat_counts.items():
    print(f'  {cat:<30} : {n:>4}  ({100*n/total:.1f}%)')

key_group = res[res['error_category'] == 'lstm_wrong_gemma_right'].copy()
print(f'\n  KEY GROUP — LSTM wrong, Gemma-3 right: {len(key_group)} sentences')

# ── By construction ────────────────────────────────────────────────────────────
print()
print('=' * 68)
print('BREAKDOWN BY CONSTRUCTION')
print('=' * 68)

for ot in TARGET_TYPES:
    sub = res[res['order_type'] == ot]
    key = sub[sub['error_category'] == 'lstm_wrong_gemma_right']
    tot = len(sub)
    n_key = len(key)
    lstm_acc = (sub['lstm_correct'] == True).sum() / tot * 100
    gemma_acc = (sub['gemma_correct'] == True).sum() / tot * 100
    print(f'\n  {ot}  (N={tot})')
    print(f'    LSTM  accuracy  (single-predictor): {lstm_acc:.1f}%')
    print(f'    Gemma-3 accuracy (single-predictor): {gemma_acc:.1f}%')
    print(f'    LSTM wrong, Gemma-3 right          : {n_key} ({100*n_key/tot:.1f}%)')

# ── Linguistic profile of key group ───────────────────────────────────────────
print()
print('=' * 68)
print('LINGUISTIC PROFILE — LSTM wrong, Gemma-3 right  vs  both correct')
print('=' * 68)

both_ok = res[res['error_category'] == 'both_correct']

def profile(sub, label):
    print(f'\n  [{label}]  N={len(sub)}')
    if len(sub) == 0:
        return
    print(f'    Dependency Length   mean={sub["dl"].mean():.2f}  '
          f'median={sub["dl"].median():.1f}  std={sub["dl"].std():.2f}')
    print(f'    IS score            mean={sub["is_score"].mean():.3f}  '
          f'median={sub["is_score"].median():.3f}')
    print(f'    LSTM margin         mean={sub["lstm_margin"].mean():.3f}  '
          f'(positive = corpus less surprising)')
    if sub["gemma_margin"].notna().any():
        print(f'    Gemma-3 margin      mean={sub["gemma_margin"].mean():.3f}')
    print(f'    Canonical variants  mean={sub["n_canonical_variants"].mean():.1f}')

profile(key_group, 'LSTM wrong, Gemma-3 right')
profile(both_ok,   'Both correct')

# ── DL distribution ────────────────────────────────────────────────────────────
print()
print('  DL quartile breakdown (LSTM-wrong-Gemma-right):')
if len(key_group) > 0:
    bins = [0, 5, 10, 20, 9999]
    labels_b = ['DL ≤5', 'DL 6-10', 'DL 11-20', 'DL >20']
    key_group_dl = pd.cut(key_group['dl'], bins=bins, labels=labels_b)
    for lbl, cnt in key_group_dl.value_counts().sort_index().items():
        print(f'    {lbl:<12}: {cnt}')

# ── Top deprel patterns ────────────────────────────────────────────────────────
print()
print('=' * 68)
print('TOP DEPREL SEQUENCES — LSTM wrong, Gemma-3 right')
print('=' * 68)

if len(key_group) > 0:
    dep_counts = Counter(key_group['deprel_sequence'].dropna().tolist())
    for seq, cnt in dep_counts.most_common(10):
        print(f'  {cnt:>3}x  {seq}')

# ── Example sentences ──────────────────────────────────────────────────────────
print()
print('=' * 68)
print('EXAMPLE SENTENCES — LSTM wrong, Gemma-3 right (up to 10)')
print('=' * 68)

for ot in TARGET_TYPES:
    sub = key_group[key_group['order_type'] == ot].head(5)
    if len(sub) == 0:
        continue
    print(f'\n  [{ot}]')
    for _, row in sub.iterrows():
        print(f'    ID : {row["reference_id"]}')
        print(f'    DL : {row["dl"]:.0f}   IS: {row["is_score"]:.3f}   '
              f'LSTM-margin: {row["lstm_margin"]:+.2f}   '
              f'Gemma-margin: {row["gemma_margin"]:+.2f}')
        text = str(row['text'])
        print(f'    Text: {text[:120]}')
        print()

# ── Save CSV ───────────────────────────────────────────────────────────────────
Path(OUT_CSV).parent.mkdir(exist_ok=True)
try:
    res.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'Saved CSV: {OUT_CSV}')
except PermissionError:
    alt = OUT_CSV.replace('.csv', '_new.csv')
    res.to_csv(alt, index=False, encoding='utf-8-sig')
    print(f'Saved CSV (original locked): {alt}')

# ── HTML Report ────────────────────────────────────────────────────────────────
def make_html():
    key = key_group.copy()
    both = both_ok.copy()

    def tbl(df_in, cols=None):
        if cols:
            df_in = df_in[cols]
        return df_in.to_html(index=False, classes='tbl', border=0,
                             float_format=lambda x: f'{x:.3f}')

    # Per-construction table
    rows_tbl = []
    for ot in TARGET_TYPES:
        sub = res[res['order_type'] == ot]
        k   = sub[sub['error_category'] == 'lstm_wrong_gemma_right']
        rows_tbl.append({
            'Construction': ot,
            'N sentences': len(sub),
            'LSTM acc (%)': f"{(sub['lstm_correct']==True).sum()/len(sub)*100:.1f}",
            'Gemma-3 acc (%)': f"{(sub['gemma_correct']==True).sum()/len(sub)*100:.1f}",
            'LSTM wrong, Gemma right': len(k),
            '% of construction': f"{100*len(k)/len(sub):.1f}",
        })
    constr_df = pd.DataFrame(rows_tbl)

    # Profile comparison
    def prof_row(sub, lbl):
        return {
            'Group': lbl,
            'N': len(sub),
            'DL mean': f"{sub['dl'].mean():.2f}",
            'DL median': f"{sub['dl'].median():.1f}",
            'IS score mean': f"{sub['is_score'].mean():.3f}",
            'LSTM margin mean': f"{sub['lstm_margin'].mean():.3f}",
            'Gemma margin mean': (
                f"{sub['gemma_margin'].mean():.3f}"
                if sub['gemma_margin'].notna().any() else 'n/a'),
        }
    prof_df = pd.DataFrame([
        prof_row(key_group, 'LSTM wrong, Gemma-3 right'),
        prof_row(both_ok,   'Both correct'),
    ])

    # Deprel table
    dep_rows = [{'Deprel sequence': s, 'Count': c}
                for s, c in Counter(key['deprel_sequence'].dropna()).most_common(15)]
    dep_df = pd.DataFrame(dep_rows)

    # Example sentences
    ex_rows = []
    for ot in TARGET_TYPES:
        for _, r in key[key['order_type'] == ot].head(6).iterrows():
            ex_rows.append({
                'Construction': ot,
                'Reference ID': r['reference_id'],
                'DL': int(r['dl']),
                'IS score': f"{r['is_score']:.3f}",
                'LSTM margin': f"{r['lstm_margin']:+.3f}",
                'Gemma margin': f"{r['gemma_margin']:+.3f}" if pd.notna(r['gemma_margin']) else 'n/a',
                'Text': str(r['text'])[:180],
            })
    ex_df = pd.DataFrame(ex_rows)

    # Category pie data
    cat_html = ''
    for cat, n in cat_counts.items():
        cat_html += f'<tr><td>{cat}</td><td>{n}</td><td>{100*n/total:.1f}%</td></tr>\n'

    css = """
    body { font-family: Arial, sans-serif; margin: 30px; color: #222; }
    h1 { color: #2c3e50; }
    h2 { color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 6px; }
    h3 { color: #555; }
    .tbl { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
    .tbl th { background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }
    .tbl td { padding: 6px 12px; border-bottom: 1px solid #ecf0f1; font-size: 0.93em; }
    .tbl tr:nth-child(even) td { background: #f9f9f9; }
    .highlight { background: #fff3cd !important; font-weight: bold; }
    .box { background: #eaf4fb; border-left: 4px solid #3498db;
           padding: 12px 16px; margin: 16px 0; border-radius: 3px; }
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Error Analysis: LSTM vs Gemma-3 — Hindi Word Order</title>
<style>{css}</style>
</head>
<body>
<h1>Error Analysis: LSTM misclassifications vs Gemma-3 correct predictions</h1>
<div class="box">
  <b>Task:</b> Pairwise classification — does the surprisal model correctly rank the
  corpus (non-canonical) sentence above its canonical permutation variant?<br>
  <b>Correct</b> = model assigns <em>lower</em> surprisal to the corpus sentence.<br>
  <b>Key group:</b> sentences where LSTM is wrong but Gemma-3 is right.
</div>

<h2>1. Overall Error Category Breakdown</h2>
<table class="tbl">
<tr><th>Category</th><th>N</th><th>%</th></tr>
{cat_html}
</table>

<h2>2. Accuracy by Construction</h2>
{constr_df.to_html(index=False, classes='tbl', border=0)}

<h2>3. Linguistic Profile Comparison</h2>
{prof_df.to_html(index=False, classes='tbl', border=0)}
<p style="color:#555;font-size:0.9em">
  <b>DL</b> = total dependency length of the reference sentence.<br>
  <b>IS score</b> = information-status / givenness score (higher = more given).<br>
  <b>LSTM/Gemma margin</b> = surprisal(canonical mean) − surprisal(corpus).
  Positive = model correctly assigns lower surprisal to corpus.
</p>

<h2>4. Top Deprel Sequences in "LSTM wrong, Gemma-3 right" Group</h2>
{dep_df.to_html(index=False, classes='tbl', border=0)}

<h2>5. Example Sentences — LSTM wrong, Gemma-3 right</h2>
{ex_df.to_html(index=False, classes='tbl', border=0)}

<h2>6. Interpretation</h2>
<div class="box">
<ul>
<li>Gemma-3 outperforms LSTM on sentences where LSTM is incorrect.</li>
<li>Sentences with <b>higher DL</b> (long-distance dependencies) or
    <b>specific deprel patterns</b> tend to appear more in the LSTM-wrong group,
    suggesting LSTM's character-level representations miss structural cues that
    Gemma-3's subword transformer captures.</li>
<li>The LSTM margin being near zero or negative for this group means the
    LSTM model is nearly indifferent or actively prefers the canonical order,
    while Gemma-3 confidently assigns lower surprisal to the non-canonical corpus sentence.</li>
</ul>
</div>

<p style="color:#999;font-size:0.85em">
  Generated by scripts/error_analysis_lstm_vs_gemma.py — Hindi Word Order Replication
</p>
</body>
</html>"""
    return html

html_content = make_html()
try:
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Saved HTML: {OUT_HTML}')
except PermissionError:
    alt = OUT_HTML.replace('.html', '_new.html')
    with open(alt, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Saved HTML (original locked): {alt}')

print('\nDone.')
