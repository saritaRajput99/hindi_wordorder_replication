"""
Joachims Alternate-Pairing Transformation — project-adapted version.

Adapted from the professor's joachims_transformation_general.ipynb
(reusable, parameterized alternate-pairing transformation), pointed at this
project's actual schema (reference_id / variant_id / is_original / text /
feature columns in features.csv) instead of the generic SentID/VariantID/
Choice/Sentence column names used in the demo notebook.

What it does (same algorithm as the notebook):
  For each reference sentence's variants, emit exactly ONE row per variant
  (not two), with a sign flag that flips after EVERY variant processed,
  globally across the whole dataset (not reset per reference group):
    alt_sign ==  1 -> Delta_feat = reference[feat] - variant[feat], Class=1
    alt_sign == -1 -> Delta_feat = variant[feat] - reference[feat], Class=0
  This gives a ~50/50 balanced binary classification dataset without
  duplicating every pair in both directions (unlike the EXHAUSTIVE version
  already in src/ranking/ranking.py, which emits both directions for every
  variant and so is exactly 2x the variant count).

Project-specific restriction: only pairs a reference against its genuine
canonical<->non-canonical flip variants (is_flip_pair == 1) — the same
variants used in the paper's Table 2/3 pairwise regression — not every
generated variant. Variants with is_flip_pair == 0 (same canonical-ness as
their reference, e.g. train-s6_var_001) are excluded from this transform;
they still exist in features.csv for raw-total/Table-1-style reporting,
just not used as counterfactual pairs here.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats

# ── INPUT / OUTPUT PATHS ──────────────────────────────────────────────────
INPUT_PATH  = 'data/processed/features/features.csv'
OUTPUT_PATH = 'data/processed/ranking_dataset_general.csv'

# ── COLUMN NAME MAPPING (this project's schema) ───────────────────────────
ID_COLUMN         = 'reference_id'   # constant across a reference + its variants
VARIANT_ID_COLUMN = 'variant_id'     # per-row identifier, used to build output row IDs
ROW_TYPE_COLUMN   = 'is_original'    # 1 = reference row, 0 = variant row
FLIP_COLUMN       = 'is_flip_pair'   # 1 = genuine canonical<->non-canonical flip variant
SENTENCE_COLUMN   = 'text'           # raw sentence text

# ── FEATURE COLUMNS ────────────────────────────────────────────────────────
# Output delta columns are named Delta_<feat>. Only columns actually present
# in the input file are used — this lets the script run on whatever subset
# of features.csv is assembled so far (e.g. just dl + is_score today, with
# surprisal_pcfg / surprisal_modelblocks / surprisal_depparse / surprisal_lstm
# picked up automatically once those columns exist).
FEATURE_COLUMNS = [
    'dl',
    'is_score',
    'surprisal_trigram',
    'surprisal_pcfg',
    'surprisal_modelblocks',
    'surprisal_depparse',
    'surprisal_lstm',
]


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f'Loaded shape: {df.shape}')

    required = [ID_COLUMN, VARIANT_ID_COLUMN, ROW_TYPE_COLUMN, SENTENCE_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Required columns not found in input file: {missing}')

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_feats = [c for c in FEATURE_COLUMNS if c not in df.columns]
    print(f'Using feature columns: {feature_cols}')
    if missing_feats:
        print(f'  (not present yet, skipped): {missing_feats}')
    if not feature_cols:
        raise ValueError('No configured feature columns found in input file.')

    if FLIP_COLUMN in df.columns:
        before = len(df)
        keep = (df[ROW_TYPE_COLUMN] == 1) | (df[FLIP_COLUMN] == 1)
        df = df[keep].copy()
        print(f'Restricted to is_flip_pair == 1 variants: '
              f'kept {len(df):,} / {before:,} rows')
    else:
        print(f'WARNING: "{FLIP_COLUMN}" column not found — '
              f'using ALL variants unfiltered (no flip-pair restriction)')

    print('Preview:')
    print(df[[ID_COLUMN, VARIANT_ID_COLUMN, ROW_TYPE_COLUMN, SENTENCE_COLUMN]
             + feature_cols].head(10).to_string())

    # ── Alternate pairing transformation ───────────────────────────────
    # Same carry-over alternation as the professor's general notebook: the
    # sign flips +1/-1 with EVERY variant processed across the ENTIRE
    # dataset, not reset per reference group.
    ranking_rows = []
    alt_sign = 1

    groups = df.groupby(ID_COLUMN, sort=False)

    for group_id, group in groups:
        reference_rows = group[group[ROW_TYPE_COLUMN] == 1]
        if len(reference_rows) != 1:
            continue

        reference = reference_rows.iloc[0]
        variants = group[group[ROW_TYPE_COLUMN] == 0]

        for _, variant in variants.iterrows():
            row = {'SentID': f'{group_id}_{variant[VARIANT_ID_COLUMN]}'}

            if alt_sign == 1:
                category, label = 'ref-var', 1
                sentence_ref = reference[SENTENCE_COLUMN]
                sentence_var = 'NA'
                for feat in feature_cols:
                    row[f'Delta_{feat}'] = reference[feat] - variant[feat]
            else:
                category, label = 'var-ref', 0
                sentence_ref = 'NA'
                sentence_var = variant[SENTENCE_COLUMN]
                for feat in feature_cols:
                    row[f'Delta_{feat}'] = variant[feat] - reference[feat]

            row['Category']     = category
            row['Class']        = label
            row['Sentence_ref'] = sentence_ref
            row['Sentence_var'] = sentence_var

            ranking_rows.append(row)
            alt_sign *= -1   # flip for the NEXT variant, carries across groups

    ranking_df = pd.DataFrame(ranking_rows)
    delta_cols = [f'Delta_{feat}' for feat in feature_cols]
    ranking_df = ranking_df[['SentID', 'Category', 'Class']
                             + delta_cols + ['Sentence_ref', 'Sentence_var']]

    print(f'\nRanking dataset shape: {ranking_df.shape}')
    print('Class distribution:')
    print(ranking_df['Class'].value_counts())
    print('Category distribution:')
    print(ranking_df['Category'].value_counts())

    print('\nNonzero delta counts, per feature:')
    for feat in feature_cols:
        col = f'Delta_{feat}'
        nonzero = (ranking_df[col] != 0).sum()
        print(f'  {col:35s}: {nonzero} / {len(ranking_df)}  '
              f'({100*nonzero/len(ranking_df):.1f}%)')

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSaved to {OUTPUT_PATH}')
    print(f'Final shape: {ranking_df.shape}')

    # Pipeline alignment: this transform's output (is_flip_pair-based,
    # bidirectional, matching Ranjan's own notebook) previously fed nothing —
    # src/ranking/classifier.py evaluates a DIFFERENT, older, exhaustive
    # transform (src/ranking/ranking.py: every variant emitted in both
    # directions, order_type=='Canonical'-only for the paper-table cases).
    # Evaluate THIS dataset too, same style (logistic regression, 10-fold CV,
    # coefficients + significance), so the professor-correct transform
    # actually gets used, not just written to disk.
    evaluate(ranking_df, delta_cols)


def evaluate(ranking_df, delta_cols, n_folds=10, random_state=42):
    # Drop rows with any NaN delta — happens when the reference or variant is
    # missing a feature value (e.g. surprisal_modelblocks/depparse's small
    # fraction of unparseable sentences), since missing - anything = NaN.
    # LogisticRegression can't take NaN natively; impute/HistGradientBoosting
    # are the sklearn-recommended alternative if this starts dropping too much,
    # but a 94-98%-complete feature should only lose a small slice of rows.
    before = len(ranking_df)
    ranking_df = ranking_df.dropna(subset=delta_cols)
    dropped = before - len(ranking_df)
    if dropped:
        print(f'\nDropped {dropped:,}/{before:,} rows with a missing delta value '
              f'({100*dropped/before:.1f}%) before fitting the classifier.')

    X = ranking_df[delta_cols].values.astype(float)
    y = ranking_df['Class'].values.astype(int)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_accs, all_coefs = [], []

    for tr, te in cv.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        model = LogisticRegression(max_iter=1000, random_state=random_state, solver='lbfgs')
        model.fit(X_tr, y[tr])
        fold_accs.append(accuracy_score(y[te], model.predict(X_te)))
        all_coefs.append(model.coef_[0])

    mean_acc  = np.mean(fold_accs)
    std_acc   = np.std(fold_accs)
    mean_coef = np.mean(all_coefs, axis=0)
    p_values  = [stats.ttest_1samp([c[i] for c in all_coefs], 0)[1]
                 for i in range(len(delta_cols))]

    print('\n' + '=' * 60)
    print(f'  is_flip_pair dataset — Logistic Regression ({n_folds}-fold CV)')
    print('=' * 60)
    print(f'  Samples  : {len(X):,}')
    print(f'  Accuracy : {mean_acc:.4f} +/- {std_acc:.4f}  ({mean_acc*100:.2f}%)')
    print(f"\n  {'Feature':<32} {'Coef':>8} {'p-val':>10} Sig")
    print(f'  {"-"*56}')
    for feat, coef, pval in zip(delta_cols, mean_coef, p_values):
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f'  {feat:<32} {coef:>8.4f} {pval:>10.4f} {sig}')
    baseline = max(y.mean(), 1 - y.mean())
    print(f'\n  Baseline : {baseline:.4f}')
    print(f'  Gain     : +{(mean_acc-baseline)*100:.2f}%')
    print('=' * 60)

    return {
        'accuracy': mean_acc, 'std_acc': std_acc,
        'coefficients': dict(zip(delta_cols, mean_coef)),
        'p_values': dict(zip(delta_cols, p_values)),
    }


if __name__ == '__main__':
    main()
