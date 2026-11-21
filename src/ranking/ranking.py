
"""
Pairwise Ranking Transform — Ranjan & van Schijndel (2024)
 
Paper Section Ranking Model:
  Uses Joachims (2002) pairwise ranking to handle class imbalance.
  2320 corpus sentences vs 56739 variants — severe imbalance.
 
  For each reference sentence r with variants v1 v2 vn:
    Positive pair: X = phi(r)  - phi(vi)  label=1
    Negative pair: X = phi(vi) - phi(r)   label=0
 
  This gives balanced dataset: 50% label=1, 50% label=0
 
Features (paper Table 2 and 3):
  dl
  surprisal_trigram
  surprisal_pcfg
  surprisal_adaptive_lstm
  is_score
"""
 
import pandas as pd
import numpy as np
from typing import List, Tuple
 
FEATURE_COLS = [
    'dl',
    'surprisal_trigram',
    'surprisal_pcfg',
    'surprisal_adaptive_lstm',
    'is_score',
]
 
FEATURE_COLS_NO_LSTM = [
    'dl',
    'surprisal_trigram',
    'surprisal_pcfg',
    'is_score',
]
 
 
def get_available_features(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"Available features : {available}")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"Missing features   : {missing}")
    return available
 
 
def build_pairwise_dataset(df, feature_cols=None, order_type=None,
                           var_order_type=None):
    """
    Build pairwise ranking dataset.

    For each reference sentence paired with each variant:
      X = phi(reference) - phi(variant)   y = 1  (reference preferred)
      X = phi(variant)   - phi(reference) y = 0  (variant preferred)

    Args:
        order_type     : if set, only keep reference sentences whose CORPUS
                         order matches this type (e.g. 'DO-fronted').
        var_order_type : if set, only pair corpus sentences against variants
                         of this order type (e.g. 'Canonical').
                         Paper Tables 2 & 3 use var_order_type='Canonical':
                           Table 2: DOSV corpus  vs SDOV (canonical) variants
                           Table 3: IOSV corpus  vs SIOV (canonical) variants

    Returns X, y, meta DataFrame
    """
    if feature_cols is None:
        feature_cols = get_available_features(df)

    if order_type is not None:
        ref_ids = df[
            (df['is_original'] == 1) & (df['order_type'] == order_type)
        ]['reference_id'].unique()
        df = df[df['reference_id'].isin(ref_ids)].copy()
        print(f"Corpus order={order_type}: {len(ref_ids):,} refs, "
              f"{len(df):,} total rows")

    X_list = []; y_list = []; meta_list = []

    for ref_id, group in df.groupby('reference_id'):
        corpus   = group[group['is_original'] == 1]
        variants = group[group['is_original'] == 0]

        # Paper Tables 2 & 3: pair only against canonical variants
        if var_order_type is not None:
            variants = variants[variants['order_type'] == var_order_type]
 
        if len(corpus) == 0 or len(variants) == 0:
            continue
 
        ref_feats = corpus.iloc[0][feature_cols].values.astype(float)
 
        for _, var_row in variants.iterrows():
            var_feats = var_row[feature_cols].values.astype(float)
 
            # Positive pair
            X_list.append(ref_feats - var_feats)
            y_list.append(1)
            meta_list.append({
                'reference_id': ref_id,
                'variant_id'  : var_row['variant_id'],
                'order_type'  : var_row['order_type'],
                'pair_type'   : 'ref_minus_var',
            })
 
            # Negative pair
            X_list.append(var_feats - ref_feats)
            y_list.append(0)
            meta_list.append({
                'reference_id': ref_id,
                'variant_id'  : var_row['variant_id'],
                'order_type'  : var_row['order_type'],
                'pair_type'   : 'var_minus_ref',
            })
 
    X    = np.array(X_list)
    y    = np.array(y_list)
    meta = pd.DataFrame(meta_list)
 
    print(f"Pairs     : {len(X):,}")
    print(f"Label=1   : {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"Label=0   : {(1-y).sum():,} ({(1-y).mean()*100:.1f}%)")
 
    return X, y, meta
 
 
def build_per_order_datasets(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = get_available_features(df)
    datasets = {}
    for ot in df['order_type'].unique():
        print(f"\n── {ot} ──────────────────")
        X, y, meta = build_pairwise_dataset(df, feature_cols, ot)
        datasets[ot] = (X, y, meta)
    return datasets
