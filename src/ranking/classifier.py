
"""
Logistic Regression Classifier 
 
Paper Section Results:
  Logistic regression on pairwise difference vectors.
  10-fold cross-validation.
  Reports accuracy, coefficients, significance.
 
  Paper Table 2: DO-fronted constructions
  Paper Table 3: IO-fronted constructions
 
Key findings from paper:
  Trigram surprisal: strongest predictor (negative coefficient)
  DL: significant for DO-fronted only
  IS score: positive coefficient (Given-New order preferred)
"""
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats
 
 
class PairwiseClassifier:
 
    def __init__(self, n_folds=10, random_state=42):
        self.n_folds      = n_folds
        self.random_state = random_state
        self.scaler       = StandardScaler()
        self.model        = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',
        )
 
    def fit_evaluate(self, X, y, feature_cols, order_type='All'):
        """Train and evaluate with 10-fold CV."""
 
        print(f"\n{'='*55}")
        print(f"  {order_type} — Logistic Regression (10-fold CV)")
        print(f"{'='*55}")
        print(f"  Samples  : {len(X):,}")
        print(f"  Features : {feature_cols}")
 
        cv        = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        fold_accs = []
        all_coefs = []

        for fold, (tr, te) in enumerate(cv.split(X, y)):
            # Fit scaler on training fold only — prevents test-set leakage
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            self.model.fit(X_tr, y[tr])
            y_pred = self.model.predict(X_te)
            fold_accs.append(accuracy_score(y[te], y_pred))
            all_coefs.append(self.model.coef_[0])
 
        mean_acc  = np.mean(fold_accs)
        std_acc   = np.std(fold_accs)
        mean_coef = np.mean(all_coefs, axis=0)
 
        # Significance via t-test
        p_values = []
        for i in range(len(feature_cols)):
            vals = [c[i] for c in all_coefs]
            _, p = stats.ttest_1samp(vals, 0)
            p_values.append(p)
 
        # Print results
        print(f"\n  Accuracy : {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"  ({mean_acc*100:.2f}%)")
        print(f"\n  {'Feature':<28} {'Coef':>8} {'p-val':>10} Sig")
        print(f"  {'-'*52}")
 
        for feat, coef, pval in zip(feature_cols, mean_coef, p_values):
            sig = ('***' if pval < 0.001 else
                   '**'  if pval < 0.01  else
                   '*'   if pval < 0.05  else '')
            print(f"  {feat:<28} {coef:>8.4f} {pval:>10.4f} {sig}")
 
        baseline = max(y.mean(), 1-y.mean())
        print(f"\n  Baseline : {baseline:.4f}")
        print(f"  Gain     : +{(mean_acc-baseline)*100:.2f}%")
 
        return {
            'order_type'  : order_type,
            'accuracy'    : mean_acc,
            'std_acc'     : std_acc,
            'coefficients': dict(zip(feature_cols, mean_coef)),
            'p_values'    : dict(zip(feature_cols, p_values)),
            'fold_accs'   : fold_accs,
            'baseline'    : baseline,
        }
 
    def run_all(self, df, feature_cols, results_dir):
        """Run classifier on all + per order type."""
        try:
            from ranking.ranking import build_pairwise_dataset
        except ImportError:
            from .ranking import build_pairwise_dataset
 
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = results_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)
 
        all_results = {}
 
        # All combined
        print("\n" + "="*55)
        print("ALL ORDER TYPES COMBINED")
        print("="*55)
        X, y, _ = build_pairwise_dataset(df, feature_cols)
        all_results['All'] = self.fit_evaluate(X, y, feature_cols, 'All')
 
        # Paper Tables 2 & 3: non-canonical corpus vs CANONICAL variants only
        # Table 2: DOSV corpus vs SDOV (canonical) variants  (N≈1663)
        # Table 3: IOSV corpus vs SIOV (canonical) variants  (N≈1353)
        for ot in ['DO-fronted', 'IO-fronted']:
            if ot not in df['order_type'].values:
                continue
            X_ot, y_ot, _ = build_pairwise_dataset(
                df, feature_cols,
                order_type=ot, var_order_type='Canonical')
            if len(X_ot) < 20:
                print(f"  {ot}: too few pairs ({len(X_ot)}) — skipping")
                continue
            all_results[ot] = self.fit_evaluate(X_ot, y_ot, feature_cols, ot)

        # Additional analyses (all variants, not just canonical)
        for ot in ['O-fronted', 'Canonical']:
            if ot not in df['order_type'].values:
                continue
            X_ot, y_ot, _ = build_pairwise_dataset(df, feature_cols, ot)
            if len(X_ot) < 20:
                continue
            all_results[ot] = self.fit_evaluate(
                X_ot, y_ot, feature_cols, ot)
 
        # Save results
        self._save_results(all_results, feature_cols, results_dir)
        self._plot_coefficients(all_results, feature_cols, fig_dir)
        self._plot_accuracy(all_results, fig_dir)
 
        return all_results
 
    def _save_results(self, results, feature_cols, results_dir):
        rows = []
        for ot, res in results.items():
            row = {
                'order_type': ot,
                'accuracy'  : f"{res['accuracy']:.4f}",
                'std'       : f"{res['std_acc']:.4f}",
                'baseline'  : f"{res['baseline']:.4f}",
            }
            for feat in feature_cols:
                coef = res['coefficients'].get(feat, 0)
                pval = res['p_values'].get(feat, 1)
                sig  = ('***' if pval<0.001 else
                        '**'  if pval<0.01  else
                        '*'   if pval<0.05  else '')
                row[f'{feat}_coef'] = f"{coef:.4f}{sig}"
            rows.append(row)
 
        csv_path = results_dir / 'classifier_results.csv'
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nResults saved: {csv_path}")
        print(pd.DataFrame(rows).to_string(index=False))
 
    def _plot_coefficients(self, results, feature_cols, fig_dir):
        order_types = [k for k in results if k != 'All']
        if not order_types: return
 
        x      = np.arange(len(feature_cols))
        width  = 0.8 / max(len(order_types),1)
        colors = ['steelblue','coral','green','purple']
 
        fig, ax = plt.subplots(figsize=(12,6))
        for i, ot in enumerate(order_types):
            coefs  = [results[ot]['coefficients'].get(f,0) for f in feature_cols]
            offset = (i - len(order_types)/2 + 0.5) * width
            bars   = ax.bar(x+offset, coefs, width,
                            label=ot, color=colors[i%len(colors)], alpha=0.8)
            for bar, feat in zip(bars, feature_cols):
                pval = results[ot]['p_values'].get(feat, 1)
                sig  = ('***' if pval<0.001 else '**' if pval<0.01
                        else '*' if pval<0.05 else '')
                if sig:
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height()+0.005, sig,
                            ha='center', va='bottom', fontsize=9)
 
        ax.set_xlabel('Feature')
        ax.set_ylabel('Coefficient')
        ax.set_title('Logistic Regression Coefficients by Order Type\n'
                     '(* p<0.05  ** p<0.01  *** p<0.001)')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_cols, rotation=20, ha='right')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.legend()
        plt.tight_layout()
        path = fig_dir / 'coefficients.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figure saved: {path}")
 
    def _plot_accuracy(self, results, fig_dir):
        ots  = list(results.keys())
        accs = [results[o]['accuracy'] for o in ots]
        base = [results[o]['baseline'] for o in ots]
        stds = [results[o]['std_acc']  for o in ots]
        x    = np.arange(len(ots))
 
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(x,     accs, 0.4, label='Model', color='steelblue',
               alpha=0.8, yerr=stds, capsize=4)
        ax.bar(x+0.4, base, 0.4, label='Baseline', color='lightgray', alpha=0.8)
        ax.set_xlabel('Order type')
        ax.set_ylabel('Accuracy')
        ax.set_title('Classifier Accuracy by Order Type')
        ax.set_xticks(x+0.2)
        ax.set_xticklabels(ots, rotation=15, ha='right')
        ax.set_ylim(0,1)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8)
        ax.legend()
        plt.tight_layout()
        path = fig_dir / 'accuracy_by_order.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figure saved: {path}")
 
 
def print_paper_comparison(results):
    print("\n" + "="*60)
    print("COMPARISON WITH PAPER RESULTS")
    print("="*60)
    print("""
Paper Table 2 — DO-fronted (DOSV corpus vs SDOV canonical, N=1663):
  Predictor               Estimate  SE      Z
  dl                      -0.58     0.091  -6.35 ***
  is_score                +0.26     0.055  +4.68 ***
  surprisal_pcfg          +0.52     0.135  +3.88 ***
  surprisal_trigram       -2.08     0.245  -8.48 ***
  surprisal_adaptive_lstm -1.65     0.240  -6.86 ***

Paper Table 3 — IO-fronted (IOSV corpus vs SIOV canonical, N=1353):
  Predictor               Estimate  SE      Z
  dl                      +0.15     0.142  +1.045  (n.s.)
  is_score                +0.13     0.074  +1.767  (n.s.)
  surprisal_pcfg          -0.89     0.238  -3.725 ***
  surprisal_trigram       -3.05     0.382  -7.976 ***
  surprisal_adaptive_lstm -2.83     0.365  -7.780 ***
""")
    print("Our Results:")
    for ot in ['DO-fronted','IO-fronted','All']:
        if ot not in results: continue
        res = results[ot]
        print(f"\n  {ot}:")
        print(f"    Accuracy: {res['accuracy']:.4f}")
        for feat, coef in res['coefficients'].items():
            pval = res['p_values'][feat]
            sig  = ('***' if pval<0.001 else '**' if pval<0.01
                    else '*' if pval<0.05 else '')
            print(f"    {feat:<32}: {coef:>8.4f} {sig}")
    print("="*60)
