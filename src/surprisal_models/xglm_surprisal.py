"""
XGLM-1.7B Surprisal Scorer
facebook/xglm-1.7B — autoregressive causal LM, 30 languages including Hindi.

Replaces Adaptive LSTM surprisal with a transformer-based LM.
Same interface as AdaptiveScorer: compute_all() → dict {variant_id: surprisal}

Surprisal = mean cross-entropy = -mean log P(token_i | token_1..token_{i-1})
           (in nats, same unit as LSTM scorer)

Install:
    pip install transformers accelerate
"""

import os
import sys
import json
import math
import time
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List

os.environ['PYTHONIOENCODING'] = 'utf-8'


class XGLMSurprisal:

    MODEL_NAME = "facebook/xglm-1.7B"

    def __init__(self,
                 model_dir: str,
                 model_name: str = None,
                 device: str = None,
                 batch_size: int = 1):
        """
        model_dir  : where to save scores JSON (e.g. models/xglm/)
        model_name : HuggingFace model ID (default: facebook/xglm-1.7B)
        device     : 'cuda' / 'cpu' / None (auto-detect)
        batch_size : sentences per forward pass (keep 1 for variable-length)
        """
        self.model_dir  = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or self.MODEL_NAME
        self.batch_size = batch_size
        self.device     = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._model     = None
        self._tok       = None
        print(f'XGLMSurprisal: device={self.device}  model={self.model_name}')

    @property
    def scores_path(self) -> Path:
        return self.model_dir / 'xglm_surprisals.json'

    def _load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f'Loading {self.model_name} ...')
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
        ).to(self.device)
        self._model.eval()
        n_params = sum(p.numel() for p in self._model.parameters())
        print(f'Loaded: {n_params/1e6:.0f}M params  dtype={next(self._model.parameters()).dtype}')

    def score_text(self, text: str) -> float:
        """
        Compute mean per-token surprisal (nats) for a text string.
        Uses model(**inputs, labels=inputs) which returns mean cross-entropy.
        """
        if self._model is None:
            self._load()

        inputs = self._tok(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        ).to(self.device)

        if inputs['input_ids'].shape[1] < 2:
            return 100.0

        with torch.no_grad():
            out = self._model(**inputs, labels=inputs['input_ids'])
        return out.loss.item()   # mean cross-entropy in nats

    def compute_all(self,
                    features_df: pd.DataFrame,
                    text_col: str = 'text',
                    id_col: str = 'variant_id',
                    checkpoint_every: int = 500) -> Dict[str, float]:
        """
        Score all variants. Resumes from checkpoint if scores_path exists.

        features_df : DataFrame with id_col and text_col columns
        Returns     : dict {variant_id: surprisal}
        """
        if self._model is None:
            self._load()

        scores: Dict[str, float] = {}

        # Resume from checkpoint
        if self.scores_path.exists():
            with open(self.scores_path, encoding='utf-8') as f:
                scores = json.load(f)
            print(f'Resumed: {len(scores):,} already scored')

        todo = [(row[id_col], row[text_col])
                for _, row in features_df.iterrows()
                if row[id_col] not in scores]
        print(f'To score: {len(todo):,}  (total: {len(features_df):,})')

        if not todo:
            print('All already scored.')
            return scores

        t0 = time.time()
        for i, (vid, text) in enumerate(todo):
            try:
                s = self.score_text(str(text))
                if not math.isfinite(s):
                    s = 100.0
            except Exception as e:
                print(f'  Warning [{vid}]: {e}')
                s = 100.0

            scores[vid] = s

            if (i + 1) % checkpoint_every == 0:
                self._save(scores)
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed
                eta     = (len(todo) - i - 1) / max(rate, 1e-6)
                pct     = (i + 1) / len(todo) * 100
                print(f'  {i+1:>6,}/{len(todo):,}  ({pct:.1f}%)  '
                      f'ETA: {eta/60:.1f} min')

        self._save(scores)
        print(f'\nDone: {len(scores):,} variants scored')
        print(f'Saved: {self.scores_path}')
        return scores

    def _save(self, scores: Dict[str, float]):
        with open(self.scores_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False)

    def add_to_df(self,
                  df: pd.DataFrame,
                  scores: Dict[str, float],
                  col: str = 'surprisal_xglm') -> pd.DataFrame:
        df = df.copy()
        df[col] = df['variant_id'].map(lambda v: scores.get(v, 100.0))
        real = (df[col] < 100.0).sum()
        print(f'XGLM: {real:,} real scores, {len(df)-real:,} defaults (100.0)')
        return df

    def print_stats(self, df: pd.DataFrame, col: str = 'surprisal_xglm'):
        valid = df[df[col] < 100.0]
        c = valid[valid['is_original'] == 1][col]
        v = valid[valid['is_original'] == 0][col]
        print('\n── XGLM SURPRISAL STATS ──────────────────')
        print(f'  Valid scores     : {len(valid):,}/{len(df):,}')
        print(f'  Corpus  mean     : {c.mean():.3f}')
        print(f'  Variant mean     : {v.mean():.3f}')
        print(f'  Corpus < Variant : {c.mean() < v.mean()}  (expected True)')
        print('──────────────────────────────────────────')

    def verify(self):
        """Sanity check: SOV should have lower surprisal than OSV in Hindi."""
        sov = 'राम ने किताब पढ़ी'
        osv = 'किताब राम ने पढ़ी'
        s_sov = self.score_text(sov)
        s_osv = self.score_text(osv)
        print(f'\nVerification (XGLM):')
        print(f'  SOV surprisal : {s_sov:.3f}')
        print(f'  OSV surprisal : {s_osv:.3f}')
        print(f'  SOV < OSV     : {s_sov < s_osv}  (expected True for canonical)')
        return s_sov < s_osv
