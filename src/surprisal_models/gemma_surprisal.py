"""
Gemma-3-12B Surprisal Scorer
google/gemma-3-12b-pt — causal LM (base/pretrained), strong Hindi support.

IMPORTANT: Use the -pt (pretrained) variant, NOT -it (instruction-tuned).
Instruction-tuned models distort raw token probabilities.

Same interface as XGLMSurprisal / AdaptiveScorer.

Surprisal = mean cross-entropy = -mean log P(token_i | token_1..token_{i-1})
           (in nats)

Install:
    pip install transformers accelerate bitsandbytes
    huggingface-cli login   (Gemma requires accepting licence on HuggingFace)

Memory requirements:
    FP16  : ~24 GB VRAM  (A100)
    4-bit : ~8  GB VRAM  (RTX 3090 / A10)   ← use load_in_4bit=True
    CPU   : ~24 GB RAM   (slow but works)
"""

import os
import sys
import json
import math
import time
import torch
import pandas as pd
from pathlib import Path
from typing import Dict

os.environ['PYTHONIOENCODING'] = 'utf-8'


class GemmaSurprisal:

    MODEL_NAME = "google/gemma-3-12b-pt"

    def __init__(self,
                 model_dir: str,
                 model_name: str = None,
                 device: str = None,
                 load_in_4bit: bool = False):
        """
        model_dir    : where to save scores JSON (e.g. models/gemma3/)
        model_name   : HuggingFace model ID. Override to use 4B variant:
                       'google/gemma-3-4b-pt'
        device       : 'cuda' / 'cpu' / None (auto-detect)
        load_in_4bit : 4-bit quantization via bitsandbytes (saves ~75% VRAM)
                       Requires: pip install bitsandbytes
        """
        self.model_dir   = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name  = model_name or self.MODEL_NAME
        self.load_in_4bit = load_in_4bit
        self.device      = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._model      = None
        self._tok        = None

        tag = '(4-bit)' if load_in_4bit else ''
        print(f'GemmaSurprisal: device={self.device}  model={self.model_name} {tag}')

    @property
    def scores_path(self) -> Path:
        # Use model size in filename so 4B and 12B don't overwrite each other
        name = self.model_name.replace('/', '_').replace('-', '_')
        return self.model_dir / f'{name}_surprisals.json'

    def _load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        print(f'Loading {self.model_name} ...')

        self._tok = AutoTokenizer.from_pretrained(self.model_name)

        if self.load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_cfg,
                device_map='auto',
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            ).to(self.device)

        self._model.eval()
        n_params = sum(p.numel() for p in self._model.parameters())
        print(f'Loaded: {n_params/1e9:.1f}B params')

    def score_text(self, text: str) -> float:
        """
        Compute mean per-token surprisal (nats).
        """
        if self._model is None:
            self._load()

        inputs = self._tok(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        )
        # When using device_map='auto' (4-bit), don't move inputs manually
        if not self.load_in_4bit:
            inputs = inputs.to(self.device)

        if inputs['input_ids'].shape[1] < 2:
            return 100.0

        with torch.no_grad():
            out = self._model(**inputs, labels=inputs['input_ids'])
        return out.loss.item()

    def compute_all(self,
                    features_df: pd.DataFrame,
                    text_col: str = 'text',
                    id_col: str = 'variant_id',
                    checkpoint_every: int = 500) -> Dict[str, float]:
        """
        Score all variants. Resumes from checkpoint if scores_path exists.
        """
        if self._model is None:
            self._load()

        scores: Dict[str, float] = {}

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
                  col: str = 'surprisal_gemma3') -> pd.DataFrame:
        df = df.copy()
        df[col] = df['variant_id'].map(lambda v: scores.get(v, 100.0))
        real = (df[col] < 100.0).sum()
        print(f'Gemma-3: {real:,} real scores, {len(df)-real:,} defaults (100.0)')
        return df

    def print_stats(self, df: pd.DataFrame, col: str = 'surprisal_gemma3'):
        valid = df[df[col] < 100.0]
        c = valid[valid['is_original'] == 1][col]
        v = valid[valid['is_original'] == 0][col]
        print('\n── GEMMA-3 SURPRISAL STATS ───────────────')
        print(f'  Valid scores     : {len(valid):,}/{len(df):,}')
        print(f'  Corpus  mean     : {c.mean():.3f}')
        print(f'  Variant mean     : {v.mean():.3f}')
        print(f'  Corpus < Variant : {c.mean() < v.mean()}  (expected True)')
        print('──────────────────────────────────────────')

    def verify(self):
        sov = 'राम ने किताब पढ़ी'
        osv = 'किताब राम ने पढ़ी'
        s_sov = self.score_text(sov)
        s_osv = self.score_text(osv)
        print(f'\nVerification (Gemma-3):')
        print(f'  SOV surprisal : {s_sov:.3f}')
        print(f'  OSV surprisal : {s_osv:.3f}')
        print(f'  SOV < OSV     : {s_sov < s_osv}  (expected True for canonical)')
        return s_sov < s_osv
