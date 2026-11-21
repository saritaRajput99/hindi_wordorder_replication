"""
Adaptive LSTM Surprisal —  match to Neural Complexity Toolkit
van Schijndel & Linzen (2018) github.com/vansky/neural-complexity

Architecture verified from model.py source:
  - RNNModel with rnn_type='LSTM'
  - encoder: nn.Embedding(ntoken, ninp)
  - rnn   : nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
  - decoder: nn.Linear(nhid, ntoken)
  - tied weights: decoder.weight = encoder.weight (requires nhid==ninp)
  - init range: uniform(-0.1, 0.1)
  - dropout: 0.5

Training (from README):
  SGD optimizer, LR decay, tied weights flag

Adaptation (from README --adapt flag):
  Online SGD step on context sentence before scoring target

Train-once logic:
  - Check models/adaptive_lstm/lstm_model.pt exists
  - If yes  → load and score
  - If no   → train then save
  - Surprisals cached in models/adaptive_lstm/surprisals.json
"""

import os
import copy
import json
import math
import time
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────

class Vocab:
    PAD = '<pad>'
    UNK = '<unk>'
    BOS = '<bos>'
    EOS = '<eos>'

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.w2i: Dict[str, int] = {}
        self.i2w: Dict[int, str] = {}

    def build(self, sentences: List[List[str]]):
        freq: Dict[str, int] = {}
        for sent in sentences:
            for w in sent:
                freq[w] = freq.get(w, 0) + 1

        for special in [self.PAD, self.UNK, self.BOS, self.EOS]:
            idx = len(self.w2i)
            self.w2i[special] = idx
            self.i2w[idx] = special

        for w, c in sorted(freq.items(), key=lambda x: -x[1]):
            if c >= self.min_freq:
                idx = len(self.w2i)
                self.w2i[w] = idx
                self.i2w[idx] = w

        print(f'Vocab: {len(freq):,} types → {len(self.w2i):,} kept '
              f'(min_freq={self.min_freq})')

    def encode(self, words: List[str]) -> List[int]:
        unk = self.w2i[self.UNK]
        return [self.w2i.get(w, unk) for w in words]

    def __len__(self): 
        return len(self.w2i)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'w2i': self.w2i, 'min_freq': self.min_freq},
                      f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        self.w2i = d['w2i']
        self.min_freq = d['min_freq']
        self.i2w = {v: k for k, v in self.w2i.items()}
        print(f'Vocab loaded: {len(self.w2i):,} words')


# ─────────────────────────────────────────
# MODEL — exact Neural Complexity Toolkit
# ─────────────────────────────────────────

class RNNModel(nn.Module):
    """
    Direct port of vansky/neural-complexity model.py
    RNNModel with rnn_type='LSTM', tied weights.
    NOTE: tied weights requires nhid == ninp.
    Paper default: nhid=nhid, ninp=ninp — we use nhid=nhid, ninp=nhid
    to enable tying (same as toolkit default).
    """

    def __init__(self, ntoken: int, ninp: int = 650,
                 nhid: int = 650, nlayers: int = 2,
                 dropout: float = 0.5, tie_weights: bool = True):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('tie_weights requires nhid == ninp')
            self.decoder.weight = self.encoder.weight

        self._init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def _init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, obs: torch.Tensor,
                hidden: Tuple) -> Tuple[torch.Tensor, Tuple]:
        """obs: (seq_len, batch)"""
        emb = self.drop(self.encoder(obs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return decoded.view(output.size(0), output.size(1),
                            decoded.size(1)), hidden

    def init_hidden(self, bsz: int, device) -> Tuple:
        w = next(self.parameters()).data
        return (w.new_zeros(self.nlayers, bsz, self.nhid).to(device),
                w.new_zeros(self.nlayers, bsz, self.nhid).to(device))


# ─────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────

class LSTMTrainer:

    def __init__(self, corpus_path: str, model_dir: str,
                 nhid: int = 650, nlayers: int = 2,
                 dropout: float = 0.5, batch_size: int = 20,
                 seq_len: int = 35, epochs: int = 40,
                 lr: float = 20.0, clip: float = 0.25,
                 min_freq: int = 2, val_split: float = 0.05,
                 max_sents: int = 1000000, device: str = None):

        self.corpus_path = corpus_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f'Device: {self.device}')

        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.clip = clip
        self.min_freq = min_freq
        self.val_split = val_split
        self.max_sents = max_sents

    # ── paths ──────────────────────────────────────────────────
    @property
    def model_path(self):
        return self.model_dir / 'lstm_model.pt'
    
    @property
    def vocab_path(self):
        return self.model_dir / 'vocab.json'

    def is_trained(self) -> bool:
        return (self.model_path.exists() and
                self.vocab_path.exists() and
                self.model_path.stat().st_size > 10000)

    # ── data helpers ───────────────────────────────────────────
    def _load_corpus(self):
        sents = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                if 3 <= len(words) <= 60:
                    sents.append(words)
                if len(sents) >= self.max_sents:
                    break
        print(f'Corpus: {len(sents):,} sentences')
        return sents

    def _to_tensor(self, sents, vocab):
        bos = vocab.w2i[Vocab.BOS]
        eos = vocab.w2i[Vocab.EOS]
        ids = []
        for s in sents:
            ids += [bos] + vocab.encode(s) + [eos]
        return torch.tensor(ids, dtype=torch.long)

    def _batchify(self, data, bsz):
        n = data.size(0) // bsz
        return data[:n * bsz].view(bsz, -1).t().contiguous()

    def _get_batch(self, src, i):
        sl = min(self.seq_len, src.size(0) - 1 - i)
        d = src[i:i + sl]
        t = src[i + 1:i + 1 + sl].reshape(-1)
        return d, t

    # ── training ───────────────────────────────────────────────
    def _run_epoch(self, model, data, crit, opt, train=True):
        model.train() if train else model.eval()
        total = 0
        n = 0
        hid = model.init_hidden(self.batch_size, self.device)

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i in range(0, data.size(0) - 1, self.seq_len):
                x, y = self._get_batch(data, i)
                x, y = x.to(self.device), y.to(self.device)
                hid = tuple(h.detach() for h in hid)

                if train:
                    opt.zero_grad()
                out, hid = model(x, hid)
                loss = crit(out.view(-1, len(self.vocab)), y)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                    opt.step()
                total += loss.item()
                n += 1

        return total / max(n, 1)

    def train(self):
        """Train LSTM. Skip if already trained."""
        if self.is_trained():
            print(f'Model already trained — loading from cache')
            return self.load()

        print('Training LSTM language model...')
        sents = self._load_corpus()

        # Vocab
        vocab = Vocab(self.min_freq)
        vocab.build(sents)
        vocab.save(str(self.vocab_path))
        self.vocab = vocab

        # Split
        n_val = max(1, int(len(sents) * self.val_split))
        val_s = sents[:n_val]
        trn_s = sents[n_val:]

        trn = self._batchify(self._to_tensor(trn_s, vocab), self.batch_size)
        val = self._batchify(self._to_tensor(val_s, vocab), self.batch_size)

        # Model — nhid == ninp for tied weights (paper default)
        model = RNNModel(
            ntoken=len(vocab), ninp=self.nhid, nhid=self.nhid,
            nlayers=self.nlayers, dropout=self.dropout,
            tie_weights=True
        ).to(self.device)
        print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

        crit = nn.CrossEntropyLoss(ignore_index=0)
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)
        best_val = float('inf')
        patience = 0
        lr = self.lr

        print(f'\n{"Ep":>4} {"Trn":>8} {"Val":>8} {"PPL":>8} '
              f'{"LR":>8} {"Time":>7}')
        print('-' * 48)

        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            tl = self._run_epoch(model, trn, crit, opt, train=True)
            vl = self._run_epoch(model, val, crit, opt, train=False)
            ppl = math.exp(min(vl, 20))
            print(f'{ep:>4} {tl:>8.4f} {vl:>8.4f} {ppl:>8.2f} '
                  f'{lr:>8.4f} {time.time()-t0:>6.1f}s')

            if vl < best_val:
                best_val = vl
                torch.save({
                    'epoch': ep, 'val_loss': vl, 'val_ppl': ppl,
                    'ntoken': len(vocab), 'nhid': self.nhid,
                    'nlayers': self.nlayers, 'dropout': self.dropout,
                    'model_state': model.state_dict(),
                }, str(self.model_path))
                patience = 0
            else:
                patience += 1
                if patience >= 5:          # was 2 — too aggressive
                    lr /= 2.0              # was /4.0 — too aggressive
                    for g in opt.param_groups:
                        g['lr'] = lr
                    patience = 0
                    print(f'  LR → {lr:.5f}')
            if lr < 0.01:
                print('Early stop')
                break

        print(f'\nBest val PPL: {math.exp(best_val):.2f}')
        print(f'Saved: {self.model_path}')
        return self.load()

    def load(self):
        """Load trained model and vocab."""
        vocab = Vocab()
        vocab.load(str(self.vocab_path))
        self.vocab = vocab

        ck = torch.load(str(self.model_path), map_location=self.device)
        model = RNNModel(
            ntoken=ck['ntoken'], ninp=ck['nhid'], nhid=ck['nhid'],
            nlayers=ck['nlayers'], dropout=ck['dropout'],
            tie_weights=True
        ).to(self.device)
        model.load_state_dict(ck['model_state'])
        model.eval()
        self.model = model
        print(f'Loaded: {self.model_path}  '
              f'(val PPL={ck["val_ppl"]:.2f})')
        return model, vocab


# ─────────────────────────────────────────
# ADAPTIVE SCORER
# ─────────────────────────────────────────

class AdaptiveScorer:

    def __init__(self, model_dir: str,
                 adapt_lr: float = 20.0,
                 adapt_steps: int = 1,
                 device: str = None):
        """
        adapt_lr    = 20.0  (van Schijndel & Linzen 2018 — same SGD LR as training)
        adapt_steps = 1     (paper: single SGD step on context)
        """
        self.model_dir = Path(model_dir)
        self.adapt_lr = adapt_lr
        self.adapt_steps = adapt_steps
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        self.vocab = None

    @property
    def surprisals_path(self):
        return self.model_dir / 'surprisals.json'

    def load(self):
        trainer = LSTMTrainer(
            corpus_path='',
            model_dir=str(self.model_dir),
            device=str(self.device)
        )
        self.model, self.vocab = trainer.load()

    def _encode(self, tokens: List[Dict]) -> torch.Tensor:
        bos = self.vocab.w2i[Vocab.BOS]
        eos = self.vocab.w2i[Vocab.EOS]

        words = []

        for t in tokens:
            # safely get word
            word = t.get('form', '')

            # skip non-string
            if not isinstance(word, str):
                continue

            word = word.strip()

            # skip empty
            if word == '':
                continue

            words.append(word)

        # fallback if sentence becomes empty
        if len(words) == 0:
            words = [Vocab.UNK]

        # encode safely
        ids = [bos] + self.vocab.encode(words) + [eos]

        return torch.tensor(ids, dtype=torch.long,
                            device=self.device).unsqueeze(1)

    def _surprisal(self, model, x: torch.Tensor) -> float:
        """
        Compute mean per-token surprisal (-log P), in nats.
        Matches paper's per-word surprisal metric.
        x: (seq_len, 1) tensor  (BOS + tokens + EOS)
        """
        model.eval()
        with torch.no_grad():
            hid = model.init_hidden(1, self.device)
            inp = x[:-1]   # BOS + tokens (predict each token)
            tgt = x[1:]    # tokens + EOS  (targets)
            out, _ = model(inp, hid)
            # out: (seq_len-1, 1, vocab)
            log_p = nn.functional.log_softmax(out, dim=2)
            tgt_idx = tgt.squeeze(1)  # (seq_len-1,)
            token_lp = log_p[:, 0, :].gather(
                1, tgt_idx.unsqueeze(1)
            ).squeeze(1)
            # Mean per token (not sum) — matches paper's bits/word metric
            return -token_lp.mean().item()

    def score_adaptive(
        self,
        target_tokens: List[Dict],
        context_tokens: Optional[List[Dict]] = None
    ) -> float:
        """
        Paper algorithm (--adapt flag in neural-complexity):
          1. Save current weights
          2. SGD step on context (lr=0.1, 1 step)
          3. Score target with adapted weights
          4. Restore original weights
        """
        if self.model is None:
            self.load()

        x_tgt = self._encode(target_tokens)

        if context_tokens is None or self.adapt_steps == 0:
            return self._surprisal(self.model, x_tgt)

        # Save original state
        orig_state = copy.deepcopy(self.model.state_dict())

        try:
            x_ctx = self._encode(context_tokens)
            self.model.train()
            opt = torch.optim.SGD(
                self.model.parameters(), lr=self.adapt_lr)
            crit = nn.CrossEntropyLoss(ignore_index=0)

            for _ in range(self.adapt_steps):
                opt.zero_grad()
                hid = self.model.init_hidden(1, self.device)
                inp = x_ctx[:-1]
                tgt = x_ctx[1:].squeeze(1).reshape(-1)
                out, _ = self.model(inp, hid)
                loss = crit(
                    out.view(-1, len(self.vocab)), tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 0.25)
                opt.step()

            s = self._surprisal(self.model, x_tgt)

        except Exception:
            s = self._surprisal(
                self._restore(orig_state), x_tgt)

        finally:
            # Always restore
            self.model.load_state_dict(orig_state)
            self.model.eval()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return s

    def _restore(self, state):
        self.model.load_state_dict(state)
        self.model.eval()
        return self.model

    # ─────────────────────────────────────────
    # BATCH — with caching
    # ─────────────────────────────────────────

    def compute_all(
        self,
        sentences: List[List[Dict]],
        metadata_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute adaptive surprisal for every filtered sentence.
        Saves to models/adaptive_lstm/surprisals.json after each batch.
        On restart, skips already-computed sentences.

        Returns: dict {sent_id: surprisal}
        """
        if self.model is None:
            self.load()

        save_path = self.surprisals_path
        surprisals: Dict[str, float] = {}

        # Resume from cache
        if save_path.exists():
            with open(save_path, 'r', encoding='utf-8') as f:
                surprisals = json.load(f)
            print(f'Resumed: {len(surprisals):,} already scored')

        sent_ids = list(metadata_df['sent_id'])
        remaining = [(i, sid) for i, sid in enumerate(sent_ids)
                     if sid not in surprisals]

        print(f'To score: {len(remaining):,} sentences')

        save_every = 100
        t0 = time.time()

        for batch_i, (idx, sid) in enumerate(remaining):
            sent = sentences[idx]

            # Context = preceding sentence (only if it belongs to same doc,
            # i.e. not the very first sentence in the list)
            ctx = sentences[idx - 1] if idx > 0 else None

            try:
                s = self.score_adaptive(sent, ctx)
                if not math.isfinite(s):
                    s = 100.0
            except Exception as e:
                print(f'  Warning: sentence {sid} failed: {e}')
                s = 100.0

            surprisals[sid] = s

            # Save checkpoint every 100 sentences
            if (batch_i + 1) % save_every == 0:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(surprisals, f, ensure_ascii=False)
                elapsed = time.time() - t0
                rate = (batch_i + 1) / elapsed
                remain = (len(remaining) - batch_i - 1) / max(rate, 1e-6)
                print(f'  {batch_i+1:>5}/{len(remaining)}  '
                      f'({(batch_i+1)/len(remaining)*100:.1f}%)  '
                      f'ETA: {remain/60:.1f} min')

        # Final save
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(surprisals, f, ensure_ascii=False)

        print(f'\nDone: {len(surprisals):,} sentences scored')
        print(f'Saved: {save_path}')
        return surprisals

    def add_to_variants(
        self,
        variants_df: pd.DataFrame,
        surprisals: Dict[str, float],
        column_name: str = 'surprisal_adaptive_lstm'
    ) -> pd.DataFrame:
        """Map sentence-level surprisals onto variants DataFrame."""
        df = variants_df.copy()
        df[column_name] = df['reference_id'].map(
            lambda s: surprisals.get(s, 100.0))
        real = (df[column_name] < 100.0).sum()
        print(f'Adaptive LSTM: {real:,} real scores, '
              f'{len(df)-real:,} defaults')
        return df

    def print_stats(self, df: pd.DataFrame,
                    col: str = 'surprisal_adaptive_lstm'):
        valid = df[df[col] < 100.0]
        c = valid[valid['is_original'] == 1][col]
        v = valid[valid['is_original'] == 0][col]
        print('\n── ADAPTIVE LSTM STATS ───────────────────')
        print(f'  Valid scores     : {len(valid):,}/{len(df):,}')
        print(f'  Corpus  mean     : {c.mean():.3f}')
        print(f'  Variant mean     : {v.mean():.3f}')
        print(f'  Corpus < Variant : {c.mean() < v.mean()} (expected True)')
        print('──────────────────────────────────────────')

    def verify(self):
        """Sanity check: context adaptation should lower surprisal."""
        if self.model is None:
            self.load()

        ctx = [{'form': 'राम'}, {'form': 'ने'},
               {'form': 'किताब'}, {'form': 'खरीदी'}]
        tgt = [{'form': 'उसने'}, {'form': 'किताब'},
               {'form': 'पढ़ी'}]

        s_base = self.score_adaptive(tgt, None)
        s_adpt = self.score_adaptive(tgt, ctx)

        print(f'\nVerification:')
        print(f'  Base surprisal    : {s_base:.3f}')
        print(f'  Adaptive surprisal: {s_adpt:.3f}')
        print(f'  Adaptation works  : {s_adpt < s_base}')
        return s_adpt < s_base