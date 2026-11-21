# ═══════════════════════════════════════════════════════════════
# KAGGLE NOTEBOOK — Plain (non-adaptive) LSTM surprisal
# Run each block as a separate cell in Kaggle.
#
# WHY THIS EXISTS:
#   The paper's "adaptive" LSTM does a single SGD step on the
#   preceding context sentence before scoring the target (simulating
#   a reader updating expectations as they go). Per professor's
#   instruction, we drop that adaptation step and just use the plain
#   LSTM forward pass — same trained model, no retraining needed.
#
# Before running, upload ONE Kaggle dataset, e.g. "lstm-model", with:
#   - lstm_model.pt   (from archive (1).zip in your Downloads)
#   - vocab.json       (same zip)
#   - variants.csv     (same zip, or your latest regenerated one —
#                        make sure it's the CORRECTED version with
#                        fixed sent_ids / punct, not the stale one)
#   - variants_tokens.jsonl   (same zip / latest regenerated)
# No GPU strictly required (plain forward pass is cheap), but GPU
# will still speed it up a lot over ~79k rows — enable it if available.
# ═══════════════════════════════════════════════════════════════


# ── CELL 1: Model classes (self-contained, no repo import needed) ─
import json, math, time
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict


class Vocab:
    PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'

    def __init__(self):
        self.w2i: Dict[str, int] = {}

    def load(self, path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        self.w2i = d['w2i']
        print(f'Vocab loaded: {len(self.w2i):,} words')

    def encode(self, words):
        unk = self.w2i[self.UNK]
        return [self.w2i.get(w, unk) for w in words]

    def __len__(self):
        return len(self.w2i)


class RNNModel(nn.Module):
    def __init__(self, ntoken, ninp=650, nhid=650, nlayers=2,
                 dropout=0.5, tie_weights=True):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            self.decoder.weight = self.encoder.weight
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, obs, hidden):
        emb = self.drop(self.encoder(obs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, device):
        w = next(self.parameters()).data
        return (w.new_zeros(self.nlayers, bsz, self.nhid).to(device),
                w.new_zeros(self.nlayers, bsz, self.nhid).to(device))


# ── CELL 2: Load the already-trained model (no retraining) ────────
MODEL_DIR = Path('/kaggle/input/lstm-model')   # match your dataset name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

vocab = Vocab()
vocab.load(MODEL_DIR / 'vocab.json')

ck = torch.load(MODEL_DIR / 'lstm_model.pt', map_location=device)
model = RNNModel(
    ntoken=ck['ntoken'], ninp=ck['nhid'], nhid=ck['nhid'],
    nlayers=ck['nlayers'], dropout=ck['dropout'], tie_weights=True
).to(device)
model.load_state_dict(ck['model_state'])
model.eval()
print(f'Loaded lstm_model.pt (val PPL={ck["val_ppl"]:.2f}) — same weights as the adaptive run')


# ── CELL 3: Plain (non-adaptive) scoring function ──────────────────
def encode(tokens: List[Dict]) -> torch.Tensor:
    bos, eos = vocab.w2i[Vocab.BOS], vocab.w2i[Vocab.EOS]
    words = [t['form'].strip() for t in tokens
             if isinstance(t.get('form'), str) and t['form'].strip()]
    if not words:
        words = [Vocab.UNK]
    ids = [bos] + vocab.encode(words) + [eos]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)


def plain_surprisal(tokens: List[Dict]) -> float:
    """
    No SGD / no context adaptation — pure forward pass.
    This is the ONLY difference from score_adaptive(): that function
    takes a context_tokens arg and does an SGD step on it first;
    here we just never do that step at all.
    """
    x = encode(tokens)
    with torch.no_grad():
        hid = model.init_hidden(1, device)
        inp, tgt = x[:-1], x[1:]
        out, _ = model(inp, hid)
        log_p = nn.functional.log_softmax(out, dim=2)
        tgt_idx = tgt.squeeze(1)
        token_lp = log_p[:, 0, :].gather(1, tgt_idx.unsqueeze(1)).squeeze(1)
        return -token_lp.mean().item()


# ── CELL 4: Score every row of variants.csv (with checkpointing) ──
import pandas as pd

VAR_CSV    = '/kaggle/input/lstm-model/variants.csv'           # adjust path
VAR_JSONL  = '/kaggle/input/lstm-model/variants_tokens.jsonl'  # adjust path
OUT_JSON   = '/kaggle/working/lstm_plain_surprisals.json'

df = pd.read_csv(VAR_CSV)
tokens_map = {}
with open(VAR_JSONL, encoding='utf-8') as f:
    for line in f:
        vid, tjson = line.rstrip('\n').split('\t', 1)
        tokens_map[vid] = tjson
print(f'Variants: {len(df):,}  Tokens loaded: {len(tokens_map):,}')

scores = {}
import os
if os.path.exists(OUT_JSON):
    with open(OUT_JSON, encoding='utf-8') as f:
        scores = json.load(f)
    print(f'Resumed: {len(scores):,} already scored')

todo = [vid for vid in df['variant_id'] if vid not in scores]
print(f'To score: {len(todo):,}')

t0 = time.time()
for i, vid in enumerate(todo):
    try:
        tokens = json.loads(tokens_map[vid])
        scores[vid] = plain_surprisal(tokens)
    except Exception as e:
        scores[vid] = 100.0
    if (i + 1) % 2000 == 0:
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(todo) - i - 1) / max(rate, 1e-6)
        print(f'  {i+1:>6,}/{len(todo):,}  ETA: {eta/60:.1f} min')

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(scores, f, ensure_ascii=False)
print(f'Done: {len(scores):,} scores -> {OUT_JSON}')


# ── CELL 5: Attach to the dataframe + quick sanity stats ──────────
df['surprisal_lstm'] = df['variant_id'].map(lambda v: scores.get(v, 100.0))
valid = df[df['surprisal_lstm'] < 100.0]
c = valid[valid['is_original'] == 1]['surprisal_lstm']
v = valid[valid['is_original'] == 0]['surprisal_lstm']
print(f'Valid: {len(valid):,}/{len(df):,}')
print(f'Corpus  mean: {c.mean():.3f}')
print(f'Variant mean: {v.mean():.3f}')
print(f'Corpus < Variant: {c.mean() < v.mean()} (expected True, same direction as adaptive version)')

df.to_csv('/kaggle/working/variants_with_plain_lstm.csv', index=False)
print('Download variants_with_plain_lstm.csv (or just lstm_plain_surprisals.json)')
print('from the Kaggle Output panel, then locally merge surprisal_lstm into features.csv')
print('alongside the existing surprisal_adaptive_lstm column for comparison.')
