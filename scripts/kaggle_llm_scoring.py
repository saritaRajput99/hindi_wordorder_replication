# ═══════════════════════════════════════════════════════════════
# KAGGLE NOTEBOOK — Hindi Word Order LLM Surprisal Scoring
# Run each block as a separate cell in Kaggle
#
# Before running:
#   1. Upload features.csv as a Kaggle Dataset
#   2. Enable GPU (T4 x1) in Notebook settings
#   3. Enable Internet in Notebook settings
#   4. For Gemma: Add HuggingFace token in Kaggle Secrets
#      Kaggle → Add-ons → Secrets → Name: HF_TOKEN → Value: hf_xxxx
# ═══════════════════════════════════════════════════════════════


# ── CELL 1: Install dependencies ────────────────────────────────
# !pip install transformers accelerate bitsandbytes -q


# ── CELL 2: Imports and load features.csv ───────────────────────
import os, json, math, time, torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Path to your uploaded features.csv
# If uploaded as Kaggle dataset named "hindi-features":
#   FEAT_CSV = '/kaggle/input/hindi-features/features.csv'
FEAT_CSV = '/kaggle/input/hindi-features/features.csv'
OUT_DIR  = Path('/kaggle/working')

df = pd.read_csv(FEAT_CSV)
print(f'Loaded: {len(df):,} variants')
print(f'Columns: {df.columns.tolist()}')
print(f'Sample text: {df["text"].iloc[0]}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ── CELL 3: Scoring function (used for both models) ─────────────
def score_all(model, tokenizer, df, out_json, checkpoint_every=500):
    """
    Score all variants. Saves checkpoint every 500 rows.
    Resumes automatically if out_json already exists.
    Returns dict {variant_id: surprisal_nats}
    """
    scores = {}

    # Resume from checkpoint
    if os.path.exists(out_json):
        with open(out_json, encoding='utf-8') as f:
            scores = json.load(f)
        print(f'Resumed: {len(scores):,} already scored')

    todo = [(row['variant_id'], row['text'])
            for _, row in df.iterrows()
            if str(row['variant_id']) not in scores]
    print(f'To score: {len(todo):,}')

    model.eval()
    t0 = time.time()

    for i, (vid, text) in enumerate(todo):
        try:
            inputs = tokenizer(
                str(text),
                return_tensors='pt',
                truncation=True,
                max_length=256,
            ).to(model.device)

            with torch.no_grad():
                loss = model(**inputs, labels=inputs['input_ids']).loss.item()

            scores[str(vid)] = loss if math.isfinite(loss) else 100.0

        except Exception as e:
            scores[str(vid)] = 100.0

        if (i + 1) % checkpoint_every == 0:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(scores, f, ensure_ascii=False)
            elapsed = time.time() - t0
            eta = (len(todo) - i - 1) / ((i + 1) / elapsed)
            print(f'  {i+1:>6,}/{len(todo):,}  ({(i+1)/len(todo)*100:.1f}%)  '
                  f'ETA: {eta/60:.1f} min')

    # Final save
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False)

    real = sum(1 for v in scores.values() if v < 100.0)
    print(f'\nDone: {len(scores):,} scored  ({real:,} real, '
          f'{len(scores)-real} fallback)')
    return scores


# ── CELL 4: XGLM-1.7B scoring (~30 min on T4) ──────────────────
XGLM_OUT = str(OUT_DIR / 'xglm_surprisals.json')

print('Loading XGLM-1.7B ...')
xglm_tok   = AutoTokenizer.from_pretrained('facebook/xglm-1.7B')
xglm_model = AutoModelForCausalLM.from_pretrained(
    'facebook/xglm-1.7B',
    torch_dtype=torch.float16,
).to(device)
print('XGLM loaded')

xglm_scores = score_all(xglm_model, xglm_tok, df, XGLM_OUT)

# Free GPU memory before loading Gemma
del xglm_model, xglm_tok
torch.cuda.empty_cache()
print('XGLM done. GPU memory freed.')


# ── CELL 5: Gemma-3-4B scoring, float16 (~45 min on T4) ────────
#
# Requires HuggingFace token — add it in Kaggle Secrets as HF_TOKEN
# Steps: Accept Gemma licence at huggingface.co/google/gemma-3-4b-pt
#
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')

GEMMA_MODEL = 'google/gemma-3-4b-pt'    # base pretrained — NOT -it
GEMMA_OUT   = str(OUT_DIR / 'gemma3_4b_surprisals.json')

print(f'Loading {GEMMA_MODEL} (float16) ...')
gemma_tok   = AutoTokenizer.from_pretrained(GEMMA_MODEL, token=HF_TOKEN)
gemma_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_MODEL,
    torch_dtype=torch.float16,
    device_map='auto',
    token=HF_TOKEN,
)
print('Gemma loaded')

gemma_scores = score_all(gemma_model, gemma_tok, df, GEMMA_OUT)

del gemma_model, gemma_tok
torch.cuda.empty_cache()
print('Gemma done.')


# ── CELL 6: Verify output files ─────────────────────────────────
for f in [XGLM_OUT, GEMMA_OUT]:
    if os.path.exists(f):
        with open(f, encoding='utf-8') as fp:
            d = json.load(fp)
        print(f'{Path(f).name}: {len(d):,} entries')
    else:
        print(f'MISSING: {f}')

print('\nDownload these two files from /kaggle/working/:')
print('  xglm_surprisals.json')
print('  gemma3_4b_surprisals.json')
print('\nThen place them locally at:')
print('  models/xglm/xglm_surprisals.json')
print('  models/gemma3/gemma3_4b_surprisals.json')
