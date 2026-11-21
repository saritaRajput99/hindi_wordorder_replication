"""Stream ARPA, keep only n-grams whose words are in our corpus vocab."""
import os, sys, json, time, math
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_CSV = os.path.join(BASE, 'models', 'adaptive_lstm', 'final_var_score.csv')
ARPA      = os.path.join(BASE, 'models', 'trigram', 'hindi_trigram.arpa')
OUT_JSON  = os.path.join(BASE, 'models', 'trigram', 'trigram_filtered.json')
SCORES_OUT= os.path.join(BASE, 'models', 'trigram', 'trigram_surprisals.json')

# --- vocab from corpus ---
df = pd.read_csv(FINAL_CSV)
vocab = set()
for text in df['text'].dropna():
    vocab.update(text.split())
vocab.update(['<s>', '</s>', '<unk>'])
print(f'Corpus vocab: {len(vocab)} words')

# --- stream ARPA ---
print(f'Streaming ARPA (filter to corpus vocab)...')
t0 = time.time()
unigrams = {}
bigrams  = {}
trigrams = {}
section  = None
lc = 0

with open(ARPA, encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.strip()
        lc += 1
        if lc % 2000000 == 0:
            print(f'  {lc//1000000}M lines  {time.time()-t0:.0f}s  1g={len(unigrams)} 2g={len(bigrams)} 3g={len(trigrams)}')

        if line.startswith('\\1-gram'):
            section = 1; continue
        elif line.startswith('\\2-gram'):
            section = 2; continue
        elif line.startswith('\\3-gram'):
            section = 3; continue
        elif line.startswith('\\') or not line:
            continue

        parts = line.split('\t')
        if len(parts) < 2:
            continue
        try:
            lp = float(parts[0])
        except ValueError:
            continue

        if section == 1:
            w  = parts[1]
            bo = float(parts[2]) if len(parts) > 2 else 0.0
            unigrams[w] = (lp, bo)
        elif section == 2:
            ws = parts[1].split()
            if len(ws) == 2 and ws[0] in vocab and ws[1] in vocab:
                bo = float(parts[2]) if len(parts) > 2 else 0.0
                bigrams[(ws[0], ws[1])] = (lp, bo)
        elif section == 3:
            ws = parts[1].split()
            if len(ws) == 3 and ws[0] in vocab and ws[1] in vocab and ws[2] in vocab:
                trigrams[(ws[0], ws[1], ws[2])] = lp

print(f'ARPA streamed in {time.time()-t0:.0f}s')
print(f'  unigrams={len(unigrams):,}  bigrams={len(bigrams):,}  trigrams={len(trigrams):,}')

# --- scoring function ---
LOG10 = math.log(10)
UNK_LP = unigrams.get('<unk>', (-10.0, 0.0))[0]

def score_sent(words):
    context = ['<s>']
    total = 0.0
    for w in words + ['</s>']:
        w_lp = unigrams.get(w, (UNK_LP, 0.0))[0]
        if len(context) >= 2:
            tri = (context[-2], context[-1], w)
            if tri in trigrams:
                total += trigrams[tri] * LOG10
                context.append(w); continue
        bi = (context[-1], w)
        if bi in bigrams:
            total += bigrams[bi][0] * LOG10
            context.append(w); continue
        # back-off
        bo = 0.0
        if len(context) >= 1:
            bo_entry = bigrams.get((context[-2], context[-1])) if len(context) >= 2 else None
            if bo_entry:
                bo = bo_entry[1] * LOG10
        total += bo + w_lp * LOG10
        context.append(w)
    return -total  # surprisal = -log P

# --- score all variants ---
print(f'Scoring {len(df):,} variants...')
t1 = time.time()
scores = {}
for i, (_, row) in enumerate(df.iterrows()):
    vid   = row['sent_id']
    words = str(row['text']).split()
    try:
        scores[vid] = score_sent(words)
    except Exception:
        scores[vid] = 0.0
    if (i+1) % 10000 == 0:
        rate = (i+1)/(time.time()-t1)
        eta  = (len(df)-i-1)/max(rate,1)
        print(f'  {i+1:,}/{len(df):,}  ETA {eta/60:.1f} min')

with open(SCORES_OUT, 'w', encoding='utf-8') as f:
    json.dump(scores, f, ensure_ascii=False)

real = sum(1 for v in scores.values() if v > 0)
print(f'Done. {real:,} real scores -> {SCORES_OUT}')
