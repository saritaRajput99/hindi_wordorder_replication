"""
Score PCFG surprisal, Modelblocks surprisal, and Dependency Parser surprisal
for every row in variants.csv, using Muraliedhar's hosted tool:
  https://huggingface.co/spaces/Muraliedhar/SyntacticTreeSurprisal

Runs locally (plain HTTP calls — no GPU/Kaggle needed).

Mapping (verified from the Space's flask_server.py / dep_incremental_backend.py):
  PCFG surprisal              -> POST /parse        -> response['pcfg_surprisal']   (Berkeley Parser)
  Modelblocks surprisal       -> POST /parse        -> response['surprisal']        (synproc)
  Dependency Parser surprisal -> POST /depparse      -> response['total_surprisal']  (arc-eager cv5 parser)

Sentence-level score = sum of per-word surprisal (bits), matching the
paper's own convention ("sum the per-word scores within each sentence").

Resumable: progress is checkpointed to RESULTS_JSON after every batch, and
re-running this script skips variant_ids already scored.

NOTE on scope: both /parse's "hdtb" model and /depparse's default
"hi_hdtb_modified_v1" model are trained on the full UD_Hindi-HDTB training
file — the same file our reference sentences come from. That's a known
train/test leakage caveat (same issue flagged for the adaptive LSTM /
SRILM steps) — flag it in the paper, don't treat these as held-out scores
without checking.

Run a SMALL TEST FIRST: set TEST_LIMIT below to e.g. 200 before committing
to the full ~78,865-row run against a shared, free community Space.
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────
BASE_URL     = 'https://muraliedhar-syntactictreesurprisal.hf.space'
VARIANTS_CSV = 'output/variants_final/variants.csv'
RESULTS_JSON = 'results/syntactic_surprisal.json'

BATCH_SIZE    = 100      # matches server's MAX_SENTENCES_PER_REQUEST
POLL_INTERVAL = 3        # seconds between status polls
POLL_TIMEOUT  = 1800     # give up on a single batch after 30 min
REQUEST_RETRIES = 3

TEST_LIMIT = 200   # <- set to None for the full run, once the test batch looks right

Path('results').mkdir(exist_ok=True)


# ── HTTP helpers ─────────────────────────────────────────────────────────

def _post(path, payload):
    for attempt in range(REQUEST_RETRIES):
        try:
            r = requests.post(f'{BASE_URL}{path}', json=payload, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f'  POST {path} failed (attempt {attempt+1}/{REQUEST_RETRIES}): {e}')
            time.sleep(5)
    raise RuntimeError(f'POST {path} failed after {REQUEST_RETRIES} attempts')


def _poll(status_path, job_id):
    t0 = time.time()
    while True:
        if time.time() - t0 > POLL_TIMEOUT:
            raise TimeoutError(f'{status_path}/{job_id} did not finish in {POLL_TIMEOUT}s')
        try:
            r = requests.get(f'{BASE_URL}{status_path}/{job_id}', timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f'  Poll failed, retrying: {e}')
            time.sleep(POLL_INTERVAL)
            continue

        if data.get('status') == 'done':
            return data
        if data.get('status') == 'error':
            raise RuntimeError(f'Job {job_id} errored: {data.get("error")}')
        time.sleep(POLL_INTERVAL)


def score_parse_batch(sentences):
    """Returns list of (pcfg_surprisal_sum, modelblocks_surprisal_sum) per sentence."""
    payload = {
        'sentence'  : '\n'.join(sentences),
        'model'     : 'hdtb',
        'surprisal' : True,
        'input_type': 'sentence',
    }
    job = _post('/parse/batch', payload)
    data = _poll('/parse/status', job['job_id'])
    out = []
    for res in data.get('results', []):
        if 'error' in res:
            out.append((None, None))
            continue
        pcfg = res.get('pcfg_surprisal') or {}
        mb   = res.get('surprisal') or {}
        pcfg_sum = sum(v for v in pcfg.values() if isinstance(v, (int, float)))
        mb_sum   = sum(v for v in mb.values() if isinstance(v, (int, float)))
        out.append((pcfg_sum, mb_sum))
    return out


def score_depparse_batch(sentences):
    """Returns list of dep_surprisal_sum per sentence."""
    payload = {
        'sentence'  : '\n'.join(sentences),
        'input_type': 'sentence',
    }
    job = _post('/depparse/batch', payload)
    data = _poll('/depparse/status', job['job_id'])
    out = []
    for res in data.get('results', []):
        if 'error' in res:
            out.append(None)
            continue
        if 'total_surprisal' in res:
            out.append(res['total_surprisal'])
        else:
            # fall back to summing per-token surprisal if total_surprisal absent
            rows = res.get('rows') or res.get('tokens') or []
            out.append(sum(r.get('surprisal', 0.0) for r in rows) if rows else None)
    return out


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(VARIANTS_CSV)
    if TEST_LIMIT:
        df = df.head(TEST_LIMIT)
        print(f'TEST MODE: scoring only the first {TEST_LIMIT} rows')

    print(f'Loaded {len(df):,} rows from {VARIANTS_CSV}')

    scores = {}
    if Path(RESULTS_JSON).exists():
        with open(RESULTS_JSON, encoding='utf-8') as f:
            scores = json.load(f)
        print(f'Resumed: {len(scores):,} already scored')

    todo = df[~df['variant_id'].isin(scores.keys())]
    print(f'To score: {len(todo):,}')

    rows = todo[['variant_id', 'text']].to_dict('records')

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        vids  = [r['variant_id'] for r in batch]
        texts = [str(r['text']) for r in batch]

        print(f'\nBatch {start//BATCH_SIZE + 1}/{-(-len(rows)//BATCH_SIZE)} '
              f'({len(batch)} sentences)...')

        try:
            parse_results = score_parse_batch(texts)
            dep_results   = score_depparse_batch(texts)
        except Exception as e:
            print(f'  Batch failed, skipping for now (will retry on next run): {e}')
            continue

        for vid, (pcfg, mb), dep in zip(vids, parse_results, dep_results):
            scores[vid] = {
                'surprisal_pcfg'       : pcfg,
                'surprisal_modelblocks': mb,
                'surprisal_depparse'   : dep,
            }

        with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False)
        print(f'  Checkpointed: {len(scores):,} total scored')

    print(f'\nDone. {len(scores):,} rows scored -> {RESULTS_JSON}')
    print('Merge into features.csv with:')
    print("  import json, pandas as pd")
    print(f"  scores = json.load(open('{RESULTS_JSON}', encoding='utf-8'))")
    print("  df['surprisal_pcfg']        = df['variant_id'].map(lambda v: scores.get(v, {}).get('surprisal_pcfg'))")
    print("  df['surprisal_modelblocks'] = df['variant_id'].map(lambda v: scores.get(v, {}).get('surprisal_modelblocks'))")
    print("  df['surprisal_depparse']    = df['variant_id'].map(lambda v: scores.get(v, {}).get('surprisal_depparse'))")


if __name__ == '__main__':
    main()
