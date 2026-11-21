"""
Berkeley Parser 5-fold CV PCFG surprisal — Ranjan & van Schijndel (2024).

Grammar: 2-level PCFG from HDTB ChunkId annotations
  ROOT → C₁ C₂ … Cₖ   (chunk-label sequence)
  Cᵢ  → (W w₁) … (W wₘ) (word sequence, 'W' = dummy POS)

5-fold CV: reference sentences held out, scored with fold grammar.
           All their variants scored with SAME fold grammar.
Scoring: -sentence_likelihood = log P(w) (inside sum over all parses).
Surprisal = -log P(w).

Output: models/pcfg/pcfg_surprisals.json  {variant_id → surprisal}

Speed: -nThreads 4 (uses all cores of i5 4th gen within Berkeley Parser).
       Batch size 5000 sentences per scoring run.
Checkpoint: saved after every fold.
"""

import os, sys, re, json, math, time, subprocess
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONLLU     = os.path.join(BASE, 'data', 'ud_hindi', 'train.conllu')
VARIANTS   = os.path.join(BASE, 'data', 'processed', 'variants', 'variants.csv')
PCFG_DIR   = os.path.join(BASE, 'models', 'pcfg')
JAR        = os.path.join(PCFG_DIR, 'BerkeleyParser-1.7.jar')
OUT_JSON   = os.path.join(PCFG_DIR, 'pcfg_surprisals.json')
CKPT_JSON  = os.path.join(PCFG_DIR, 'pcfg_surprisals_ckpt.json')
TMP_DIR    = os.path.join(PCFG_DIR, 'tmp')

os.makedirs(TMP_DIR, exist_ok=True)

N_FOLDS    = 5
SEED       = 42
BATCH_SIZE = 3000      # sentences per Berkeley Parser subprocess
N_WORKERS  = 2         # parallel Berkeley Parser processes (each single-threaded)
                       # Note: -nThreads crashes with -sentence_likelihood; use Python parallelism
SM_CYCLES  = 4         # EM split-merge cycles (default=6; 4=faster)
JVM_MEM    = '5g'      # -Xmx heap per process (2 workers × 5g = 10g, fits 12 GB RAM)
SENTINEL   = 0.0       # flags unscored sentence (real surprisal always > 0)


# ── 1. CoNLL-U → PTB ────────────────────────────────────────────────────────

def _chunk_base(cid):
    return re.sub(r'\d+$', '', cid) if cid else 'UNK'


def _to_ptb(chunks):
    """[(label, [words])] → '(ROOT (NP (W w1) (W w2)) (VGF (W w3)))'"""
    return '(ROOT ' + ' '.join(
        '(' + lbl + ' ' + ' '.join('(W ' + w.lower() + ')' for w in ws) + ')'
        for lbl, ws in chunks
    ) + ')'


def extract_ptb_trees(conllu_path):
    """Returns list of (surface_str, ptb_str) for every sentence."""
    trees  = []
    tokens = []

    def flush(toks):
        if not toks:
            return
        seen = OrderedDict()
        for w, cid in toks:
            key  = cid if cid else f'UNK_{len(seen)}'
            base = _chunk_base(cid)
            if key not in seen:
                seen[key] = (base, [])
            seen[key][1].append(w)
        surface = ' '.join(w for w, _ in toks)
        chunks  = list(seen.values())
        trees.append((surface, _to_ptb(chunks)))

    with open(conllu_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                flush(tokens); tokens = []; continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 10 or '-' in parts[0] or '.' in parts[0]:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            cid = None
            for kv in parts[9].split('|'):
                if kv.startswith('ChunkId='):
                    cid = kv.split('=', 1)[1]; break
            tokens.append((parts[1], cid))
    flush(tokens)
    return trees


# ── 2. Train Berkeley Parser grammar ────────────────────────────────────────

def train_grammar(ptb_file, grammar_out):
    if os.path.exists(grammar_out):
        print(f'    Grammar already exists: {grammar_out}')
        return
    cmd = [
        'java', f'-Xmx{JVM_MEM}', '-Dfile.encoding=UTF-8',
        '-cp', JAR, 'edu.berkeley.nlp.PCFGLA.GrammarTrainer',
        '-path', ptb_file,
        '-out', grammar_out,
        '-treebank', 'SINGLEFILE',
        '-SMcycles', str(SM_CYCLES),
    ]
    t0 = time.time()
    r  = subprocess.run(cmd, capture_output=True, text=True,
                        encoding='utf-8', errors='replace', timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(
            f'GrammarTrainer failed (rc={r.returncode}):\n{r.stderr[-500:]}')
    print(f'    Grammar trained in {(time.time()-t0)/60:.1f} min → {grammar_out}')


# ── 3. Score sentences in batches ───────────────────────────────────────────

def _run_berkeley_batch(grammar, sentences, batch_id=''):
    """
    Score one batch using a single Berkeley Parser subprocess.
    Returns list of log_prob per sentence (SENTINEL if failed).
    Output format: 'log_prob\\t(parse)' per line.
    Note: -nThreads crashes with -sentence_likelihood; use single-threaded.
    """
    input_text = '\n'.join(s.lower() for s in sentences) + '\n'
    cmd = [
        'java', f'-Xmx{JVM_MEM}', '-Dfile.encoding=UTF-8',
        '-jar', JAR,
        '-gr', grammar,
        '-sentence_likelihood',
    ]
    try:
        r = subprocess.run(
            cmd, input=input_text, capture_output=True,
            text=True, encoding='utf-8', errors='replace', timeout=1800
        )
        out_lines = r.stdout.strip().split('\n') if r.stdout.strip() else []
    except subprocess.TimeoutExpired:
        print(f'      [{batch_id}] TIMEOUT')
        return [SENTINEL] * len(sentences)
    except Exception as e:
        print(f'      [{batch_id}] ERROR: {e}')
        return [SENTINEL] * len(sentences)

    log_probs = []
    for i in range(len(sentences)):
        lp = SENTINEL
        if i < len(out_lines):
            line = out_lines[i].strip()
            if line and line != '(())':
                try:
                    lp = float(line.split('\t')[0])
                    if not math.isfinite(lp):
                        lp = SENTINEL
                except (ValueError, IndexError):
                    lp = SENTINEL
        log_probs.append(lp)
    return log_probs


def score_all(grammar, sentences, batch_size=BATCH_SIZE, n_workers=N_WORKERS):
    """
    Score all sentences using N_WORKERS parallel Berkeley Parser processes.
    Returns dict: {sentence → log_prob}
    """
    unique = list(dict.fromkeys(s.lower() for s in sentences))
    n_batches = math.ceil(len(unique) / batch_size)
    print(f'    Scoring {len(unique):,} unique sentences  '
          f'{n_batches} batches × {batch_size}  ({n_workers} parallel workers) ...')

    batches = [unique[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    sent_to_logp = {}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futs = {
            exe.submit(_run_berkeley_batch, grammar, b, f'batch{i+1}'): i
            for i, b in enumerate(batches)
        }
        done = 0
        for fut in as_completed(futs):
            bi    = futs[fut]
            lps   = fut.result()
            for s, lp in zip(batches[bi], lps):
                sent_to_logp[s] = lp
            done += 1
            real = sum(1 for v in sent_to_logp.values() if v != SENTINEL)
            eta  = (time.time()-t0) / done * (n_batches - done)
            print(f'    {done}/{n_batches} batches done  '
                  f'({real:,} real scores)  ETA {eta/60:.1f} min')

    return sent_to_logp


# ── 4. Main ──────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('PCFG Surprisal — Berkeley Parser 5-fold CV')
    print('=' * 65)

    if not os.path.exists(JAR):
        print(f'\nERROR: Berkeley Parser jar not found at:\n  {JAR}')
        sys.exit(1)
    print(f'\nJar: {JAR}  ({os.path.getsize(JAR)//1024} KB)')

    # 1. Extract PTB trees from CoNLL-U
    print(f'\n[1] Extracting PTB trees from train.conllu ...')
    t0    = time.time()
    trees = extract_ptb_trees(CONLLU)
    surface_to_ptb = dict(trees)
    print(f'    {len(trees):,} trees in {time.time()-t0:.1f}s')
    print(f'    Sample: {trees[0][1][:95]}')

    # 2. Load variants (latest run only)
    print(f'\n[2] Loading variants (current run) ...')
    df = pd.read_csv(VARIANTS)
    print(f'    {len(df):,} rows  |  {df["reference_id"].nunique():,} unique refs')
    print(f'    order_type counts:\n' +
          df['order_type'].value_counts().to_string())

    vid_to_text  = dict(zip(df['variant_id'], df['text']))
    ref_to_vids  = df.groupby('reference_id')['variant_id'].apply(list).to_dict()
    ref_df       = df[df['is_original'] == 1][['reference_id', 'text']].drop_duplicates()
    ref_surface  = dict(zip(ref_df['reference_id'], ref_df['text']))

    # 3. 5-fold split by reference_id
    ref_ids = sorted(ref_df['reference_id'].tolist())
    rng = random.Random(SEED)
    rng.shuffle(ref_ids)
    fold_size = max(1, len(ref_ids) // N_FOLDS)
    folds = [
        ref_ids[k*fold_size : (k+1)*fold_size if k < N_FOLDS-1 else len(ref_ids)]
        for k in range(N_FOLDS)
    ]
    print(f'\n    5-fold split: {[len(f) for f in folds]} refs/fold')

    # 4. Resume checkpoint
    surprisals  = {}
    done_folds  = set()
    if os.path.exists(CKPT_JSON):
        with open(CKPT_JSON, encoding='utf-8') as f:
            surprisals = json.load(f)
        scored_set = set(surprisals)
        for k, fold_refs in enumerate(folds):
            fold_vids = {v for r in fold_refs for v in ref_to_vids.get(r, [])}
            if fold_vids.issubset(scored_set):
                done_folds.add(k)
        print(f'    Resumed: {len(surprisals):,} scores, '
              f'folds already done: {sorted(done_folds)}')

    # 5. 5-fold CV loop
    print('\n[3] 5-fold CV ...')
    for fold_idx, test_ref_ids in enumerate(folds):
        if fold_idx in done_folds:
            print(f'\n  Fold {fold_idx+1}/{N_FOLDS}: skip (done)')
            continue

        print(f'\n  {"─"*55}')
        print(f'  Fold {fold_idx+1}/{N_FOLDS}  |  {len(test_ref_ids):,} held-out refs')
        print(f'  {"─"*55}')

        # surfaces of held-out refs (to exclude from training)
        held_surfaces = {
            ref_surface[r] for r in test_ref_ids if r in ref_surface
        }

        # ── 5a. Write training PTB
        ptb_train = os.path.join(TMP_DIR, f'train_fold{fold_idx}.ptb')
        grammar   = os.path.join(PCFG_DIR, f'grammar_fold{fold_idx}.gr')

        if not os.path.exists(grammar):
            train_trees = [
                ptb for surf, ptb in trees if surf not in held_surfaces
            ]
            print(f'  Writing {len(train_trees):,} training trees ...')
            with open(ptb_train, 'w', encoding='utf-8') as f:
                f.write('\n'.join(train_trees) + '\n')
            print(f'  Training Berkeley Parser (SMcycles={SM_CYCLES}) ...')
            train_grammar(ptb_train, grammar)

        # ── 5b. Collect sentences for this fold
        fold_vids   = [v for r in test_ref_ids for v in ref_to_vids.get(r, [])]
        fold_texts  = list(dict.fromkeys(
            vid_to_text[v] for v in fold_vids if v in vid_to_text
        ))
        print(f'  Sentences to score: {len(fold_texts):,} unique '
              f'from {len(fold_vids):,} variant IDs')

        # ── 5c. Score
        t0 = time.time()
        text_to_logp = score_all(grammar, fold_texts)
        print(f'  Scoring done in {(time.time()-t0)/60:.1f} min')

        # ── 5d. Map log_prob → surprisal per variant_id
        real = 0
        for vid in fold_vids:
            text = vid_to_text.get(vid, '').lower()
            lp   = text_to_logp.get(text, SENTINEL)
            if lp != SENTINEL:
                surprisals[vid] = -lp    # surprisal = -log P
                real += 1
            else:
                surprisals[vid] = SENTINEL
        print(f'  Mapped: {real:,}/{len(fold_vids):,} real scores '
              f'({100*real/max(len(fold_vids),1):.1f}%)')

        # ── 5e. Checkpoint
        with open(CKPT_JSON, 'w', encoding='utf-8') as f:
            json.dump(surprisals, f, ensure_ascii=False)
        print(f'  Checkpoint: {len(surprisals):,} total scores saved')

    # 6. Save final
    print(f'\n[4] Saving final output ...')
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(surprisals, f, ensure_ascii=False)

    real  = sum(1 for v in surprisals.values() if v > SENTINEL)
    total = len(surprisals)
    vals  = [v for v in surprisals.values() if v > SENTINEL]
    print(f'    {OUT_JSON}')
    print(f'    Real scores : {real:,}/{total:,}  ({100*real/total:.1f}%)')
    if vals:
        print(f'    Surprisal   mean={np.mean(vals):.2f}  '
              f'min={np.min(vals):.2f}  max={np.max(vals):.2f}')

    print('\n' + '='*65)
    print('DONE — PCFG surprisals ready.')
    print('Next: run scripts/run_analysis_full.py to include PCFG in regression.')
    print('='*65)


if __name__ == '__main__':
    main()
