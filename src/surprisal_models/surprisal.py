import os, json, math
import pandas as pd
from typing import List, Dict


# ── Pure Python ARPA Language Model ──────────────────────────────────────────

class ARPAModel:
    """
    Pure Python ARPA n-gram language model reader.
    Supports KenLM-style ARPA format. No C++ dependencies — works on Windows.

    Implements Katz back-off:
      P(w|context) = P_n(w, context)          if trigram seen
                   = bow(context_n-1) * P(w|context_n-2)  otherwise
    """

    LN10 = math.log(10)   # conversion factor: log10 → natural log

    def __init__(self):
        self.unigrams: Dict[str, tuple] = {}   # word → (log10_p, backoff)
        self.bigrams:  Dict[str, tuple] = {}   # "w1 w2" → (log10_p, backoff)
        self.trigrams: Dict[str, float] = {}   # "w1 w2 w3" → log10_p
        self.unk_logp = -99.0

    def load(self, arpa_path: str) -> None:
        """Load ARPA file. Displays progress every 1M lines."""
        print(f"Loading ARPA model: {arpa_path}")
        section = None
        n_lines = 0

        with open(arpa_path, encoding='utf-8', errors='replace') as f:
            for raw in f:
                line = raw.rstrip('\n')
                n_lines += 1

                if n_lines % 1_000_000 == 0:
                    print(f"  {n_lines//1_000_000}M lines read  "
                          f"(1g={len(self.unigrams):,}  "
                          f"2g={len(self.bigrams):,}  "
                          f"3g={len(self.trigrams):,})")

                if not line or line == '\\data\\' or line.startswith('ngram '):
                    continue
                if line == '\\end\\':
                    break
                if line == '\\1-grams:':
                    section = 1; continue
                if line == '\\2-grams:':
                    section = 2; continue
                if line == '\\3-grams:':
                    section = 3; continue

                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                try:
                    lp = float(parts[0])
                except ValueError:
                    continue

                if section == 1:
                    word = parts[1]
                    bow  = float(parts[2]) if len(parts) > 2 else 0.0
                    self.unigrams[word] = (lp, bow)
                    if word == '<unk>':
                        self.unk_logp = lp
                elif section == 2:
                    key = parts[1]   # "w1 w2" already space-joined in ARPA
                    bow = float(parts[2]) if len(parts) > 2 else 0.0
                    self.bigrams[key] = (lp, bow)
                elif section == 3:
                    key = parts[1]   # "w1 w2 w3"
                    self.trigrams[key] = lp

        print(f"Loaded: {len(self.unigrams):,} unigrams, "
              f"{len(self.bigrams):,} bigrams, "
              f"{len(self.trigrams):,} trigrams")

    def _logp_trigram(self, w1: str, w2: str, w3: str) -> float:
        """Log10 P(w3 | w1, w2) with Katz back-off."""
        key3 = f"{w1} {w2} {w3}"
        if key3 in self.trigrams:
            return self.trigrams[key3]

        # Back off to bigram
        key2_ctx = f"{w1} {w2}"
        bow2 = self.bigrams[key2_ctx][1] if key2_ctx in self.bigrams else 0.0
        return bow2 + self._logp_bigram(w2, w3)

    def _logp_bigram(self, w1: str, w2: str) -> float:
        """Log10 P(w2 | w1) with back-off."""
        key2 = f"{w1} {w2}"
        if key2 in self.bigrams:
            return self.bigrams[key2][0]

        bow1 = self.unigrams[w1][1] if w1 in self.unigrams else 0.0
        return bow1 + self._logp_unigram(w2)

    def _logp_unigram(self, w: str) -> float:
        if w in self.unigrams:
            return self.unigrams[w][0]
        return self.unk_logp

    def sentence_surprisal(self, words: List[str]) -> float:
        """
        Sentence surprisal in nats = -sum log P(w_i | context_i).
        Uses BOS/EOS markers and trigram context.
        """
        tokens = ['<s>', '<s>'] + [w.lower() for w in words] + ['</s>']
        total_log10 = 0.0
        for i in range(2, len(tokens)):
            total_log10 += self._logp_trigram(tokens[i-2], tokens[i-1], tokens[i])
        return -total_log10 * self.LN10   # convert to nats


# ── KenLM-backed Trigram Surprisal ────────────────────────────────────────────

class TrigramSurprisal:
    '''
    Trigram surprisal using KenLM.
    Paper used SRILM — KenLM is identical algorithm, open-source.
    '''

    def __init__(self, model_path: str):
        try:
            import kenlm
            self.kenlm = kenlm
        except ImportError:
            raise ImportError(
                'Install kenlm: pip install '
                'https://github.com/kpu/kenlm/archive/master.zip'
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}')
        self.model      = self.kenlm.Model(model_path)
        self.model_path = model_path
        print(f'Loaded: {model_path}')

    def sentence_surprisal(self, tokens: List[Dict]) -> float:
        '''
        Sentence surprisal = -log P(sentence) in nats.
        KenLM returns log10 prob, convert: -log10_p * log(10).
        '''
        words     = ' '.join(t['form'] for t in tokens)
        log10_p   = self.model.score(words, bos=True, eos=True)
        return -log10_p * math.log(10)

    def text_surprisal(self, text: str) -> float:
        log10_p = self.model.score(text, bos=True, eos=True)
        return -log10_p * math.log(10)

    def add_to_variants(
        self,
        variants_df: pd.DataFrame,
        column_name: str = 'surprisal_trigram'
    ) -> pd.DataFrame:
        '''
        Add trigram surprisal column to variants DataFrame.
        Requires 'tokens_json' column from generator.py output.
        '''
        df         = variants_df.copy()
        surprisals = []
        errors     = 0

        print(f'Computing surprisal for {len(df):,} variants...')

        for i, (_, row) in enumerate(df.iterrows()):
            try:
                tokens = json.loads(row['tokens_json'])
                s      = self.sentence_surprisal(tokens)
            except Exception:
                s      = 0.0
                errors += 1
            surprisals.append(s)

            if (i + 1) % 5000 == 0:
                print(f'  {i+1:>6,}/{len(df):,} '
                      f'({(i+1)/len(df)*100:.1f}%)')

        df[column_name] = surprisals
        if errors:
            print(f'  Errors (set to 0.0): {errors}')
        return df

    def verify(self):
        sov = [{'form':'राम'},{'form':'ने'},
               {'form':'किताब'},{'form':'पढ़ी'}]
        osv = [{'form':'किताब'},{'form':'राम'},
               {'form':'ने'},{'form':'पढ़ी'}]
        ss  = self.sentence_surprisal(sov)
        os_ = self.sentence_surprisal(osv)
        print(f'SOV surprisal : {ss:.3f}')
        print(f'OSV surprisal : {os_:.3f}')
        print(f'SOV < OSV     : {ss < os_}  (expected True)')
        return ss < os_

    def save_variant_scores(
        self,
        variants_df : pd.DataFrame,
        save_path   : str,
        column_name : str = 'surprisal_trigram'
    ) -> dict:
        """
        Score all variants and save to JSON keyed by variant_id.
        Run this on Colab, then download the JSON for local use.

        Usage on Colab:
            trigram = TrigramSurprisal(model_path)
            trigram.save_variant_scores(variants_df, save_path)
        """
        scores = {}
        errors = 0
        print(f'Scoring {len(variants_df):,} variants...')

        for i, (_, row) in enumerate(variants_df.iterrows()):
            vid = row['variant_id']
            try:
                tokens   = json.loads(row['tokens_json'])
                scores[vid] = self.sentence_surprisal(tokens)
            except Exception:
                scores[vid] = 0.0
                errors += 1

            if (i + 1) % 5000 == 0:
                # Save checkpoint every 5000
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(scores, f, ensure_ascii=False)
                print(f'  {i+1:>6,}/{len(variants_df):,}  checkpoint saved')

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False)

        real = sum(1 for v in scores.values() if v > 0)
        print(f'Saved {len(scores):,} scores ({real:,} real, {errors} errors)')
        print(f'Path: {save_path}')
        return scores

    @staticmethod
    def load_scores(
        variants_df : pd.DataFrame,
        json_path   : str,
        column_name : str = 'surprisal_trigram'
    ) -> pd.DataFrame:
        """
        Load pre-computed trigram scores from JSON (saved by save_variant_scores).
        Use this locally instead of running KenLM.

        json_path: models/trigram/trigram_surprisals.json
        """
        with open(json_path, encoding='utf-8') as f:
            scores = json.load(f)

        df = variants_df.copy()
        df[column_name] = df['variant_id'].map(lambda vid: scores.get(vid, 0.0))
        real = (df[column_name] > 0).sum()
        print(f'Loaded trigram scores: {real:,}/{len(df):,} non-zero')
        return df

    def print_stats(self, df: pd.DataFrame,
                    col: str = 'surprisal_trigram'):
        c = df[df['is_original']==1][col]
        v = df[df['is_original']==0][col]
        print('\n── SURPRISAL STATS ───────────────────────')
        print(f'  Corpus   mean={c.mean():.3f} std={c.std():.3f} n={len(c):,}')
        print(f'  Variants mean={v.mean():.3f} std={v.std():.3f} n={len(v):,}')
        print(f'  Corpus < Variants : {c.mean() < v.mean()} (expected True)')
        print('\n  By order type:')
        for order, mean_s in df.groupby('order_type')[col].mean().items():
            print(f'    {order:<14} : {mean_s:.3f}')
        print('──────────────────────────────────────────')
