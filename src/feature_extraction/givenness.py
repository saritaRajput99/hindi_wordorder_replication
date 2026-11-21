"""
Information Status (IS) / Givenness Scorer
 Section: Information Status

Paper definition:
  - Examine subject and object phrases in the target sentence
  - Check if any CONTENT WORDS within those phrases appear
    in the PRECEDING context sentence
  - If yes  → tag constituent as GIVEN
  - If no   → tag constituent as NEW
  - Also tag as GIVEN if the constituent HEAD is a pronoun

IS score per sentence (based on LINEAR surface order):
  Given-New  order → +1   (first argument in sentence is Given, second is New)
  New-Given  order → -1   (first argument in sentence is New, second is Given)
  Given-Given      →  0
  New-New          →  0

For DO-fronted (DOSV): the fronted given-object appears first → IS = +1.
For canonical (SDOV): the new-subject appears first → IS = -1.
This gives IS_corpus > IS_variant, consistent with the paper's positive coefficient (+0.26).

NOTE: IS score is ORDER-BASED (linear position), NOT role-based.
Do NOT confuse with a role-based definition where +1 would mean
"subject is given" — that would give the opposite sign for non-canonical orders.

Content words = NOUN, PROPN, VERB, ADJ, ADV
(excludes function words: ADP, AUX, PART, PUNCT, DET, CCONJ, SCONJ)
"""

import json
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# POS tags considered content words (paper convention)
CONTENT_POS = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'}

# Hindi pronoun forms — marked as Given regardless of context
HINDI_PRONOUNS = {
    # 3rd person
    'वह', 'वो', 'उसने', 'उसको', 'उसे', 'उसका', 'उसकी', 'उसके',
    'वे', 'उन्होंने', 'उन्हें', 'उनको', 'उनका', 'उनकी', 'उनके',
    # 1st person
    'मैं', 'मुझे', 'मुझको', 'मेरा', 'मेरी', 'मेरे', 'हम',
    'हमें', 'हमको', 'हमारा', 'हमारी', 'हमारे',
    # 2nd person
    'तुम', 'तुम्हें', 'तुम्हारा', 'आप', 'आपको', 'आपका',
    # Demonstrative
    'यह', 'इसने', 'इसे', 'इसको', 'इसका', 'इसकी', 'इसके',
    'ये', 'इन्होंने', 'इन्हें',
    # Relative/interrogative
    'जो', 'जिसने', 'जिसे', 'कौन', 'क्या',
}


class GivennessScorer:

    def __init__(self):
        self.subject_deps = {'nsubj', 'nsubj:pass'}
        self.object_deps  = {'obj', 'iobj'}

    # ─────────────────────────────────────────
    # CONSTITUENT EXTRACTION
    # ─────────────────────────────────────────

    def get_root(self, tokens: List[Dict]) -> Optional[Dict]:
        return next((t for t in tokens if t['head'] == 0), None)

    def get_constituent_tokens(
        self,
        head_token : Dict,
        all_tokens : List[Dict]
    ) -> List[Dict]:
        """
        Recursively get all tokens in the subtree rooted at head_token.
        Same DFS as generator.py — preserves full NP.
        """
        subtree = [head_token]
        for t in all_tokens:
            if t['head'] == head_token['id']:
                subtree.extend(self.get_constituent_tokens(t, all_tokens))
        return subtree

    def get_subject_object(
        self,
        tokens: List[Dict]
    ) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """
        Extract subject and object constituents (full subtrees).
        Returns (subject_tokens, object_tokens) or (None, None).
        """
        root = self.get_root(tokens)
        if not root:
            return None, None

        subj_head = next(
            (t for t in tokens
             if t['head'] == root['id']
             and t['deprel'] in self.subject_deps),
            None
        )
        obj_head = next(
            (t for t in tokens
             if t['head'] == root['id']
             and t['deprel'] in self.object_deps),
            None
        )

        subj_tokens = self.get_constituent_tokens(subj_head, tokens) \
                      if subj_head else None
        obj_tokens  = self.get_constituent_tokens(obj_head,  tokens) \
                      if obj_head  else None

        return subj_tokens, obj_tokens

    # ─────────────────────────────────────────
    # GIVEN / NEW CLASSIFICATION
    # ─────────────────────────────────────────

    def get_content_words(self, tokens: List[Dict]) -> set:
        """
        Extract content word forms from a token list.
        Lowercased for case-insensitive matching.
        """
        return {
            t['form'].lower()
            for t in tokens
            if t.get('upostag', '') in CONTENT_POS
        }

    def is_pronoun_head(self, tokens: List[Dict]) -> bool:
        """
        Returns True if the HEAD token of this constituent
        is a pronoun (PRON POS or in Hindi pronoun list).
        Paper: pronoun head = Given regardless of context.

        HEAD is the token in the subtree whose own head is OUTSIDE
        the subtree (i.e. it attaches to the root verb). We identify
        it as the token with a deprel in subject/object deps, or else
        the token with the smallest id (leftmost = most likely head).
        """
        if not tokens:
            return False
        token_ids = {t['id'] for t in tokens}
        # Find the constituent head: the token whose 'head' points outside subtree
        head_tok = None
        for t in tokens:
            if t['head'] not in token_ids:
                head_tok = t
                break
        if head_tok is None:
            head_tok = tokens[0]
        return (head_tok.get('upostag', '') == 'PRON' or
                head_tok['form'] in HINDI_PRONOUNS)

    def classify_constituent(
        self,
        constituent_tokens : List[Dict],
        context_forms      : set
    ) -> str:
        """
        Classify a constituent as GIVEN or NEW.

        Rules (paper):
          1. If head is a pronoun → GIVEN
          2. If any content word in constituent appears
             in the preceding context sentence → GIVEN
          3. Otherwise → NEW

        Args:
            constituent_tokens : tokens of the constituent
            context_forms      : set of content word forms
                                 from the preceding sentence

        Returns:
            'given' or 'new'
        """
        # Rule 1 — pronoun head
        if self.is_pronoun_head(constituent_tokens):
            return 'given'

        # Rule 2 — lexical overlap with context
        constituent_content = self.get_content_words(constituent_tokens)
        if constituent_content & context_forms:
            return 'given'

        # Rule 3 — no overlap
        return 'new'

    # ─────────────────────────────────────────
    # IS SCORE COMPUTATION
    # ─────────────────────────────────────────

    def compute_is_score(
        self,
        tokens         : List[Dict],
        context_tokens : Optional[List[Dict]]
    ) -> int:
        """
        Compute IS score for a sentence given its context.

        IS score encoding (paper) — based on LINEAR surface order:
          Given-New  → +1  (first-in-sentence=given, second=new)
          New-Given  → -1  (first-in-sentence=new,   second=given)
          Given-Given→  0
          New-New    →  0

        For DO-fronted (DOSV): given-object is first → +1 (corpus).
        For canonical (SDOV):  new-subject  is first → -1 (variant).
        → Pairwise diff positive → matches paper's +0.26 coefficient.

        Args:
            tokens         : target sentence tokens
            context_tokens : preceding sentence tokens (or None)

        Returns:
            int: -1, 0, or +1
        """
        subj_tokens, obj_tokens = self.get_subject_object(tokens)

        # Cannot compute IS without both S and O
        if subj_tokens is None or obj_tokens is None:
            return 0

        # Get content words from context sentence
        context_forms = set()
        if context_tokens:
            context_forms = self.get_content_words(context_tokens)

        # Classify each constituent
        subj_status = self.classify_constituent(subj_tokens, context_forms)
        obj_status  = self.classify_constituent(obj_tokens,  context_forms)

        # Determine LINEAR order: which constituent appears first in the sentence?
        # IS score depends on the order of FIRST vs SECOND argument, not
        # grammatical role. For OSV variants, object appears before subject.
        subj_head_id = min(t['id'] for t in subj_tokens)
        obj_head_id  = min(t['id'] for t in obj_tokens)

        if subj_head_id < obj_head_id:
            # Subject before Object (canonical SOV / SVO)
            first_status  = subj_status
            second_status = obj_status
        else:
            # Object before Subject (non-canonical OSV)
            first_status  = obj_status
            second_status = subj_status

        # IS score: +1 if first=Given and second=New (Given-New order)
        if first_status == 'given' and second_status == 'new':
            return +1   # Given-New — canonical information order
        elif first_status == 'new' and second_status == 'given':
            return -1   # New-Given — non-canonical information order
        else:
            return 0    # Given-Given or New-New

    # ─────────────────────────────────────────
    # BATCH PROCESSING
    # ─────────────────────────────────────────

    def add_to_variants(
        self,
        variants_df    : pd.DataFrame,
        sentences      : List[List[Dict]],
        metadata_df    : pd.DataFrame,
        column_name    : str = 'is_score'
    ) -> pd.DataFrame:
        """
        Add IS score column to variants DataFrame.

        For each variant:
          - Identify the reference sentence (sent_id)
          - Find the preceding sentence in the corpus (context)
          - Compute IS score using paper rules

        Args:
            variants_df : DataFrame with 'reference_id' and 'tokens_json'
            sentences   : list of filtered sentence token lists
                          (same order as metadata_df)
            metadata_df : metadata DataFrame with 'sent_id' column
            column_name : output column name

        Returns:
            DataFrame with IS score column added
        """
        df = variants_df.copy()

        # Build sent_id → index mapping for fast lookup
        sent_id_to_idx = {
            row['sent_id']: i
            for i, row in metadata_df.iterrows()
        }

        is_scores = []
        errors    = 0

        print(f"Computing IS scores for {len(df):,} variants...")

        for i, (_, row) in enumerate(df.iterrows()):
            try:
                # Get variant tokens
                tokens = json.loads(row['tokens_json'])

                # Get index of reference sentence
                ref_id  = row['reference_id']
                ref_idx = sent_id_to_idx.get(ref_id, None)

                # Get context = preceding sentence only (paper: "preceding context sentence")
                # Paper checks if content words from S/O appear in the immediately
                # preceding sentence, not a wider window.
                context_tokens = None
                if ref_idx is not None and ref_idx > 0:
                    context_tokens = sentences[ref_idx - 1]

                # Compute IS score
                score = self.compute_is_score(tokens, context_tokens)

            except Exception:
                score  = 0
                errors += 1

            is_scores.append(score)

            if (i + 1) % 5000 == 0:
                pct = (i + 1) / len(df) * 100
                print(f"  {i+1:>6,}/{len(df):,}  ({pct:.1f}%)")

        df[column_name] = is_scores

        if errors:
            print(f"  Errors (set to 0): {errors}")

        return df

    # ─────────────────────────────────────────
    # VERIFICATION
    # ─────────────────────────────────────────

    def verify(self):
        """
        Test with a hand-crafted example matching the paper's Example 2.

        Paper example:
          Context: Air India and Indian Airlines services have been notified
          Target:  This notice [DO=given] has been issued by Director-General [S=given]
          Expected IS score: 0 (Given-Given)
        """
        # Minimal mock tokens
        def tok(id, form, upos, head, deprel):
            return {'id': id, 'form': form, 'lemma': form,
                    'upostag': upos, 'head': head, 'deprel': deprel,
                    'feats': '_', 'misc': '_'}

        # Context: "राम ने किताब खरीदी" (Ram bought a book)
        context = [
            tok(1, 'राम',    'PROPN', 4, 'nsubj'),
            tok(2, 'ने',     'ADP',   1, 'case'),
            tok(3, 'किताब',  'NOUN',  4, 'obj'),
            tok(4, 'खरीदी', 'VERB',  0, 'root'),
        ]

        # Target SOV: "राम ने किताब पढ़ी" — राम=given(overlap), किताब=given(overlap)
        target_sov = [
            tok(1, 'राम',   'PROPN', 3, 'nsubj'),
            tok(2, 'किताब', 'NOUN',  3, 'obj'),
            tok(3, 'पढ़ी',  'VERB',  0, 'root'),
        ]

        # Target with new object: "राम ने अखबार पढ़ा" — राम=given, अखबार=new
        target_given_new = [
            tok(1, 'राम',    'PROPN', 3, 'nsubj'),
            tok(2, 'अखबार', 'NOUN',  3, 'obj'),
            tok(3, 'पढ़ा',  'VERB',  0, 'root'),
        ]

        # Target with pronoun subject + NEW object (not in context)
        # subj=given (pronoun), obj=new (अखबार not in context)
        target_pronoun = [
            tok(1, 'वह',    'PRON', 3, 'nsubj'),
            tok(2, 'अखबार', 'NOUN', 3, 'obj'),   # not in context → new
            tok(3, 'पढ़ा',  'VERB', 0, 'root'),
        ]

        # No context
        target_no_context = [
            tok(1, 'सीता',  'PROPN', 3, 'nsubj'),
            tok(2, 'खाना',  'NOUN',  3, 'obj'),
            tok(3, 'बनाती', 'VERB',  0, 'root'),
        ]

        print("IS Score verification:\n")

        score1 = self.compute_is_score(target_sov, context)
        print(f"  Given-Given (both in context) : {score1}  (expected  0)")

        score2 = self.compute_is_score(target_given_new, context)
        print(f"  Given-New   (subj in context) : {score2}  (expected +1)")

        score3 = self.compute_is_score(target_pronoun, context)
        print(f"  Pronoun subj + new obj        : {score3}  (expected +1)")

        score4 = self.compute_is_score(target_no_context, None)
        print(f"  No context (new-new)          : {score4}  (expected  0)")

        all_pass = (score1 == 0 and score2 == 1 and
                    score3 == 1 and score4 == 0)
        print(f"\n  All tests passed: {all_pass}")
        return all_pass

    # ─────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────

    def print_stats(
        self,
        df  : pd.DataFrame,
        col : str = 'is_score'
    ):
        """Print IS score statistics."""
        corpus  = df[df['is_original'] == 1][col]
        variant = df[df['is_original'] == 0][col]

        print('\n── IS SCORE STATS ────────────────────────')
        print(f'  Score distribution:')
        print(f'    +1 (Given-New)  : {(df[col]==1).sum():>6,}')
        print(f'     0 (GG or NN)   : {(df[col]==0).sum():>6,}')
        print(f'    -1 (New-Given)  : {(df[col]==-1).sum():>6,}')
        print(f'\n  Corpus   mean={corpus.mean():.3f}  '
              f'n={len(corpus):,}')
        print(f'  Variants mean={variant.mean():.3f}  '
              f'n={len(variant):,}')
        print(f'\n  Corpus IS > Variant IS : '
              f'{corpus.mean() > variant.mean()}')
        print(f'  Expected               : True')
        print(f'  (corpus follows Given-Before-New more than variants)')

        print(f'\n  By order type (corpus sentences only):')
        corp_df = df[df['is_original'] == 1]
        for order, grp in corp_df.groupby('order_type'):
            mean_is = grp[col].mean()
            print(f'    {order:<14} : {mean_is:+.3f}')
        print('──────────────────────────────────────────')
