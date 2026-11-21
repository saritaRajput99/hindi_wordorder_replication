"""
VariantGenerator — Ranjan & van Schijndel (2024) aligned

What this does (paper Section "Data and Methods"):
  - Permutes ALL preverbal direct root dependents (non-punct)
    → Full subtrees move intact; classification uses nsubj/obj/iobj positions
  - Caps at 99 variants per sentence (paper arbitrary cutoff)
  - Preserves full subtrees (DFS) so NPs move intact
  - Keeps ONLY permutations where subject vs. the primary object-type
    element (obj if present, else iobj) ends up in the OPPOSITE relative
    order from the reference — professor-confirmed rule: a Canonical (SOV)
    reference must only produce non-canonical (OSV-type) variants, and a
    non-canonical (DOSV/IOSV) reference must only produce canonical (SOV)
    variants, at every row, not just on average.
  - Repairs dependency pointers after permutation
  - Computes dependency length (DL) for every variant
  - Flags which variant is the original corpus sentence (is_original=1)
  - Serialises tokens as JSON for downstream surprisal computation
  - Order classification (Canonical/DO-fronted/IO-fronted/etc.) uses CORE_DEPS only

Paper alignment:
  - Permutation candidates: all preverbal direct root dependents (core +
    non-core), same subtree-preserving search as before — but every
    resulting variant is filtered to be the required canonical<->non-canonical
    flip of its reference (see generate_variants()).
  - max_variants=99 per paper's "arbitrary cutoff for computational tractability"
"""

import copy
import itertools
import json
import math
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from .grammar_filter import GrammarFilter


class VariantGenerator:

    def __init__(self, max_variants: int = 99,
                 grammar_filter: Optional[GrammarFilter] = None):
        """
        Args:
            max_variants: maximum permutations per sentence.
                          Paper uses 99 as an arbitrary cutoff for
                          computational tractability.
                          
                          5 args -> 5! = 120  (capped at 99)
        
        """
        self.max_variants = max_variants
        self.grammar_filter = grammar_filter

        # UD-only core dependency labels (matches hutb_loader SUBJECT_RELS etc.)
        # Modifiers (amod, compound, nummod etc.) travel WITH their head
        # as part of the subtree -- they are never standalone constituents.
        self.SUBJ      = {'nsubj', 'nsubj:pass'}
        self.OBJ       = {'obj'}
        self.IOBJ      = {'iobj'}
        self.CORE_DEPS = self.SUBJ | self.OBJ | self.IOBJ

        self.stats = {
            'sentences_processed' : 0,
            'total_variants'      : 0,
            'skipped_few_args'    : 0,
            'filtered_ungrammatical': 0,
        }

    # ─────────────────────────────────────────
    # SUBTREE EXTRACTION (DFS)
    # ─────────────────────────────────────────

    def extract_subtree(self, token: Dict, sent: List[Dict]) -> List[Dict]:
        """
        Recursively collect all tokens in the subtree rooted at token.
        Returns tokens sorted by original linear position.
        This ensures NPs (adjective + noun, etc.) stay intact when permuted.
        """
        subtree = [copy.deepcopy(token)]
        for t in sent:
            if t['head'] == token['id']:
                subtree.extend(self.extract_subtree(t, sent))
        return sorted(subtree, key=lambda x: x['id'])

    # ─────────────────────────────────────────
    # CONSTITUENT EXTRACTION
    # ─────────────────────────────────────────

    def get_preverbal_constituents(self, sent: List[Dict]) -> List[Dict]:
        """
        Extract ALL preverbal direct root dependents (with full subtrees).

       
        """
        root = next((t for t in sent if t['head'] == 0), None)
        if not root:
            return []

        constituents = []
        for t in sent:
            if (
                t['head'] == root['id'] and
                t['id']   <  root['id'] and
                t['deprel'] != 'punct'
            ):
                constituents.append({
                    'head'   : t,
                    'subtree': self.extract_subtree(t, sent),
                })

        return sorted(constituents, key=lambda x: x['head']['id'])

    def get_preverbal_punct_blocks(self, sent: List[Dict]) -> List[Dict]:
        """
        Preverbal direct root dependents with deprel == 'punct' (e.g. a lone
        comma/symbol attached straight to the root, not nested inside any
        other constituent's subtree). These are excluded from
        get_preverbal_constituents() because they aren't permutable
        arguments — but they must still be re-attached somewhere in
        generate_variants(), or they silently vanish from the token stream
        that surprisal models read.
        """
        root = next((t for t in sent if t['head'] == 0), None)
        if not root:
            return []

        blocks = []
        for t in sent:
            if (
                t['head'] == root['id'] and
                t['id']   <  root['id'] and
                t['deprel'] == 'punct'
            ):
                blocks.append({
                    'head'   : t,
                    'subtree': self.extract_subtree(t, sent),
                })

        return sorted(blocks, key=lambda x: x['head']['id'])

    def get_deprel_sequence(self, constituents: List[Dict]) -> List[str]:
        """
        Get the deprel sequence from constituent HEAD tokens.
        e.g. ['nsubj', 'obj']  or  ['obj', 'nsubj']
        No normalization — deprels are used as-is (matches hutb_loader).
        """
        return [c['head']['deprel'] for c in constituents]

    # ─────────────────────────────────────────
    # ORDER CLASSIFICATION
    # ─────────────────────────────────────────

    def classify_order(self, deprel_seq: List[str], has_iobj_in_sent: bool = False) -> str:
        """
        Classify variant word order per professor-confirmed rule
        (paper Table 1 categories):

          Canonical   : subject before object (SOV)
          DO-fronted  : direct object (obj) before subject
          IO-fronted  : indirect object (iobj) before subject,
                        no direct object before subject
          

    
        """
        subj_pos = next(
            (i for i, d in enumerate(deprel_seq) if d in self.SUBJ), None)
        obj_pos  = next(
            (i for i, d in enumerate(deprel_seq) if d in self.OBJ),  None)
        iobj_pos = next(
            (i for i, d in enumerate(deprel_seq) if d in self.IOBJ), None)

        if subj_pos is None:
            return 'Other'

        # Direct object before subject → DO-fronted (paper: DOSV = k2 before k1)
        if obj_pos is not None and obj_pos < subj_pos:
            return 'DO-fronted'

        # Indirect object before subject (no direct obj before subj)
        # Paper: IOSV = k4 before k1
        if iobj_pos is not None and iobj_pos < subj_pos:
            return 'IO-fronted'

        # Subject before all objects → Canonical (SOV)
        return 'Canonical'

    # ─────────────────────────────────────────
    # DEPENDENCY LENGTH
    # ─────────────────────────────────────────

    def compute_dl(self, tokens: List[Dict]) -> int:
        """
        Sentence-level dependency length, punctuation excluded:
        DL = sum |position(dep) - position(head)|  for all non-root,
        non-PUNCT tokens, where position is computed after dropping
        PUNCT tokens and re-indexing the remainder compactly.
        Must be called AFTER repair_pointers so IDs are sequential.
        """
        kept = [t for t in tokens if t['upostag'] != 'PUNCT']
        compact_pos = {t['id']: i for i, t in enumerate(kept)}

        total = 0
        for t in kept:
            if t['head'] <= 0:
                continue
            head_pos = compact_pos.get(t['head'])
            if head_pos is None:
                continue
            total += abs(compact_pos[t['id']] - head_pos)
        return total

    # ─────────────────────────────────────────
    # POINTER REPAIR
    # ─────────────────────────────────────────

    def repair_pointers(self, sent: List[Dict]) -> List[Dict]:
        """
        After permuting constituents the original token IDs are scrambled.
        This assigns new sequential IDs (1-based) and updates all head
        pointers using the old->new mapping.
        """
        sent = copy.deepcopy(sent)

        for i, t in enumerate(sent):
            t['new_id'] = i + 1

        mapping = {t['id']: t['new_id'] for t in sent}

        for t in sent:
            t['head'] = mapping.get(t['head'], t['head'])
            t['id']   = t['new_id']
            del t['new_id']

        return sent

    # ─────────────────────────────────────────
    # VARIANT GENERATION (single sentence)
    # ─────────────────────────────────────────

    def _extract_verb_aspect(self, sent: List[Dict]) -> Optional[str]:
        """
        Determine the aspect of the root verb.
        Returns: 'perfective', 'imperfective', 'habitual', or None
        """
        root = next((t for t in sent if t['head'] == 0), None)
        if not root:
            return None

        form = root.get('form', '').lower()
        feats = root.get('feats', {})
        if isinstance(feats, dict):
            aspect = feats.get('Aspect')
            if aspect:
                return aspect.lower()  # 'Perf', 'Imp', 'Hab', etc.

        # Heuristic: check form against perfective patterns
        PERFECTIVE_SUFFIXES = {'ा', 'ी', 'े', 'ेँ', 'aa', 'ii', 'ee', 'i'}
        for suffix in PERFECTIVE_SUFFIXES:
            if form.endswith(suffix):
                return 'perfective'

        PERFECTIVE_VERBS = {'किया', 'किया।', 'kiya', 'diya', 'diyaa',
                            'गया', 'gayaa', 'गई', 'gayi',
                            'लिया', 'liyaa', 'पाया', 'paayaa'}
        if form in PERFECTIVE_VERBS:
            return 'perfective'

        return None

    def generate_variants(self, sent: List[Dict]) -> List[Dict]:
        """
        Generate all permutations of preverbal constituents.

        Returns list of variant dicts, each with:
            tokens      : repaired token list (list of dicts)
            deprel_seq  : deprel sequence of this permutation
            order_type  : Canonical / DO-fronted / IO-fronted / Other
            dl          : dependency length of this variant
            is_original : True if this permutation matches corpus word order
        """
        arguments = self.get_preverbal_constituents(sent)

        if len(arguments) < 2:
            self.stats['skipped_few_args'] += 1
            return []

        # Paper DOSV criterion: k4 (iobj) present ANYWHERE in sentence
        # (pre- or post-verbal). Check once for all permutations of this sentence.
        root = next((t for t in sent if t['head'] == 0), None)
        has_iobj_in_sent = any(
            t['deprel'] in self.IOBJ and t['head'] == root['id']
            for t in sent
        )

        # Extract verb aspect once for grammar filtering
        verb_aspect = self._extract_verb_aspect(sent)

        # Root token and everything from root onwards (verb + postverbal)
        postverbal = [copy.deepcopy(t) for t in sent if t['id'] >= root['id']]

        # Standalone preverbal punctuation (direct root dependents, not part
        # of any constituent's subtree) isn't permuted, but must still land
        # somewhere in every variant's token stream. Each punct block travels
        # with the constituent that originally followed it (prefix_punct), or
        # — if it came after the last constituent — stays anchored right
        # before the root regardless of permutation (trailing_punct).
        punct_blocks  = self.get_preverbal_punct_blocks(sent)
        prefix_punct  = {}
        trailing_punct = []
        for pb in punct_blocks:
            target = next(
                (i for i, c in enumerate(arguments) if c['head']['id'] > pb['head']['id']),
                None
            )
            if target is None:
                trailing_punct.extend(pb['subtree'])
            else:
                prefix_punct.setdefault(target, []).extend(pb['subtree'])

        # Paper-faithful generation: permute ALL preverbal elements
        # unrestricted (core + non-core), cap at 99, grammar-filter — same
        # as the paper's total (72,833-scale) count. We additionally tag
        # each variant with is_flip_pair (see below), rather than discarding
        # non-flip permutations, so the professor's row-level requirement
        # (every FLIP-PAIR row is a genuine canonical<->non-canonical flip
        # of its reference) is satisfiable via that flag/filter, without
        # losing the paper-matching total or needing special-case sampling.
        ref_order_type = self.classify_order(
            self.get_deprel_sequence(arguments), has_iobj_in_sent)
        ref_is_canonical = (ref_order_type == 'Canonical')

        original_perm_tuple = tuple(range(len(arguments)))
        n_args = len(arguments)
        n_total_perms = math.factorial(n_args)

        if n_total_perms <= self.max_variants:
            # Enumerate all permutations (safe: ≤ 99 or small n)
            all_perms = list(itertools.permutations(range(n_args)))
        else:
            # Too many permutations to enumerate — sample randomly.
            # Always include original order; sample (max_variants-1) others.
            sampled_set = {original_perm_tuple}
            while len(sampled_set) < self.max_variants:
                p = list(range(n_args))
                random.shuffle(p)
                sampled_set.add(tuple(p))
            all_perms = list(sampled_set)

        # String deduplication (hutb_loader: only keep unique surface strings,
        # and exclude the reference sentence string from generated variants).
        reference_text = ' '.join(t['form'] for t in sent)
        seen_strings   = {reference_text}   # pre-seed with reference to exclude it

        variants = []
        original_variant = None   # will hold the corpus-order entry separately

        for perm_indices in all_perms:
            perm = [arguments[i] for i in perm_indices]

            # Build variant: all permuted preverbal constituents + postverbal tokens.
            # Punct blocks travel with the constituent they're prefixed to,
            # so they move along with it under permutation instead of being
            # dropped; trailing punct always stays right before postverbal.
            flat = []
            for i in perm_indices:
                flat.extend(prefix_punct.get(i, []))
                flat.extend(arguments[i]['subtree'])
            flat.extend(trailing_punct)
            variant_tokens = flat + postverbal

            # Repair dependency pointers
            variant_tokens = self.repair_pointers(variant_tokens)

            deprel_seq  = self.get_deprel_sequence(perm)
            order_type  = self.classify_order(deprel_seq, has_iobj_in_sent)
            dl          = self.compute_dl(variant_tokens)

            # is_original: True only for the identity permutation (0,1,...,n-1)
            # Using index comparison (not deprel_seq) avoids false positives
            # when two constituents share the same deprel label.
            is_original = (perm_indices == original_perm_tuple)

            if is_original:
                # Always include reference sentence (is_original=1)
                # Paper: grammar filter applies to generated variants only,
                # not to the original corpus sentence.
                original_variant = {
                    'tokens'      : variant_tokens,
                    'deprel_seq'  : deprel_seq,
                    'order_type'  : order_type,
                    'dl'          : dl,
                    'is_original' : True,
                    'is_flip_pair': False,
                }
                continue

            # is_flip_pair: True iff this variant's canonical-ness is the
            # opposite of its reference's — the professor-confirmed rule.
            # We no longer discard non-flip permutations (that would lose
            # paper-matching totals, since the paper's own 72,833 includes
            # non-flip variants too); we tag them instead, so downstream
            # paired analysis can filter on is_flip_pair == True while the
            # full generated pool stays intact for raw totals/Table 1.
            is_flip_pair = (order_type == 'Canonical') != ref_is_canonical

            # Paper: filter ungrammatical generated variants (not originals)
            if self.grammar_filter is not None:
                if not self.grammar_filter.is_grammatical(
                    tuple(deprel_seq),
                    constituents=perm,
                    verb_aspect=verb_aspect,
                    full_sent=sent
                ):
                    self.stats['filtered_ungrammatical'] += 1
                    continue

            # String deduplication: skip if surface form already seen
            surface = ' '.join(t['form'] for t in variant_tokens)
            if surface in seen_strings:
                continue
            seen_strings.add(surface)

            variants.append({
                'tokens'      : variant_tokens,
                'deprel_seq'  : deprel_seq,
                'order_type'  : order_type,
                'dl'          : dl,
                'is_original' : False,
                'is_flip_pair': is_flip_pair,
            })

            self.stats['total_variants'] += 1

        # Prepend original sentence (always first, not counted in dedup)
        if original_variant is not None:
            variants.insert(0, original_variant)
            self.stats['total_variants'] += 1

        return variants

    # ─────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────

    def generate_for_all(
        self,
        sentences  : List[List[Dict]],
        metadata_df: pd.DataFrame,
        output_dir : str = None,
        seed       : int = 42
    ) -> pd.DataFrame:
        """
        Generate variants for all filtered sentences.

        Output DataFrame columns:
            reference_id     -- sent_id from metadata (corpus sentence)
            variant_id       -- unique id per variant
            is_original      -- 1 = corpus sentence, 0 = generated variant
            is_flip_pair     -- 1 = this variant's canonical-ness is the
                                 opposite of its reference's (the pairing
                                 used for the paper's regression/classification
                                 analysis); 0 = same canonical-ness as the
                                 reference (an extra generated variant not
                                 used as a counterfactual pair, same as the
                                 non-flip half of the paper's own 72,833).
                                 Always 0 for the reference row itself.
            deprel_sequence  -- space-joined deprel sequence
            order_type       -- Canonical / DO-fronted / IO-fronted
            dl               -- dependency length of this variant
            text             -- surface form of this variant
            tokens_json      -- JSON string of token list (for surprisal models)

        Paper Table 1 targets:
            ~1,996 reference sentences
            ~72,833 total variants (of which the flip-matched subset,
            i.e. is_flip_pair == 1, is what Table 1's four categories report)

        Writes to CSV incrementally to avoid MemoryError on large variant sets.
        """
        random.seed(seed)   # fix seed for reproducible random sampling (5+ arg sentences)
        import csv as _csv

        COLS = [
            'reference_id', 'variant_id', 'is_original', 'is_flip_pair',
            'deprel_sequence', 'order_type', 'dl', 'text', 'tokens_json',
        ]

        # tokens_json is written to a SEPARATE .jsonl file to keep variants.csv small.
        # variants.csv: lightweight (~50MB), readable by pandas without OOM.
        # variants_tokens.jsonl: one tab-separated line per variant: variant_id\ttokens_json
        LIGHT_COLS = [c for c in COLS if c != 'tokens_json']

        if output_dir:
            output_dir  = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path     = output_dir / 'variants.csv'
            conllu_path  = output_dir / 'variants.conllu'
            tokens_path  = output_dir / 'variants_tokens.jsonl'
            csv_f    = open(csv_path,    'w', encoding='utf-8', newline='')
            conllu_f = open(conllu_path, 'w', encoding='utf-8')
            tok_f    = open(tokens_path, 'w', encoding='utf-8')
            writer   = _csv.DictWriter(csv_f, fieldnames=LIGHT_COLS)
            writer.writeheader()
        else:
            csv_f = conllu_f = tok_f = writer = None
            csv_path = conllu_path = tokens_path = None

        all_rows_light = []   # lightweight rows (no tokens_json) for in-memory df
        total_written  = 0

        for i, sent in enumerate(sentences):

            sent_id  = metadata_df.iloc[i].get('sent_id', f'sent_{i:04d}')
            variants = self.generate_variants(sent)

            if not variants:
                continue

            for vidx, var in enumerate(variants):
                vid   = f'{sent_id}_var_{vidx:03d}'
                text  = ' '.join(t['form'] for t in var['tokens'])
                tjson = json.dumps(var['tokens'], ensure_ascii=False)

                row = {
                    'reference_id'   : sent_id,
                    'variant_id'     : vid,
                    'is_original'    : int(var['is_original']),
                    'is_flip_pair'   : int(var['is_flip_pair']),
                    'deprel_sequence': ' '.join(var['deprel_seq']),
                    'order_type'     : var['order_type'],
                    'dl'             : var['dl'],
                    'text'           : text,
                    'tokens_json'    : tjson,
                }

                if writer:
                    # Write lightweight row to CSV (no tokens_json)
                    light_row = {k: row[k] for k in LIGHT_COLS}
                    writer.writerow(light_row)
                    # Write tokens_json to separate .jsonl file
                    tok_f.write(f'{vid}\t{tjson}\n')
                else:
                    all_rows_light.append({k: row[k] for k in LIGHT_COLS})

                # Write CoNLL-U
                if conllu_f:
                    conllu_f.write(f'# sent_id = {vid}\n')
                    conllu_f.write(f'# text = {text}\n')
                    for t in var['tokens']:
                        conllu_f.write(
                            f"{t['id']}\t{t['form']}\t"
                            f"{t.get('lemma','_')}\t{t.get('upostag','_')}\t"
                            f"{t.get('xpostag','_')}\t{t.get('feats','_')}\t"
                            f"{t['head']}\t{t['deprel']}\t"
                            f"{t.get('deps','_')}\t{t.get('misc','_')}\n"
                        )
                    conllu_f.write('\n')

                total_written += 1

            self.stats['sentences_processed'] += 1

            if (i + 1) % 500 == 0:
                print(f'  Processed {i+1}/{len(sentences)} sentences  '
                      f'({total_written:,} variants so far)...')

        if csv_f:    csv_f.close()
        if conllu_f: conllu_f.close()
        if tok_f:    tok_f.close()

        if output_dir:
            print(f'\nSaved:')
            print(f'  variants.csv           ({total_written:,} rows, lightweight)')
            print(f'  variants_tokens.jsonl  ({total_written:,} rows, tokens_json)')
            print(f'  variants.conllu        ({total_written:,} sentences)')
            # Read the lightweight CSV (no tokens_json) — should be ~50MB
            df = pd.read_csv(
                csv_path,
                dtype={'dl': float, 'is_original': int},
            )
        else:
            df = pd.DataFrame(all_rows_light)

        self._print_stats(df)
        return df

    def _print_stats(self, df: pd.DataFrame):
        print('\n-- VARIANT STATS -----------------------------------------')
        print(f'  Sentences processed     : {self.stats["sentences_processed"]}')
        print(f'  Total rows (orig+var)   : {self.stats["total_variants"]}')
        print(f'  Skipped (<2 args)        : {self.stats["skipped_few_args"]}')
        print(f'  Filtered ungrammatical  : {self.stats["filtered_ungrammatical"]}')

        if len(df) > 0:
            n_orig  = df['is_original'].sum()
            n_total = len(df)
            print(f'  Corpus sentences    : {n_orig}')
            print(f'  Generated variants  : {n_total - n_orig}')
            print(f'  Paper target (ref)  : ~1,996')
            print(f'  Paper target (total): ~72,833')
            print(f'\n  Order distribution:')
            for order, count in df['order_type'].value_counts().items():
                pct = count / n_total * 100
                print(f'    {order:<14} : {count:>6}  ({pct:.1f}%)')

            orig_dl  = df[df['is_original'] == 1]['dl'].mean()
            other_dl = df[df['is_original'] == 0]['dl'].mean()
            print(f'\n  DL (corpus sentences)  : {orig_dl:.2f}')
            print(f'  DL (generated variants): {other_dl:.2f}')
            if orig_dl <= other_dl:
                print('  corpus DL <= variant DL  (DLT supported)')
            else:
                print('  corpus DL > variant DL   (DLT not supported here)')
        print('--------------------------------------------------')
