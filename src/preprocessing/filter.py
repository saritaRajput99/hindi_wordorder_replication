"""
SentenceFilter 



"""

import os
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path


class SentenceFilter:

    def __init__(self):

        self.subject_deps         = {'nsubj', 'nsubj:pass'}
        self.object_deps          = {'obj'}
        self.indirect_object_deps = {'iobj'}

        self.allowed_core = (
            self.subject_deps |
            self.object_deps |
            self.indirect_object_deps
        )

        self.negation_words = {'नहीं', 'न', 'मत'}   # paper: exactly these three

        self.stats = {
            'total': 0,
            'passed': 0,
            'rejected': {
                'no_root': 0,
                'non_declarative': 0,   # interrogative or imperative
                'negation': 0,
                'invalid_ids': 0,
                'non_projective': 0,
                'no_verb_root': 0,
                'no_subj_obj': 0,
                'few_preverbal': 0,
                'so_not_preverbal': 0,  # S or O not preverbal — cannot permute
            }
        }

    # --------------------------------------------------
    # BASIC CHECKS
    # --------------------------------------------------
    def get_root(self, sentence):
        return next((t for t in sentence if t['head'] == 0), None)

    def is_declarative(self, sentence):
        # Exclude interrogative sentences
        if any(t['form'] in {'?', '؟'} for t in sentence):
            return False
        # Exclude imperative sentences (root verb has Mood=Imp)
        root = self.get_root(sentence)
        if root:
            feats = root.get('feats', {})
            mood  = feats.get('Mood') if isinstance(feats, dict) else (
                'Imp' if 'Mood=Imp' in str(feats) else None
            )
            if mood == 'Imp':
                return False
        return True

    def has_no_negation(self, sentence):
        return all(t['form'] not in self.negation_words for t in sentence)

    def has_valid_ids(self, sentence):
        return all(isinstance(t['id'], int) and isinstance(t['head'], int) for t in sentence)

    def is_projective(self, sentence):
        edges = [
            (min(t['id'], t['head']), max(t['id'], t['head']))
            for t in sentence if t['head'] != 0
        ]
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                a, b = edges[i]
                c, d = edges[j]
                if (a < c < b < d) or (c < a < d < b):
                    return False
        return True

    def root_is_valid(self, sentence):
        """Root must be a verb (VERB or AUX) — paper criterion."""
        root = self.get_root(sentence)
        return root is not None and root['upostag'] in {'VERB', 'AUX'}

    def norm_dep(self, t, root_id):
        """Treat preverbal, core-argument 'obl' with Case=Acc as 'obj' (HDTB quirk)."""
        dep = t['deprel']
        feats = t.get('feats', {})
        case = feats.get('Case') if isinstance(feats, dict) else (
            'Acc' if 'Case=Acc' in str(feats) else None
        )
        if (
            dep == 'obl'
            and case == 'Acc'
            and t['head'] == root_id
            and t['id'] < root_id
        ):
            return 'obj'
        return dep

    # --------------------------------------------------
    # PREVERBAL ARGUMENTS (with norm_dep)
    # --------------------------------------------------
    def preverbal_dependents(self, sentence):
        root = self.get_root(sentence)
        if not root:
            return []

        return sorted(
            [
                t for t in sentence
                if t['head'] == root['id']
                and t['id'] < root['id']
                and t['deprel'] in self.allowed_core
            ],
            key=lambda x: x['id']
        )

    # --------------------------------------------------
    # PAPER CRITERIA
    # --------------------------------------------------
    def has_subject_object(self, sentence):
        """
        Paper criterion (a): trees contain both well-defined subjects and objects.
        Check ALL direct root dependents (any position).
        Uses raw deprel labels — obj in HDTB is always the true direct object.
        """
        root = self.get_root(sentence)
        if not root:
            return False
        root_deps = [t for t in sentence if t['head'] == root['id']]
        has_subj = any(t['deprel'] in self.subject_deps for t in root_deps)
        has_obj  = any(t['deprel'] in (self.object_deps | self.indirect_object_deps)
                       for t in root_deps)
        return has_subj and has_obj

    def has_valid_preverbal(self, sentence):
        """
        Paper criterion (d): root has at least two preverbal dependents.
        Paper says 'preverbal dependents' — any type, not just core args.
        Count ALL non-punct direct root deps that appear before the root.
        """
        root = self.get_root(sentence)
        if not root:
            return False, 0
        pre = [
            t for t in sentence
            if t['head'] == root['id']
            and t['id'] < root['id']
            and t['deprel'] != 'punct'
        ]
        return len(pre) >= 2, len(pre)

    def has_preverbal_subj_and_obj(self, sentence):
        """
        Both subject AND object must appear as preverbal direct root dependents.
        Required for valid permutations — if S or O is postverbal it cannot be permuted.
        """
        preverbal = self.preverbal_dependents(sentence)
        has_subj = any(t['deprel'] in self.subject_deps for t in preverbal)
        has_obj  = any(t['deprel'] in (self.object_deps | self.indirect_object_deps)
                       for t in preverbal)
        return has_subj and has_obj

    def is_canonical(self, sentence):
        preverbal = self.preverbal_dependents(sentence)
        subj = next((t for t in preverbal if t['deprel'] in self.subject_deps), None)
        obj  = next((t for t in preverbal if t['deprel'] in self.object_deps), None)
        return subj and obj and subj['id'] < obj['id']

    # --------------------------------------------------
    # MAIN FILTER
    # --------------------------------------------------
    def filter_with_rejection_tracking(self, sentences, sent_ids=None):
        """
        sent_ids: optional list, same length/order as `sentences`, giving each
        sentence's original sent_id. When provided, the original id for every
        sentence that passes is returned in `filtered_ids`, aligned with
        `filtered` — this is what lets callers recover the TRUE source
        sent_id for a filtered sentence instead of guessing it from position.
        """
        filtered = []
        rejected = []
        filtered_ids = [] if sent_ids is not None else None

        self.stats['total'] = len(sentences)
        self.stats['passed'] = 0

        for idx, sent in enumerate(sentences):
            text = ' '.join(t['form'] for t in sent)

            if self.get_root(sent) is None:
                self.stats['rejected']['no_root'] += 1
                rejected.append({'text': text, 'reason': 'no_root'})
                continue

            if not self.is_declarative(sent):
                self.stats['rejected']['non_declarative'] += 1
                rejected.append({'text': text, 'reason': 'non_declarative'})
                continue

            if not self.has_no_negation(sent):
                self.stats['rejected']['negation'] += 1
                rejected.append({'text': text, 'reason': 'negation'})
                continue

            if not self.has_valid_ids(sent):
                self.stats['rejected']['invalid_ids'] += 1
                rejected.append({'text': text, 'reason': 'invalid_ids'})
                continue

            if not self.is_projective(sent):
                self.stats['rejected']['non_projective'] += 1
                rejected.append({'text': text, 'reason': 'non_projective'})
                continue

            if not self.root_is_valid(sent):
                self.stats['rejected']['no_verb_root'] += 1
                rejected.append({'text': text, 'reason': 'no_verb_root'})
                continue

            if not self.has_subject_object(sent):
                self.stats['rejected']['no_subj_obj'] += 1
                rejected.append({'text': text, 'reason': 'no_subj_obj'})
                continue

            ok, count = self.has_valid_preverbal(sent)
            if not ok:
                self.stats['rejected']['few_preverbal'] += 1
                rejected.append({'text': text, 'reason': f'preverbal_{count}'})
                continue

            if not self.has_preverbal_subj_and_obj(sent):
                self.stats['rejected']['so_not_preverbal'] += 1
                rejected.append({'text': text, 'reason': 'so_not_preverbal'})
                continue

            filtered.append(sent)
            if sent_ids is not None:
                filtered_ids.append(sent_ids[idx])
            self.stats['passed'] += 1

        if sent_ids is not None:
            return filtered, pd.DataFrame(rejected), filtered_ids
        return filtered, pd.DataFrame(rejected)

    def print_stats(self):
        s = self.stats
        rej = s['rejected']
        print(f"\nFilter stats:")
        print(f"  Total    : {s['total']:,}")
        print(f"  Passed   : {s['passed']:,}")
        print(f"  Rejected : {s['total'] - s['passed']:,}")
        for reason, count in rej.items():
            if count:
                print(f"    {reason:<20}: {count:,}")

    def create_metadata(self, filtered, sent_ids=None):
        """
        Create metadata DataFrame for filtered sentences.
        Each row: sent_id, text (surface form).

        sent_ids: optional list aligned with `filtered` (e.g. the
        `filtered_ids` returned by filter_with_rejection_tracking) giving the
        TRUE original sent_id for each sentence. Falls back to a sequential
        placeholder id only when the caller has no real ids to pass.
        """
        rows = []
        for i, sent in enumerate(filtered):
            text = ' '.join(t['form'] for t in sent)
            sid = sent_ids[i] if sent_ids is not None else f'sent_{i:04d}'
            rows.append({'sent_id': sid, 'text': text})
        return pd.DataFrame(rows)

    # --------------------------------------------------
    # ORDER (with norm_dep)
    # --------------------------------------------------
    def get_order(self, sentence):
        preverbal = self.preverbal_dependents(sentence)
        subj = next((t for t in preverbal if t['deprel'] in self.subject_deps), None)
        obj  = next((t for t in preverbal if t['deprel'] in self.object_deps), None)
        iobj = next((t for t in preverbal if t['deprel'] in self.indirect_object_deps), None)

        if subj and obj:
            return 'SOV' if subj['id'] < obj['id'] else 'OSV'
        if subj and iobj:
            return 'SOV' if subj['id'] < iobj['id'] else 'OSV'
        return 'unknown'