#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BLEU measurements
"""

from __future__ import unicode_literals
from collections import defaultdict
import math


class BLEUMeasure(object):
    """An accumulator object capable of computing BLEU score using multiple references.

    We assume computing BLEU over TreeData instances (flat trees may be used to compute
    "real" BLEU over words).

    The BLEU score is smoothed a bit so that it's not undefined when there are zero matches
    for a particular n-gram count.
    """

    MAX_NGRAM = 4

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.ref_len = 0
        self.cand_lens = [0] * self.MAX_NGRAM
        self.hits = [0] * self.MAX_NGRAM

    def append(self, pred_tree, ref_trees):
        """Append a sentence for measurements, increase counters.

        @param pred_tree: the system output tree
        @param ref_trees: the corresponding reference trees (list/tuple)
        """

        for i in xrange(self.MAX_NGRAM):
            self.hits[i] += self.compute_hits(i+1, pred_tree, ref_trees)
            self.cand_lens[i] += len(pred_tree) - i

        # take the reference that is closest in length to the candidate
        closest_ref = min(ref_trees, key=lambda ref_tree: abs(len(ref_tree) - len(pred_tree)))
        self.ref_len += len(closest_ref)

    def compute_hits(self, n, pred_tree, ref_trees):
        """Compute clipped n-gram hits for the given trees and the given N

        @param n: n-gram 'N' (1 for unigrams, 2 for bigrams etc.)
        @param pred_tree: the system output tree
        @param ref_trees: the corresponding reference trees (list/tuple)
        """
        merged_ref_ngrams = {}

        for ref_tree in ref_trees:
            ref_ngrams = defaultdict(int)

            for ngram in self.ngrams(n, ref_tree):
                ref_ngrams[ngram] += 1
            for ngram, cnt in ref_ngrams.iteritems():
                merged_ref_ngrams[ngram] = max((merged_ref_ngrams.get(ngram, 0), cnt))

        pred_ngrams = defaultdict(int)
        for ngram in self.ngrams(n, pred_tree):
            pred_ngrams[ngram] += 1

        hits = 0
        for ngram, cnt in pred_ngrams.iteritems():
            hits += min(merged_ref_ngrams.get(ngram, 0), cnt)

        return hits

    def ngrams(self, n, tree):
        """Given a trees, return n-grams of nodes for the given N

        @param n: n-gram 'N' (1 for unigrams, 2 for bigrams etc.)
        @param tree: the tree in question
        @return: n-grams of nodes, as tuples of tuples (t-lemma & formeme)
        """
        return zip(*[tree.nodes[i:] for i in range(n)])

    def bleu(self):
        """Return the current BLEU score, according to the accumulated counts."""

        # brevity penalty
        bp = 1.0
        if (self.cand_lens[0] <= self.ref_len):
            bp = math.exp(1.0 - self.ref_len / float(self.cand_lens[0]))

        # n-gram precision is smoothed a bit: 0 hits for a given n-gram count are
        # changed to 1e-5 to make BLEU defined everywhere
        prec_avg = sum(1.0 / self.MAX_NGRAM *
                       math.log((n_hits if n_hits != 0 else 1e-5) / float(max(n_lens, 1.0)))
                       for n_hits, n_lens in zip(self.hits, self.cand_lens))

        return bp * math.exp(prec_avg)
