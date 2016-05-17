#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BLEU measurements
"""

from __future__ import unicode_literals
from collections import defaultdict
import math

from tgen.tree import TreeData


class BLEUMeasure(object):
    """An accumulator object capable of computing BLEU score using multiple references.

    BLEU may be computed over TreeData instances (with flat trees equivalent to "real" BLEU
    over words), or over tokens -- lists of pairs form-tag.

    The BLEU score is smoothed a bit so that it's not undefined when there are zero matches
    for a particular n-gram count, or when the predicted sentence is empty.
    """

    def __init__(self, max_ngram=4):
        self.max_ngram=max_ngram
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.ref_len = 0
        self.cand_lens = [0] * self.max_ngram
        self.hits = [0] * self.max_ngram

    def append(self, pred_sent, ref_sents):
        """Append a sentence for measurements, increase counters.

        @param pred_sent: the system output sentence (tree/tokens)
        @param ref_sents: the corresponding reference sentences (list/tuple of trees/tokens)
        """

        for i in xrange(self.max_ngram):
            self.hits[i] += self.compute_hits(i+1, pred_sent, ref_sents)
            self.cand_lens[i] += len(pred_sent) - i

        # take the reference that is closest in length to the candidate
        closest_ref = min(ref_sents, key=lambda ref_sent: abs(len(ref_sent) - len(pred_sent)))
        self.ref_len += len(closest_ref)

    def compute_hits(self, n, pred_sent, ref_sents):
        """Compute clipped n-gram hits for the given sentences and the given N

        @param n: n-gram 'N' (1 for unigrams, 2 for bigrams etc.)
        @param pred_sent: the system output sentence (tree/tokens)
        @param ref_sents: the corresponding reference sentences (list/tuple of trees/tokens)
        """
        merged_ref_ngrams = {}

        for ref_sent in ref_sents:
            ref_ngrams = defaultdict(int)

            for ngram in self.ngrams(n, ref_sent):
                ref_ngrams[ngram] += 1
            for ngram, cnt in ref_ngrams.iteritems():
                merged_ref_ngrams[ngram] = max((merged_ref_ngrams.get(ngram, 0), cnt))

        pred_ngrams = defaultdict(int)
        for ngram in self.ngrams(n, pred_sent):
            pred_ngrams[ngram] += 1

        hits = 0
        for ngram, cnt in pred_ngrams.iteritems():
            hits += min(merged_ref_ngrams.get(ngram, 0), cnt)

        return hits

    def ngrams(self, n, sent):
        """Given a sentence, return n-grams of nodes for the given N

        @param n: n-gram 'N' (1 for unigrams, 2 for bigrams etc.)
        @param sent: the sent in question
        @return: n-grams of nodes, as tuples of tuples (t-lemma & formeme)
        """
        # with sents
        if isinstance(sent, TreeData):
            return zip(*[sent.nodes[i:] for i in range(n)])
        # with tokens (as lists of pairs form+tag, or plain forms only)
        if sent and isinstance(sent[0], tuple):
            sent = [form for (form, _) in sent]  # ignore tags, use just forms
        return zip(*[sent[i:] for i in range(n)])

    def bleu(self):
        """Return the current BLEU score, according to the accumulated counts."""

        # brevity penalty (smoothed a bit: if candidate length is 0, we change it to 1e-5
        # to avoid division by zero)
        bp = 1.0
        if (self.cand_lens[0] <= self.ref_len):
            bp = math.exp(1.0 - self.ref_len /
                          (float(self.cand_lens[0]) if self.cand_lens[0] else 1e-5))

        return bp * self.ngram_precision()

    def ngram_precision(self):
        """Return the current n-gram precision (harmonic mean of n-gram precisions up to max_ngram)
        according to the accumulated counts."""

        # n-gram precision is smoothed a bit: 0 hits for a given n-gram count are
        # changed to 1e-5 to make BLEU defined everywhere
        prec_avg = sum(1.0 / self.max_ngram *
                       math.log((n_hits if n_hits != 0 else 1e-5) / float(max(n_lens, 1.0)))
                       for n_hits, n_lens in zip(self.hits, self.cand_lens))

        return math.exp(prec_avg)

