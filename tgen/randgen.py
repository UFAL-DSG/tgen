#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Random generator, according to training data distribution.
"""

from __future__ import unicode_literals
from collections import defaultdict, Counter
import cPickle as pickle
import random

from flect.logf import log_info
from alex.components.nlg.tectotpl.core.util import file_stream

from futil import read_das, read_ttrees
from tgen import CandidateGenerator, Ranker


class RandomGenerator(CandidateGenerator, Ranker):

    def __init__(self):
        self.form_counts = None
        self.child_cdfs = None

    def load_model(self, fname):
        log_info('Loading model from ' + fname)
        with file_stream(fname, mode='rb', encoding=None) as fh:
            self.form_counts = pickle.load(fh)
            self.child_cdfs = pickle.load(fh)

    def save_model(self, fname):
        log_info('Saving model to ' + fname)
        with file_stream(fname, mode='wb', encoding=None) as fh:
            pickle.dump(self.form_counts, fh, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.child_cdfs, fh, pickle.HIGHEST_PROTOCOL)

    def train(self, da_file, t_file):
        """``Train'' the generator (collect counts of DAs and corresponding t-nodes).

        Will load a Treex YAML file or a pickle (to speed up loading of YAML).
        """
        # read training data
        log_info('Reading ' + t_file)
        ttrees = read_ttrees(t_file)
        log_info('Reading ' + da_file)
        das = read_das(da_file)
        # collect counts
        log_info('Collecting counts')
        form_counts = {}
        child_counts = defaultdict(Counter)
        for ttree, da in zip(ttrees.bundles, das):
            ttree = ttree.get_zone('en', '').ttree
            # counts for formeme/lemma given dai
            for dai in da:
                for tnode in ttree.get_descendants():
                    if not dai in form_counts:
                        form_counts[dai] = defaultdict(Counter)
                    form_counts[dai][tnode.parent.formeme][(tnode.formeme, tnode.t_lemma, tnode > tnode.parent)] += 1
            # counts for number of children
            for tnode in ttree.get_descendants():
                child_counts[tnode.formeme][len(tnode.get_children())] += 1
        self.form_counts = form_counts
        self.child_cdfs = self.cdfs_from_counts(child_counts)

    def get_merged_cdfs(self, da):
        """Get merged CDFs for the DAIs in the given DA."""
        merged_counts = defaultdict(Counter)
        for dai in da:
            for parent_formeme in self.form_counts[dai]:
                merged_counts[parent_formeme].update(self.form_counts[dai][parent_formeme])
        return self.cdfs_from_counts(merged_counts)

    def cdfs_from_counts(self, counts):
        """Given a dictionary of counts, create a dictionary of corresponding CDFs."""
        cdfs = {}
        for key in counts:
            tot = 0
            cdf = []
            for subkey in counts[key]:
                tot += counts[key][subkey]
                cdf.append((subkey, tot))
            cdfs[key] = cdf
        return cdfs

    def sample(self, cdf):
        """Return a sample from the distribution, given a CDF (as a list)."""
        total = cdf[-1][1]
        rand = random.random() * total  # get a random number in [0,total)
        for key, ubound in cdf:
            if ubound > rand:
                return key
        raise Exception('Unable to generate from CDF!')

    def get_number_of_children(self, formeme):
        if formeme not in self.child_cdfs:
            return 0
        return self.sample(self.child_cdfs[formeme])

    def get_best_child(self, parent, cdf):
        return self.sample(cdf)
