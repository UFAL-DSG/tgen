#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ranker based on logistic regression.
"""

from __future__ import unicode_literals
import cPickle as pickle

from flect.logf import log_info
from flect.model import Model
from alex.components.nlg.tectotpl.core.util import file_stream

from futil import read_das, read_ttrees
from tgen import Ranker
import operator
from flect.dataset import DataSet


# TODO: Feature Generator


class LogisticRegressionRanker(Ranker):

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.features = None

    def create_training_data(self, t_file, da_file, candgen, train_arff_fname):
        """Create an ARFF file to train the ranker classifier.

        @param t_file: training data file with t-trees (YAML/Pickle)
        @param da_file: training data file with dialogue acts
        @param candgen: (trained) candidate generator
        @param train_arff_fname: output training data file (with features)
        """
        # read training data
        log_info('Reading ' + t_file)
        ttrees = read_ttrees(t_file)
        log_info('Reading ' + da_file)
        das = read_das(da_file)
        # collect features
        log_info('Generating features')
        train = []
        for ttree, da in zip(ttrees.bundles, das):
            ttree = ttree.get_zone('en', '').ttree
            cdfs = candgen.get_merged_cdfs(da)
            for node in ttree.get_descendants():
                true_children = set([(c.formeme, c.t_lemma, c > node) for c in node.get_children()])
                for cand in cdfs[node.formeme]:
                    feats = self.features.get_features(cand, node)
                    feats['sel'] = cand in true_children
                    train.append(feats)
        # save to file
        log_info('Writing ' + train_arff_fname)
        train_set = DataSet()
        train_set.load_from_dict(train)
        train_set.save_to_arff(train_arff_fname)

    def train(self, train_arff_fname):
        """Train on the given training data file."""
        self.model = Model(self.cfg['model'])
        self.model.train(train_arff_fname)

    def load_model(self, model_fname):
        """Load a pre-trained model from a file."""
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            self.model = pickle.load(fh)

    def save_model(self, model_fname):
        """Save the model to a file."""
        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self.model, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def cdf_to_dist(self, cdf):
        """Convert a CDF to a distribution (keep the list format, just discount lower bounds)."""
        lo_bound = 0.0
        dist = []
        for cand, hi_bound in cdf:
            dist.append((cand, hi_bound - lo_bound))
            lo_bound = hi_bound
        return dist

    def get_best_child(self, parent, cdf):
        """Predicting the best child of the given node."""
        candidates = [self.features.get_features(candidate, parent)
                      for candidate in self.cdf_to_dist(cdf)]
        ranks = [prob[1] for prob in self.model.classify(candidates, pdist=True)]
        best_index, _ = max(enumerate(ranks), key=operator.itemgetter(1))
        return cdf[best_index][0]
