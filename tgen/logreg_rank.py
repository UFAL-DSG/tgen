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
from interface import Ranker
import operator
from flect.dataset import DataSet
from features import Features


class LogisticRegressionRanker(Ranker):

    LO_PROB = 1e-4  # probability of unseen children
    TARGET_FEAT_NAME = 'sel'  # name of the target feature

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.model = None
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        if cfg and 'features' in cfg:
            self.features = Features(cfg['features'])
        self.attrib_types = {self.TARGET_FEAT_NAME: 'numeric'}
        self.attrib_order = [self.TARGET_FEAT_NAME]
        if 'attrib_types' in cfg:
            self.attrib_types.update(self.cfg['attrib_types'])
        if 'attrib_order' in cfg:
            self.attrib_order.extend(self.cfg['attrib_order'])

    def create_training_data(self, t_file, da_file, candgen, train_arff_fname, header_file=None):
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
            ttree = ttree.get_zone(self.language, self.selector).ttree
            cdfs = candgen.get_merged_cdfs(da)
            for node in ttree.get_descendants():
                # find true children of the given node
                true_children = [(c.formeme, c.t_lemma, c > node) for c in node.get_children()]
                # generate false candidate instances
                pdist = {}
                if node.formeme in cdfs:
                    true_children_set = set(true_children)
                    pdist = self.cdf_to_dist(cdfs[node.formeme])
                    for cand, prob in pdist.iteritems():
                        if cand in true_children_set:
                            continue
                        feats = self.features.get_features((cand, prob), node)
                        feats[self.TARGET_FEAT_NAME] = 0
                        train.append(feats)
                # generate true instances
                for true_child in true_children:
                    feats = self.features.get_features((true_child, pdist.get(true_child, self.LO_PROB)), node)
                    feats[self.TARGET_FEAT_NAME] = 1
                    train.append(feats)
        # create the ARFF file
        log_info('Writing ' + train_arff_fname)
        train_set = DataSet()
        if header_file is None:  # create headers on-the-fly
            train_set.load_from_dict(train,
                                     attrib_types=self.attrib_types,
                                     attrib_order=self.attrib_order,
                                     sparse=True,
                                     default_val=0.0)
        else:  # use given headers
            train_set.load_from_arff(header_file, headers_only=True)
            train_set.is_sparse = True
            train_set.append_from_dict(train, add_values=True, default_val=0.0)
        # save the ARFF file
        train_set.save_to_arff(train_arff_fname)

    def train(self, train_arff_fname):
        """Train on the given training data file."""
        self.model = Model(self.cfg['model'])
        self.model.train(train_arff_fname)

    @staticmethod
    def load_from_file(model_fname):
        """Load a pre-trained model from a file."""
        log_info("Loading ranker from %s..." % model_fname)
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            return pickle.load(fh)

    def save_to_file(self, model_fname):
        """Save the model to a file."""
        log_info("Saving ranker to %s..." % model_fname)
        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def cdf_to_dist(self, cdf):
        """Convert a CDF to a distribution (keep the list format, just discount lower bounds)."""
        lo_bound = 0.0
        dist = {}
        for cand, hi_bound in cdf:
            dist[cand] = hi_bound - lo_bound
            lo_bound = hi_bound
        return dist

    def get_best_child(self, parent, da, cdf):
        """Predicting the best child of the given node."""
        log_info('Predicting candidates for %s | %s' % (unicode(da), unicode(parent.t_lemma) + '/' + unicode(parent.formeme)))
        candidates = [self.features.get_features((cand, prob), parent)
                      for cand, prob in self.cdf_to_dist(cdf).iteritems()]
        ranks = [prob[1] for prob in self.model.classify(candidates, pdist=True)]
        best_index, _ = max(enumerate(ranks), key=operator.itemgetter(1))
        for index, rank in sorted(enumerate(ranks), key=operator.itemgetter(1), reverse=True)[0:10]:
            log_info('Child: %s, score: %s' % (unicode(cdf[index][0]), unicode(rank)))
        log_info('Best child: %s, score: %s' % (unicode(cdf[best_index][0]), unicode(ranks[best_index])))
        return cdf[best_index][0]
