#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ranker based on logistic regression.
"""

from __future__ import unicode_literals
import cPickle as pickle
import re
from functools import partial

from flect.logf import log_info
from flect.model import Model
from alex.components.nlg.tectotpl.core.util import file_stream

from futil import read_das, read_ttrees
from interface import Ranker
import operator
from flect.dataset import DataSet
from feature_func import find_nodes, value, same_as_current, prob


class Features(object):

    def __init__(self, cfg):
        self.features = self.parse_feature_spec(cfg)

    def parse_feature_spec(self, spec):
        """Prepares feature feature function from specifications in the following format:

        Label: value/same_as_current scope param1, ...

        Scope may be: parent, siblings, grandpa, uncles, or their combinations (connected
        with '+', no spaces). Always applies only to the part of the tree that is already
        built (i.e. to the top/left only).
        """
        features = {}
        for feat in spec:
            label, func_name = re.split(r'[:\s]+', feat, 1)
            if func_name == 'prob':
                features[label] = prob
            else:
                func_name, func_scope, func_params = re.split(r'[:\s]+', func_name, 2)
                func_params = re.split(r'[,\s]+', func_params)
                feat_func = None
                scope_func = partial(find_nodes, scope=func_scope.split('+'))
                if func_name.lower() == 'same_as_current':
                    feat_func = partial(same_as_current, scope_func=scope_func, attrib=func_params[0])
                elif func_name.lower() == 'value':
                    feat_func = partial(value, scope_func=scope_func, attrib=func_params[0])
                else:
                    raise Exception('Unknown feature function:' + feat)
                features[label] = feat_func
        return features

    def get_features(self, node, parent):
        feats = {}
        for name, func in self.features.iteritems():
            val = func(node, parent)
            for subname, subval in val.iteritems():
                feats[name + '_' + subname] = subval
        return feats


class LogisticRegressionRanker(Ranker):

    LO_PROB = 1e-4

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.model = None
        if cfg and 'features' in cfg:
            self.features = Features(cfg['features'])
        self.attrib_types = {'sel': 'numeric'}
        self.attrib_order = ['sel']
        if 'attrib_types' in cfg:
            self.attrib_types.update(self.cfg['attrib_types'])
        if 'attrib_order' in cfg:
            self.attrib_order.extend(self.cfg['attrib_order'])

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
                        feats['sel'] = 0
                        train.append(feats)
                # generate true instances
                for true_child in true_children:
                    feats = self.features.get_features((true_child, pdist.get(true_child, self.LO_PROB)), node)
                    feats['sel'] = 1
                    train.append(feats)
        # save to file
        log_info('Writing ' + train_arff_fname)
        train_set = DataSet()
        train_set.load_from_dict(train,
                                 attrib_types=self.attrib_types,
                                 attrib_order=self.attrib_order,
                                 sparse=True,
                                 default_val=0.0)
        train_set.save_to_arff(train_arff_fname)

    def train(self, train_arff_fname):
        """Train on the given training data file."""
        self.model = Model(self.cfg['model'])
        self.model.train(train_arff_fname)

    @staticmethod
    def load_from_file(model_fname):
        """Load a pre-trained model from a file."""
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            return pickle.load(fh)

    def save_to_file(self, model_fname):
        """Save the model to a file."""
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

    def get_best_child(self, parent, cdf):
        """Predicting the best child of the given node."""
        candidates = [self.features.get_features((cand, prob), parent)
                      for cand, prob in self.cdf_to_dist(cdf).iteritems()]
        ranks = [prob[1] for prob in self.model.classify(candidates, pdist=True)]
        best_index, _ = max(enumerate(ranks), key=operator.itemgetter(1))
        return cdf[best_index][0]
