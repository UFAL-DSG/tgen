#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ranker based on logistic regression.
"""

from __future__ import unicode_literals
import cPickle as pickle
import re
from functools import partial
from collections import defaultdict

from flect.logf import log_info
from flect.model import Model
from alex.components.nlg.tectotpl.core.util import file_stream

from futil import read_das, read_ttrees
from tgen import Ranker
import operator
from flect.dataset import DataSet


class Features(object):

    def __init__(self, cfg):
        self.features = self.parse_feature_spec(cfg['features'])

    def parse_feature_spec(self, spec):
        """Prepares feature feature function from specifications in the following format:

        Label: value/same_as_current scope param1, ...

        Scope may be: parent, siblings, grandpa, uncles, or their combinations (connected
        with '+', no spaces). Always applies only to the part of the tree that is already
        built (i.e. to the top/left only).
        """
        features = {}
        for feat in spec:
            label, func_name, func_scope, func_params = re.split(r'[:\s]+', feat, 3)
            func_params = re.split(r'[,\s]+', func_params)
            feat_func = None
            scope_func = partial(self.find_nodes, func_scope.split('+'))
            if func_name.lower() == 'same_as_current':
                feat_func = partial(self.same_as_current, scope_func=scope_func, attrib=func_params[0])
            elif func_name.lower() == 'value':
                feat_func = partial(self.get_value, scope_func=scope_func, attrib=func_params[0])
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

    @staticmethod
    def find_nodes(parent, scope):
        """Given a parent node and scope specifications (in a list), this returns the
        corresponding nodes.
        """
        nodes = []
        for scope_spec in scope:
            if scope_spec == 'parent':
                nodes.append(parent)
            elif scope_spec == 'grandpa' and parent.parent:
                nodes.append(parent.parent)
            elif scope_spec == 'siblings':  # TODO: use left siblings only ?
                nodes.extend(parent.get_children())
            elif scope_spec == 'uncles' and parent.parent:
                nodes.extend([uncle for uncle in parent.parent.get_children() if uncle != parent])
        return nodes

    @staticmethod
    def same_as_current(node, parent, scope_func, attrib):
        """Return the number of nodes in the given scope that have the same value
        of the given attribute as the current node.

        @rtype: dict
        @return: dictionary with one key ('') and the number of matching values as a value
        """
        value = node[['formeme', 't_lemma', 'right'].index(attrib)]  # TODO more attributes / more flexible ?
        num_matching = 0
        for node in scope_func(parent):
            if attrib == 'right':  # special handling for 'right'
                if node.parent and (node > node.parent) == value:
                    num_matching += 1
            elif node.get_attr(attrib) == value:  # any other attribute
                num_matching += 1
        return {'': num_matching}

    @staticmethod
    def value(node, parent, scope_func, attrib):
        """Return the number of nodes holding the individual values of the given attribute
        in the given scope.

        @rtype dict
        @return: dictionary with keys for values of the attribute, values for counts of matching nodes
        """
        ret = defaultdict(int)
        for node in scope_func(parent):
            if attrib == 'right':
                if node.parent and node > node.parent:
                    ret['True'] += 1
                elif node.parent:
                    ret['False'] += 1
            else:
                ret[unicode(node.get_attr(attrib))] += 1
        return ret


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
