#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Candidate tree rankers.

"""
from __future__ import unicode_literals
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
import numpy as np
import cPickle as pickle
import operator

from flect.logf import log_info
from flect.model import Model
from alex.components.nlg.tectotpl.core.util import file_stream
from flect.dataset import DataSet

from features import Features
from futil import read_das, read_ttrees


class Ranker(object):

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


class PerceptronRanker(Ranker):

    def __init__(self, cfg):
        self.w = None
        self.features = ['bias']
        self.vectorizer = None
        self.alpha = 1
        self.passes = 5
        self.train_cands = 10
        if cfg:
            if 'features' in cfg:
                self.features.extend(cfg['features'])
            if 'alpha' in cfg:
                self.alpha = cfg['alpha']
            if 'passes' in cfg:
                self.passes = cfg['passes']
            if 'train_cands' in cfg:
                self.train_cands = cfg['train_cands']
        # initialize feature functions
        self.features = Features(self.features)

    def score(self, cand_ttree, da):
        feats = self.vectorizer.transform(self.features.get_features(cand_ttree, {'da': da}))
        return self._score(feats)

    def _score(self, cand_feats):
        return self.w * cand_feats.toarray()[0]

    def train(self, das_file, ttree_file):
        # read input
        das = read_das(das_file)
        ttrees = read_ttrees(ttree_file)
        # compute features for trees
        X = []
        for da, ttree in zip(das, ttrees.bundles):
            ttree = ttree.get_zone(self.language, self.selector).ttree
            X.append(self.features.get_features(ttree, {'da': da}))
        # vectorize
        self.vectorizer = DictVectorizer()
        X = self.vectorizer.fit_transform(X, sparse=True)
        # initialize weights
        self.w = np.zeros(X.get_shape[1])  # number of columns
        # 1st pass over training data -- just add weights
        for inst in X:
            self.w += self.alpha * inst.toarray()[0]
        # further passes over training data -- compare the right instance to other, wrong ones
        for _ in xrange(self.passes):
            for inst in X:
                # get some random 'other' candidates and score them along with the right one
                cands = [inst] + [cand for cand in np.random.choice(X, self.train_cands)
                                  if not np.array_equal(cand.toarray(), inst.toarray())]
                scores = [self.score(cand) for cand in cands]
                top_cand_idx = scores.index(max(scores))
                # update weights if the system doesn't give the highest score to the right one
                if top_cand_idx != 0:
                    self.w += (self.alpha * inst.toarray()[0] -
                               self.alpha * cands[top_cand_idx].toarray()[0])
