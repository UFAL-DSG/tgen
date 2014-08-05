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

from flect.logf import log_info, log_debug
from flect.model import Model
from alex.components.nlg.tectotpl.core.util import file_stream
from flect.dataset import DataSet

from features import Features
from futil import read_das, read_ttrees, ttrees_from_doc
from planner import SamplingPlanner, ASearchPlanner
from candgen import RandomCandidateGenerator


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
        if not cfg:
            cfg = {}
        self.w = None
        self.feats = ['bias: bias']
        self.vectorizer = None
        self.alpha = cfg.get('alpha', 1)
        self.passes = cfg.get('passes', 5)
        self.rival_number = cfg.get('rival_number', 10)
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        self.asearch_planner = None
        self.sampling_planner = None
        self.candgen = None
        self.rival_gen_strategy = cfg.get('rival_gen_strategy', ['other_inst'])
        self.rival_gen_max_iter = cfg.get('rival_gen_max_iter', 50)
        self.rival_gen_max_defic_iter = cfg.get('rival_gen_max_defic_iter', 3)
        self.rival_gen_beam_size = cfg.get('rival_gen_beam_size', 100)
        # initialize random candidate generator if needed
        if 'candgen_model' in cfg:
            self.candgen = RandomCandidateGenerator({})
            self.candgen.load_model(cfg['candgen_model'])
            self.sampling_planner = SamplingPlanner({'langugage': self.language,
                                                     'selector': self.selector,
                                                     'candgen': self.candgen})
        # initialize feature functions
        if 'features' in cfg:
            self.feats.extend(cfg['features'])
        self.feats = Features(self.feats)
        # initialize planner if needed
        if 'gen_cur_weights' in self.rival_gen_strategy:
            assert self.candgen is not None
            self.asearch_planner = ASearchPlanner({'candgen': self.candgen,
                                                   'language': self.language,
                                                   'selector': self.selector,
                                                   'ranker': self, })

    def score(self, cand_ttree, da):
        feats = self.vectorizer.transform(self.feats.get_features(cand_ttree, {'da': da}))
        return self._score(feats)

    def _score(self, cand_feats):
        return np.dot(self.w, cand_feats.toarray()[0])

    def train(self, das_file, ttree_file):
        # read input
        das = read_das(das_file)
        ttrees = ttrees_from_doc(read_ttrees(ttree_file), self.language, self.selector)
        # compute features for trees
        X = []
        for da, ttree in zip(das, ttrees):
            X.append(self.feats.get_features(ttree, {'da': da}))
        # vectorize
        self.vectorizer = DictVectorizer()
        X = self.vectorizer.fit_transform(X)
        # initialize weights
        self.w = np.zeros(X.get_shape()[1])  # number of columns

        # 1st pass over training data -- just add weights
        for inst in X:
            self.w += self.alpha * inst.toarray()[0]

        log_debug('\n***\nTR %05d:' % 0)
        log_debug(self._feat_val_str(self.w))

        # further passes over training data -- compare the right instance to other, wrong ones
        for iter_no in xrange(1, self.passes + 1):

            iter_errs = 0
            log_debug('\n***\nTR %05d:' % iter_no)

            for ttree_no, da in enumerate(das):
                # obtain some 'rival', alternative incorrect candidates
                gold_ttree, gold_feats = ttrees[ttree_no], X[ttree_no]
                rival_ttrees, rival_feats = self._get_rival_candidates(da, ttrees, ttree_no)
                cands = [gold_feats] + rival_feats

                # score them along with the right one
                scores = [self._score(cand) for cand in cands]
                top_cand_idx = scores.index(max(scores))

                log_debug('TTREE-NO: %04d, SEL_CAND: %04d, LEN: %02d' % (ttree_no, top_cand_idx, len(cands)))
                log_debug('ALL CAND TTREES:')
                for ttree, score in zip([gold_ttree] + rival_ttrees, scores):
                    log_debug("%.3f" % score, "\t", ttree)
                # log_debug('GOLD CAND -- ', self._feat_val_str(cands[0].toarray()[0], '\t'))
                # log_debug('SEL  CAND -- ', self._feat_val_str(cands[top_cand_idx].toarray()[0], '\t'))

                # update weights if the system doesn't give the highest score to the right one
                if top_cand_idx != 0:
                    self.w += (self.alpha * X[ttree_no].toarray()[0] -
                               self.alpha * cands[top_cand_idx].toarray()[0])
                    iter_errs += 1

            iter_acc = (1.0 - (iter_errs / float(len(ttrees))))
            log_debug(self._feat_val_str(self.w), '\n***')
            log_debug('ITER ACCURACY: %.3f' % iter_acc)

            log_info('Iteration %05d -- tree-level accuracy: %.3f' % (iter_no, iter_acc))

    def _feat_val_str(self, vec, sep='\n'):
        return sep.join(['%s: %.3f' % (name, weight)
                         for name, weight in zip(self.vectorizer.get_feature_names(), vec)])

    def _get_rival_candidates(self, da, train_ttrees, gold_ttree_no):
        """Generate some rival candidates for a DA and the correct (gold) t-tree,
        given the current rival generation strategy (self.rival_gen_strategy).

        TODO: checking for trees identical to the gold one slows down the process

        @param da: the current input dialogue act
        @param train_ttrees: training t-trees
        @param gold_ttree_no: the index of the gold t-tree in train_ttrees
        @rtype: tuple
        @return: an array of rival t-trees and an array of the corresponding features
        """
        rival_ttrees, rival_feats = [], []

        # use current DA but change trees when computing features
        if 'other_inst' in self.rival_gen_strategy:
            # use alternative indexes, avoid the correct one
            rival_idxs = map(lambda idx: len(train_ttrees) - 1 if idx == gold_ttree_no else idx,
                             np.random.choice(len(train_ttrees) - 1, self.rival_number))
            other_inst_ttrees = [train_ttrees[rival_idx] for rival_idx in rival_idxs]
            rival_ttrees.extend(other_inst_ttrees)
            rival_feats.extend([self.vectorizer.transform(self.feats.get_features(ttree, {'da': da}))
                                for ttree in other_inst_ttrees])

        # candidates generated using the random planner (use the current DA)
        if 'random' in self.rival_gen_strategy:
            gen_doc = None
            while gen_doc is None or (len(gen_doc.bundles) < self.rival_number):
                gen_doc = self.sampling_planner.generate_tree(da, gen_doc)
                if (gen_doc.bundles[-1].get_zone(self.language, self.selector).ttree
                    == train_ttrees[gold_ttree_no]):  # don't generate trees identical to the gold one
                    del gen_doc.bundles[-1]
            random_ttrees = ttrees_from_doc(gen_doc, self.language, self.selector)
            rival_ttrees.extend(random_ttrees)
            rival_feats.extend([self.vectorizer.transform(self.feats.get_features(ttree, {'da': da}))
                                for ttree in random_ttrees])

        # candidates generated using the A*search planner, which uses this ranker with current
        # weights to guide the search, and the current DA as the input
        # TODO: use just one!, others are meaningless
        if 'gen_cur_weights' in self.rival_gen_strategy:
            gen_ttrees = [t for t in self.asearch_planner.get_best_candidates(da, self.rival_number + 1,
                                                                              self.rival_gen_max_iter,
                                                                              self.rival_gen_max_defic_iter,
                                                                              self.rival_gen_beam_size)
                          if t != train_ttrees[gold_ttree_no]]
            rival_ttrees.extend(gen_ttrees[:self.rival_number])
            rival_feats.extend([self.vectorizer.transform(self.feats.get_features(ttree, {'da': da}))
                                for ttree in gen_ttrees[:self.rival_number]])

        # return all resulting candidates
        return rival_ttrees, rival_feats
