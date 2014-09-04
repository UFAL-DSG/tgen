#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Candidate tree rankers.

"""
from __future__ import unicode_literals
import numpy as np
import cPickle as pickle
import random
import time
import datetime

from alex.components.nlg.tectotpl.core.util import file_stream

from ml import DictVectorizer, StandardScaler
from logf import log_info, log_debug
from features import Features
from futil import read_das, read_ttrees, trees_from_doc, sentences_from_doc
from planner import SamplingPlanner, ASearchPlanner
from candgen import RandomCandidateGenerator
from eval import Evaluator, EvalTypes
from tree import TreeNode
from tgen.eval import ASearchListsAnalyzer


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


class PerceptronRanker(Ranker):
    """Global ranker for whole trees, based on linear Perceptron by Collins & Duffy (2002)."""

    def __init__(self, cfg):
        if not cfg:
            cfg = {}
        self.w = None
        self.feats = ['bias: bias']
        self.vectorizer = None
        self.normalizer = None
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
        self.rival_gen_beam_size = cfg.get('rival_gen_beam_size')
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
        return self._score(self._extract_feats(cand_ttree, da))

    def _score(self, cand_feats):
        return np.dot(self.w, cand_feats)

    def _extract_feats(self, ttree, da):
        return self.normalizer.transform(
                        self.vectorizer.transform(
                                [self.feats.get_features(ttree, {'da': da})]))[0]

    def train(self, das_file, ttree_file, data_portion=1.0):
        # read input
        log_info('Reading DAs from ' + das_file + '...')
        das = read_das(das_file)
        log_info('Reading t-trees from ' + ttree_file + '...')
        ttree_doc = read_ttrees(ttree_file)
        sentences = sentences_from_doc(ttree_doc, self.language, self.selector)
        ttrees = trees_from_doc(ttree_doc, self.language, self.selector)
        # make training data smaller if necessary
        train_size = int(round(data_portion * len(ttrees)))
        ttrees = ttrees[:train_size]
        das = das[:train_size]
        log_info('Using %d training instances.' % train_size)
        log_info('Training ...')
        # compute features for trees
        X = []
        for da, ttree in zip(das, ttrees):
            X.append(self.feats.get_features(ttree, {'da': da}))
        # vectorize and normalize (+train normalizer and vectorizer)
        self.vectorizer = DictVectorizer(sparse=False)
        self.normalizer = StandardScaler(copy=False)
        X = self.normalizer.fit_transform(self.vectorizer.fit_transform(X))

#        for ttree_no, da in enumerate(das):
#            ttree, feats = ttrees[ttree_no], X[ttree_no]
#             log_debug('------\nDATA %d:' % ttree_no)
#             log_debug('DA: %s' % unicode(da))
#             log_debug('SENT: %s' % sentences[ttree_no])
#             log_debug('TTREE: %s' % unicode(ttree))
#             log_debug('FEATS:', self._feat_val_str(feats, '\t', nonzero=True))

        # initialize weights
        # self.w = np.zeros(X.shape[1])  # can't be zeroes (will never update)
        self.w = np.ones(X.shape[1])

        # 1st pass over training data -- just add weights
        # TODO irrelevant with normalization?
        # for inst in X:
        #    self.w += self.alpha * inst

        log_debug('\n***\nTR %05d:' % 0)
        log_debug(self._feat_val_str(self.w))

        # further passes over training data -- compare the right instance to other, wrong ones
        for iter_no in xrange(1, self.passes + 1):

            iter_start_time = time.clock()

            iter_errs = 0
            log_debug('\n***\nTR %05d:' % iter_no)
            evaler = Evaluator()
            lists_analyzer = ASearchListsAnalyzer()

            for ttree_no, da in enumerate(das):
                # obtain some 'rival', alternative incorrect candidates
                gold_ttree, gold_feats = ttrees[ttree_no], X[ttree_no]
                rival_ttrees, rival_feats = self._get_rival_candidates(da, ttrees, ttree_no,
                                                                       lists_analyzer)
                cands = [gold_feats] + rival_feats

                # score them along with the right one
                scores = [self._score(cand) for cand in cands]
                top_cand_idx = scores.index(max(scores))

                # find the top-scoring generated tree, evaluate F-score against gold t-tree
                # (disregarding whether it was selected as the best one)
                evaler.append(TreeNode(gold_ttree),
                              TreeNode(rival_ttrees[scores[1:].index(max(scores[1:]))]))

                log_debug('TTREE-NO: %04d, SEL_CAND: %04d, LEN: %02d' % (ttree_no, top_cand_idx, len(cands)))
                log_debug('SENT: %s' % sentences[ttree_no])
                log_debug('ALL CAND TREES:')
                for ttree, score in zip([gold_ttree] + rival_ttrees, scores):
                    log_debug("%.3f" % score, "\t", ttree)
                # log_debug('GOLD CAND -- ', self._feat_val_str(cands[0], '\t'))
                # log_debug('SEL  CAND -- ', self._feat_val_str(cands[top_cand_idx], '\t'))

                # update weights if the system doesn't give the highest score to the right one
                if top_cand_idx != 0:
                    self.w += (self.alpha * X[ttree_no] -
                               self.alpha * cands[top_cand_idx])
                    iter_errs += 1

            iter_acc = (1.0 - (iter_errs / float(len(ttrees))))
            log_debug(self._feat_val_str(self.w), '\n***')
            log_debug('ITER ACCURACY: %.3f' % iter_acc)

            iter_end_time = time.clock()

            log_info('Iteration %05d -- tree-level accuracy: %.4f' % (iter_no, iter_acc))
            log_info(' * Generated trees NODE scores: P: %.4f, R: %.4f, F: %.4f' % evaler.p_r_f1())
            log_info(' * Generated trees DEP  scores: P: %.4f, R: %.4f, F: %.4f' %
                     evaler.p_r_f1(EvalTypes.DEP))
            log_info(' * Gold tree BEST: %.4f, on CLOSE: %.4f, on ANY list: %4f' %
                     lists_analyzer.stats())
            log_info(' * Duration: %s' % str(datetime.timedelta(seconds=(iter_end_time - iter_start_time))))

    def _feat_val_str(self, vec, sep='\n', nonzero=False):
        return sep.join(['%s: %.3f' % (name, weight)
                         for name, weight in zip(self.vectorizer.get_feature_names(), vec)
                         if not nonzero or weight != 0])

    def _get_rival_candidates(self, da, train_trees, gold_tree_idx, lists_analyzer=None):
        """Generate some rival candidates for a DA and the correct (gold) t-tree,
        given the current rival generation strategy (self.rival_gen_strategy).

        TODO: checking for trees identical to the gold one slows down the process

        @param da: the current input dialogue act
        @param train_trees: training data trees
        @param gold_tree_idx: the index of the gold tree in train_trees
        @rtype: tuple
        @return: an array of rival t-trees and an array of the corresponding features
        """
        rival_trees, rival_feats = [], []

        # use current DA but change trees when computing features
        if 'other_inst' in self.rival_gen_strategy:
            # use alternative indexes, avoid the correct one
            rival_idxs = map(lambda idx: len(train_trees) - 1 if idx == gold_tree_idx else idx,
                             random.sample(xrange(len(train_trees) - 1), self.rival_number))
            other_inst_ttrees = [train_trees[rival_idx] for rival_idx in rival_idxs]
            rival_trees.extend(other_inst_ttrees)
            rival_feats.extend([self._extract_feats(ttree, da) for ttree in other_inst_ttrees])

        # candidates generated using the random planner (use the current DA)
        if 'random' in self.rival_gen_strategy:
            random_trees = []
            while len(random_trees) < self.rival_number:
                tree = self.sampling_planner.generate_tree(da)
                if (tree != train_trees[gold_tree_idx]):  # don't generate trees identical to the gold one
                    random_trees.append(tree)
            rival_trees.extend(random_trees)
            rival_feats.extend([self._extract_feats(ttree, da) for ttree in random_trees])

        # candidates generated using the A*search planner, which uses this ranker with current
        # weights to guide the search, and the current DA as the input
        # TODO: use just one!, others are meaningless
        if 'gen_cur_weights' in self.rival_gen_strategy:
            open_list, close_list = self.asearch_planner.run(da,
                                                             self.rival_gen_max_iter,
                                                             self.rival_gen_max_defic_iter,
                                                             self.rival_gen_beam_size)
            if lists_analyzer:
                lists_analyzer.append(train_trees[gold_tree_idx], open_list, close_list)
            gen_ttrees = []
            while close_list and len(gen_ttrees) < self.rival_number:
                tree = close_list.pop()[0]
                if tree != train_trees[gold_tree_idx]:
                    gen_ttrees.append(tree)
            rival_trees.extend(gen_ttrees[:self.rival_number])
            rival_feats.extend([self._extract_feats(ttree, da)
                                for ttree in gen_ttrees[:self.rival_number]])

        # return all resulting candidates
        return rival_trees, rival_feats
