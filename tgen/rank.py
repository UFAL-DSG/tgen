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
from collections import defaultdict

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
        self.w_after_iter = []
        self.w = None
        self.w_sum = 0.0
        self.feats = ['bias: bias']
        self.vectorizer = None
        self.normalizer = None
        self.alpha = cfg.get('alpha', 1)
        self.passes = cfg.get('passes', 5)
        self.prune_feats = cfg.get('prune_feats', 1)
        self.rival_number = cfg.get('rival_number', 10)
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        self.averaging = cfg.get('averaging', False)
        self.future_promise_weight = cfg.get('future_promise_weight', 1.0)
        self.future_promise_type = cfg.get('future_promise_type', 'expected_children')
        self.rival_gen_strategy = cfg.get('rival_gen_strategy', ['other_inst'])
        self.rival_gen_max_iter = cfg.get('rival_gen_max_iter', 50)
        self.rival_gen_max_defic_iter = cfg.get('rival_gen_max_defic_iter', 3)
        self.rival_gen_beam_size = cfg.get('rival_gen_beam_size')
        self.candgen_model = cfg.get('candgen_model')
        self.diffing_trees = cfg.get('diffing_trees', False)
        # initialize feature functions
        if 'features' in cfg:
            self.feats.extend(cfg['features'])
        self.feats = Features(self.feats)

    def score(self, cand_tree, da):
        """Score the given tree in the context of the given dialogue act.
        @param cand_tree: the candidate tree to be scored, as a TreeData object
        @param da: a DialogueAct object representing the input dialogue act
        """
        return self._score(self._extract_feats(cand_tree, da))

    def _score(self, cand_feats):
        return np.dot(self.w, cand_feats)

    def _extract_feats(self, tree, da):
        return self.normalizer.transform(
            self.vectorizer.transform([self.feats.get_features(tree, {'da': da})]))[0]

    def get_future_promise(self, cand_tree):
        """Compute expected future cost for a tree."""
        if self.future_promise_type == 'num_nodes':
            return self.w_sum * self.future_promise_weight * max(0, 10 - len(cand_tree))
        else:  # expected children (default)
            return self.candgen.get_future_promise(cand_tree) * self.w_sum * self.future_promise_weight

    def train(self, das_file, ttree_file, data_portion=1.0):
        """Run training on the given training data."""
        self._init_training(das_file, ttree_file, data_portion)
        for iter_no in xrange(1, self.passes + 1):
            self._training_iter(iter_no)
        # averaged perceptron – average the weights obtained after each iteration
        if self.averaging is True:
            self.w = np.average(self.w_after_iter, axis=0)

    def _init_training(self, das_file, ttree_file, data_portion):
        """Initialize training (read input files, reset weights, reset
        training data features.)"""
        # read input
        log_info('Reading DAs from ' + das_file + '...')
        das = read_das(das_file)
        log_info('Reading t-trees from ' + ttree_file + '...')
        ttree_doc = read_ttrees(ttree_file)
        sents = sentences_from_doc(ttree_doc, self.language, self.selector)
        trees = trees_from_doc(ttree_doc, self.language, self.selector)

        # make training data smaller if necessary
        train_size = int(round(data_portion * len(trees)))
        self.train_trees = trees[:train_size]
        self.train_das = das[:train_size]
        self.train_sents = sents[:train_size]
        log_info('Using %d training instances.' % train_size)

        # precompute training data features
        X = []
        for da, tree in zip(self.train_das, self.train_trees):
            X.append(self.feats.get_features(tree, {'da': da}))
        if self.prune_feats > 1:
            self._prune_features(X)
        # vectorize and normalize (+train normalizer and vectorizer)
        self.vectorizer = DictVectorizer(sparse=False)
        self.normalizer = StandardScaler(copy=False)
        self.train_feats = self.normalizer.fit_transform(self.vectorizer.fit_transform(X))
        log_info('Features matrix shape: %s' % str(self.train_feats.shape))

        # initialize candidate generator + planner if needed
        if self.candgen_model is not None:
            self.candgen = RandomCandidateGenerator.load_from_file(self.candgen_model)
            self.sampling_planner = SamplingPlanner({'langugage': self.language,
                                                     'selector': self.selector,
                                                     'candgen': self.candgen})
        if 'gen_cur_weights' in self.rival_gen_strategy:
            assert self.candgen is not None
            self.asearch_planner = ASearchPlanner({'candgen': self.candgen,
                                                   'language': self.language,
                                                   'selector': self.selector,
                                                   'ranker': self, })

        # initialize diagnostics
        self.lists_analyzer = None
        self.evaluator = None

        # initialize weights
        self.w = np.ones(self.train_feats.shape[1])
        self.w_sum = sum(self.w)
        # self.w = np.array([random.gauss(0, self.alpha) for _ in xrange(self.train_feats.shape[1])])

        log_debug('\n***\nINIT:')
        log_debug(self._feat_val_str(self.w))
        log_info('Training ...')

    def _training_iter(self, iter_no):
        """Run one training iteration, update weights (store them for possible averaging),
        and return statistics.

        @return: a tuple of Evaluator and ListsAnalyzer objects containing iteration statistics."""

        iter_start_time = time.clock()
        self.evaluator = Evaluator()
        self.lists_analyzer = ASearchListsAnalyzer()
        self.w_sum = sum(self.w)

        log_debug('\n***\nTR %05d:' % iter_no)

        for tree_no in xrange(len(self.train_trees)):
            # obtain some 'rival', alternative incorrect candidates
            gold_tree, gold_feats = self.train_trees[tree_no], self.train_feats[tree_no]
            rival_trees, rival_feats = self._get_rival_candidates(tree_no)
            cands = [gold_feats] + rival_feats

            # score them along with the right one
            scores = [self._score(cand) for cand in cands]
            top_cand_idx = scores.index(max(scores))
            top_rival_tree = rival_trees[scores[1:].index(max(scores[1:]))]

            # find the top-scoring generated tree, evaluate against gold t-tree
            # (disregarding whether it was selected as the best one)
            self.evaluator.append(TreeNode(gold_tree), TreeNode(top_rival_tree), scores[0], max(scores[1:]))

            # debug print: candidate trees
            log_debug('TTREE-NO: %04d, SEL_CAND: %04d, LEN: %02d' % (tree_no, top_cand_idx, len(cands)))
            log_debug('SENT: %s' % self.train_sents[tree_no])
            log_debug('ALL CAND TREES:')
            for ttree, score in zip([gold_tree] + rival_trees, scores):
                log_debug("%.3f" % score, "\t", ttree)

            # update weights if the system doesn't give the highest score to the right one
            if top_cand_idx != 0:
                self._update_weights(self.train_das[tree_no], gold_tree, top_rival_tree,
                                     gold_feats, cands[top_cand_idx])

        # store a copy of the current weights for averaging
        self.w_after_iter.append(np.copy(self.w))

        # debug print: current weights and iteration accuracy
        log_debug(self._feat_val_str(self.w), '\n***')
        log_debug('ITER ACCURACY: %.3f' % self.evaluator.tree_accuracy())

        # print and return statistics
        self._print_iter_stats(iter_no, datetime.timedelta(seconds=(time.clock() - iter_start_time)))
        return self.evaluator, self.lists_analyzer

    def _update_weights(self, da, good_tree, bad_tree, good_feats, bad_feats):
        # discount trees leading to the generated one and add trees leading to the gold one
        if self.diffing_trees:
            good_sts, bad_sts = good_tree.diffing_trees(bad_tree,
                                                        symmetric=self.diffing_trees.startswith('sym'))
            # TODO discount common subtree off all features
            discount = None
            if 'nocom' in self.diffing_trees:
                discount = self._extract_feats(good_tree.get_common_subtree(bad_tree), da)
            # add good trees (leading to gold)
            for good_st in good_sts:
                good_feats = self._extract_feats(good_st, da)
                if discount is not None:
                    good_feats -= discount
                good_tree_w = 1
                if self.diffing_trees.endswith('weighted'):
                    good_tree_w = len(good_st) / float(len(good_tree))
                self.w += self.alpha * good_tree_w * good_feats
            # discount bad trees (leading to the generated one)
            if 'nobad' in self.diffing_trees:
                bad_sts = []
            elif 'onebad' in self.diffing_trees:
                bad_sts = [bad_tree]
            for bad_st in bad_sts:
                bad_feats = self._extract_feats(bad_st, da)
                if discount is not None:
                    bad_feats -= discount
                bad_tree_w = 1
                if self.diffing_trees.endswith('weighted'):
                    bad_tree_w = len(bad_st) / float(len(bad_tree))
                self.w -= self.alpha * bad_tree_w * bad_feats
        # just discount the best generated tree and add the gold tree
        else:
            self.w += (self.alpha * good_feats - self.alpha * bad_feats)
        # # log_debug('Updated  w: ' + str(np.frombuffer(self.w, "uint8").sum()))

    def _print_iter_stats(self, iter_no, iter_duration):
        """Print iteration statistics from internal evaluator fields and given iteration duration."""
        log_info('Iteration %05d -- tree-level accuracy: %.4f' % (iter_no, self.evaluator.tree_accuracy()))
        log_info(' * Generated trees NODE scores: P: %.4f, R: %.4f, F: %.4f' %
                 self.evaluator.p_r_f1())
        log_info(' * Generated trees DEP  scores: P: %.4f, R: %.4f, F: %.4f' %
                 self.evaluator.p_r_f1(EvalTypes.DEP))
        log_info(' * Gold tree BEST: %.4f, on CLOSE: %.4f, on ANY list: %4f' %
                 self.lists_analyzer.stats())
        log_info(' * Tree size stats:\n -- GOLD: %s\n -- PRED: %s\n -- DIFF: %s' %
                 self.evaluator.tree_size_stats())
        log_info(' * Common subtree stats:\n -- SIZE: %s\n -- ΔGLD: %s\n -- ΔPRD: %s' %
                 self.evaluator.common_subtree_stats())
        log_info(' * Score stats\n -- GOLD: %s\n -- PRED: %s\n -- DIFF: %s'
                 % self.evaluator.score_stats())
        log_info(' * Duration: %s' % str(iter_duration))

    def _feat_val_str(self, vec, sep='\n', nonzero=False):
        return sep.join(['%s: %.3f' % (name, weight)
                         for name, weight in zip(self.vectorizer.get_feature_names(), vec)
                         if not nonzero or weight != 0])

    def _get_rival_candidates(self, tree_no):
        """Generate some rival candidates for a DA and the correct (gold) tree,
        given the current rival generation strategy (self.rival_gen_strategy).

        TODO: checking for trees identical to the gold one slows down the process

        @param tree_no: the index of the current training data item (tree, DA)
        @rtype: tuple of two lists: one of TreeData's, one of arrays
        @return: an array of rival trees and an array of the corresponding features
        """
        da = self.train_das[tree_no]
        train_trees = self.train_trees

        rival_trees, rival_feats = [], []

        # use current DA but change trees when computing features
        if 'other_inst' in self.rival_gen_strategy:
            # use alternative indexes, avoid the correct one
            rival_idxs = map(lambda idx: len(train_trees) - 1 if idx == tree_no else idx,
                             random.sample(xrange(len(train_trees) - 1), self.rival_number))
            other_inst_trees = [train_trees[rival_idx] for rival_idx in rival_idxs]
            rival_trees.extend(other_inst_trees)
            rival_feats.extend([self._extract_feats(tree, da) for tree in other_inst_trees])

        # candidates generated using the random planner (use the current DA)
        if 'random' in self.rival_gen_strategy:
            random_trees = []
            while len(random_trees) < self.rival_number:
                tree = self.sampling_planner.generate_tree(da)
                if (tree != train_trees[tree_no]):  # don't generate trees identical to the gold one
                    random_trees.append(tree)
            rival_trees.extend(random_trees)
            rival_feats.extend([self._extract_feats(tree, da) for tree in random_trees])

        # candidates generated using the A*search planner, which uses this ranker with current
        # weights to guide the search, and the current DA as the input
        # TODO: use just one!, others are meaningless
        if 'gen_cur_weights' in self.rival_gen_strategy:
            open_list, close_list = self.asearch_planner.run(da,
                                                             self.rival_gen_max_iter,
                                                             self.rival_gen_max_defic_iter,
                                                             self.rival_gen_beam_size)
            self.lists_analyzer.append(train_trees[tree_no], open_list, close_list)
            gen_trees = []
            while close_list and len(gen_trees) < self.rival_number:
                tree = close_list.pop()[0]
                if tree != train_trees[tree_no]:
                    gen_trees.append(tree)
            rival_trees.extend(gen_trees[:self.rival_number])
            rival_feats.extend([self._extract_feats(tree, da)
                                for tree in gen_trees[:self.rival_number]])

        # return all resulting candidates
        return rival_trees, rival_feats

    def _prune_features(self, X):
        """Prune features – remove all entries from X that involve features not having a
        specified minimum occurrence count.
        """
        counts = defaultdict(int)
        for inst in X:
            for key in inst.iterkeys():
                counts[key] += 1
        for inst in X:
            for key in inst.keys():
                if counts[key] < self.prune_feats:
                    del inst[key]
