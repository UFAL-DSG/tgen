#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Candidate tree rankers.

"""
from __future__ import unicode_literals
import numpy as np
import cPickle as pickle
import time
import datetime
from collections import defaultdict, namedtuple

from pytreex.core.util import file_stream

from ml import DictVectorizer, StandardScaler
from logf import log_info, log_debug
from features import Features
from futil import read_das, read_ttrees, trees_from_doc, sentences_from_doc
from planner import ASearchPlanner
from candgen import RandomCandidateGenerator
from eval import Evaluator, EvalTypes
from tree import TreeNode
from tgen.eval import ASearchListsAnalyzer
from tgen.rnd import rnd


class Inst(namedtuple('Inst', ['tree', 'da', 'feats', 'score'])):
    """A holder for one data instance (input tree, output DA, extracted features, ranker score)."""
    pass


class Ranker(object):
    """Base class for rankers."""

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


class BasePerceptronRanker(Ranker):

    def __init__(self, cfg):
        if not cfg:
            cfg = {}
        self.passes = cfg.get('passes', 5)
        self.alpha = cfg.get('alpha', 1)
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        # initialize diagnostics
        self.lists_analyzer = None
        self.evaluator = None
        self.prune_feats = cfg.get('prune_feats', 1)
        self.rival_number = cfg.get('rival_number', 10)
        self.averaging = cfg.get('averaging', False)
        self.randomize = cfg.get('randomize', False)
        self.future_promise_weight = cfg.get('future_promise_weight', 1.0)
        self.future_promise_type = cfg.get('future_promise_type', 'expected_children')
        self.rival_gen_strategy = cfg.get('rival_gen_strategy', ['other_inst'])
        self.rival_gen_max_iter = cfg.get('rival_gen_max_iter', 50)
        self.rival_gen_max_defic_iter = cfg.get('rival_gen_max_defic_iter', 3)
        self.rival_gen_beam_size = cfg.get('rival_gen_beam_size')
        self.rival_gen_prune_size = cfg.get('rival_gen_prune_size')
        self.candgen_model = cfg.get('candgen_model')
        self.diffing_trees = cfg.get('diffing_trees', False)

    def score(self, cand_tree, da):
        """Score the given tree in the context of the given dialogue act.
        @param cand_tree: the candidate tree to be scored, as a TreeData object
        @param da: a DA object representing the input dialogue act
        """
        return self._score(self._extract_feats(cand_tree, da))

    def score_all(self, cand_trees, da):
        """Array version of the score() function"""
        return [self.score(cand_tree, da) for cand_tree in cand_trees]

    def _extract_feats(self, tree, da):
        raise NotImplementedError

    def train(self, das_file, ttree_file, data_portion=1.0):
        """Run training on the given training data."""
        self._init_training(das_file, ttree_file, data_portion)
        for iter_no in xrange(1, self.passes + 1):
            self.train_order = range(len(self.train_trees))
            if self.randomize:
                rnd.shuffle(self.train_order)
            log_info("Train order: " + str(self.train_order))
            self._training_pass(iter_no)
            if self.evaluator.tree_accuracy() == 1:  # if tree accuracy is 1, we won't learn anything anymore
                break
        # averaged perceptron – average the weights obtained after each pass
        if self.averaging is True:
            self.set_weights_iter_average()

    def _init_training(self, das_file, ttree_file, data_portion):
        """Initialize training (read input data, fix size, initialize candidate generator
        and planner)"""
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
        self.train_order = range(len(self.train_trees))
        log_info('Using %d training instances.' % train_size)

        # initialize candidate generator
        if self.candgen_model is not None:
            self.candgen = RandomCandidateGenerator.load_from_file(self.candgen_model)
#             self.sampling_planner = SamplingPlanner({'language': self.language,
#                                                      'selector': self.selector,
#                                                      'candgen': self.candgen})

        # check if A*search planner is needed (i.e., any rival generation strategy requires it)
        # and initialize it
        if isinstance(self.rival_gen_strategy[0], tuple):
            asearch_needed = any([s in ['gen_cur_weights', 'gen_update']
                                  for _, ss in self.rival_gen_strategy
                                  for s in ss])
        else:
            asearch_needed = any([s in ['gen_cur_weights', 'gen_update']
                                  for s in self.rival_gen_strategy])
        if asearch_needed:
            assert self.candgen is not None
            self.asearch_planner = ASearchPlanner({'candgen': self.candgen,
                                                   'language': self.language,
                                                   'selector': self.selector,
                                                   'ranker': self, })

    def _training_pass(self, pass_no):
        """Run one training pass, update weights (store them for possible averaging),
        and store diagnostic values."""

        pass_start_time = time.time()
        self.reset_diagnostics()
        self.update_weights_sum()

        log_debug('\n***\nTR %05d:' % pass_no)

        rgen_max_iter = self._get_num_iters(pass_no, self.rival_gen_max_iter)
        rgen_max_defic_iter = self._get_num_iters(pass_no, self.rival_gen_max_defic_iter)
        rgen_beam_size = self.rival_gen_beam_size
        rgen_prune_size = self.rival_gen_prune_size
        rgen_strategy = self._get_rival_gen_strategy(pass_no)

        for tree_no in self.train_order:

            log_debug('TREE-NO: %d' % tree_no)
            log_debug('SENT: %s' % self.train_sents[tree_no])

            gold = Inst(da=self.train_das[tree_no],
                        tree=self.train_trees[tree_no],
                        score=self._score(self.train_feats[tree_no]),
                        feats=self.train_feats[tree_no])

            # obtain some 'rival', alternative incorrect candidates
            for strategy in rgen_strategy:

                # generate using current weights
                if strategy == 'gen_cur_weights':
                    gen = self._gen_cur_weights(gold, rgen_max_iter, rgen_max_defic_iter,
                                                rgen_prune_size, rgen_beam_size)

                # generate while trying to update weights
                elif strategy == 'gen_update':
                    gen = self._gen_update(gold, rgen_max_iter, rgen_max_defic_iter,
                                           rgen_prune_size, rgen_beam_size)

                # check against other possible candidates/combinations
                else:
                    gen = self._get_rival_candidates(gold, tree_no, strategy)

                # evaluate the top-scoring generated tree against gold t-tree
                # (disregarding whether it was selected as the best one)
                self.evaluator.append(TreeNode(gold.tree), TreeNode(gen.tree), gold.score, gen.score)

                # update weights if the system doesn't give the highest score to the gold standard tree
                if gold.score < gen.score:
                    self._update_weights(gold, gen)

        # store a copy of the current weights for averaging
        self.store_iter_weights()

        # debug print: current weights and pass accuracy
        log_debug(self._feat_val_str(), '\n***')
        log_debug('PASS ACCURACY: %.3f' % self.evaluator.tree_accuracy())

        # print and return statistics
        self._print_pass_stats(pass_no, datetime.timedelta(seconds=(time.time() - pass_start_time)))

    def diffing_trees_with_scores(self, da, good_tree, bad_tree):
        """For debugging purposes. Return a printout of diffing trees between the chosen candidate
        and the gold tree, along with scores."""
        good_sts, bad_sts = good_tree.diffing_trees(bad_tree, symmetric=False)
        comm_st = good_tree.get_common_subtree(bad_tree)
        ret = 'Common subtree: %.3f' % self.score(comm_st, da) + "\t" + unicode(comm_st) + "\n"
        ret += "Good subtrees:\n"
        for good_st in good_sts:
            ret += "%.3f" % self.score(good_st, da) + "\t" + unicode(good_st) + "\n"
        ret += "Bad subtrees:\n"
        for bad_st in bad_sts:
            ret += "%.3f" % self.score(bad_st, da) + "\t" + unicode(bad_st) + "\n"
        return ret

    def _get_num_iters(self, cur_pass_no, iter_setting):
        """Return the maximum number of iterations (total/deficit) given the current pass.
        Used to keep track of variable iteration number setting in configuration.

        @param cur_pass_no: the number of the current pass
        @param iter_setting: number of iteration setting (self.max_iter or self.max_defic_iter)
        """
        if isinstance(iter_setting, (list, tuple)):
            ret = 0
            for set_pass_no, set_iter_no in iter_setting:
                if set_pass_no > cur_pass_no:
                    break
                ret = set_iter_no
            return ret
        else:
            return iter_setting  # a single setting for all passes

    def _get_rival_gen_strategy(self, cur_pass_no):
        """Return the rival generation strategy/strategies for the current pass.
        Used to keep track of variable rival generation setting setting in configuration.

        @param cur_pass_no: the number of the current pass
        """
        if isinstance(self.rival_gen_strategy[0], tuple):
            ret = []
            for set_pass_no, strategies in self.rival_gen_strategy:
                if set_pass_no > cur_pass_no:
                    break
                ret = strategies
            return ret
        else:
            return self.rival_gen_strategy  # a single setting for all passes

    def _print_pass_stats(self, pass_no, pass_duration):
        """Print pass statistics from internal evaluator fields and given pass duration."""
        log_info('Pass %05d -- tree-level accuracy: %.4f' % (pass_no, self.evaluator.tree_accuracy()))
        log_info(' * Generated trees NODE scores: P: %.4f, R: %.4f, F: %.4f' %
                 self.evaluator.p_r_f1())
        log_info(' * Generated trees DEP  scores: P: %.4f, R: %.4f, F: %.4f' %
                 self.evaluator.p_r_f1(EvalTypes.DEP))
        log_info(' * Gold tree BEST: %.4f, on CLOSE: %.4f, on ANY list: %.4f' %
                 self.lists_analyzer.stats())
        log_info(' * Tree size stats:\n -- GOLD: %s\n -- PRED: %s\n -- DIFF: %s' %
                 self.evaluator.size_stats())
        log_info(' * Common subtree stats:\n -- SIZE: %s\n -- ΔGLD: %s\n -- ΔPRD: %s' %
                 self.evaluator.common_substruct_stats())
        log_info(' * Score stats\n -- GOLD: %s\n -- PRED: %s\n -- DIFF: %s'
                 % self.evaluator.score_stats())
        log_info(' * Duration: %s' % str(pass_duration))

    def _feat_val_str(self, sep='\n', nonzero=False):
        """Return feature names and values for printing. To be overridden in base classes."""
        return ''

    def _get_rival_candidates(self, gold, tree_no, strategy):
        """Generate some rival candidates for a DA and the correct (gold) tree,
        given a strategy; using other DAs for the correct tree, other trees for the correct
        DA, or random trees.

        NB: This has not been shown to be usable in practice; use _gen_cur_weights() instead.

        TODO: checking for trees identical to the gold one slows down the process

        @param tree_no: the index of the current training data item (tree, DA)
        @rtype: tuple of two lists: one of TreeData's, one of arrays
        @return: an array of rival trees and an array of the corresponding features
        """
        train_trees = self.train_trees

        rival_das, rival_trees, rival_feats = [], [], []

        if strategy != 'other_da':
            rival_das = [gold.da] * self.rival_number

        # use current DA but change trees when computing features
        if strategy == 'other_inst':
            # use alternative indexes, avoid the correct one
            rival_idxs = map(lambda idx: len(train_trees) - 1 if idx == tree_no else idx,
                             rnd.sample(xrange(len(train_trees) - 1), self.rival_number))
            other_inst_trees = [train_trees[rival_idx] for rival_idx in rival_idxs]
            rival_trees.extend(other_inst_trees)
            rival_feats.extend([self._extract_feats(tree, gold.da) for tree in other_inst_trees])

        # use the current gold tree but change DAs when computing features
        if strategy == 'other_da':
            rival_idxs = map(lambda idx: len(train_trees) - 1 if idx == tree_no else idx,
                             rnd.sample(xrange(len(train_trees) - 1), self.rival_number))
            other_inst_das = [self.train_das[rival_idx] for rival_idx in rival_idxs]
            rival_das.extend(other_inst_das)
            rival_trees.extend([self.train_trees[tree_no]] * self.rival_number)
            rival_feats.extend([self._extract_feats(self.train_trees[tree_no], da)
                                for da in other_inst_das])

#         # candidates generated using the random planner (use the current DA)
#         if strategy == 'random':
#             random_trees = []
#             while len(random_trees) < self.rival_number:
#                 tree = self.sampling_planner.generate_tree(da)
#                 if (tree != train_trees[tree_no]):  # don't generate trees identical to the gold one
#                     random_trees.append(tree)
#             rival_trees.extend(random_trees)
#             rival_feats.extend([self._extract_feats(tree, da) for tree in random_trees])

        # score them along with the right one
        rival_scores = [self._score(r) for r in rival_feats]
        top_rival_idx = rival_scores.index(max(rival_scores))
        gen = Inst(tree=rival_trees[top_rival_idx],
                   da=rival_das[top_rival_idx],
                   score=rival_scores[top_rival_idx],
                   feats=rival_feats[top_rival_idx])

        # debug print: candidate trees
        log_debug('#RIVALS: %02d' % len(rival_feats))
        log_debug('SEL: GOLD' if gold.score >= gen.score else ('SEL: RIVAL#%d' % top_rival_idx))
        log_debug('ALL CAND TREES:')
        for ttree, score in zip([gold.tree] + rival_trees, [gold.score] + rival_scores):
            log_debug("%12.5f" % score, "\t", ttree)

        return gen

    def _gen_cur_weights(self, gold, max_iter, max_defic_iter, prune_size, beam_size):
        """
        Get the best candidate generated using the A*search planner, which uses this ranker with current
        weights to guide the search, and the current DA as the input.

        @param gold: the gold-standard Inst holding the input DA for generation and the reference tree
        @param max_iter: maximum number of A*-search iterations to run
        @param max_defic_iter: maximum number of deficit A*-search iterations (stopping criterion)
        @param prune_size: beam size for open list pruning
        @param beam_size: beam size for candidate expansion (expand more per iteration if > 1)
        @return: The best generated tree that is different from the gold-standard tree
        @rtype: Inst
        """
        log_debug('GEN-CUR-WEIGHTS')
        # TODO make asearch_planner remember features (for last iteration, maybe)
        self.asearch_planner.run(gold.da, max_iter, max_defic_iter, prune_size, beam_size)
        return self.get_best_generated(gold)

    def get_best_generated(self, gold):
        """Return the best generated tree that is different from the gold-standard tree
        (to be used for updates, if it scores better). Also, keep track of logging and
        update analyzer lists.

        @param gold: the gold-standard Inst from which the generated tree must differ
        @rtype: Inst
        """
        self.lists_analyzer.append(gold.tree,
                                   self.asearch_planner.open_list,
                                   self.asearch_planner.close_list)

        gen_tree = gold.tree
        while self.asearch_planner.close_list and gen_tree == gold.tree:
            gen_tree, gen_score = self.asearch_planner.close_list.pop()

        # scores are negative on the close list – reverse the sign
        gen = Inst(tree=gen_tree, da=gold.da, score=-gen_score,
                   feats=self._extract_feats(gen_tree, gold.da))
        log_debug('SEL: GOLD' if gold.score >= gen.score else 'SEL: GEN')
        log_debug("GOLD:\t", "%12.5f" % gold.score, "\t", gold.tree)
        log_debug("GEN :\t", "%12.5f" % gen.score, "\t", gen.tree)
        return gen

    def _gen_update(self, gold, max_iter, max_defic_iter, prune_size, beam_size):
        """Try generating using the current weights, but update the weights after each
        iteration if the result is not going in the right direction (not a subtree of the
        gold-standard tree).

        @param gold: the gold-standard Inst holding the input DA for generation and the reference tree
        @param max_iter: maximum number of A*-search iterations to run
        @param max_defic_iter: maximum number of deficit A*-search iterations (stopping criterion)
        @param prune_size: beam size for open list pruning
        @param beam_size: beam size for candidate expansion (expand more per iteration if > 1)
        @return: The best generated tree that is different from the gold-standard tree
        @rtype: Inst
        """

        log_debug('GEN-UPDATE')
        self.asearch_planner.init_run(gold.da, max_iter, max_defic_iter, prune_size, beam_size)

        while not self.asearch_planner.check_finalize():
            # run one A*search iteration
            self.asearch_planner.run_iter()

            # stop if there's nothing on the open list
            if not self.asearch_planner.open_list:
                break

            # look if we are on the right track to the gold tree
            cur_top, score = self.asearch_planner.open_list.peek()
            csi, _ = gold.tree.common_subtree_idxs(cur_top)

            # if not, update
            if len(csi) != len(cur_top):

                feats = self._extract_feats(cur_top, gold.da)
                gen = Inst(tree=cur_top, da=gold.da, feats=feats, score=score)

                # for small wrong trees,
                # fake the current open list to only include a subtree of the gold tree
                # TODO fake it better, include more variants
                # update using a subtree of the gold tree
                if len(cur_top) < len(gold.tree):
                    diff = sorted(list(set(range(len(gold.tree))) - set(csi)),
                                  cmp=gold.tree._compare_node_depth)
                    gold_sub = gold.tree.get_subtree(csi + diff[0:len(cur_top) - len(gold.tree)])

                    self.asearch_planner.open_list.clear()
                    self.asearch_planner.open_list.push(gold_sub, score)
                    # TODO speed up by remembering the features in planner
                    feats = self._extract_feats(gold_sub, gold.da)
                    gold_sub = Inst(tree=gold_sub, da=gold.da, feats=feats, score=0)
                    self._update_weights(gold_sub, gen)

                # otherwise, update using the full gold tree
                else:
                    self._update_weights(gold, gen)

        return self.get_best_generated(gold)

    def get_weights(self):
        """Return the current ranker weights (parameters). To be overridden by derived classes."""
        raise NotImplementedError

    def set_weights(self, w):
        """Set new ranker weights. To be overridden by derived classes."""
        raise NotImplementedError

    def set_weights_average(self, ws):
        """Set the weights as the average of the given array of weights (used in parallel training).
        To be overridden by derived classes."""
        raise NotImplementedError

    def store_iter_weights(self):
        """Remember the current weights to be used for averaging.
        To be overridden by derived classes."""
        raise NotImplementedError

    def set_weights_iter_average(self):
        """Set new weights as the average of all remembered weights. To be overridden by
        derived classes."""
        raise NotImplementedError

    def get_weights_sum(self):
        """Return weights size in order to weigh future promise against them."""
        raise NotImplementedError

    def update_weights_sum(self):
        """Update the current weights size for future promise weighining."""
        raise NotImplementedError

    def reset_diagnostics(self):
        """Reset the evaluation statistics (Evaluator and ASearchListsAnalyzer objects)."""
        self.evaluator = Evaluator()
        self.lists_analyzer = ASearchListsAnalyzer()

    def get_diagnostics(self):
        """Return the current evaluation statistics (a tuple of Evaluator and ASearchListsAnalyzer
        objects."""
        return self.evaluator, self.lists_analyzer

    def set_diagnostics_average(self, diags):
        """Given an array of evaluation statistics objects, average them an store in this ranker
        instance."""
        self.reset_diagnostics()
        for evaluator, lists_analyzer in diags:
            self.evaluator.merge(evaluator)
            self.lists_analyzer.merge(lists_analyzer)

    def get_future_promise(self, cand_tree):
        """Compute expected future promise for a tree."""
        w_sum = self.get_weights_sum()
        if self.future_promise_type == 'num_nodes':
            return w_sum * self.future_promise_weight * max(0, 10 - len(cand_tree))
        elif self.future_promise_type == 'norm_exp_children':
            return (self.candgen.get_future_promise(cand_tree) / len(cand_tree)) * w_sum * self.future_promise_weight
        elif self.future_promise_type == 'ands':
            prom = 0
            for idx, node in enumerate(cand_tree.nodes):
                if node.t_lemma == 'and':
                    num_kids = cand_tree.children_num(idx)
                    prom += max(0, 2 - num_kids)
            return prom * w_sum * self.future_promise_weight
        else:  # expected children (default)
            return self.candgen.get_future_promise(cand_tree) * w_sum * self.future_promise_weight

    def get_future_promise_all(self, cand_trees):
        """Array version of get_future_promise."""
        return [self.get_future_promise(cand_tree) for cand_tree in cand_trees]


class FeaturesPerceptronRanker(BasePerceptronRanker):
    """Base class for global ranker for whole trees, based on features."""

    def __init__(self, cfg):
        super(FeaturesPerceptronRanker, self).__init__(cfg)
        if not cfg:
            cfg = {}
        self.feats = ['bias: bias']
        self.vectorizer = None
        self.normalizer = None
        self.binarize = cfg.get('binarize', False)
        # initialize feature functions
        if 'features' in cfg:
            self.feats.extend(cfg['features'])
        self.feats = Features(self.feats, cfg.get('intermediate_features', []))

    def _extract_feats(self, tree, da):
        feats = self.vectorizer.transform([self.feats.get_features(tree, {'da': da})])
        if self.normalizer:
            feats = self.normalizer.transform(feats)
        return feats[0]

    def _init_training(self, das_file, ttree_file, data_portion):

        super(FeaturesPerceptronRanker, self)._init_training(das_file, ttree_file, data_portion)

        # precompute training data features
        X = []
        for da, tree in zip(self.train_das, self.train_trees):
            X.append(self.feats.get_features(tree, {'da': da}))
        if self.prune_feats > 1:
            self._prune_features(X)
        # vectorize and binarize or normalize (+train vectorizer/normalizer)
        if self.binarize:
            self.vectorizer = DictVectorizer(sparse=False, binarize_numeric=True)
            self.train_feats = self.vectorizer.fit_transform(X)
        else:
            self.vectorizer = DictVectorizer(sparse=False)
            self.normalizer = StandardScaler(copy=False)
            self.train_feats = self.normalizer.fit_transform(self.vectorizer.fit_transform(X))

        log_info('Features matrix shape: %s' % str(self.train_feats.shape))

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


class PerceptronRanker(FeaturesPerceptronRanker):
    """Perceptron ranker based on linear Perceptron by Collins & Duffy (2002)."""

    def __init__(self, cfg):
        super(PerceptronRanker, self).__init__(cfg)
        self.w_after_iter = []
        self.w = None
        self.w_sum = 0.0

    def __setstate__(self, state):
        """Backward compatibility – adding members missing in older versions."""
        if 'normalizer' not in state:
            state['binarize'] = True
            state['normalizer'] = None
        if 'binarize' not in state:
            state['binarize'] = False
        self.__dict__ = state

    def _init_training(self, das_file, ttree_file, data_portion):
        # load data, determine number of features etc. etc.
        super(PerceptronRanker, self)._init_training(das_file, ttree_file, data_portion)
        # initialize weights
        self.w = np.ones(self.train_feats.shape[1])
        self.update_weights_sum()
        # self.w = np.array([rnd.gauss(0, self.alpha) for _ in xrange(self.train_feats.shape[1])])

        log_debug('\n***\nINIT:')
        log_debug(self._feat_val_str())
        log_info('Training ...')

    def _score(self, cand_feats):
        return np.dot(self.w, cand_feats)

    def _update_weights(self, good, bad):
        """Perform a perceptron weights update (not the check if we need to update).
        Also perform differing tree updates."""
        # discount trees leading to the generated one and add trees leading to the gold one
        if self.diffing_trees:
            good_sts, bad_sts = good.tree.diffing_trees(bad.tree,
                                                        symmetric=self.diffing_trees.startswith('sym'))
            # if set, discount common subtree's features from all subtrees' features
            discount = None
            if 'nocom' in self.diffing_trees:
                discount = self._extract_feats(good.tree.get_common_subtree(bad.tree), good.da)
            # add good trees (leading to gold)
            for good_st in good_sts:
                good_feats = self._extract_feats(good_st, good.da)
                if discount is not None:
                    good_feats -= discount
                good_tree_w = 1
                if self.diffing_trees.endswith('weighted'):
                    good_tree_w = len(good_st) / float(len(good.tree))
                self.w += self.alpha * good_tree_w * good_feats
            # discount bad trees (leading to the generated one)
            if 'nobad' in self.diffing_trees:
                bad_sts = []
            elif 'onebad' in self.diffing_trees:
                bad_sts = [bad.tree]
            for bad_st in bad_sts:
                bad_feats = self._extract_feats(bad_st, bad.da)
                if discount is not None:
                    bad_feats -= discount
                bad_tree_w = 1
                if self.diffing_trees.endswith('weighted'):
                    bad_tree_w = len(bad_st) / float(len(bad.tree))
                self.w -= self.alpha * bad_tree_w * bad_feats
        # just discount the best generated tree and add the gold tree
        else:
            self.w += (self.alpha * good.feats - self.alpha * bad.feats)
        # # log_debug('Updated  w: ' + str(np.frombuffer(self.w, "uint8").sum()))

    def _feat_val_str(self, sep='\n', nonzero=False):
        return ''

    def get_weights(self):
        """Return the current perceptron ranker weights."""
        return self.w

    def set_weights(self, w):
        """Set new perceptron ranker weights."""
        self.w = w

    def set_weights_average(self, ws):
        """Set the weights as the average of the given array of weights (used in parallel training)."""
        self.w = np.average(ws, axis=0)

    def store_iter_weights(self):
        """Remember the current weights to be used for averaged perceptron."""
        self.w_after_iter.append(np.copy(self.w))

    def set_weights_iter_average(self):
        """Average the remembered weights."""
        self.w = np.average(self.w_after_iter, axis=0)

    def get_weights_sum(self):
        """Return the sum of weights (at start of current iteration) to be used to weigh future
        promise."""
        return self.w_sum

    def update_weights_sum(self):
        """Update the current weights sum figure."""
        self.w_sum = sum(self.w)
