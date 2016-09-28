#!/usr/bin/env python
# coding=utf-8

"""
Evaluation (t-tree comparison functions).
"""

from __future__ import unicode_literals
from collections import defaultdict
from enum import Enum
from tgen.logf import log_debug, log_warn, log_info
from tgen.tree import TreeData, TreeNode
from tgen.futil import add_bundle_text
import numpy as np

try:
    from pytreex.core.node import T
except ImportError:
    log_warn('Pytreex modules not available, will not be able to evaluate trees.')


EvalTypes = Enum(b'EvalTypes', b'TOKEN NODE DEP')
EvalTypes.__doc__ = """Evaluation flavors (tokens, tree node-only, tree dependency)"""


def collect_counts(sent, eval_type=EvalTypes.NODE):
    """Collects counts of different node/dependency types in the given t-tree.

    @param sent: the tree/sentence to collect counts from
    @param eval_type: if set to EvalTypes.NODE (default), count nodes (formemes, lemmas, dependency \
        direction), if set to EvalTypes.DEP, count dependencies (including parent's formeme, lemma, \
        dependency direction), if set to EvalTypes.TOKEN, count just word forms (in list of tokens).
    @rtype: defaultdict
    """
    counts = defaultdict(int)
    nodes = sent if isinstance(sent, list) else sent.get_descendants()
    for node in nodes:
        if eval_type == EvalTypes.TOKEN:
            node_id = node[0]  # for tokens, use form only (ignore tag)
        elif eval_type == EvalTypes.NODE:
            node_id = (node.formeme, node.t_lemma, node > node.parent)
        else:
            parent = node.parent
            node_id = (node.formeme, node.t_lemma, node > node.parent,
                       parent.formeme, parent.t_lemma, (parent.parent is not None and parent > parent.parent))
        counts[node_id] += 1
    return counts


def corr_pred_gold(gold, pred, eval_type=EvalTypes.NODE):
    """Given a golden tree/sentence and a predicted tree/sentence, this counts correctly
    predicted nodes/tokens (true positives), all predicted nodes/tokens (true + false
    positives), and all golden nodes/tokens (true positives + false negatives).

    @param gold: a golden t-tree/sentence
    @param pred: a predicted t-tree/sentence
    @param eval_type: type of matching (see EvalTypes)
    @rtype: tuple
    @return: numbers of correctly predicted, total predicted, and total golden nodes/tokens
    """
    gold_counts = collect_counts(gold, eval_type)
    pred_counts = collect_counts(pred, eval_type)
    ccount, pcount = 0, 0
    for node_id, node_count in pred_counts.iteritems():
        pcount += node_count
        ccount += min(node_count, gold_counts[node_id])
    gcount = sum(node_count for node_count in gold_counts.itervalues())
    return ccount, pcount, gcount


def precision(gold, pred, eval_type=EvalTypes.NODE):
    ccount, pcount, _ = corr_pred_gold(gold, pred, eval_type)
    return ccount / float(pcount)


def recall(gold, pred, eval_type=EvalTypes.NODE):
    # # correct / # gold
    ccount, _, gcount = corr_pred_gold(gold, pred, eval_type)
    return ccount / float(gcount)


def f1(gold, pred, eval_type=EvalTypes.NODE):
    return f1_from_counts(corr_pred_gold(gold, pred, eval_type))


def f1_from_counts(correct, predicted, gold):
    return p_r_f1_from_counts(correct, predicted, gold)[2]


def p_r_f1_from_counts(correct, predicted, gold):
    """Return precision, recall, and F1 given counts of true positives (correct),
    total predicted nodes, and total gold nodes.

    @param correct: true positives (correctly predicted nodes/tokens)
    @param predicted: true + false positives (all predicted nodes/tokens)
    @param gold: true positives + false negatives (all golden nodes/tokens)
    @rtype: tuple
    @return: precision, recall, F1
    """
    if correct == 0.0:  # escape division by zero
        return 0.0, 0.0, 0.0
    precision = correct / float(predicted)
    recall = correct / float(gold)
    return precision, recall, (2 * precision * recall) / (precision + recall)


def to_treedata(t):
    if isinstance(t, TreeNode):
        return t.tree
    elif isinstance(t, T):
        return TreeData.from_ttree(t)


def common_subtree_size(a, b):
    a = to_treedata(a)
    b = to_treedata(b)
    return a.common_subtree_size(b)


def max_common_subphrase_length(a, b):
    """Return the length of the longest common subphrase of a and b; where a and b are
    lists of tokens (form+tag)."""
    longest = 0
    for sp_a in xrange(len(a)):
        for sp_b in xrange(len(b)):
            pos_a = sp_a
            pos_b = sp_b
            # disregard tags for comparison
            while pos_a < len(a) and pos_b < len(b) and a[pos_a][0] == b[pos_b][0]:
                pos_a += 1
                pos_b += 1
            if pos_a - sp_a > longest:
                longest = pos_a - sp_a
    return longest


class Stats:
    """A set of important statistic values, with simple access and printing."""

    def __init__(self, data):
        self.mean = np.mean(data)
        self.median = np.median(data)
        self.min = min(data)
        self.max = max(data)
        self.perc25 = np.percentile(data, 25)
        self.perc75 = np.percentile(data, 75)

    def __str__(self):
        return "\t".join("%s: %9.3f" % (key.capitalize(), getattr(self, key))
                         for key in ['mean', 'median', 'min', 'max', 'perc25', 'perc75'])


class Evaluator(object):
    """A fancy object-oriented interface to computing node F-scores.

    Accumulates scores over trees/sentences using append(), then can return
    a total score using f1(), precision(), recall(), and p_r_f1()."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Zero out all current statistics, start from scratch."""
        self.correct = {eval_type: 0 for eval_type in EvalTypes}
        self.predicted = {eval_type: 0 for eval_type in EvalTypes}
        self.gold = {eval_type: 0 for eval_type in EvalTypes}
        self.sizes = []
        self.scores = []

    def process_eval_doc(self, eval_doc, gen_trees, language, ref_selector, target_selector):
        """Evaluate generated trees against a reference document; save per-tree statistics
        in the reference document and print out global statistics.

        Does not reset statistics at the beginning (must be reset manually if needed).

        @param eval_doc: reference t-tree document
        @param gen_trees: a list of generated TreeData objects
        @param language: language for the reference document
        @param ref_selector: selector for reference trees in the reference document
        @param target_selector: selector for generated trees (used to save statistics)
        """
        log_info('Evaluating...')
        for eval_bundle, gen_tree, in zip(eval_doc.bundles, gen_trees):
            # add some stats about the tree directly into the output file
            eval_ttree = eval_bundle.get_zone(language, ref_selector).ttree
            gen_ttree = TreeNode(gen_tree)
            add_bundle_text(eval_bundle, language, target_selector + 'Xscore',
                            "P: %.4f R: %.4f F1: %.4f" %
                            p_r_f1_from_counts(*corr_pred_gold(eval_ttree, gen_ttree)))
            # collect overall stats
            # TODO maybe add cost somehow?
            self.append(eval_ttree, gen_ttree)

        # print out the overall stats
        log_info("NODE precision: %.4f, Recall: %.4f, F1: %.4f" % self.p_r_f1())
        log_info("DEP  precision: %.4f, Recall: %.4f, F1: %.4f" % self.p_r_f1(EvalTypes.DEP))
        log_info("Tree size stats:\n * GOLD %s\n * PRED %s\n * DIFF %s" % self.size_stats())
        log_info("Score stats:\n * GOLD %s\n * PRED %s\n * DIFF %s" % self.score_stats())
        log_info("Common subtree stats:\n -- SIZE: %s\n -- ΔGLD: %s\n -- ΔPRD: %s" %
                 self.common_substruct_stats())

    def append(self, gold, pred, gold_score=0.0, pred_score=0.0):
        """Add a pair of golden and predicted tree/sentence to the current statistics.

        @param gold: a T or TreeNode object representing the golden tree, or list of golden tokens
        @param pred: a T or TreeNode object representing the predicted tree, or list of predicted \
            tokens
        """
        if isinstance(gold, list):  # tokens
            eval_types = [EvalTypes.TOKEN]
            gold_len = len(gold)
            pred_len = len(pred)
            css = max_common_subphrase_length(gold, pred)
        else:  # trees
            eval_types = [EvalTypes.NODE, EvalTypes.DEP]
            gold_len = len(gold.get_descendants())
            pred_len = len(pred.get_descendants())
            css = common_subtree_size(gold, pred)
        self.sizes.append((gold_len, pred_len, css))

        for eval_type in eval_types:
            ccount, pcount, gcount = corr_pred_gold(gold, pred, eval_type)
            self.correct[eval_type] += ccount
            self.predicted[eval_type] += pcount
            self.gold[eval_type] += gcount
        self.scores.append((gold_score, pred_score))

    def merge(self, other):
        """Merge in statistics from another Evaluator object."""
        for eval_type in EvalTypes:
            self.correct[eval_type] += other.correct[eval_type]
            self.predicted[eval_type] += other.predicted[eval_type]
            self.gold[eval_type] += other.gold[eval_type]
        self.sizes.extend(other.sizes)
        self.scores.extend(other.scores)

    def f1(self, eval_type=EvalTypes.NODE):
        return self.p_r_f1(eval_type)[2]

    def precision(self, eval_type=EvalTypes.NODE):
        return self.p_r_f1(eval_type)[0]

    def recall(self, eval_type=EvalTypes.NODE):
        return self.p_r_f1(eval_type)[1]

    def p_r_f1(self, eval_type=EvalTypes.NODE):
        return p_r_f1_from_counts(self.correct[eval_type],
                                  self.predicted[eval_type],
                                  self.gold[eval_type])

    def size_stats(self):
        """Return current tree/sentence size statistics.
        @rtype: a 3-tuple of Stats objects
        @return: statistics for golden trees/sentences, predicted trees/sentences, and differences
        """
        return (Stats([inst[0] for inst in self.sizes]),
                Stats([inst[1] for inst in self.sizes]),
                Stats([inst[0] - inst[1] for inst in self.sizes]))

    def common_substruct_stats(self):
        """Return common subtree/subphrase size statistics.
        @rtype: a 3-tuple of Stats objects
        @return: statistics for common subtree/subphrase size + sizes of what's missing to full \
            gold/predicted tree/sentence
        """
        return (Stats([inst[2] for inst in self.sizes]),
                Stats([inst[0] - inst[2] for inst in self.sizes]),
                Stats([inst[1] - inst[2] for inst in self.sizes]))

    def score_stats(self):
        """Return tree/sentence score statistics.
        @rtype: a 3-tuple of Stats objects
        @return: statistics for golden trees/sentences, predicted trees/sentences, and differences
        """
        return (Stats([inst[0] for inst in self.scores]),
                Stats([inst[1] for inst in self.scores]),
                Stats([inst[0] - inst[1] for inst in self.scores]))

    def tree_accuracy(self):
        """Return tree-level accuracy (percentage of gold trees scored higher or equal to
        the best predicted tree."""
        return (sum(1 for gold_score, pred_score in self.scores if gold_score >= pred_score) /
                float(len(self.scores)))


class ASearchListsAnalyzer(object):
    """Analysis of the final open and close lists of the A*search generator."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Zero all statistics."""
        self.total = 0
        self.gold_best = 0
        self.gold_on_close = 0
        self.gold_on_open = 0

    def append(self, gold_tree, open_list, close_list):
        """Analyze the open and close lists of a generator for the presence of the gold-standard
        tree and add the results to statistics."""
        self.total += 1
        best_tree = close_list.peek()[0]
        if gold_tree == best_tree:
            self.gold_best += 1
            log_debug('GOLD TREE IS BEST')
        if gold_tree in close_list:
            self.gold_on_close += 1
            log_debug('GOLD TREE IS ON CLOSE LIST')
        if gold_tree in open_list:
            self.gold_on_open += 1
            log_debug('GOLD TREE IS ON OPEN LIST')

    def merge(self, other):
        """Merge in another ASearchListsAnalyzer object."""
        self.total += other.total
        self.gold_best += other.gold_best
        self.gold_on_close += other.gold_on_close
        self.gold_on_open += other.gold_on_open

    def stats(self):
        """Return statistics (as percentages): gold tree was best, gold tree was on
        close list, gold tree was on open list.
        @rtype: tuple
        """
        if self.total == 0:
            return (0.0, 0.0, 0.0)
        tot = float(self.total)
        return (self.gold_best / tot,
                self.gold_on_close / tot,
                (self.gold_on_close + self.gold_on_open) / tot)


class SlotErrAnalyzer(object):
    """Analyze slot error (as in Wen 2015 EMNLP paper), accumulator object."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Zero all statistics."""
        self.missing = 0
        self.superfluous = 0
        self.total = 0

    def append(self, da, sent):
        """Include statistics from the given sentence (assuming tokens, not trees)."""
        if sent and isinstance(sent[0], tuple):
            sent = [form for form, pos in sent]  # ignore POS
        if isinstance(da, tuple):
            da = da[1]  # ignore contexts

        slots_in_da = set([dai.value for dai in da if dai.value and dai.value.startswith('X-')])
        slots_in_sent = set([tok for tok in sent if tok.startswith('X-')])
        self.total += len(slots_in_da)
        self.missing += len(slots_in_da - slots_in_sent)
        self.superfluous += len(slots_in_sent - slots_in_da)

    def slot_error(self):
        """Return the currently accumulated slot error."""
        if self.total == 0:  # avoid zero division error
            return 0
        return (self.missing + self.superfluous) / float(self.total)
