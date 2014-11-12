#!/usr/bin/env python
# coding=utf-8

"""
Evaluation (t-tree comparison functions).
"""

from __future__ import unicode_literals
from collections import defaultdict
from enum import Enum
from tgen.logf import log_debug
from tgen.tree import TreeNode
import numpy as np


EvalTypes = Enum(b'EvalTypes', b'NODE DEP')
EvalTypes.__doc__ = """Evaluation flavors (node-only, dependency)"""


def collect_counts(ttree, eval_type=EvalTypes.NODE):
    """Collects counts of different node/dependency types in the given t-tree.

    @param ttree: the tree to collect counts from
    @param eval_type: if set to 'node' (default), count nodes (formemes, lemmas, dependency \
        direction), if set to 'dep', count dependencies (including parent's formeme, lemma, \
        dependency direction).
    @rtype: defaultdict
    """
    counts = defaultdict(int)
    for node in ttree.get_descendants():
        if eval_type == EvalTypes.NODE:
            node_id = (node.formeme, node.t_lemma, node > node.parent)
        else:
            parent = node.parent
            node_id = (node.formeme, node.t_lemma, node > node.parent,
                       parent.formeme, parent.t_lemma, (parent.parent is not None and parent > parent.parent))
        counts[node_id] += 1
    return counts


def corr_pred_gold(gold_ttree, pred_ttree, eval_type=EvalTypes.NODE):
    """Given a golden tree and a predicted tree, this counts correctly
    predicted nodes (true positives), all predicted nodes (true + false
    positives), and all golden nodes (true positives + false negatives).

    @param gold_ttree: a golden t-tree
    @param pred_ttree: a predicted t-tree
    @param eval_type: reserved for future use
    @rtype: tuple
    @return: numbers of correctly predicted, total predicted, and total golden nodes
    """
    gold_counts = collect_counts(gold_ttree, eval_type)
    pred_counts = collect_counts(pred_ttree, eval_type)
    correct, predicted = 0, 0
    for node_id, node_count in pred_counts.iteritems():
        predicted += node_count
        correct += min(node_count, gold_counts[node_id])
    gold = sum(node_count for node_count in gold_counts.itervalues())
    return correct, predicted, gold


def precision(gold_ttree, pred_ttree, eval_type=EvalTypes.NODE):
    # # correct / # predicted
    correct, predicted, _ = corr_pred_gold(gold_ttree, pred_ttree, eval_type)
    return correct / float(predicted)


def recall(gold_ttree, pred_ttree, eval_type=EvalTypes.NODE):
    # # correct / # gold
    correct, _, gold = corr_pred_gold(gold_ttree, pred_ttree, eval_type)
    return correct / float(gold)


def f1(gold_ttree, pred_ttree, eval_type=EvalTypes.NODE):
    return f1_from_counts(corr_pred_gold(gold_ttree, pred_ttree, eval_type))


def f1_from_counts(correct, predicted, gold):
    return p_r_f1_from_counts(correct, predicted, gold)[2]


def p_r_f1_from_counts(correct, predicted, gold):
    """Return precision, recall, and F1 given counts of true positives (correct),
    total predicted nodes, and total gold nodes.

    @param correct: true positives (correctly predicted nodes)
    @param predicted: true + false positives (all predicted nodes)
    @param gold: true positives + false negatives (all golden nodes)
    @rtype: tuple
    @return: precision, recall, F1
    """
    if correct == 0.0:  # escape division by zero
        return 0.0, 0.0, 0.0
    precision = correct / float(predicted)
    recall = correct / float(gold)
    return precision, recall, (2 * precision * recall) / (precision + recall)


# Adapted from http://rosettacode.org/wiki/Longest_common_subsequence#Python
def longest_common_subseq(a, b):
    """Find the longest common subsequence in an array of TreeNode-s,
    @return: a tuple of TreeNode arrays, each containing nodes from one of the original arrays.
    """
    # dynamic programming, substring
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            # check for node equality (t_lemma, formeme, precede/follow parent)
            if (x.t_lemma == y.t_lemma and
                    x.formeme == y.formeme and
                    x.is_right_child == y.is_right_child):
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix, from the end to the beginning
    res_a = []
    res_b = []
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            res_a.append(a[x - 1])
            res_b.append(b[y - 1])
            x -= 1
            y -= 1
    res_a.reverse()
    res_b.reverse()
    return res_a, res_b


def common_subtree_size(a, b):
    """Find the size of the common subtree of two TreeNode-s, starting from the root
    (but not counting it)."""
    common_ch_a, common_ch_b = longest_common_subseq(a.get_children(), b.get_children())
    return len(common_ch_a) + sum(common_subtree_size(a_sub, b_sub)
                                  for a_sub, b_sub in zip(common_ch_a, common_ch_b))


# TODO cast these from TreeData to TreeNode; build this into tgen.tree somehow
def common_subtree(a, b):
    common_ch_a, common_ch_b = longest_common_subseq(a.get_children(), b.get_children())
    append_a, append_b = [], []
    for a_sub, b_sub in zip(common_ch_a, common_ch_b):
        common_sub_a, common_sub_b = common_subtree(a_sub, b_sub)
        append_a.extend(common_sub_a)
        append_b.extend(common_sub_b)
    return common_ch_a + append_a, common_ch_b + append_b


def group_lists(l_long, l_short):
    port_size, bigger_ports = divmod(len(l_long), len(l_short))
    new_long = []
    for port_no in xrange(len(l_short)):
        if port_no < bigger_ports:
            new_long.append(l_long[(port_size + 1) * port_no: (port_size + 1) * (port_no + 1)])
        else:
            new_long.append(l_long[port_size * port_no + bigger_ports:port_size * (port_no + 1) + bigger_ports])
    return new_long, [[item] for item in l_short]


def build_subtrees(tree, start_idxs, adding_idxs):
    trees = []
    start_idxs = set(map(lambda n: n.node_idx, start_idxs))
    for add_list in adding_idxs:
        start_idxs |= set(map(lambda n: n.node_idx, add_list))
        trees.append(tree.get_subtree(start_idxs))
    return trees


def diffing_trees(a, b):
    '''a and b are TreeNode's'''
    common_a, common_b = common_subtree(a, b)
    common_a.append(a)
    common_b.append(b)
    all_a = [TreeNode(a.tree, i) for i in xrange(len(a))]
    all_b = [TreeNode(b.tree, i) for i in xrange(len(b))]
    diff_a = sorted(list(set(all_a) - set(common_a)), cmp=lambda x, y: cmp(x.get_depth(), y.get_depth()))
    diff_b = sorted(list(set(all_b) - set(common_b)), cmp=lambda x, y: cmp(x.get_depth(), y.get_depth()))
    if len(diff_a) > len(diff_b):
        diff_a, diff_b = group_lists(diff_a, diff_b)
    elif len(diff_b) > len(diff_a):
        diff_b, diff_a = group_lists(diff_b, diff_a)
    return build_subtrees(a.tree, common_a, diff_a), build_subtrees(b.tree, common_b, diff_b)


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

    Accumulates scores over trees using append(), then can return
    a total score using f1(), precision(), recall(), and p_r_f1()."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Zero out all current statistics, start from scratch."""
        self.correct = {eval_type: 0 for eval_type in EvalTypes}
        self.predicted = {eval_type: 0 for eval_type in EvalTypes}
        self.gold = {eval_type: 0 for eval_type in EvalTypes}
        self.tree_sizes = []
        self.scores = []

    def append(self, gold_tree, pred_tree, gold_tree_score=0.0, pred_tree_score=0.0):
        """Add a pair of golden and predicted tree to the current statistics.
        @param gold_tree: a T or TreeNode object representing the golden tree
        @param pred_tree: a T or TreeNode object representing the predicted tree
        """
        for eval_type in EvalTypes:
            correct, predicted, gold = corr_pred_gold(gold_tree, pred_tree, eval_type)
            self.correct[eval_type] += correct
            self.predicted[eval_type] += predicted
            self.gold[eval_type] += gold
        gold_tree_size = len(gold_tree.get_descendants())
        pred_tree_size = len(pred_tree.get_descendants())
        css = common_subtree_size(gold_tree, pred_tree)
        self.tree_sizes.append((gold_tree_size, pred_tree_size, css))
        self.scores.append((gold_tree_score, pred_tree_score))

    def merge(self, other):
        """Merge in statistics from another Evaluator object."""
        for eval_type in EvalTypes:
            self.correct[eval_type] += other.correct[eval_type]
            self.predicted[eval_type] += other.predicted[eval_type]
            self.gold[eval_type] += other.gold[eval_type]
        self.tree_sizes.extend(other.tree_sizes)
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

    def tree_size_stats(self):
        """Return current tree size statistics.
        @rtype: a 3-tuple of Stats objects
        @return: statistics for golden trees, predicted trees, and differences
        """
        return (Stats([inst[0] for inst in self.tree_sizes]),
                Stats([inst[1] for inst in self.tree_sizes]),
                Stats([inst[0] - inst[1] for inst in self.tree_sizes]))

    def common_subtree_stats(self):
        """Return common subtree size statistics.
        @rtype: a 3-tuple of Stats objects
        @return: statistics for common subtree size + sizes of what's missing to full gold/predicted tree
        """
        return (Stats([inst[2] for inst in self.tree_sizes]),
                Stats([inst[0] - inst[2] for inst in self.tree_sizes]),
                Stats([inst[1] - inst[2] for inst in self.tree_sizes]))

    def score_stats(self):
        """Return tree score statistics.
        @rtype: a 3-tuple of Stats objects
        @return: statistics for golden trees, predicted trees, and differences
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
        tot = float(self.total)
        return (self.gold_best / tot,
                self.gold_on_close / tot,
                (self.gold_on_close + self.gold_on_open) / tot)
