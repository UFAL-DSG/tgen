#!/usr/bin/env python
# coding=utf-8

"""
Evaluation (t-tree comparison functions).
"""

from __future__ import unicode_literals
from collections import defaultdict
from enum import Enum
from tgen.logf import log_debug


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


def tp_fp_fn(gold_ttree, pred_ttree, eval_type=EvalTypes.NODE):
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
    correct, predicted, _ = tp_fp_fn(gold_ttree, pred_ttree, eval_type)
    return correct / float(predicted)


def recall(gold_ttree, pred_ttree, eval_type=EvalTypes.NODE):
    # # correct / # gold
    correct, _, gold = tp_fp_fn(gold_ttree, pred_ttree, eval_type)
    return correct / float(gold)


def f1(gold_ttree, pred_ttree, eval_type=EvalTypes.NODE):
    return f1_from_counts(tp_fp_fn(gold_ttree, pred_ttree, eval_type))


def f1_from_counts(correct, gold, predicted):
    return p_r_f1_from_counts(correct, gold, predicted)[2]


def p_r_f1_from_counts(correct, gold, predicted):
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


class Evaluator(object):
    """A fancy object-oriented interface to computing node F-scores.

    Accumulates scores over trees using append(), then can return
    a total score using f1(), precision(), recall(), and p_r_f1()."""

    def __init__(self):
        self.reset()

    def append(self, gold_tree, pred_tree):
        for eval_type in EvalTypes:
            correct, predicted, gold = tp_fp_fn(gold_tree, pred_tree, eval_type)
            self.correct[eval_type] += correct
            self.predicted[eval_type] += predicted
            self.gold[eval_type] += gold

    def reset(self):
        self.correct = {eval_type: 0 for eval_type in EvalTypes}
        self.predicted = {eval_type: 0 for eval_type in EvalTypes}
        self.gold = {eval_type: 0 for eval_type in EvalTypes}

    def f1(self, eval_type=EvalTypes.NODE):
        return self.p_r_f1(eval_type)[2]

    def precision(self, eval_type=EvalTypes.NODE):
        return self.p_r_f1(eval_type)[0]

    def recall(self, eval_type=EvalTypes.NODE):
        return self.p_r_f1(eval_type)[1]

    def p_r_f1(self, eval_type=EvalTypes.NODE):
        return p_r_f1_from_counts(self.correct[eval_type],
                                  self.gold[eval_type],
                                  self.predicted[eval_type])


class ASearchListsAnalyzer(object):
    """Analysis of the final open and close lists of the A*search generator."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.gold_best = 0
        self.gold_on_close = 0
        self.gold_on_open = 0

    def append(self, gold_tree, open_list, close_list):
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

    def stats(self):
        tot = float(self.total)
        return (self.gold_best / tot,
                self.gold_on_close / tot,
                (self.gold_on_close + self.gold_on_open) / tot)
