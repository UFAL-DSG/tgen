#!/usr/bin/env python
# coding=utf-8

"""
Evaluation (t-tree comparison functions).
"""

from __future__ import unicode_literals
from collections import defaultdict


def collect_counts(ttree, count_type='node'):
    counts = defaultdict(int)
    for node in ttree.get_descendants():
        if count_type == 'node':
            node_id = (node.formeme, node.t_lemma, node > node.parent)
        else:
            parent = node.parent
            node_id = (node.formeme, node.t_lemma, node > node.parent,
                       parent.formeme, parent.t_lemma, (parent.parent is not None and parent > parent.parent))
        counts[node_id] += 1
    return counts


def tp_fp_fn(gold_ttree, pred_ttree, count_type='node'):
    """Given a golden tree and a predicted tree, this counts correctly
    predicted nodes (true positives), all predicted nodes (true + false
    positives), and all golden nodes (true positives + false negatives).

    @param gold_ttree: a golden t-tree
    @param pred_ttree: a predicted t-tree
    @param count_type: reserved for future use
    @rtype: tuple
    @return: numbers of correctly predicted, total predicted, and total golden nodes
    """
    gold_counts = collect_counts(gold_ttree, type)
    pred_counts = collect_counts(pred_ttree, type)
    correct, predicted = 0, 0
    for node_id, node_count in pred_counts.iteritems():
        predicted += node_count
        correct += min(node_count, gold_counts[node_id])
    gold = sum(node_count for node_count in gold_counts.itervalues())
    return correct, predicted, gold


def precision(gold_ttree, pred_ttree, count_type='node'):
    # # correct / # predicted
    correct, predicted, _ = tp_fp_fn(gold_ttree, pred_ttree, count_type)
    return correct / float(predicted)


def recall(gold_ttree, pred_ttree, count_type='node'):
    # # correct / # gold
    correct, _, gold = tp_fp_fn(gold_ttree, pred_ttree, count_type)
    return correct / float(gold)


def f1(gold_ttree, pred_ttree, count_type='node'):
    return f1_from_counts(tp_fp_fn(gold_ttree, pred_ttree, count_type))


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
        self.correct = 0
        self.gold = 0
        self.predicted = 0

    def append(self, gold_tree, pred_tree):
        correct, predicted, gold = tp_fp_fn(gold_tree, pred_tree)
        self.correct += correct
        self.predicted += predicted
        self.gold += gold

    def reset(self):
        self.correct = 0
        self.predicted = 0
        self.gold = 0

    def f1(self):
        return self.p_r_f1()[2]

    def precision(self):
        return self.p_r_f1()[0]

    def recall(self):
        return self.p_r_f1()[1]

    def p_r_f1(self):
        return p_r_f1_from_counts(self.correct, self.gold, self.predicted)
