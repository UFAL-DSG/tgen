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
    if correct == 0.0:  # escape division by zero
        return 0.0, 0.0, 0.0
    precision = correct / float(predicted)
    recall = correct / float(gold)
    return precision, recall, (2 * precision * recall) / (precision + recall)
