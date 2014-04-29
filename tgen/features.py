#!/usr/bin/env python
# coding=utf-8

"""
Feature functions for the ranker (must be top-level functions as they are pickled with the model).
"""

from collections import defaultdict
import re
from functools import partial


def find_nodes(parent, scope):
    """Given a parent node and scope specifications (in a list), this returns the
    corresponding nodes.
    """
    nodes = []
    for scope_spec in scope:
        if scope_spec == 'parent':
            nodes.append(parent)
        elif scope_spec == 'grandpa' and parent.parent:
            nodes.append(parent.parent)
        elif scope_spec == 'siblings':  # TODO: use left siblings only ?
            nodes.extend(parent.get_children())
        elif scope_spec == 'uncles' and parent.parent:
            nodes.extend([uncle for uncle in parent.parent.get_children() if uncle != parent])
        elif scope_spec == 'tree':
            nodes.extend(parent.root.get_descendants())
    return nodes


def same_as_current(node, parent, scope_func, attrib):
    """Return the number of nodes in the given scope that have the same value
    of the given attribute as the current node.

    @rtype: dict
    @return: dictionary with one key ('') and the number of matching values as a value
    """
    node = node[0]
    value = node[['formeme', 't_lemma', 'right'].index(attrib)]  # TODO more attributes / more flexible ?
    num_matching = 0.0
    for node in scope_func(parent):
        if attrib == 'right':  # special handling for 'right'
            if node.parent and (node > node.parent) == value:
                num_matching += 1
        elif node.get_attr(attrib) == value:  # any other attribute
            num_matching += 1
    return {'': num_matching}


def value(node, parent, scope_func, attrib):
    """Return the number of nodes holding the individual values of the given attribute
    in the given scope.

    @rtype dict
    @return: dictionary with keys for values of the attribute, values for counts of matching nodes
    """
    node = node[0]
    ret = defaultdict(float)
    for node in scope_func(parent):
        if attrib == 'right':
            if node.parent and node > node.parent:
                ret['True'] += 1
            elif node.parent:
                ret['False'] += 1
        else:
            ret[unicode(node.get_attr(attrib))] += 1
    return ret


def prob(node, parent):
    return {'': node[1]}

class Features(object):

    def __init__(self, cfg):
        self.features = self.parse_feature_spec(cfg)

    def parse_feature_spec(self, spec):
        """Prepares feature feature function from specifications in the following format:

        Label: value/same_as_current scope param1, ...

        Scope may be: parent, siblings, grandpa, uncles, or their combinations (connected
        with '+', no spaces). Always applies only to the part of the tree that is already
        built (i.e. to the top/left only).
        """
        features = {}
        for feat in spec:
            label, func_name = re.split(r'[:\s]+', feat, 1)
            if func_name == 'prob':
                features[label] = prob
            else:
                func_name, func_scope, func_params = re.split(r'[:\s]+', func_name, 2)
                func_params = re.split(r'[,\s]+', func_params)
                feat_func = None
                scope_func = partial(find_nodes, scope=func_scope.split('+'))
                if func_name.lower() == 'same_as_current':
                    feat_func = partial(same_as_current, scope_func=scope_func, attrib=func_params[0])
                elif func_name.lower() == 'value':
                    feat_func = partial(value, scope_func=scope_func, attrib=func_params[0])
                else:
                    raise Exception('Unknown feature function:' + feat)
                features[label] = feat_func
        return features

    def get_features(self, node, parent):
        feats_hier = {}
        for name, func in self.features.iteritems():
            feats_hier[name] = func(node, parent)
        feats = {}
        for name, val in feats_hier.iteritems():
            for subname, subval in val.iteritems():
                feats[name + '_' + subname if subname else name] = subval
        return feats
