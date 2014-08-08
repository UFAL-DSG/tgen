#!/usr/bin/env python
# coding=utf-8

"""
Feature functions for the ranker (must be top-level functions as they are pickled with the model),
as well as an object that handles them.

@todo: Vectorization and normalization should be handled here.
@todo: Allow conjunctions & other operations with features.
"""

from collections import defaultdict
import re
from functools import partial
from tree import TreeNode, TreeData


def find_nodes(node, scope, incremental=False):
    """Given a parent node and scope specifications (in a list), this returns the
    corresponding nodes.
    """
    nodes = []
    for scope_spec in scope:
        if scope_spec == 'node' or incremental:
            nodes.apppend(node)
        elif scope_spec == 'tree':
            nodes.extend(node.root.get_descendants())
        elif node.parent:
            parent = node.parent
            if scope_spec == 'parent':
                nodes.append(parent)
            elif scope_spec == 'grandpa' and parent.parent:
                nodes.append(parent.parent)
            elif scope_spec == 'siblings':  # TODO: use left siblings only ?
                nodes.extend(parent.get_children())
            elif scope_spec == 'uncles' and parent.parent:
                nodes.extend([uncle for uncle in parent.parent.get_children() if uncle != parent])
    return nodes


def depth(cur_node, context, scope_func, incremental=False):
    """Return the maximum tree depth in the given scope.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    nodes = scope_func(cur_node, incremental=incremental)
    if nodes:
        return {'': max(node.get_depth() for node in nodes)}
    return {'': 0}


def max_children(cur_node, context, scope_func, incremental=False):
    """Return the maximum number of children in nodes in the given scope.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    nodes = scope_func(cur_node, incremental=incremental)
    if nodes:
        return {'': max(len(node.get_children()) for node in nodes)}
    return {'': 0}


def nodes_per_dai(cur_node, context, scope_func, incremental=False):
    """Return the ratio of the number of nodes to the number of original DAIs.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    return {'': len(scope_func(cur_node, incremental=incremental)) / float(len(context['da']))}


def rep_nodes_per_rep_dai(cur_node, context, scope_func, incremental=False):
    """Return the ratio of the number of repeated nodes to the number of repeated DAIs.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    node_count = defaultdict(int)
    for node in scope_func(cur_node, incremental=incremental):
        node_count[node.t_lemma + "\n" + node.formeme] += 1
    dai_count = defaultdict(int)
    for dai in context['da']:
        dai_count[dai] += 1
    rep_nodes = sum(count for count in node_count.itervalues() if count > 1)
    # avoid division by zero + penalize repeated nodes for non-repeated DAIs
    rep_dais = sum(count for count in dai_count.itervalues() if count > 1) + 0.5
    return {'': rep_nodes / rep_dais}


def rep_nodes(cur_node, context, scope_func, incremental=False):
    """Return the number of repeated nodes in the given scope.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    node_count = defaultdict(int)
    for node in scope_func(cur_node, incremental=incremental):
        node_count[node.t_lemma + "\n" + node.formeme] += 1
    return {'': sum(count for count in node_count.itervalues() if count > 1)}


def same_as_current(cur_node, context, scope_func, attrib, incremental=False):
    """Return the number of nodes in the given scope that have the same value
    of the given attribute as the current node.

    @rtype: dict
    @return: dictionary with one key ('') and the number of matching values as a value
    """
    if attrib == 'right':
        value = True if cur_node.parent and cur_node > cur_node.parent else False
    else:
        value = getattr(cur_node, attrib)
    num_matching = 0.0
    for node in scope_func(cur_node, incremental=incremental):
        if attrib == 'right':  # special handling for 'right'
            if node.parent and (node > node.parent) == value:
                num_matching += 1
        elif node.get_attr(attrib) == value:  # any other attribute
            num_matching += 1
    return {'': num_matching}


def value(cur_node, context, scope_func, attrib, incremental=False):
    """Return the number of nodes holding the individual values of the given attribute
    in the given scope.

    @rtype: dict
    @return: dictionary with keys for values of the attribute, values for counts of matching nodes
    """
    ret = defaultdict(float)
    for node in scope_func(cur_node, incremental=incremental):
        if attrib == 'right':
            if node.parent and node > node.parent:
                ret['True'] += 1
            elif node.parent:
                ret['False'] += 1
        else:
            ret[unicode(node.get_attr(attrib))] += 1
    return ret


def presence(cur_node, context, scope_func, attrib, incremental=False):
    """Return 1 for all values of the given attribute found in the given scope.

    @rtype: dict
    @return: dictionary with keys for values of the attribute and values equal to 1
    """
    ret = defaultdict(float)
    for node in scope_func(cur_node, incremental=incremental):
        if attrib == 'right':
            if node.parent and node > node.parent:
                ret['True'] = 1
            elif node.parent:
                ret['False'] = 1
        else:
            ret[unicode(node.get_attr(attrib))] = 1
    return ret


def dai_cooc(cur_node, context, scope_func, attrib, incremental=False):
    """Return 1 for all combinations of DAIs and values of a node.

    @rtype: dict
    @return: dictionary with keys composed of DAIs and values of the given attribute, \
        and values equal to 1
    """
    ret = defaultdict(float)
    for dai in context['da']:
        for node in scope_func(cur_node, incremental=incremental):
            if attrib == 'right':
                if node.parent and node > node.parent:
                    ret[unicode(dai) + '+True'] = 1
                elif node.parent:
                    ret[unicode(dai) + '+False'] = 1
            else:
                ret[unicode(dai) + '+' + unicode(node.get_attr(attrib))] = 1
    return ret


def prob(cur_node, context):
    # TODO this won't work. Use wild attributes? Or some other structure?
    return {'': context['node_prob']}


def bias(cur_node, context):
    """A constant feature function, always returning 1"""
    return {'': 1}


class Features(object):

    def __init__(self, cfg):
        self.features = self.parse_feature_spec(cfg)

    def parse_feature_spec(self, spec):
        """Prepares feature functions from specifications in the following format:

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
            elif func_name == 'bias':
                features[label] = bias
            else:
                func_name, func_params = re.split(r'[:\s]+', func_name, 1)
                func_params = re.split(r'[,\s]+', func_params)
                func_scope = func_params[0]
                func_params = func_params[1:]
                feat_func = None
                scope_func = partial(find_nodes, scope=func_scope.split('+'))
                # node features
                if func_name.lower() == 'same_as_current':
                    feat_func = partial(same_as_current, scope_func=scope_func, attrib=func_params[0])
                elif func_name.lower() == 'value':
                    feat_func = partial(value, scope_func=scope_func, attrib=func_params[0])
                elif func_name.lower() == 'presence':
                    feat_func = partial(presence, scope_func=scope_func, attrib=func_params[0])
                elif func_name.lower() == 'dai_cooc':
                    feat_func = partial(dai_cooc, scope_func=scope_func, attrib=func_params[0])
                # tree shape features
                elif func_name.lower() == 'depth':
                    feat_func = partial(depth, scope_func=scope_func)
                elif func_name.lower() == 'max_children':
                    feat_func = partial(max_children, scope_func=scope_func)
                elif func_name.lower() == 'nodes_per_dai':
                    feat_func = partial(nodes_per_dai, scope_func=scope_func)
                elif func_name.lower() == 'rep_nodes_per_rep_dai':
                    feat_func = partial(rep_nodes_per_rep_dai, scope_func=scope_func)
                elif func_name.lower() == 'rep_nodes':
                    feat_func = partial(rep_nodes, scope_func=scope_func)
                else:
                    raise Exception('Unknown feature function:' + feat)
                features[label] = feat_func
        return features

    def get_features(self, node, context, feats=None):
        """Return features for the given node. Accumulates features from other nodes
        if given in the feats parameter.

        @param node: The current node w.r.t. to which the features should be computed
        @param feats: Previous feature values, for incremental computation.
        """
        if isinstance(node, TreeData):  # this will handle T-nodes as well as simplified TreeData
            node = TreeNode(node)
        if feats is None:
            feats = defaultdict(float)
        feats_hier = {}
        for name, func in self.features.iteritems():
            feats_hier[name] = func(node, context)
        for name, val in feats_hier.iteritems():
            for subname, subval in val.iteritems():
                feats[name + '_' + subname if subname else name] += subval
        return feats
