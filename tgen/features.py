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


# Helper functions


def dep_dir(tree, node_idx):
    """Return 'R' if the node at the given position in the given tree is
    its parent's right child, 'L' otherwise. Return None for the technical root.

    @rtype: str"""
    parent_idx = tree.parents[node_idx]
    if parent_idx >= 0 and node_idx > parent_idx:
        return 'R'
    elif parent_idx >= 0:
        return 'L'
    return None


def attribs_val(tree, idx, attribs):
    """Return the joined value of the given attributes for the idx-th node of
    the given tree.

    @rtype: unicode"""
    val = []
    for attrib in attribs:
        if attrib == 'num_children':
            val.append(str(sum(1 for i in xrange(len(tree)) if tree.parents[i] == idx)))
        else:
            val.append(unicode(getattr(tree.nodes[idx], attrib)))
    return '+'.join(val)


# Feature functions


def depth(tree, context):
    """Return the depth of the given tree.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    depths = {}  # store nodes that we've already processed
    max_depth = 0
    for node_id in xrange(len(tree)):
        pos = node_id
        depth = 0
        # go up to the root / an already processed node
        while tree.parents[pos] >= 0 and pos not in depths:
            depth += 1
            pos = tree.parents[pos]
        if pos in depths:  # processed node: add its depth
            depth += depths[pos]
        # store the depth to save computation
        depths[node_id] = depth
        if depth > max_depth:
            max_depth = depth
    return {'': max_depth}


def max_children(tree, context):
    """Return the maximum number of children in the given tree.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    children = defaultdict(int)
    for parent in tree.parents:
        children[parent] += 1
    return {'': max(children.itervalues())}


def nodes_per_dai(tree, context):
    """Return the ratio of the number of nodes to the number of original DAIs.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    return {'': len(tree) / float(len(context['da']))}


def rep_nodes_per_rep_dai(tree, context):
    """Return the ratio of the number of repeated nodes to the number of repeated DAIs.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    node_count = defaultdict(int)
    for node in tree.nodes:
        node_count[node] += 1
    dai_count = defaultdict(int)
    for dai in context['da']:
        dai_count[dai] += 1
    rep_nodes = sum(count for count in node_count.itervalues() if count > 1)
    # avoid division by zero + penalize repeated nodes for non-repeated DAIs
    rep_dais = sum(count for count in dai_count.itervalues() if count > 1) + 0.5
    return {'': rep_nodes / rep_dais}


def rep_nodes(tree, context):
    """Return the number of repeated nodes in the given tree.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    node_count = defaultdict(int)
    for node in tree.nodes:
        node_count[node] += 1
    return {'': sum(count for count in node_count.itervalues() if count > 1)}


def tree_size(tree, context):
    """Return the number of nodes in the current tree.

    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    return {'': len(tree)}


def count(tree, context, attribs):
    """Return the number of nodes holding the individual values of the given attribute
    in the given tree.

    @rtype: dict
    @return: dictionary with keys for values of the attribute, values for counts of matching nodes
    """
    ret = defaultdict(int)
    for idx in xrange(len(tree)):
        ret[attribs_val(tree, idx, attribs)] += 1
    return ret


def presence(tree, context, attribs):
    """Return 1 for all values of the given attribute found in the given tree.

    @rtype: dict
    @return: dictionary with keys for values of the attribute and values equal to 1
    """
    ret = {}
    for idx in xrange(len(tree)):
        ret[attribs_val(tree, idx, attribs)] = 1
    return ret


def dependency(tree, context, attribs):
    """Return 1 for all dependency pairs of the given attribute found in the given tree.
    @rtype: : dict
    @return dictionary with keys for values of the attribute and values equal to 1
    """
    ret = {}
    for idx, parent_idx in enumerate(tree.parents):
        if parent_idx <= 0:  # skip for technical root
            continue
        ret[attribs_val(tree, idx, attribs) + "//" + attribs_val(tree, parent_idx, attribs)] = 1
    return ret


def dir_dependency(tree, context, attribs):
    """Same as :py:func:`dependency`, but includes edge direction (L/R).
    @rtype: : dict
    @return dictionary with keys for values of the attribute and values equal to 1
    """
    ret = {}
    for idx, parent_idx in enumerate(tree.parents):
        if parent_idx <= 0:  # skip for technical root
            continue
        ret[attribs_val(tree, idx, attribs) + '/' +
            dep_dir(tree, idx) + '/' +
            attribs_val(tree, parent_idx, attribs)] = 1
    return ret


def dai_presence(tree, context):
    """Return 1 for all DAIs in the given context.

    @rtype: dict
    @return: dictionary with keys composed of DAIs and values equal to 1
    """
    ret = {}
    for dai in context['da']:
        ret[unicode(dai)] = 1
    return ret


def combine(tree, context, attribs):
    """A meta-feature combining n-tuples of other features (as found in context['feats']).

    @rtype: dict
    @return: dictionary with keys composed of combined keys of the original features \
        and values equal to 1.
    """
    cur = context['feats'][attribs[0]]
    for attrib in attribs[1:]:
        add = context['feats'][attrib]
        merged = {ckey + "-&-" + akey: 1
                  for ckey in cur.iterkeys()
                  for akey in add.iterkeys()}
        cur = merged
    return cur


def bias(tree, context):
    """A constant feature function, always returning 1"""
    return {'': 1}


class Features(object):

    def __init__(self, cfg):
        self.features = self.parse_feature_spec(cfg)

    def parse_feature_spec(self, spec):
        """Prepares feature functions from specifications in the following format:

        Label: value/same_as_current/... [param1,...]
        """
        features = []
        for feat in spec:
            label, func_name = re.split(r'[:\s]+', feat, 1)
            func_params = ''
            try:  # parse parameters if there are any (otherwise default to empty)
                func_name, func_params = re.split(r'[:\s]+', func_name, 1)
                func_params = re.split(r'[,\s]+', func_params)
            except:
                pass

            feat_func = None
            # bias
            if func_name.lower() == 'bias':
                feat_func = bias
            # node features
            elif func_name.lower() == 'count':
                feat_func = partial(count, attribs=func_params)
            elif func_name.lower() == 'presence':
                feat_func = partial(presence, attribs=func_params)
            elif func_name.lower() == 'dai_presence':
                feat_func = dai_presence
            elif func_name.lower() == 'dependency':
                feat_func = partial(dependency, attribs=func_params)
            elif func_name.lower() == 'dir_dependency':
                feat_func = partial(dir_dependency, attribs=func_params)
            # tree shape features
            elif func_name.lower() == 'depth':
                feat_func = depth
            elif func_name.lower() == 'tree_size':
                feat_func = tree_size
            elif func_name.lower() == 'max_children':
                feat_func = max_children
            elif func_name.lower() == 'nodes_per_dai':
                feat_func = nodes_per_dai
            elif func_name.lower() == 'rep_nodes_per_rep_dai':
                feat_func = rep_nodes_per_rep_dai
            elif func_name.lower() == 'rep_nodes':
                feat_func = rep_nodes
            elif func_name.lower() == 'combine':
                feat_func = partial(combine, attribs=func_params)
            else:
                raise Exception('Unknown feature function:' + feat)

            features.append((label, feat_func))

        return features

    def get_features(self, tree, context):
        """Return features for the given tree.

        @param tree: The current tree w.r.t. to which the features should be computed
        @param feats: Previous feature values, for incremental computation.
        """
        feats = defaultdict(float)
        feats_hier = {}
        context['feats'] = feats_hier  # allow features to look at previous features
        for name, func in self.features:
            feats_hier[name] = func(tree, context)
        for name, val in feats_hier.iteritems():
            for subname, subval in val.iteritems():
                feats[name + '_' + subname if subname else name] += subval
        return feats
