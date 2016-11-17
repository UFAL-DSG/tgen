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
import inspect
from functools import partial
from itertools import combinations


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
            val.append(str(tree.children_num(idx)))
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


def repeated(tree, context, attribs):
    """Return 1 for all node attribute values whose count is greater than 1.
    @rtype: dict
    @return: dictionary with keys for values of the attribute and values equal to 1
    """
    ret = count(tree, context, attribs)
    for key in ret.keys():
        if ret[key] < 2:
            del ret[key]
        else:
            ret[key] = 1
    return ret


def dependency(tree, context, attribs):
    """Return 1 for all dependency pairs of the given attribute found in the given tree.
    @rtype: dict
    @return: dictionary with keys for values of the attribute and values equal to 1
    """
    ret = {}
    for idx, parent_idx in enumerate(tree.parents):
        if parent_idx <= 0:  # skip for technical root
            continue
        ret[attribs_val(tree, idx, attribs) + "//" + attribs_val(tree, parent_idx, attribs)] = 1
    return ret


def dir_dependency(tree, context, attribs):
    """Same as :py:func:`dependency`, but includes edge direction (L/R).
    @rtype: dict
    @return: dictionary with keys for values of the attribute and values equal to 1
    """
    ret = {}
    for idx, parent_idx in enumerate(tree.parents):
        if parent_idx <= 0:  # skip for technical root
            continue
        ret[attribs_val(tree, idx, attribs) + '/' +
            dep_dir(tree, idx) + '/' +
            attribs_val(tree, parent_idx, attribs)] = 1
    return ret


def siblings(tree, context, attribs):
    """Return 1 for all node pairs that are siblings in the given tree
    @rtype:  dict
    @return: dictionary with keys for values of the attribute and values equal to 1
    """
    parents = defaultdict(list)
    for idx, parent_idx in enumerate(tree.parents):
        parents[parent_idx].append(idx)
    ret = {}
    for siblings in parents.itervalues():
        for sibl1, sibl2 in combinations(siblings, 2):
            ret[attribs_val(tree, sibl1, attribs) + '-x-' + attribs_val(tree, sibl2, attribs)] = 1
    return ret


def bigrams(tree, context, attribs):
    """Return 1 for all node bigrams (in order)
    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    attr_1 = attribs_val(tree, 0, attribs)
    ret = {}
    for idx in xrange(1, len(tree)):
        attr = attribs_val(tree, idx, attribs)
        ret[attr_1 + '->-' + attr] = 1
        attr_1 = attr
    return ret


def trigrams(tree, context, attribs):
    """Return 1 for all node trigrams (in order)
    @rtype: dict
    @return: dictionary with one key ('') and the target number as a value
    """
    if len(tree) < 3:
        return {}
    attr_1 = attribs_val(tree, 0, attribs)
    attr_2 = attribs_val(tree, 1, attribs)
    ret = {}
    for idx in xrange(2, len(tree)):
        attr = attribs_val(tree, idx, attribs)
        ret[attr_2 + '->-' + attr_1 + '->-' + attr] = 1
        attr_2 = attr_1
        attr_1 = attr
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


def svp_presence(tree, context):
    """Return 1 for all DA slot+value pairs in the given context.

    @rtype: dict
    @return: dictionary with DA slot-value pairs as keys and values equal to 1
    """
    ret = {}
    for dai in context['da']:
        if dai.slot is None:
            continue
        ret[dai.slot + '=' + str(dai.value)] = 1
    return ret


def dat_presence(tree, context):
    """Dialogue act type (assuming the same type for all DAIs).

    @rtype: dict
    @return: dictionary with one key – DA type – and a value equal to 1
    """
    return {context['da'][0].da_type: 1}


def slot_presence(tree, context):
    """Return 1 for all DA slots in the given context.

    @rtype: dict
    @return: dictionary with keys composed of DA slots and values equal to 1
    """
    ret = {}
    for dai in context['da']:
        if dai.slot is None:
            continue
        ret[dai.slot] = 1
    return ret


def slot_count(tree, context):
    """Return the number of times the specified slot occurs in the DA.

    @rtype: dict
    @return: dictionary with keys composed of DA slots and values giving their number of occurrences
    """
    ret = defaultdict(int)
    for dai in context['da']:
        if dai.slot is None:
            continue
        ret[dai.slot] += 1
    return ret


def slot_repeated(tree, context):
    """Return 1 for all DA slots that are repeated in the given context.

    @rtype: dict
    @return: dictionary with keys composed of DA slots that occur repeatedly in the DA and 1 as values
    """
    ret = slot_count(tree, context)
    for key in ret.keys():
        if ret[key] < 2:
            del ret[key]
        else:
            ret[key] = 1
    return ret


def set_difference(tree, context, attribs):
    """A meta-feature that will produce the set difference of two boolean features
    (will have keys set to 1 only for those features that occur in the first set but not in the
    second).

    @rtype: dict
    @return: dictionary with keys for key occurring with the first feature but not the second, and \
        keys equal to 1
    """
    ret = {}
    for key, val in context['feats'][attribs[0]].iteritems():
        if key not in context['feats'][attribs[1]]:
            ret[key] = val
    return ret


def difference(tree, context, attribs):
    """A meta-feature that will produce differences of two numeric features.

    @rtype: dict
    @return: dictionary with keys composed of original names and values equal to value differences
    """
    ret = defaultdict(float)
    for key1, val1 in context['feats'][attribs[0]].iteritems():
        for key2, val2 in context['feats'][attribs[1]].iteritems():
            ret[key1 + '---' + key2] = val1 - val2
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

    def __init__(self, feat_list, interm_feats=set()):
        self.features = self.parse_feature_spec(feat_list)
        self.intermediate_features = set(interm_feats)

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

            try:
                feat_func = globals()[func_name]
            except KeyError:
                raise Exception('Unknown feature function:' + feat)

            arg_num = len(inspect.getargspec(feat_func).args)
            if arg_num == 2:
                pass
            elif arg_num == 3:
                feat_func = partial(feat_func, attribs=func_params)
            else:
                raise Exception('Feature function %s has an invalid number of arguments (%d)' %
                                (feat, arg_num))

            features.append((label, feat_func))

        return features

    def get_features(self, tree, context):
        """Return features for the given tree.

        Filter out all features whose names begin with "*" (intermediate features
        used only to compose more complex features).

        @param tree: The current tree w.r.t. to which the features should be computed
        @param feats: Previous feature values, for incremental computation.
        """
        feats = defaultdict(float)
        feats_hier = {}
        context['feats'] = feats_hier  # allow features to look at previous features
        for name, func in self.features:
            feats_hier[name] = func(tree, context)
        for name, val in feats_hier.iteritems():
            if name in self.intermediate_features:  # filter intermediate features
                continue
            for subname, subval in val.iteritems():
                feats[name + '_' + subname if subname else name] += subval
        return feats
