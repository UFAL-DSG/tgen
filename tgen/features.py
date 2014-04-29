#!/usr/bin/env python
# coding=utf-8

"""
Feature functions for the ranker (must be top-level functions as they are pickled with the model).
"""

from collections import defaultdict


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
