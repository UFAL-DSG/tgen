#!/usr/bin/env python
# coding=utf-8

"""
Trees for generating.
"""

from __future__ import unicode_literals
from collections import namedtuple, deque


__author__ = "Ondřej Dušek"
__date__ = "2014"


class NodeData(namedtuple('NodeData', ['t_lemma', 'formeme'])):
    """This stores the actual data of a node, without parent-child information"""
    pass


class TreeData(object):
    """This stores all node data for a tree, as well as parent-child information."""

    __slots__ = ['nodes', 'parents']

    def __init__(self, nodes=None, parents=None, ttree=None):

        if nodes and parents:
            # copy structure if it's given
            self.nodes = list(nodes)
            self.parents = list(parents)
        else:
            # add just technical root
            self.nodes = [NodeData(None, None)]
            self.parents = [-1]

            # initialize with data from existing t-tree
            if ttree is not None:
                tnodes = ttree.get_descendants(ordered=True)
                id2ord = {tnode.id: num for num, tnode in enumerate(tnodes, start=1)}
                id2ord[ttree.id] = 0
                for tnode in tnodes:
                    self.nodes.append(NodeData(tnode.t_lemma, tnode.formeme))
                    self.parents.append(id2ord[tnode.parent.id])

    def create_child(self, parent_idx, right, child_data):
        child_idx = parent_idx + 1 if right else parent_idx
        self.nodes.insert(child_idx, child_data)
        self.parents.insert(child_idx, parent_idx)
        self.parents[:] = [idx + 1 if idx >= child_idx else idx for idx in self.parents[:]]
        return child_idx

    def children_idxs(self, parent_idx):
        return [idx for idx, val in enumerate(self.parents) if val == parent_idx]

    def node_depth(self, node_idx):
        depth = 0
        while node_idx > 0:
            node_idx = self.parents[node_idx]
            depth += 1
        return depth

    def __hash__(self):
        # TODO: this is probably slow... make it faster, possibly replace the lists with tuples?
        return hash(tuple(self.nodes)) ^ hash(tuple(self.parents))

    def __eq__(self, other):
        return self.parents == other.parents and self.nodes == other.nodes

    def __unicode__(self):
        return ' '.join(['%d|%d|%s|%s' % (idx, parent_idx, node.t_lemma, node.formeme)
                         for idx, (parent_idx, node)
                         in enumerate(zip(self.parents, self.nodes))])

    def __str__(self):
        return unicode(self).encode('UTF-8', 'replace')

    def __len__(self):
        return len(self.nodes)

    def clone(self):
        return TreeData(nodes=self.nodes, parents=self.parents)


class TreeNode(object):
    """This is a tiny wrapper over TreeData that holds a link to a tree
    and a node index and implements nice object-oriented calls for traversing
    the tree and accessing node properties.

    A new object is usually created when traversing the tree, but it contains
    a link to the same TreeData and just a different node index.
    """

    __slots__ = ['tree', 'node_idx']

    def __init__(self, tree, node_idx=0):
        self.tree = tree
        self.node_idx = node_idx

    def create_child(self, right, child_data):
        child_idx = self.tree.create_child(self.node_idx, right, child_data)
        if not right:
            self.node_idx += 1
        return TreeNode(self.tree, child_idx)

    def get_children(self):
        return [TreeNode(self.tree, child_idx)
                for child_idx in self.tree.children_idxs(self.node_idx)]

    def get_depth(self):
        return self.tree.node_depth(self.node_idx)

    @property
    def parent(self):
        return TreeNode(self.tree, self.tree.parents[self.tree.node_idx])

    @property
    def root(self):
        return TreeNode(self.tree, 0)

    @property
    def formeme(self):
        return self.tree.nodes[self.node_idx].formeme

    @property
    def t_lemma(self):
        return self.tree.nodes[self.node_idx].t_lemma

    def get_attr(self, attr_name):
        return getattr(self.tree.nodes[self.node_idx], attr_name)

    def get_descendants(self, add_self=False, ordered=True):
        # fast descendants of root, no recursion (will be always ordered)
        if self.node_idx == 0:
            return [TreeNode(self.tree, idx) for idx in xrange(0 if add_self else 1,
                                                               len(self.tree))]
        # slow descendants of any other node
        else:
            processed = []
            to_process = deque([self.node_idx])
            while to_process:
                idx = to_process.popleft()
                processed.append(idx)
                to_process.extend(self.tree.children_idxs(idx))

            if not add_self:
                processed = processed[1:]
            if ordered:
                processed.sort()
            return [TreeNode(self.tree, idx) for idx in processed]

    def __lt__(self, other):
        return self.node_idx < other.node_idx

    def __gt__(self, other):
        return self.node_idx > other.node_idx

    def __ge__(self, other):
        return self.node_idx >= other.node_idx

    def __le__(self, other):
        return self.node_idx <= other.node_idx
