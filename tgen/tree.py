#!/usr/bin/env python
# coding=utf-8

"""
Trees for generating.
"""

from __future__ import unicode_literals
from collections import namedtuple, deque
from pytreex.core.node import T


__author__ = "Ondřej Dušek"
__date__ = "2014"


class NodeData(namedtuple('NodeData', ['t_lemma', 'formeme'])):
    """This stores the actual data of a node, without parent-child information"""
    pass


def _group_lists(l_long, l_short):
    """Take two lists, a longer and a shorter one, and group them into equally long lists of
    sublists. One of the resulting sublists is always composed of one-element sublists and
    the other of longer sublists.

    @param l_long: a "longer" list
    @param l_short: a "shorter" list
    @return: a pair of lists of sublists (in the order of parameters given)
    """
    port_size, bigger_ports = divmod(len(l_long), len(l_short))
    if port_size == 0:  # call itself the other way round if l_long is actually shorter than l_short
        l_short, l_long = _group_lists(l_short, l_long)
        return l_long, l_short
    new_long = []
    for port_no in xrange(len(l_short)):
        if port_no < bigger_ports:
            new_long.append(l_long[(port_size + 1) * port_no: (port_size + 1) * (port_no + 1)])
        else:
            new_long.append(l_long[port_size * port_no + bigger_ports:port_size * (port_no + 1) + bigger_ports])
    return new_long, [[item] for item in l_short]


class TreeData(object):
    """This stores all node data for a tree, as well as parent-child information."""

    __slots__ = ['nodes', 'parents']

    def __init__(self, nodes=None, parents=None):

        if nodes and parents:
            # copy structure if it's given
            self.nodes = list(nodes)
            self.parents = list(parents)
        else:
            # add just technical root
            self.nodes = [NodeData(None, None)]
            self.parents = [-1]

    @staticmethod
    def from_ttree(ttree):
        """Copy the tree from a T-tree representation (just structure, t-lemmas and formemes)."""
        tree = TreeData()
        tnodes = ttree.get_descendants(ordered=True)
        id2ord = {tnode.id: num for num, tnode in enumerate(tnodes, start=1)}
        id2ord[ttree.id] = 0
        for tnode in tnodes:
            tree.nodes.append(NodeData(tnode.t_lemma, tnode.formeme))
            tree.parents.append(id2ord[tnode.parent.id])
        return tree

    @staticmethod
    def from_string(string):
        """Parse a string representation of the tree, as returned by `__unicode__`."""
        tree = TreeData()
        for node in string.split(' ')[1:]:
            _, parent, t_lemma, formeme = node.split('|')
            tree.parents.append(int(parent))
            tree.nodes.append(NodeData(t_lemma, formeme))
        return tree

    def create_child(self, parent_idx, child_idx, child_data):
        """Create a child of the given node at the given position, shifting remaining nodes
        to the right.

        @param parent_idx: index of the parent node
        @param child_idx: index of the newly created child node (or boolean: left/right of the current parent?)
        @param child_data: the child node itself as a `NodeData` instance
        @return: the new child index
        """
        if isinstance(child_idx, bool):
            child_idx = parent_idx + 1 if child_idx else parent_idx
        self.nodes.insert(child_idx, child_data)
        self.parents.insert(child_idx, parent_idx)
        self.parents = [idx + 1 if idx >= child_idx else idx for idx in self.parents]
        return child_idx

    def move_node(self, node_idx, target_pos):
        """Move the node on the given position to another position, shifting nodes in between
        and updating parent indexes along the way.

        @param node_idx: the index of the node to be moved
        @param target_pos: the desired target position (index after the moving)
        @return: None
        """
        if node_idx > target_pos:
            self.nodes = (self.nodes[:target_pos] + [self.nodes[node_idx]] +
                          self.nodes[target_pos:node_idx] + self.nodes[node_idx + 1:])
            self.parents = (self.parents[:target_pos] + [self.parents[node_idx]] +
                            self.parents[target_pos:node_idx] + self.parents[node_idx + 1:])
            for pos in xrange(len(self)):
                if self.parents[pos] == node_idx:
                    self.parents[pos] = target_pos
                elif self.parents[pos] >= target_pos and self.parents[pos] < node_idx:
                    self.parents[pos] += 1
        elif node_idx < target_pos:
            self.nodes = (self.nodes[:node_idx] + self.nodes[node_idx + 1:target_pos + 1] +
                          [self.nodes[node_idx]] + self.nodes[target_pos + 1:])
            self.parents = (self.parents[:node_idx] + self.parents[node_idx + 1:target_pos + 1] +
                            [self.parents[node_idx]] + self.parents[target_pos + 1:])
            for pos in xrange(len(self)):
                if self.parents[pos] == node_idx:
                    self.parents[pos] = target_pos
                elif self.parents[pos] > node_idx and self.parents[pos] <= target_pos:
                    self.parents[pos] -= 1

    def remove_node(self, node_idx):
        """Remove a node, rehang all its children to its parent."""
        for pos in xrange(len(self)):
            if self.parents[pos] == node_idx:
                self.parents[pos] = self.parents[node_idx]
        self.move_node(node_idx, len(self)-1)
        del self.parents[-1]
        del self.nodes[-1]

    def subtree_bound(self, parent_idx, right):
        """Return the subtree bound of the given node (furthermost index belonging to the subtree),
        going left or right.

        NB: This assumes the trees are projective.

        @param parent_idx: index of the node to examine
        @param right: if True, return rightmost subtree bound; if False, return leftmost bound
        @return: the furthermost index belonging to the subtree of the given node in the given \
            direction
        """
        move = 1 if right else -1
        cur_idx = parent_idx + move
        while cur_idx >= 0 and cur_idx < len(self):
            # the current node is not in the subtree => the last position was the boundary
            if not self.is_descendant(parent_idx, cur_idx):
                return cur_idx - move
            cur_idx += move
        # return 0 or len(self-1)
        return cur_idx - move

    def children_idxs(self, parent_idx, left_only=False, right_only=False):
        """Return the indexes of the children of the given node.

        @param parent_idx: the node whose children should be found
        @param left_only: only look for left children (preceding the parent)
        @param right_only: only look for right children (following the parent)
        @return: an array of node indexes matching the criteria (may be empty)
        """
        if left_only:
            return [idx for idx, val in enumerate(self.parents[:parent_idx]) if val == parent_idx]
        if right_only:
            return [idx for idx, val in enumerate(self.parents[parent_idx + 1:], start=parent_idx + 1)
                    if val == parent_idx]
        return [idx for idx, val in enumerate(self.parents) if val == parent_idx]

    def children_num(self, parent_idx):
        return sum(1 for val in self.parents if val == parent_idx)

    def node_depth(self, node_idx):
        """Return the depth of the given node (the technical root has a depth=0).

        @param node_idx: index of the node to examine
        @return: An integer indicating the length of the path from the node to the technical root
        """
        depth = 0
        while node_idx > 0:
            node_idx = self.parents[node_idx]
            depth += 1
        return depth

    def is_descendant(self, anc_idx, desc_idx):
        """Check if a node is a descendant of another node (there is a directed
        path between them.

        @param anc_idx: the "ancestor node" index – the node where the path should begin
        @param desc_idx: the "descendant node" index – the node where the path should end
        @return: True if the path between the nodes exist, False otherwise
        """
        node_idx = desc_idx
        while node_idx > 0:
            node_idx = self.parents[node_idx]
            if node_idx == anc_idx:
                return True
        return anc_idx == node_idx

    def is_right_child(self, node_idx):
        return self.parents[node_idx] < node_idx

    def __hash__(self):
        # TODO: this is probably slow... make it faster, possibly replace the lists with tuples?
        return hash(tuple(self.nodes)) ^ hash(tuple(self.parents))

    def __eq__(self, other):
        return (self is other or
                ((self.parents is other.parents or self.parents == other.parents) and
                 (self.nodes is other.nodes or self.nodes == other.nodes)))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __unicode__(self):
        return ' '.join(['%d|%d|%s|%s' % (idx, parent_idx, node.t_lemma, node.formeme)
                         for idx, (parent_idx, node)
                         in enumerate(zip(self.parents, self.nodes))])

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __str__(self):
        return unicode(self).encode('UTF-8', 'replace')

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.nodes)

    def clone(self):
        return TreeData(nodes=self.nodes, parents=self.parents)

    def to_tok_list(self):
        """Convert the tree to a list of tokens -- (word, empty tag) pairs."""
        return [(n.t_lemma, None) for n in self.nodes[1:]]

    def create_ttree(self):
        """Convert the TreeData structure to a regular t-tree."""
        tnodes = [T(data={'ord': 0})] + [T(data={'t_lemma': node.t_lemma,
                                                 'formeme': node.formeme,
                                                 'ord': i})
                                         for i, node in enumerate(self.nodes[1:], start=1)]
        for parent_idx, tnode in zip(self.parents[1:], tnodes[1:]):
            tnode.parent = tnodes[parent_idx]
        return tnodes[0]

    def get_subtree(self, node_idxs):
        """Return a subtree of the current tree that only contains the nodes whose indexes
        are given in the parameter.

        @param node_idxs: a set or list of valid node indexes (integers) that are to be included \
            in the returned subtree. If a list is given, it may be changed.
        """
        if isinstance(node_idxs, set):
            node_idxs |= set([0])
            node_idxs = sorted(node_idxs)
        else:
            if 0 not in node_idxs:
                node_idxs.append(0)
            node_idxs.sort()
        idx_mapping = {old_idx: new_idx for old_idx, new_idx in zip(node_idxs, range(len(node_idxs)))}
        idx_mapping[-1] = -1  # “mapping” for technical roots
        new_parents = [idx_mapping[parent] for idx, parent in enumerate(self.parents)
                       if idx in node_idxs]
        new_nodes = [node for idx, node in enumerate(self.nodes) if idx in node_idxs]
        return TreeData(new_nodes, new_parents)

    def get_subtrees_list(self, start_idxs, adding_idxs):
        """Return a list of subtrees that originate from the current tree; starting from a
        subtree composed of nodes specified in start_idxs and gradually adding nodes specified
        in adding_idxs.

        Will not give the subtree with start_idxs only, starts with start_idxs and the first
        element of adding_idxs.

        @param start_idxs: list of indexes for a subtree to start with
        @param adding_idxs: list of lists of indexes that are to be added stepwise
        @return: a list of growing subtrees
        """
        trees = []
        start_idxs = set(start_idxs)
        for add_list in adding_idxs:
            start_idxs |= set(add_list)
            trees.append(self.get_subtree(start_idxs))
        return trees

    def __lt__(self, other):
        """Comparing by node values, then by parents. Actually only needed to make
        calls to heapq in CandidateList deterministic."""
        return (self.nodes, self.parents) < (other.nodes, other.parents)

    # Adapted from http://rosettacode.org/wiki/Longest_common_subsequence#Python
    @staticmethod
    def _longest_common_subseq(tree_a, idxs_a, tree_b, idxs_b):
        """Return the common subsequence of tree indexes (out of the given indexes).
        This is a helper method for `_common_subtree_size` and `_common_subtree_idxs`.

        @param tree_a: first tree
        @param idxs_a: node indexes of the first tree to examine
        @param tree_a: second tree
        @param idxs_a: node indexes of the second tree to examine
        @return: a tuple of list of node indexes (nodes which are common to both trees, \
            possibly under different indexes)
        """
        # dynamic programming, substring
        lengths = [[0 for j in range(len(idxs_b) + 1)] for i in range(len(idxs_a) + 1)]
        # row 0 and column 0 are initialized to 0 already
        for i, idx_a in enumerate(idxs_a):
            for j, idx_b in enumerate(idxs_b):
                # check for node equality (t_lemma, formeme, precede/follow parent)
                if (tree_a[idx_a] == tree_b[idx_b] and
                        tree_a.is_right_child(idx_a) == tree_b.is_right_child(idx_b)):
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
        # read the substring out from the matrix, from the end to the beginning
        res_a = []
        res_b = []
        i, j = len(idxs_a), len(idxs_b)
        while i != 0 and j != 0:
            if lengths[i][j] == lengths[i - 1][j]:
                i -= 1
            elif lengths[i][j] == lengths[i][j - 1]:
                j -= 1
            else:
                res_a.append(idxs_a[i - 1])
                res_b.append(idxs_b[j - 1])
                i -= 1
                j -= 1
        res_a.reverse()
        res_b.reverse()
        return res_a, res_b

    @staticmethod
    def _common_subtree_size(tree_a, idx_a, tree_b, idx_b):
        com_ch_a, com_ch_b = TreeData._longest_common_subseq(tree_a, tree_a.children_idxs(idx_a),
                                                             tree_b, tree_b.children_idxs(idx_b))
        return len(com_ch_a) + sum(TreeData._common_subtree_size(tree_a, idx_sub_a, tree_b, idx_sub_b)
                                   for idx_sub_a, idx_sub_b in zip(com_ch_a, com_ch_b))

    def common_subtree_size(self, other):
        """Return the common subtree size of the two trees; the technical root is counted,
        i.e. the common subtree size >= 1.
        @rtype: integer
        """
        return TreeData._common_subtree_size(self, -1, other, -1)

    @staticmethod
    def _common_subtree_idxs(tree_a, idx_a, tree_b, idx_b):
        com_ch_a, com_ch_b = TreeData._longest_common_subseq(tree_a, tree_a.children_idxs(idx_a),
                                                             tree_b, tree_b.children_idxs(idx_b))
        append_a, append_b = [], []
        for idx_sub_a, idx_sub_b in zip(com_ch_a, com_ch_b):
            com_sub_a, com_sub_b = TreeData._common_subtree_idxs(tree_a, idx_sub_a, tree_b, idx_sub_b)
            append_a.extend(com_sub_a)
            append_b.extend(com_sub_b)
        return com_ch_a + append_a, com_ch_b + append_b

    def common_subtree_idxs(self, other):
        """Return indexes of nodes belonging to the common subtree of the two trees.
        @return: a pair of lists of node indexes
        """
        return TreeData._common_subtree_idxs(self, -1, other, -1)

    def get_common_subtree(self, other):
        """Create a new subtree, composed of nodes that this tree shares with the other tree."""
        idxs, _ = self.common_subtree_idxs(other)
        return self.get_subtree(idxs)

    def _compare_node_depth(self, idx_a, idx_b):
        """Compare the depth of nodes at given indexes."""
        return cmp(self.node_depth(idx_a), self.node_depth(idx_b))

    def diffing_trees(self, other, symmetric=False):
        """Given two trees, find their common subtree and return a pair of lists of trees
        that start with the common subtree and gradually diverge towards the original trees.

        If `symmetric' is set, both lists will have equal length; if one of the trees is bigger,
        the gradual subtree changes in its list will be bigger.

        @param symmetric: return tree lists of the same length?
        @rtype: a pair of lists of TreeData
        @return: diverging lists of subtrees of self, other (in this order)
        """
        com_self, com_other = self.common_subtree_idxs(other)
        diff_self = sorted(list(set(range(len(self))) - set(com_self)), cmp=self._compare_node_depth)
        diff_other = sorted(list(set(range(len(other))) - set(com_other)), cmp=other._compare_node_depth)
        # symmetric list lengths
        if symmetric:
            # one tree is a subtree of the other – back-off to returning self, other
            if not diff_other or not diff_self:
                return ([self], [other])
            diff_self, diff_other = _group_lists(diff_self, diff_other)
        # just all possible trees on the paths (lists may be empty)
        else:
            diff_self = [[i] for i in diff_self]
            diff_other = [[i] for i in diff_other]

        return (self.get_subtrees_list(com_self, diff_self),
                other.get_subtrees_list(com_other, diff_other))


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

    # @jit
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
        return TreeNode(self.tree, self.tree.parents[self.node_idx])

    @property
    def root(self):
        return TreeNode(self.tree, 0)

    @property
    def formeme(self):
        return self.tree.nodes[self.node_idx].formeme

    @property
    def t_lemma(self):
        return self.tree.nodes[self.node_idx].t_lemma

    @property
    def is_right_child(self):
        return self.tree.parents[self.node_idx] < self.node_idx

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

    def __eq__(self, other):
        return (self is other or
                (self.node_idx == other.node_idx and (self.tree is other.tree or self.tree == other.tree)))

    def __len__(self):
        return len(self.tree)

    def __hash__(self):
        return hash(self.tree) ^ hash(self.node_idx)
