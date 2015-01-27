#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generating candidate subtrees to enhance the current candidate tree.
"""

from __future__ import unicode_literals
from collections import defaultdict, Counter
import cPickle as pickle
import random

from logf import log_info
from alex.components.nlg.tectotpl.core.util import file_stream

from futil import read_das, read_ttrees, ttrees_from_doc
from tree import TreeNode, NodeData
from tgen.logf import log_warn
from tgen.tree import TreeData
from tgen.planner import CandidateList


class RandomCandidateGenerator(object):
    """Random candidate generation according to a parent-child distribution.
    Its main function now is to generate all possible successors of a tree --
    see get_all_successors().
    """

    def __init__(self, cfg):
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        # possible children given parent ID and DAIs
        self.child_type_counts = None
        # CDFs on the number of children given parent ID
        self.child_num_cdfs = None
        # maximum number of children given parent ID
        self.max_children = None
        # expected number of children given parent ID
        self.exp_child_num = None
        self.prune_threshold = cfg.get('prune_threshold', 1)
        self.parent_lemmas = cfg.get('parent_lemmas', False)
        # limits on tree size / number of nodes at depth level, given DAIs
        # (initialized to boolean, which then triggers training)
        self.node_limits = cfg.get('node_limits', False)

    @staticmethod
    def load_from_file(fname):
        log_info('Loading model from ' + fname)
        with file_stream(fname, mode='rb', encoding=None) as fh:
            candgen = pickle.load(fh)
            if type(candgen) == dict:  # backward compatibility
                child_type_counts = candgen
                candgen = RandomCandidateGenerator({})
                candgen.child_type_counts = child_type_counts
                candgen.child_num_cdfs = pickle.load(fh)
                candgen.max_children = pickle.load(fh)
            if not hasattr(candgen, 'node_limits'):  # backward compatibility
                candgen.node_limits = None
            if not hasattr(candgen, 'child_type_counts'):  # backward compatibility
                candgen.child_type_counts = candgen.form_counts
                candgen.child_num_cdfs = candgen.child_cdfs
            if not hasattr(candgen, 'exp_child_num'):
                candgen.exp_child_num = candgen.exp_from_cdfs(candgen.child_num_cdfs)
            return candgen

    def save_to_file(self, fname):
        log_info('Saving model to ' + fname)
        with file_stream(fname, mode='wb', encoding=None) as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)

    def train(self, da_file, t_file):
        """``Training'' the generator (collect counts of DAIs and corresponding t-nodes).

        @param da_file: file with training DAs
        @param t_file: file with training t-trees (YAML or pickle)
        """
        # read training data
        log_info('Reading ' + t_file)
        ttrees = ttrees_from_doc(read_ttrees(t_file), self.language, self.selector)
        log_info('Reading ' + da_file)
        das = read_das(da_file)

        # collect counts
        log_info('Collecting counts')
        child_type_counts = {}
        child_num_counts = defaultdict(Counter)
        max_total_nodes = defaultdict(int)
        max_level_nodes = defaultdict(Counter)

        for ttree, da in zip(ttrees, das):
            # counts for formeme/lemma given DAI
            for dai in da:
                for tnode in ttree.get_descendants():
                    if dai not in child_type_counts:
                        child_type_counts[dai] = defaultdict(Counter)
                    parent_id = self._parent_node_id(tnode.parent)
                    child_id = (tnode.formeme, tnode.t_lemma, tnode > tnode.parent)
                    child_type_counts[dai][parent_id][child_id] += 1

            # counts for number of children
            for tnode in ttree.get_descendants(add_self=1):
                child_num_counts[self._parent_node_id(tnode)][len(tnode.get_children())] += 1

            # counts for max. number of nodes
            total_nodes = len(ttree.get_descendants(add_self=True))
            for dai in da:
                max_total_nodes[dai] = max((max_total_nodes[dai], total_nodes))
            level_nodes = defaultdict(int)
            for tnode in ttree.get_descendants(add_self=True):
                level_nodes[tnode.get_depth()] += 1
            for dai in da:
                for level in level_nodes.iterkeys():
                    max_level_nodes[dai][level] = max((max_level_nodes[dai][level],
                                                       level_nodes[level]))

        # prune counts
        if self.prune_threshold > 1:
            for dai, forms in child_type_counts.items():
                self.prune(forms)
                if not forms:
                    del child_type_counts[dai]
            self.prune(child_num_counts)

        # transform counts
        self.child_type_counts = child_type_counts
        self.child_num_cdfs = self.cdfs_from_counts(child_num_counts)
        self.max_children = {par_id: max(child_num_counts[par_id].keys())
                             for par_id in child_num_counts.keys()}
        self.exp_child_num = self.exp_from_cdfs(self.child_num_cdfs)

        if self.node_limits:
            self.node_limits = {dai: {'total': max_total}
                                for dai, max_total in max_total_nodes.iteritems()}
            for dai, max_levels in max_level_nodes.iteritems():
                self.node_limits[dai].update(max_levels)
        else:
            self.node_limits = None

    def _parent_node_id(self, node):
        """Parent node id according to the self.parent_lemmas setting (either
        only formeme, or lemma + formeme)

        @param node: a node that is considered a parent in the current situation
        """
        if self.parent_lemmas:
            return (node.t_lemma, node.formeme)
        return node.formeme

    def prune(self, counts):
        """Prune a counts dictionary, keeping only items with counts above
        the prune_threshold given in the constructor."""
        for parent_type, child_types in counts.items():
            for child_form, child_count in child_types.items():
                if child_count < self.prune_threshold:
                    del child_types[child_form]
            if not counts[parent_type]:
                del counts[parent_type]

    def get_merged_child_type_cdfs(self, da):
        """Get merged child CDFs for the DAIs in the given DA.

        @param da: the current dialogue act
        """
        merged_counts = defaultdict(Counter)
        for dai in da:
            try:
                for parent_id in self.child_type_counts[dai]:
                    merged_counts[parent_id].update(self.child_type_counts[dai][parent_id])
            except KeyError:
                log_warn('DAI ' + unicode(dai) + ' unknown, adding nothing to CDF.')
        return self.cdfs_from_counts(merged_counts)

    def get_merged_limits(self, da):
        """Return merged limits on node counts (total and on each tree level). Uses a
        maximum for all DAIs in the given DA.

        Returns none if the given candidate generator does not have any node limits.

        @param da: the current dialogue act
        @rtype: defaultdict(Counter)
        """
        if not self.node_limits:
            return None
        merged_limits = defaultdict(int)
        for dai in da:
            try:
                for level, level_limit in self.node_limits[dai].iteritems():
                    merged_limits[level] = max((level_limit, merged_limits[level]))
            except KeyError:
                log_warn('DAI ' + unicode(dai) + ' unknown, limits unchanged.')
        return merged_limits

    def cdfs_from_counts(self, counts):
        """Given a dictionary of counts, create a dictionary of corresponding CDFs."""
        cdfs = {}
        for key in counts:
            tot = 0
            cdf = []
            for subkey in counts[key]:
                tot += counts[key][subkey]
                cdf.append((subkey, tot))
            # normalize
            cdf = [(subkey, val / float(tot)) for subkey, val in cdf]
            cdfs[key] = cdf
        return cdfs

    def exp_from_cdfs(self, cdfs):
        """Given a dictionary of CDFs (with numeric subkeys), create a dictionary of
        corresponding expected values. Used for children counts."""
        exps = {}
        for key, cdf in cdfs.iteritems():
            # convert the CDF -- array of tuples (value, cumulative probability) into an
            # array of rising cumulative probabilities (duplicate last probability
            # if there is no change)
            cdf_arr = [-1] * (max(i for i, _ in cdf) + 1)
            for i, v in cdf:
                cdf_arr[i] = v
            for i, v in enumerate(cdf_arr):
                if v == -1:
                    if i == 0:
                        cdf_arr[0] = 0.0
                    else:
                        cdf_arr[i] = cdf_arr[i - 1]
            # compute E[X] = sum _x=1^inf ( 1 - CDF(x) )
            exps[key] = sum(1 - cdf_val for cdf_val in cdf_arr)
        return exps

    def sample(self, cdf):
        """Return a sample from the distribution, given a CDF (as a list)."""
        total = cdf[-1][1]
        rand = random.random() * total  # get a random number in [0,total)
        for key, ubound in cdf:
            if ubound > rand:
                return key
        raise Exception('Unable to generate from CDF!')

    def get_number_of_children(self, parent_id):
        if parent_id not in self.child_num_cdfs:
            return 0
        return self.sample(self.child_num_cdfs[parent_id])

    def get_best_child(self, parent, da, cdf):
        return self.sample(cdf)

    def get_all_successors(self, cand_tree, cdfs, node_limits=None):
        """Get all possible successors of a candidate tree, given CDFS and node number limits.

        NB: This assumes projectivity (will never create a non-projective tree).

        @param cand_tree: The current candidate tree to be expanded
        @param cdfs: Merged CDFs of children given the current DA (obtained using get_merged_child_type_cdfs)
        @param node_limits: limits on the number of nodes (total and on different child_depth levels, \
            obtained via get_merged_limits)
        """
        # TODO possibly avoid creating TreeNode instances for iterating
        nodes = TreeNode(cand_tree).get_descendants(add_self=1, ordered=1)
        nodes_on_level = defaultdict(int)
        res = []
        if node_limits is not None:
            # stop if maximum number of nodes is reached
            if len(nodes) >= node_limits['total']:
                return []
            # remember number of nodes on all levels
            for node in nodes:
                nodes_on_level[node.get_depth()] += 1

        # try adding one node to all possible places
        for node_num, node in enumerate(nodes):
            # skip nodes that can't have more children
            parent_id = self._parent_node_id(node)
            if (len(node.get_children()) >= self.max_children.get(parent_id, 0) or
                    parent_id not in cdfs):
                continue
            # skip nodes above child_depth levels where the maximum number of nodes has been reached
            if node_limits is not None:
                child_depth = node.get_depth() + 1
                if nodes_on_level[child_depth] >= node_limits[child_depth]:
                    continue
            # try all formeme/t-lemma/direction variants of a new child under the given parent node
            for formeme, t_lemma, right in map(lambda item: item[0], cdfs[parent_id]):
                # place the child directly following/preceding the parent
                succ_tree = cand_tree.clone()
                succ_tree.create_child(node_num, right, NodeData(t_lemma, formeme))
                res.append(succ_tree)
                # if the parent already has some left/right children, try to place the new node
                # in all possible positions before/after their subtrees (for left/right child,
                # respectively)
                children_idxs = cand_tree.children_idxs(node_num, left_only=not right, right_only=right)
                for child_idx in children_idxs:
                    succ_tree = cand_tree.clone()
                    subtree_bound = succ_tree.subtree_bound(child_idx, right)
                    succ_tree.create_child(node_num, subtree_bound + (1 if right else 0),
                                           NodeData(t_lemma, formeme))
                    res.append(succ_tree)

        # return all created successors
        return res

    def get_future_promise(self, cand_tree):
        """Get the total expected number of future children in the given tree (based on the
        expectations for the individual node types)."""
        promise = 0.0
        for node_idx in xrange(len(cand_tree)):
            exp_child_num = self.exp_child_num[self._parent_node_id(cand_tree.nodes[node_idx])]
            promise += max(cand_tree.children_num(node_idx) - exp_child_num, 0)
        return promise

    def can_generate(self, tree, da):
        """Check if the candidate generator can generate a given tree at all.

        This is for debugging purposes only.
        Tries if get_all_successors always returns a successor that leads to the given tree
        (puts on the open list only successors that are subtrees of the given tree).
        """
        cdfs = self.get_merged_child_type_cdfs(da)
        node_limits = self.get_merged_limits(da)
        open_list = CandidateList({TreeData(): 1})
        found = False
        tree_no = 0

        while open_list and not found:
            cur_st, _ = open_list.pop()
            if cur_st == tree:
                found = True
                break
            for succ in self.get_all_successors(cur_st, cdfs, node_limits):
                tree_no += 1
                # only push on the open list if the successor is still a subtree of the target tree
                if tree.common_subtree_size(succ) == len(succ):
                    open_list.push(succ, len(succ))

        if not found:
            log_info('Did not find tree: ' + unicode(tree) + ' for DA: ' + unicode(da) + ('(total %d trees)' % tree_no))
            return False
        log_info('Found tree: %s for DA: %s (as %d-th tree)' % (unicode(tree), unicode(da), tree_no))
        return True

    def can_generate_greedy(self, tree, da):
        """Check if the candidate generator can generate a given tree greedily, always
        pursuing the first viable path.

        This is for debugging purposes only.
        Uses `get_all_successors` and always goes on with the first one that increases coverage
        of the current tree.
        """
        cdfs = self.get_merged_child_type_cdfs(da)
        node_limits = self.get_merged_limits(da)
        cur_subtree = TreeData()
        found = True

        while found and cur_subtree != tree:
            found = False
            for succ in self.get_all_successors(cur_subtree, cdfs, node_limits):
                # use the first successor that is still a subtree of the target tree
                if tree.common_subtree_size(succ) == len(succ):
                    cur_subtree = succ
                    found = True
                    break

        # we have hit a dead end
        if cur_subtree != tree:
            log_info('Did not find tree: ' + unicode(tree) + ' for DA: ' + unicode(da))
            return False

        # everything alright
        log_info('Found tree: %s for DA: %s' % (unicode(tree), unicode(da)))
        return True
