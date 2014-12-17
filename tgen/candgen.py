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

from futil import read_das, read_ttrees
from tree import TreeNode, NodeData
from tgen.logf import log_warn


class RandomCandidateGenerator(object):
    """Random candidate generation according to a parent-child distribution.
    Its main function now is to generate all possible successors of a tree --
    see get_all_successors().
    """

    def __init__(self, cfg):
        self.form_counts = None
        self.child_cdfs = None
        self.max_children = None
        self.prune_threshold = cfg.get('prune_threshold', 1)
        self.parent_lemmas = cfg.get('parent_lemmas', False)

    @staticmethod
    def load_from_file(fname):
        log_info('Loading model from ' + fname)
        with file_stream(fname, mode='rb', encoding=None) as fh:
            candgen = pickle.load(fh)
            if type(candgen) == dict:  # backward compatibility
                form_counts = candgen
                candgen = RandomCandidateGenerator({})
                candgen.form_counts = form_counts
                candgen.child_cdfs = pickle.load(fh)
                candgen.max_children = pickle.load(fh)
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
        ttrees = read_ttrees(t_file)
        log_info('Reading ' + da_file)
        das = read_das(da_file)

        # collect counts
        log_info('Collecting counts')
        form_counts = {}
        child_counts = defaultdict(Counter)

        for ttree, da in zip(ttrees.bundles, das):
            ttree = ttree.get_zone('en', '').ttree
            # counts for formeme/lemma given dai
            for dai in da:
                for tnode in ttree.get_descendants():
                    if dai not in form_counts:
                        form_counts[dai] = defaultdict(Counter)
                    parent_id = self._parent_node_id(tnode.parent)
                    child_id = (tnode.formeme, tnode.t_lemma, tnode > tnode.parent)
                    form_counts[dai][parent_id][child_id] += 1
            # counts for number of children
            for tnode in ttree.get_descendants(add_self=1):
                child_counts[self._parent_node_id(tnode)][len(tnode.get_children())] += 1

        # prune counts
        if self.prune_threshold > 1:
            for dai, forms in form_counts.items():
                self.prune(forms)
                if not forms:
                    del form_counts[dai]
            self.prune(child_counts)

        # transform counts
        self.form_counts = form_counts
        self.child_cdfs = self.cdfs_from_counts(child_counts)
        self.max_children = {par_id: max(child_counts[par_id].keys())
                             for par_id in child_counts.keys()}

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

    def get_merged_cdfs(self, da):
        """Get merged child CDFs for the DAIs in the given DA.

        @param da: the current dialogue act
        """
        merged_counts = defaultdict(Counter)
        for dai in da:
            try:
                for parent_id in self.form_counts[dai]:
                    merged_counts[parent_id].update(self.form_counts[dai][parent_id])
            except KeyError:
                log_warn('DAI ' + unicode(dai) + ' unknown, adding nothing to CDF.')
        return self.cdfs_from_counts(merged_counts)

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

    def sample(self, cdf):
        """Return a sample from the distribution, given a CDF (as a list)."""
        total = cdf[-1][1]
        rand = random.random() * total  # get a random number in [0,total)
        for key, ubound in cdf:
            if ubound > rand:
                return key
        raise Exception('Unable to generate from CDF!')

    def get_number_of_children(self, parent_id):
        if parent_id not in self.child_cdfs:
            return 0
        return self.sample(self.child_cdfs[parent_id])

    def get_best_child(self, parent, da, cdf):
        return self.sample(cdf)

    def get_all_successors(self, cand_tree, cdfs):
        """Get all possible successors of a candidate tree.

        @param cand_tree: The current candidate tree to be expanded
        @param cdfs: Merged CDFs of children given the current DA (obtained using get_merged_cdfs)
        """
        # always try adding one node to all possible places
        # TODO possibly avoid creating TreeNode instances for iterating
        nodes = TreeNode(cand_tree).get_descendants(add_self=1, ordered=1)
        res = []
        for node_num, node in enumerate(nodes):
            parent_id = self._parent_node_id(node)
            # skip nodes that can't have more children
            if (len(node.get_children()) >= self.max_children.get(parent_id, 0) or
                    parent_id not in cdfs):
                continue
            # try all formeme/t-lemma/direction variants of the children at the given spot
            for formeme, t_lemma, right in map(lambda item: item[0], cdfs[parent_id]):
                succ_tree = cand_tree.clone()
                succ_tree.create_child(node_num, right, NodeData(t_lemma, formeme))
                res.append(succ_tree)
        # return all successors
        return res
