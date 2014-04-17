#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentence planning: Generating T-trees from dialogue acts.
"""

from alex.components.nlg.tectotpl.core.document import Document
from alex.components.nlg.tectotpl.core.node import T
from collections import deque


class SentencePlanner(object):
    """Common ancestor of sentence planners."""

    def __init__(self, cfg):
        """Initialize, setting language, selector, and successor generator"""        
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        # candidate generator
        self.candgen = cfg['candgen']

    def generate_tree(self, da, gen_doc=None):
        """Generate a tree given input DA.
        @param gen_doc: if this is None, return the tree in a new Document object, otherwise append
        """
        raise NotImplementedError

    def get_target_zone(self, gen_doc):
        """Create a document if necessary, add a new bundle and a new zone into it,
        and return the newly created zone."""
        if gen_doc is None:
            gen_doc = Document()
        bundle = gen_doc.create_bundle()
        zone = bundle.create_zone(self.language, self.selector)
        return zone


class SamplingPlanner(SentencePlanner):
    """Random t-tree generator given DAs.

    Trainable from DA distributions
    """

    MAX_TREE_SIZE = 50

    def __init__(self, cfg):
        super(SamplingPlanner, self).__init__(cfg)
        # ranker (selecting the best candidate)
        self.ranker = cfg['ranker']

    def generate_tree(self, da, gen_doc=None):
        zone = self.get_target_zone(gen_doc)
        root = zone.create_ttree()
        cdfs = self.candgen.get_merged_cdfs(da)
        nodes = deque([self.generate_child(root, da, cdfs[root.formeme])])
        treesize = 1
        while nodes and treesize < self.MAX_TREE_SIZE:
            node = nodes.popleft()
            if node.formeme not in cdfs:  # skip weirdness
                continue
            for _ in xrange(self.candgen.get_number_of_children(node.formeme)):
                child = self.generate_child(node, da, cdfs[node.formeme])
                nodes.append(child)
                treesize += 1
        return gen_doc

    def generate_child(self, parent, da, cdf):
        """Generate one t-node, given its parent and the CDF for the possible children."""
        formeme, t_lemma, right = self.ranker.get_best_child(parent, da, cdf)
        child = parent.create_child()
        child.t_lemma = t_lemma
        child.formeme = formeme
        if right:
            child.shift_after_subtree(parent)
        else:
            child.shift_before_subtree(parent)
        return child


class ASearchPlanner(SentencePlanner):
    """Sentence planner using A*-search."""

    def __init__(self, cfg):
        super(ASearchPlanner, self).__init__(cfg)

    def generate_tree(self, da, gen_doc=None):
        # TODO tree hashing
        # TODO TreeList – search by hash + some queue sorted according to scores
        # TODO add scoring
        # TODO add future cost ?
        # initialization
        open_list, close_list = deque([T()]), deque()
        # main search loop
        while open_list:
            cand = open_list.popleft()
            successors = self.candgen.get_all_successors(open)
            open_list.append([s for s in successors
                              if not s in open_list and not s in close_list])
        # return the result
        return close_list.popleft()
