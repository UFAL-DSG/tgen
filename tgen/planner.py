#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentence planning: Generating T-trees from dialogue acts.
"""

import heapq

from alex.components.nlg.tectotpl.core.document import Document
import alex.components.nlg.tectotpl.core.node
from collections import deque
from UserDict import DictMixin


class T(alex.components.nlg.tectotpl.core.node.T):
    """Just a copy the T class from Alex with my override for __hash__
    that works with the sentence planner

    TODO make this work well for subtrees and not-so-nice trees (the parent
    orders should be relative to their position in the array).
    """

    def __eq__(self, other):
        """Return true if t-lemmas, formemes, and parent orders in the whole
        tree are equal."""
        self_desc = self.get_descendants(add_self=1, ordered=1)
        other_desc = self.get_descendants(add_self=1, ordered=1)
        if len(self_desc) != len(other_desc):
            return False
        for self_node, other_node in zip(self_desc, other_desc):
            if (self_node.t_lemma != other_node.t_lemma or
                    self_node.formeme != other_node.formeme or
                    self_node.parent.ord != other_node.parent.ord):
                return False
        return True

    def __hash__(self):
        """Return hash of the tree that is composed of t-lemmas, formemes,
        and parent orders of all nodes in the tree (ordered)."""
        desc = self.get_descendants(add_self=1, ordered=1)
        return hash(unicode(self))

    def __unicode__(self):
        desc = self.get_descendants(add_self=1, ordered=1)
        return ' '.join([n.t_lemma + '|' + n.formeme + '|' + str(n.parent.ord)
                         for n in desc])


class CandidateList(DictMixin):
    """List of candidate trees that can be quickly checked for membership and
    is sorted according to scores."""

    def __init__(self, members=None):
        self.queue = []
        self.members = {}
        if members:
            for key, value in members.iteritems():
                self.push(key, value)
        pass

    def __nonzero__(self):
        return len(self.members) > 0

    def __contains__(self, key):
        return key in self.members

    def __getitem__(self, key):
        return self.members[key]

    def __setitem__(self, key, value):
        # slow if key is in list
        if key in self:
            if value == self[key]:
                return
            queue_index = (i for i, v in enumerate(self.queue) if i[1] == key).next()
            self.queue[queue_index] = (value, key)
            self.__fix_queue(queue_index)
        else:
            heapq.heappush(self.queue, (value, key))
        self.members[key] = value

    def __delitem__(self, key):
        del self.members[key]  # this will raise an exception if the key is not there
        queue_index = (i for i, v in enumerate(self.queue) if i[1] == key).next()
        self.queue[queue_index] = self.queue[-1]
        del self.queue[-1]
        self.__fix_queue(queue_index)

    def keys(self):
        return self.members.keys()

    def pop(self):
        value, key = heapq.heappop(self.queue)
        del self.members[key]
        return key, value

    def push(self, key, value):
        self[key] = value  # calling __setitem__; it will test for membership

    def pushall(self, members):
        for key, value in members.iteritems():
            self[key] = value

    def __fix_queue(self, index):
        """Fixing a heap after change in one element (swapping elements until heap
        condition is satisfied.)"""
        down = index / 2 - 1
        up = index * 2 + 1
        if index > 0 and self.queue[index][0] < self.queue[down][0]:
            self.queue[index], self.queue[down] = self.queue[down], self.queue[index]
            self.__fix_queue(down)
        elif up < len(self.queue):
            if self.queue[index][0] > self.queue[up][0]:
                self.queue[index], self.queue[up] = self.queue[up], self.queue[index]
                self.__fix_queue(up)
            elif self.queue[index][0] > self.queue[up + 1][0]:
                self.queue[index], self.queue[up + 1] = self.queue[up + 1], self.queue[index]
                self.__fix_queue(up + 1)


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
        and return the newly created zone and the document (original or new)."""
        if gen_doc is None:
            gen_doc = Document()
        bundle = gen_doc.create_bundle()
        zone = bundle.create_zone(self.language, self.selector)
        return zone, gen_doc


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
        zone, gen_doc = self.get_target_zone(gen_doc)
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

    MAX_ITER = 10000

    def __init__(self, cfg):
        super(ASearchPlanner, self).__init__(cfg)
        self.debug_out = None
        if 'debug_out' in cfg:
            self.debug_out = cfg['debug_out']

    def generate_tree(self, da, gen_doc=None, gold_ttree=None):
        # TODO add future cost ?
        # initialization
        open_list, close_list = CandidateList({T(): 0.0}), CandidateList()
        num_iter = 0
        # main search loop
        while open_list and num_iter < self.MAX_ITER:
            cand = open_list.pop()
            score = 0.0
            if gold_ttree and cand == gold_ttree:
                score = 1.0
            close_list.push(cand, score)
            if self.debug_out:
                print >> self.debug_out, "IT %05d:" % num_iter, cand
            successors = self.candgen.get_all_successors(cand)
            # TODO add real scoring here
            open_list.pushall({s: 0.0 for s in successors if not s in close_list})
            num_iter += 1
        # return the result
        zone, gen_doc = self.get_target_zone(gen_doc)
        zone.ttree = close_list.pop()
        return gen_doc
