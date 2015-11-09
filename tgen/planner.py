#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentence planning: Generating T-trees from dialogue acts.
"""

from __future__ import unicode_literals
from collections import deque
from UserDict import DictMixin

from logf import log_debug
from tree import TreeData, TreeNode, NodeData
from alex.components.nlg.tectotpl.core.util import first


class CandidateList(DictMixin):
    """List of candidate trees that can be quickly checked for membership and
    can yield the best-scoring candidate quickly.

    The implementation involves a dictionary and a heap. The heap is sorted by
    value (score), the dictionary is indexed by the keys (candidate trees).

    The heap is a min-heap; negative score values are therefore used to keep
    the highest score on top."""

    def __init__(self, members=None):
        self.queue = []
        self.members = {}
        if members:
            self.push_all(members)
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
            queue_index = (i for i, v in enumerate(self.queue) if v[1] == key).next()
            self.queue[queue_index] = (value, key)
            self._siftup(queue_index)
        else:
            self.queue.append((value, key))
            self._siftdown(0, len(self.queue) - 1)
        self.members[key] = value

    def __delitem__(self, key):
        del self.members[key]  # this will raise an exception if the key is not there
        queue_index = (i for i, v in enumerate(self.queue) if v[1] == key).next()
        self.queue[queue_index] = self.queue[-1]
        del self.queue[-1]
        if queue_index < len(self.queue):  # skip if we deleted the last item
            self._siftup(queue_index)

    def keys(self):
        return self.members.keys()

    def pop(self):
        """Return the first item on the heap and remove it."""
        last = self.queue.pop()  # raises appropriate IndexError if heap is empty
        if self.queue:
            value, key = self.queue[0]
            self.queue[0] = last
            self._siftup(0)
        else:
            value, key = last
        del self.members[key]
        return key, value

    def peek(self):
        """Return the first item on the heap, but do not remove it."""
        value, key = self.queue[0]
        return key, value

    def push(self, key, value):
        """Push one key-value pair to the heap."""
        self[key] = value  # calling __setitem__; it will test for membership

    def push_all(self, members):
        """Push all members of the given structure to the heap
        (a list of pairs key-value or a dictionary are accepted)."""
        if isinstance(members, dict):
            members = members.iteritems()
        for key, value in members:
            self[key] = value

    def prune(self, size):
        """Trim the list to the given size, return the rest."""
        if len(self.queue) <= size:  # don't do anything if we're small enough already
            return {}
        pruned_queue = []
        pruned_members = {}
        for _ in xrange(size):
            key, val = self.pop()
            pruned_queue.append((val, key))
            pruned_members[key] = val
        remain_members = self.members
        self.members = pruned_members
        self.queue = pruned_queue
        return remain_members

    def __repr__(self):
        return ' '.join(['%6.3f' % val for val, _ in self.queue])

    def _siftdown(self, startpos, pos):
        """Copied from heapq._siftdown, with custom comparison (comparing *just* by 1st element)"""
        heap = self.queue
        newitem = heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = heap[parentpos]
            if newitem[0] < parent[0]:
                heap[pos] = parent
                pos = parentpos
                continue
            break
        heap[pos] = newitem

    def _siftup(self, pos):
        """Copied from heapq._siftup, with custom comparison (comparing *just* by 1st element)"""
        heap = self.queue
        endpos = len(heap)
        startpos = pos
        newitem = heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not heap[childpos] < heap[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            heap[pos] = heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        heap[pos] = newitem
        self._siftdown(startpos, pos)


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

        @param gen_doc: if this is None, return the tree as a TreeData object, otherwise append \
            to a t-tree document
        """
        raise NotImplementedError

    def get_target_zone(self, gen_doc):
        """Find the first bundle in the given document that does not have the target
        zone (or create it), then create the target zone and return it.

        @rtype: Zone
        """
        bundle = first(lambda bundle: not bundle.has_zone(self.language, self.selector),
                       gen_doc.bundles) or gen_doc.create_bundle()
        zone = bundle.create_zone(self.language, self.selector)
        return zone


class SamplingPlanner(SentencePlanner):
    """Random t-tree generator given DAs.

    TODO: This is obsolete, it will not work after the introduction of TreeData.
    Fix it or remove it (there's probably no point in having it now).
    """

    MAX_TREE_SIZE = 50

    def __init__(self, cfg):
        super(SamplingPlanner, self).__init__(cfg)
        # ranker (selecting the best candidate)
        self.ranker = None
        if 'ranker' in cfg:
            self.ranker = cfg['ranker']

    def generate_tree(self, da, gen_doc=None):
        root = TreeNode(TreeData())
        cdfs = self.candgen.get_merged_child_type_cdfs(da)
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
        if gen_doc:
            zone = self.get_target_zone(gen_doc)
            zone.ttree = root.create_ttree()
            return
        return root.tree

    def generate_child(self, parent, da, cdf):
        """Generate one t-node, given its parent and the CDF for the possible children."""
        if self.ranker:
            formeme, t_lemma, right = self.ranker.get_best_child(parent, da, cdf)
        else:
            formeme, t_lemma, right = self.candgen.sample(cdf)
        child = parent.create_child(right, NodeData(t_lemma, formeme))
        return child


class ASearchPlanner(SentencePlanner):
    """Sentence planner using A*-search."""

    MAX_ITER = 10000

    def __init__(self, cfg):
        super(ASearchPlanner, self).__init__(cfg)
        self.ranker = cfg['ranker']
        self.max_iter = cfg.get('max_iter', self.MAX_ITER)
        self.max_defic_iter = cfg.get('max_defic_iter')
        self.open_list = None
        self.close_list = None
        self.defic_iter = 0
        self.input_da = None
        self.num_iter = -1
        self.cdfs = None
        self.node_limits = None
        self.beam_size = None

    def generate_tree(self, da, gen_doc=None):
        """Generate a tree for the given DA.
        @param da: The input DA
        @param gen_doc: Save the generated tree into this PyTreex document, if given
        @return: the generated tree
        """
        # generate and use only 1-best
        self.run(da)
        best_tree, best_score = self.close_list.peek()
        log_debug("RESULT: %12.5f %s" % (best_score, unicode(best_tree)))
        # if requested, append the result
        if gen_doc:
            zone = self.get_target_zone(gen_doc)
            zone.ttree = best_tree.create_ttree()
            zone.sentence = unicode(da)
        # return the result
        return best_tree

    def reset(self):
        """Clear all data structures from the last run, most notably open and close lists."""
        self.open_list = None
        self.close_list = None
        self.defic_iter = 0
        self.num_iter = -1
        self.input_da = None
        self.cdfs = None
        self.node_limits = None

    def init_run(self, input_da, max_iter=None, max_defic_iter=None, beam_size=None):
        """Init the A*-search generation for the given input DA, with the given parameters
        (the parameters override any previously set parameters, such as those loaded from
        configuration upon class creation).

        @param input_da: The input DA for which a tree is to be generated
        @param max_iter: Maximum number of iteration (hard termination)
        @param max_defic_iter: Maximum number of deficit iteration (soft termination)
        @param beam_size: Beam size if pruning is to be used
        """
        log_debug('GEN TREE for DA: %s' % unicode(input_da))

        # initialization
        empty_tree = TreeData()
        et_score = self.ranker.score(empty_tree, input_da)
        et_futpr = self.ranker.get_future_promise(empty_tree)

        self.open_list = CandidateList({empty_tree: (-(et_score + et_futpr), -et_score, -et_futpr)})
        self.close_list = CandidateList()
        self.input_da = input_da
        self.defic_iter = 0
        self.num_iter = 0
        self.cdfs = self.candgen.get_merged_child_type_cdfs(input_da)
        self.node_limits = self.candgen.get_merged_limits(input_da)

        if max_iter is not None:
            self.max_iter = max_iter
        if max_defic_iter is not None:
            self.max_defic_iter = max_defic_iter
        if beam_size is not None:
            self.beam_size = beam_size

    def run(self, input_da, max_iter=None, max_defic_iter=None, beam_size=None):
        """Run the A*-search generation. The open and close lists of this object
        will contain the results (the best tree is the best one on the close list).

        @param input_da: the input dialogue act
        @param max_iter: maximum number of iterations for generation
        @param gold_ttree: a gold t-tree to check if it matches the current candidate
        @return: None
        """
        self.init_run(input_da, max_iter, max_defic_iter, beam_size)
        # main search loop
        while not self.check_finalize():
            self.run_iter()

    def run_iter(self):
        """Run one iteration of the A*-search generation algorithm. Move the best candidate
        from open list to close list and try to expand it in all ways possible, then put the results
        on the open list. Keep track of the number of iteration and deficit iteration so that
        the termination condition evaluates properly.
        """

        # log_debug("   OPEN : %s" % str(open_list))
        # log_debug("   CLOSE: %s" % str(close_list))
        cand, score = self.open_list.pop()
        self.close_list.push(cand, score[1])  # only use score without future promise
        log_debug("-- IT %4d: O %5d S %12.5f -- %s" %
                  (self.num_iter, len(self.open_list), -score[1], unicode(cand)))
        successors = [succ for succ
                      in self.candgen.get_all_successors(cand, self.cdfs, self.node_limits)
                      if succ not in self.close_list]

        if successors:
            # add candidates with score (negative for the min-heap)
            scores = self.ranker.score_all(successors, self.input_da)
            futprs = self.ranker.get_future_promise_all(successors)
            self.open_list.push_all([(succ, (-(score + futpr), -score, -futpr))
                                     for succ, score, futpr in zip(successors, scores, futprs)])
            # pruning (if supposed to do it)
            # TODO do not even add them on the open list when pruning
            if self.beam_size is not None:
                pruned = self.open_list.prune(self.beam_size)
                self.close_list.push_all(pruned)
        self.num_iter += 1

        # check where the score is higher -- on the open or on the close list
        # keep track of 'deficit' iterations (and do not allow more than the threshold)
        # TODO decide how to check this: should we check the combined score against the close list?
        if self.open_list and self.close_list:
            open_best_score, close_best_score = self.open_list.peek()[1][1], self.close_list.peek()[1]
            if open_best_score <= close_best_score:  # scores are negative, less is better
                self.defic_iter = 0
            else:
                self.defic_iter += 1

        if self.num_iter == self.max_iter:
            log_debug('ITERATION LIMIT REACHED')
        elif self.defic_iter == self.max_defic_iter:
            log_debug('DEFICIT ITERATION LIMIT REACHED')

    def check_finalize(self):
        """Check if the termination criterion is met. If it is met, push everything from
        open list to close list, getting rid of future cost, and return True. Otherwise
        just return false.
        """

        # check termination criterion, return false if it is not met
        if (self.open_list and self.num_iter < self.max_iter and
                (self.max_defic_iter is None or self.defic_iter <= self.max_defic_iter)):
            return False

        # if we should terminate,
        # push everything from open list to close list, getting rid of future cost
        while self.open_list:
            cand, score = self.open_list.pop()
            self.close_list.push(cand, score[1])

        return True
