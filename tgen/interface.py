#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals


class CandidateGenerator(object):
    pass


class Ranker(object):

    def get_best_child(self, parent, cdf):
        raise NotImplementedError


class SentencePlanner(object):
    """Abstract interface for sentence planners."""

    def generate_tree(self, da, gen_doc=None):
        """Generate a tree given input DA.
        @param gen_doc: if this is None, return the tree in a new Document object, otherwise append
        """
        raise NotImplementedError
