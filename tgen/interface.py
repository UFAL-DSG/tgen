#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals


class CandidateGenerator(object):
    pass


class Ranker(object):

    def get_best_child(self, parent, cdf):
        raise NotImplementedError


