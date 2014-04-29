#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Perceptron candidate tree ranker.
"""
from __future__ import unicode_literals
from features import Features


class PerceptronRanker(object):

    def __init__(self, cfg):
        if cfg and 'features' in cfg:
            self.features = Features(cfg['features'])

    def rank(self, cand_feats, succ, new_node):
        pass

    def train(self, das_file, ttrees_file):
        pass
