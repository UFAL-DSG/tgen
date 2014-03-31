#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generating T-trees from dialogue acts.

Usage: ./tgen train_generator-das train_generator-t-trees test-das output-t-trees
"""

from __future__ import unicode_literals
import random
from collections import deque
import sys

from alex.components.nlg.tectotpl.core.document import Document
from alex.components.nlg.tectotpl.block.write.yaml import YAML as YAMLWriter
from flect.logf import log_info

from futil import read_das
from randgen import RandomGenerator
from logreg_rank import LogisticRegressionRanker
from flect.config import Config


class TTreeGenerator(object):
    """Random t-tree generator given DAs.

    Trainable from DA distributions
    """

    MAX_TREE_SIZE = 50

    def __init__(self, cfg):
        """Initialize the generator (just language and selector, distributions are empty)"""
        self.language = cfg.get('language', 'en')
        self.selector = cfg.get('selector', '')
        # candidate generator
        self.candgen = cfg['candgen']
        # ranker (selecting the best candidate)
        self.ranker = cfg['ranker']

    def generate_tree(self, da, doc=None):
        """Generate one tree given DA.

        If doc is None, will create a new Treex/TectoTpl document. If doc is set, will
        add to the end of the given document.
        """
        # create a document
        if doc is None:
            doc = Document()
        bundle = doc.create_bundle()
        zone = bundle.create_zone(self.language, self.selector)
        # creating a tree
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
        return doc

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


if __name__ == '__main__':

    random.seed(1206)

    if len(sys.argv) < 3:
        sys.exit(__doc__)

    action = sys.argv[1]
    args = sys.argv[2:]

    if action == 'candgen_train':
        if len(args) != 3:
            sys.exit(__doc__)
        fname_da_train, fname_ttrees_train, fname_cand_model = args

        log_info('Training candidate generator...')
        candgen = RandomGenerator()
        candgen.train(fname_da_train, fname_ttrees_train)
        candgen.save_model(fname_cand_model)

    elif action == 'rank_create_data':
        if len(args) != 5:
            sys.exit(__doc__)
        fname_da_train, fname_ttrees_train, fname_cand_model, fname_rank_config, fname_rank_train = args

        log_info('Creating training data for ranker...')
        candgen = RandomGenerator()
        candgen.load_model(fname_cand_model)
        rank_config = Config(fname_rank_config)
        ranker = LogisticRegressionRanker(rank_config)
        ranker.create_training_data(fname_ttrees_train, fname_da_train, candgen, fname_rank_train)

    elif action == 'rank_train':
        if len(args) != 3:
            sys.exit(__doc__)

        fname_rank_config, fname_rank_train, fname_rank_model = args
        log_info('Training ranker...')

        rank_config = Config(fname_rank_config)
        ranker = LogisticRegressionRanker(rank_config)
        ranker.train(fname_rank_train)
        ranker.save_model(fname_rank_model)

    elif action == 'generate':
        if len(args) != 3:
            sys.exit(__doc__)
        fname_cand_model, fname_da_test, fname_ttrees_out = args

        # load model
        log_info('Initializing...')
        candgen = RandomGenerator()
        candgen.load_model(fname_cand_model)
        tgen = TTreeGenerator({'candgen': candgen, 'ranker': candgen})
        # generate
        log_info('Generating...')
        doc = None
        das = read_das(fname_da_test)
        for da in das:
            doc = tgen.generate_tree(da, doc)
        # write output
        log_info('Writing output...')
        writer = YAMLWriter(scenario=None, args={'to': fname_ttrees_out})
        writer.process_document(doc)

    log_info('Done.')
