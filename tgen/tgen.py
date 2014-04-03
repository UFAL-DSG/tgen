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

from futil import read_das, read_ttrees, chunk_list
from randgen import RandomGenerator
from logreg_rank import LogisticRegressionRanker
from flect.config import Config
from getopt import getopt
from eval import tp_fp_fn, f1_from_counts, p_r_f1_from_counts


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

    def generate_tree(self, da, gen_doc=None):
        """Generate one tree given DA.

        If gen_doc is None, will create a new Treex/TectoTpl document. If gen_doc is set, will
        add to the end of the given document.
        """
        # create a document
        if gen_doc is None:
            gen_doc = Document()
        bundle = gen_doc.create_bundle()
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

        opts, files = getopt(args, 'h:')
        header_file = None
        for opt, arg in opts:
            if opt == '-h':
                header_file = arg
        if len(files) != 5:
            sys.exit(__doc__)
        fname_da_train, fname_ttrees_train, fname_cand_model, fname_rank_config, fname_rank_train = files

        log_info('Creating data for ranker...')
        candgen = RandomGenerator()
        candgen.load_model(fname_cand_model)
        rank_config = Config(fname_rank_config)
        ranker = LogisticRegressionRanker(rank_config)
        ranker.create_training_data(fname_ttrees_train, fname_da_train, candgen, fname_rank_train,
                                    header_file=header_file)

    elif action == 'rank_train':
        if len(args) != 3:
            sys.exit(__doc__)

        fname_rank_config, fname_rank_train, fname_rank_model = args
        log_info('Training ranker...')

        rank_config = Config(fname_rank_config)
        ranker = LogisticRegressionRanker(rank_config)
        ranker.train(fname_rank_train)
        ranker.save_to_file(fname_rank_model)

    elif action == 'generate':

        opts, files = getopt(args, 'r:n:o:')
        num_to_generate = 1
        ranker_model = None
        oracle_eval_file = None
        fname_ttrees_out = None

        for opt, arg in opts:
            if opt == '-r':
                ranker_model = arg
            elif opt == '-n':
                num_to_generate = int(arg)
            elif opt == '-o':
                oracle_eval_file = arg
            elif opt == '-w':
                fname_ttrees_out = arg

        if len(files) != 2:
            sys.exit(__doc__)
        fname_cand_model, fname_da_test = files

        # load model
        log_info('Initializing...')
        candgen = RandomGenerator()
        candgen.load_model(fname_cand_model)

        if ranker_model is not None:
            ranker = LogisticRegressionRanker.load_from_file(ranker_model)
        else:
            ranker = candgen

        tgen = TTreeGenerator({'candgen': candgen, 'ranker': ranker})
        # generate
        log_info('Generating...')
        gen_doc = None
        das = read_das(fname_da_test)
        for da in das:
            for _ in xrange(num_to_generate):  # repeat generation n times
                gen_doc = tgen.generate_tree(da, gen_doc)

        # evaluate if needed
        if oracle_eval_file is not None:
            log_info('Evaluating oracle F1...')
            log_info('Loading gold data from ' + oracle_eval_file)
            gold_trees = read_ttrees(oracle_eval_file)
            log_info('Gold data loaded.')
            correct, predicted, gold = 0, 0, 0
            for gold_tree, gen_trees in zip(gold_trees.bundles, chunk_list(gen_doc.bundles, num_to_generate)):
                # find best of predicted trees (in terms of F1)
                _, tc, tp, tg = max([(f1_from_counts(c, p, g), c, p, g) for c, p, g
                                     in map(lambda gen_tree: tp_fp_fn(gold_tree.get_zone(tgen.language, tgen.selector).ttree,
                                                                      gen_tree.get_zone(tgen.language, tgen.selector).ttree),
                                            gen_trees)],
                                    key=lambda x: x[0])
                correct += tc
                predicted += tp
                gold += tg
            # evaluate oracle F1
            log_info("Oracle Precision: %.6f, Recall: %.6f, F1: %.6f" % p_r_f1_from_counts(correct, gold, predicted))
        # write output
        if fname_ttrees_out is not None:
            log_info('Writing output...')
            writer = YAMLWriter(scenario=None, args={'to': fname_ttrees_out})
            writer.process_document(gen_doc)

    log_info('Done.')
