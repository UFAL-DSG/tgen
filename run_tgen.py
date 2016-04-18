#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generating T-trees from dialogue acts.

Usage: ./tgen.py <action> <argument1 ...>

Actions:

candgen_train -- train candidate generator (probability distributions)
    - arguments: [-l] [-n] [-p prune_threshold] [-c <lemma|node>:limit] [-s] train-das train-ttrees output-model
                 * l = create lexicalized candgen (limit using parent lemmas as well as formemes)
                 * n = engage limits on number of nodes (on depth levels + total number)
                 * s = enforce compatibility for slots as well

percrank_train -- train perceptron global ranker
    - arguments: [-d debug-output] [-c candgen-model] [-s data-portion] [-j parallel-jobs] [-w parallel-work-dir] \\
                 [-r rand_seed] [-e experiment_id] ranker-config train-das train-ttrees output-model
                 * r = random seed is used as a string; no seed change if empty string is passed

sample_gen -- sampling generation (oracle experiment; rather obsolete)
    - arguments: [-n trees-per-da] [-o oracle-eval-ttrees] [-w output-ttrees] candgen-model test-das

asearch_gen -- generate using the A*search sentence planner
    - arguments: [-e eval-ttrees-file] [-s eval-ttrees-selector] [-d debug-output] [-w output-ttrees] \\
                 [-c config] candgen-model percrank-model test-das

treecl_train -- train a tree classifier (part of candidate generator, accessible externally here)
    - arguments: config train-das train-trees treecl-model-file

seq2seq_train -- train a seq2seq generator (via trees or strings)
    - arguments: [-d debug-output] [-s data-portion] [-r rand-seed] [-j parallel-models] [-w parallel-work-dir] \\
                 [-e experiment-id] config train-das train-trees seq2seq-model

seq2seq_gen -- evaluate the seq2seq generator
    - arguments: [-e eval-ttrees-file] [-r eval-ttrees-selector] [-t target-selector] [-d debug-output]
                 [-w output-ttrees] [-b beam-size-override] seq2seq-model test-das

rerank_cl_train -- train the reranking classifier (part of seq2seq generator, accessible externally here)
    - arguments:  config train-das train-trees rerank-cl-model

rerank_cl_eval -- evaluate the reranking classifier (part of seq2seq generator, accessible externally here)
    - arguments: [-l language] [-s selector] [-t] rerank-cl-model test-das test-sents
                 * -t = flatten trees to tokens
"""

from __future__ import unicode_literals
import sys
from getopt import getopt
import platform
import os

from pytreex.core.util import file_stream
from pytreex.core.document import Document

from flect.config import Config

from tgen.logf import log_info, set_debug_stream, log_debug
from tgen.futil import read_das, read_ttrees, chunk_list, add_bundle_text, \
    trees_from_doc, ttrees_from_doc, write_ttrees, tokens_from_doc
from tgen.candgen import RandomCandidateGenerator
from tgen.rank import PerceptronRanker
from tgen.planner import ASearchPlanner, SamplingPlanner
from tgen.eval import p_r_f1_from_counts, corr_pred_gold, f1_from_counts, ASearchListsAnalyzer, \
    EvalTypes, Evaluator
from tgen.tree import TreeData
from tgen.parallel_percrank_train import ParallelRanker
from tgen.debug import exc_info_hook
from tgen.rnd import rnd
from tgen.seq2seq import Seq2SeqBase, Seq2SeqGen
from tgen.parallel_seq2seq_train import ParallelSeq2SeqTraining
from tgen.tfclassif import RerankingClassifier

# Start IPdb on error in interactive mode
sys.excepthook = exc_info_hook


def candgen_train(args):
    opts, files = getopt(args, 'p:lnc:sd:t:')

    prune_threshold = 1
    parent_lemmas = False
    node_limits = False
    comp_type = None
    comp_limit = None
    comp_slots = False
    tree_classif = False

    for opt, arg in opts:
        if opt == '-p':
            prune_threshold = int(arg)
        elif opt == '-d':
            set_debug_stream(file_stream(arg, mode='w'))
        elif opt == '-l':
            parent_lemmas = True
        elif opt == '-n':
            node_limits = True
        elif opt == '-c':
            comp_type = arg
            if ':' in comp_type:
                comp_type, comp_limit = comp_type.split(':', 1)
                comp_limit = int(comp_limit)
        elif opt == '-t':
            tree_classif = Config(arg)
        elif opt == '-s':
            comp_slots = True

    if len(files) != 3:
        sys.exit("Invalid arguments.\n" + __doc__)
    fname_da_train, fname_ttrees_train, fname_cand_model = files

    log_info('Training candidate generator...')
    candgen = RandomCandidateGenerator({'prune_threshold': prune_threshold,
                                        'parent_lemmas': parent_lemmas,
                                        'node_limits': node_limits,
                                        'compatible_dais_type': comp_type,
                                        'compatible_dais_limit': comp_limit,
                                        'compatible_slots': comp_slots,
                                        'tree_classif': tree_classif})
    candgen.train(fname_da_train, fname_ttrees_train)
    candgen.save_to_file(fname_cand_model)


def rerank_cl_train(args):

    opts, files = getopt(args, 'a:')

    load_seq2seq_model = None
    for opt, arg in opts:
        if opt == '-a':
            load_seq2seq_model = arg

    if len(files) != 4:
        sys.exit("Invalid arguments.\n" + __doc__)
    fname_config, fname_da_train, fname_trees_train, fname_cl_model = files

    if load_seq2seq_model:
        tgen = Seq2SeqBase.load_from_file(load_seq2seq_model)

    config = Config(fname_config)
    rerank_cl = RerankingClassifier(config)
    rerank_cl.train(fname_da_train, fname_trees_train)

    if load_seq2seq_model:
        tgen.classif_filter = rerank_cl
        tgen.save_to_file(fname_cl_model)
    else:
        rerank_cl.save_to_file(fname_cl_model)


def treecl_train(args):
    from tgen.classif import TreeClassifier

    opts, files = getopt(args, '')

    if len(files) != 4:
        sys.exit("Invalid arguments.\n" + __doc__)
    fname_config, fname_da_train, fname_trees_train, fname_cl_model = files

    config = Config(fname_config)
    treecl = TreeClassifier(config)

    treecl.train(fname_da_train, fname_trees_train)
    treecl.save_to_file(fname_cl_model)


def percrank_train(args):
    opts, files = getopt(args, 'c:d:s:j:w:e:r:')
    candgen_model = None
    train_size = 1.0
    parallel = False
    jobs_number = 0
    work_dir = None
    experiment_id = None

    for opt, arg in opts:
        if opt == '-d':
            set_debug_stream(file_stream(arg, mode='w'))
        elif opt == '-s':
            train_size = float(arg)
        elif opt == '-c':
            candgen_model = arg
        elif opt == '-j':
            parallel = True
            jobs_number = int(arg)
        elif opt == '-w':
            work_dir = arg
        elif opt == '-e':
            experiment_id = arg
        elif opt == '-r' and arg:
            rnd.seed(arg)

    if len(files) != 4:
        sys.exit(__doc__)

    fname_rank_config, fname_train_das, fname_train_ttrees, fname_rank_model = files
    log_info('Training perceptron ranker...')

    rank_config = Config(fname_rank_config)
    if candgen_model:
        rank_config['candgen_model'] = candgen_model
    if rank_config.get('nn'):
        from tgen.rank_nn import SimpleNNRanker, EmbNNRanker
        if rank_config['nn'] in ['emb', 'emb_trees', 'emb_prev']:
            ranker_class = EmbNNRanker
        else:
            ranker_class = SimpleNNRanker
    else:
        ranker_class = PerceptronRanker

    log_info('Using %s for ranking' % ranker_class.__name__)

    if not parallel:
        ranker = ranker_class(rank_config)
    else:
        rank_config['jobs_number'] = jobs_number
        if work_dir is None:
            work_dir, _ = os.path.split(fname_rank_config)
        ranker = ParallelRanker(rank_config, work_dir, experiment_id, ranker_class)

    ranker.train(fname_train_das, fname_train_ttrees, data_portion=train_size)

    # avoid the "maximum recursion depth exceeded" error
    sys.setrecursionlimit(100000)
    ranker.save_to_file(fname_rank_model)


def seq2seq_train(args):

    train_size = 1.0
    parallel = False
    jobs_number = 0
    work_dir = None
    experiment_id = None
    fname_contexts = None

    opts, files = getopt(args, 'd:s:r:j:w:e:c:')

    for opt, arg in opts:
        if opt == '-d':
            set_debug_stream(file_stream(arg, mode='w'))
        if opt == '-s':
            train_size = float(arg)
        elif opt == '-j':
            parallel = True
            jobs_number = int(arg)
        elif opt == '-w':
            work_dir = arg
        elif opt == '-e':
            experiment_id = arg
        elif opt == '-r' and arg:
            rnd.seed(arg)
        elif opt == '-c':
            fname_contexts = arg

    if len(files) != 4:
        sys.exit(__doc__)

    fname_gen_config, fname_train_das, fname_train_ttrees, fname_gen_model = files
    log_info('Training sequence-to-sequence generator...')

    config = Config(fname_gen_config)
    if parallel:
        config['jobs_number'] = jobs_number
        if work_dir is None:
            work_dir, _ = os.path.split(fname_gen_config)
        generator = ParallelSeq2SeqTraining(config, work_dir, experiment_id)
    else:
        generator = Seq2SeqGen(config)

    generator.train(fname_train_das, fname_train_ttrees,
                    data_portion=train_size, context_file=fname_contexts)
    sys.setrecursionlimit(100000)
    generator.save_to_file(fname_gen_model)


def sample_gen(args):
    opts, files = getopt(args, 'r:n:o:w:')
    num_to_generate = 1
    oracle_eval_file = None
    fname_ttrees_out = None

    for opt, arg in opts:
        if opt == '-n':
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
    candgen = RandomCandidateGenerator.load_from_file(fname_cand_model)

    ranker = candgen

    tgen = SamplingPlanner({'candgen': candgen, 'ranker': ranker})
    # generate
    log_info('Generating...')
    gen_doc = Document()
    das = read_das(fname_da_test)
    for da in das:
        for _ in xrange(num_to_generate):  # repeat generation n times
            tgen.generate_tree(da, gen_doc)

    # evaluate if needed
    if oracle_eval_file is not None:
        log_info('Evaluating oracle F1...')
        log_info('Loading gold data from ' + oracle_eval_file)
        gold_trees = ttrees_from_doc(read_ttrees(oracle_eval_file), tgen.language, tgen.selector)
        gen_trees = ttrees_from_doc(gen_doc, tgen.language, tgen.selector)
        log_info('Gold data loaded.')
        correct, predicted, gold = 0, 0, 0
        for gold_tree, gen_trees in zip(gold_trees, chunk_list(gen_trees, num_to_generate)):
            # find best of predicted trees (in terms of F1)
            _, tc, tp, tg = max([(f1_from_counts(c, p, g), c, p, g) for c, p, g
                                 in map(lambda gen_tree: corr_pred_gold(gold_tree, gen_tree),
                                        gen_trees)],
                                key=lambda x: x[0])
            correct += tc
            predicted += tp
            gold += tg
        # evaluate oracle F1
        log_info("Oracle Precision: %.6f, Recall: %.6f, F1: %.6f" % p_r_f1_from_counts(correct, predicted, gold))
    # write output
    if fname_ttrees_out is not None:
        log_info('Writing output...')
        write_ttrees(gen_doc, fname_ttrees_out)


def asearch_gen(args):
    """A*search generation"""

    opts, files = getopt(args, 'e:d:w:c:s:')
    eval_file = None
    fname_ttrees_out = None
    cfg_file = None
    eval_selector = ''

    for opt, arg in opts:
        if opt == '-e':
            eval_file = arg
        elif opt == '-s':
            eval_selector = arg
        elif opt == '-d':
            set_debug_stream(file_stream(arg, mode='w'))
        elif opt == '-w':
            fname_ttrees_out = arg
        elif opt == '-c':
            cfg_file = arg

    if len(files) != 3:
        sys.exit('Invalid arguments.\n' + __doc__)
    fname_cand_model, fname_rank_model, fname_da_test = files

    log_info('Initializing...')
    candgen = RandomCandidateGenerator.load_from_file(fname_cand_model)
    ranker = PerceptronRanker.load_from_file(fname_rank_model)
    cfg = Config(cfg_file) if cfg_file else {}
    cfg.update({'candgen': candgen, 'ranker': ranker})
    tgen = ASearchPlanner(cfg)

    log_info('Generating...')
    das = read_das(fname_da_test)

    if eval_file is None:
        gen_doc = Document()
    else:
        eval_doc = read_ttrees(eval_file)
        if eval_selector == tgen.selector:
            gen_doc = Document()
        else:
            gen_doc = eval_doc

    # generate and evaluate
    if eval_file is not None:
        # generate + analyze open&close lists
        lists_analyzer = ASearchListsAnalyzer()
        for num, (da, gold_tree) in enumerate(zip(das,
                                                  trees_from_doc(eval_doc, tgen.language, eval_selector)),
                                              start=1):
            log_debug("\n\nTREE No. %03d" % num)
            gen_tree = tgen.generate_tree(da, gen_doc)
            lists_analyzer.append(gold_tree, tgen.open_list, tgen.close_list)
            if gen_tree != gold_tree:
                log_debug("\nDIFFING TREES:\n" + tgen.ranker.diffing_trees_with_scores(da, gold_tree, gen_tree) + "\n")

        log_info('Gold tree BEST: %.4f, on CLOSE: %.4f, on ANY list: %4f' % lists_analyzer.stats())

        # evaluate the generated trees against golden trees
        eval_ttrees = ttrees_from_doc(eval_doc, tgen.language, eval_selector)
        gen_ttrees = ttrees_from_doc(gen_doc, tgen.language, tgen.selector)

        log_info('Evaluating...')
        evaler = Evaluator()
        for eval_bundle, eval_ttree, gen_ttree, da in zip(eval_doc.bundles, eval_ttrees, gen_ttrees, das):
            # add some stats about the tree directly into the output file
            add_bundle_text(eval_bundle, tgen.language, tgen.selector + 'Xscore',
                            "P: %.4f R: %.4f F1: %.4f" % p_r_f1_from_counts(*corr_pred_gold(eval_ttree, gen_ttree)))

            # collect overall stats
            evaler.append(eval_ttree,
                          gen_ttree,
                          ranker.score(TreeData.from_ttree(eval_ttree), da),
                          ranker.score(TreeData.from_ttree(gen_ttree), da))
        # print overall stats
        log_info("NODE precision: %.4f, Recall: %.4f, F1: %.4f" % evaler.p_r_f1())
        log_info("DEP  precision: %.4f, Recall: %.4f, F1: %.4f" % evaler.p_r_f1(EvalTypes.DEP))
        log_info("Tree size stats:\n * GOLD %s\n * PRED %s\n * DIFF %s" % evaler.tree_size_stats())
        log_info("Score stats:\n * GOLD %s\n * PRED %s\n * DIFF %s" % evaler.score_stats())
        log_info("Common subtree stats:\n -- SIZE: %s\n -- ΔGLD: %s\n -- ΔPRD: %s" %
                 evaler.common_subtree_stats())
    # just generate
    else:
        for da in das:
            tgen.generate_tree(da, gen_doc)

    # write output
    if fname_ttrees_out is not None:
        log_info('Writing output...')
        write_ttrees(gen_doc, fname_ttrees_out)


def seq2seq_gen(args):
    """Sequence-to-sequence generation"""

    opts, files = getopt(args, 'e:d:w:r:t:b:c:')
    eval_file = None
    fname_ttrees_out = None
    ref_selector = ''
    target_selector = ''
    beam_size_override = None
    fname_contexts = None

    for opt, arg in opts:
        if opt == '-e':
            eval_file = arg
        elif opt == '-r':
            ref_selector = arg
        elif opt == '-t':
            target_selector = arg
        elif opt == '-d':
            set_debug_stream(file_stream(arg, mode='w'))
        elif opt == '-w':
            fname_ttrees_out = arg
        elif opt == '-b':
            beam_size_override = int(arg)
        elif opt == '-c':
            fname_contexts = arg

    if len(files) != 2:
        sys.exit('Invalid arguments.\n' + __doc__)
    fname_seq2seq_model, fname_da_test = files

    tgen = Seq2SeqBase.load_from_file(fname_seq2seq_model)
    if beam_size_override:
        tgen.beam_size = beam_size_override

    log_info('Generating...')
    das = read_das(fname_da_test)
    if fname_contexts:
        contexts = tokens_from_docs(read_ttrees(fname_contexts), tgen.language, tgen.selector)
        das = [(context, da) for context, da in zip(contexts, das)]

    if eval_file is None:
        gen_doc = Document()
    else:
        eval_doc = read_ttrees(eval_file)
        if ref_selector == target_selector:
            gen_doc = Document()
        else:
            gen_doc = eval_doc

    tgen.selector = target_selector

    # generate and evaluate
    if eval_file is not None:
        for num, da in enumerate(das, start=1):
            log_debug("\n\nTREE No. %03d" % num)
            tgen.generate_tree(da, gen_doc)

        # evaluate the generated trees against golden trees
        eval_ttrees = ttrees_from_doc(eval_doc, tgen.language, ref_selector)
        gen_ttrees = ttrees_from_doc(gen_doc, tgen.language, target_selector)

        log_info('Evaluating...')
        evaler = Evaluator()
        for eval_bundle, eval_ttree, gen_ttree, da in zip(eval_doc.bundles, eval_ttrees, gen_ttrees, das):
            # add some stats about the tree directly into the output file
            add_bundle_text(eval_bundle, tgen.language, tgen.selector + 'Xscore',
                            "P: %.4f R: %.4f F1: %.4f" % p_r_f1_from_counts(*corr_pred_gold(eval_ttree, gen_ttree)))

            # collect overall stats
            # TODO maybe add cost ??
            evaler.append(eval_ttree, gen_ttree)
        # print overall stats
        log_info("NODE precision: %.4f, Recall: %.4f, F1: %.4f" % evaler.p_r_f1())
        log_info("DEP  precision: %.4f, Recall: %.4f, F1: %.4f" % evaler.p_r_f1(EvalTypes.DEP))
        log_info("Tree size stats:\n * GOLD %s\n * PRED %s\n * DIFF %s" % evaler.tree_size_stats())
        log_info("Score stats:\n * GOLD %s\n * PRED %s\n * DIFF %s" % evaler.score_stats())
        log_info("Common subtree stats:\n -- SIZE: %s\n -- ΔGLD: %s\n -- ΔPRD: %s" %
                 evaler.common_subtree_stats())
    # just generate
    else:
        for da in das:
            tgen.generate_tree(da, gen_doc)

    # write output
    if fname_ttrees_out is not None:
        log_info('Writing output...')
        write_ttrees(gen_doc, fname_ttrees_out)


def rerank_cl_eval(args):

    opts, files = getopt(args, 's:l:t')

    language = None
    selector = None
    for opt, arg in opts:
        if opt == '-l':
            language = arg
        elif opt == '-s':
            selector = arg

    if len(files) != 3:
        sys.exit("Invalid arguments.\n" + __doc__)
    fname_cl_model, fname_test_da, fname_test_sent = files

    log_info("Loading reranking classifier...")
    rerank_cl = RerankingClassifier.load_from_file(fname_cl_model)
    if language is not None:
        rerank_cl.language = language
    if selector is not None:
        rerank_cl.selector = selector

    log_info("Evaluating...")
    tot_len, dist = rerank_cl.evaluate_file(fname_test_da, fname_test_sent)
    log_info("Penalty: %d, Total DAIs %d." % (dist, tot_len))


if __name__ == '__main__':

    if len(sys.argv) < 3:
        sys.exit(__doc__)

    action = sys.argv[1]
    args = sys.argv[2:]

    log_info('Running on %s version %s' % (platform.python_implementation(),
                                           platform.python_version()))

    if action == 'candgen_train':
        candgen_train(args)
    elif action == 'percrank_train':
        percrank_train(args)
    elif action == 'sample_gen':
        sample_gen(args)
    elif action == 'asearch_gen':
        asearch_gen(args)
    elif action == 'seq2seq_train':
        seq2seq_train(args)
    elif action == 'seq2seq_gen':
        seq2seq_gen(args)
    elif action == 'treecl_train':
        treecl_train(args)
    elif action == 'rerank_cl_train':
        rerank_cl_train(args)
    elif action == 'rerank_cl_eval':
        rerank_cl_eval(args)
    else:
        # Unknown action
        sys.exit(("\nERROR: Unknown Tgen action: %s\n\n---" % action) + __doc__)

    log_info('Done.')
