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

rerank_cl_train -- train the reranking classifier (part of seq2seq generator, accessible
        externally here for debugging purposes)
    - arguments:  config train-das train-trees rerank-cl-model

rerank_cl_eval -- evaluate the reranking classifier (part of seq2seq generator, accessible
        externally here for debugging purposes)
    - arguments: [-l language] [-s selector] rerank-cl-model test-das test-sents
"""

from __future__ import unicode_literals
import sys
from getopt import getopt
import platform
import os
from argparse import ArgumentParser

from tgen.config import Config
from tgen.logf import log_info, set_debug_stream, log_debug, log_warn
from tgen.futil import file_stream, read_das, read_ttrees, chunk_list, add_bundle_text, \
    trees_from_doc, ttrees_from_doc, write_ttrees, tokens_from_doc, read_tokens, write_tokens, \
    postprocess_tokens, create_ttree_doc
from tgen.candgen import RandomCandidateGenerator
from tgen.rank import PerceptronRanker
from tgen.planner import ASearchPlanner, SamplingPlanner
from tgen.eval import p_r_f1_from_counts, corr_pred_gold, f1_from_counts, ASearchListsAnalyzer, \
    EvalTypes, Evaluator
from tgen.tree import TreeData
from tgen.parallel_percrank_train import ParallelRanker
from tgen.debug import exc_info_hook
from tgen.rnd import rnd
from tgen.bleu import BLEUMeasure
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

    ap = ArgumentParser(prog=' '.join(sys.argv[0:2]))
    ap.add_argument('-a', '--add-to-seq2seq', type=str,
                    help='Replace trained classifier in an existing seq2seq model (path to file)')
    ap.add_argument('fname_config', type=str, help='Reranking classifier configuration file path')
    ap.add_argument('fname_da_train', type=str, help='Training DAs file path')
    ap.add_argument('fname_trees_train', type=str, help='Training trees file path (must be trees!)')
    ap.add_argument('fname_cl_model', type=str, help='Path for the output trained model')
    args = ap.parse_args(args)

    if args.add_to_seq2seq:
        tgen = Seq2SeqBase.load_from_file(args.add_to_seq2seq)

    config = Config(args.fname_config)
    rerank_cl = RerankingClassifier(config)
    rerank_cl.train(args.fname_da_train, args.fname_trees_train)

    if args.add_to_seq2seq:
        tgen.classif_filter = rerank_cl
        tgen.save_to_file(args.fname_cl_model)
    else:
        rerank_cl.save_to_file(args.fname_cl_model)


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

    ap = ArgumentParser(prog=' '.join(sys.argv[0:2]))

    ap.add_argument('-s', '--train-size', type=float,
                    help='Portion of the training data to use (default: 1.0)', default=1.0)
    ap.add_argument('-d', '--debug-logfile', type=str, help='Debug output file name')
    ap.add_argument('-j', '--jobs', type=int, help='Number of parallel jobs to use')
    ap.add_argument('-w', '--work-dir', type=str, help='Main working directory for parallel jobs')
    ap.add_argument('-e', '--experiment-id', type=str,
                    help='Experiment ID for parallel jobs (used as job name prefix)')
    ap.add_argument('-r', '--random-seed', type=str,
                    help='Initial random seed (used as string).')
    ap.add_argument('-c', '--context-file', type=str,
                    help='Input ttree/text file with context utterances')
    ap.add_argument('-v', '--valid-data', type=str,
                    help='Validation data paths (2-3 comma-separated files: DAs, trees/sentences, contexts)')
    ap.add_argument('-l', '--lexic-data', type=str,
                    help='Lexicalization data paths (1-2 comma-separated files: surface forms,' +
                    'training lexic. instructions)')
    ap.add_argument('-t', '--tb-summary-dir', '--tensorboard-summary-dir', '--tensorboard', type=str,
                    help='Directory where Tensorboard summaries are saved during training')

    ap.add_argument('seq2seq_config_file', type=str, help='Seq2Seq generator configuration file')
    ap.add_argument('da_train_file', type=str, help='Input training DAs')
    ap.add_argument('tree_train_file', type=str, help='Input training trees/sentences')
    ap.add_argument('seq2seq_model_file', type=str,
                    help='File name where to save the trained Seq2Seq generator model')

    args = ap.parse_args(args)

    if args.debug_logfile:
        set_debug_stream(file_stream(args.debug_logfile, mode='w'))
    if args.random_seed:
        rnd.seed(args.random_seed)

    log_info('Training sequence-to-sequence generator...')

    config = Config(args.seq2seq_config_file)

    if args.tb_summary_dir:  # override Tensorboard setting
        config['tb_summary_dir'] = args.tb_summary_dir
    if args.jobs:  # parallelize when training
        config['jobs_number'] = args.jobs
        if not args.work_dir:
            work_dir, _ = os.path.split(args.seq2seq_config_file)
        generator = ParallelSeq2SeqTraining(config, args.work_dir or work_dir, args.experiment_id)
    else:  # just a single training instance
        generator = Seq2SeqGen(config)

    generator.train(args.da_train_file, args.tree_train_file,
                    data_portion=args.train_size, context_file=args.context_file,
                    validation_files=args.valid_data, lexic_files=args.lexic_data)

    sys.setrecursionlimit(100000)
    generator.save_to_file(args.seq2seq_model_file)


def sample_gen(args):
    from pytreex.core.document import Document
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
    from pytreex.core.document import Document

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
        log_info("Tree size stats:\n * GOLD %s\n * PRED %s\n * DIFF %s" % evaler.size_stats())
        log_info("Score stats:\n * GOLD %s\n * PRED %s\n * DIFF %s" % evaler.score_stats())
        log_info("Common subtree stats:\n -- SIZE: %s\n -- ΔGLD: %s\n -- ΔPRD: %s" %
                 evaler.common_substruct_stats())
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

    ap = ArgumentParser(prog=' '.join(sys.argv[0:2]))

    ap.add_argument('-e', '--eval-file', type=str, help='A ttree/text file for evaluation')
    ap.add_argument('-a', '--abstr-file', type=str,
                    help='Lexicalization file (a.k.a. abstraction instructions, for postprocessing)')
    ap.add_argument('-r', '--ref-selector', type=str, default='',
                    help='Selector for reference trees in the evaluation file')
    ap.add_argument('-t', '--target-selector', type=str, default='',
                    help='Target selector for generated trees in the output file')
    ap.add_argument('-d', '--debug-logfile', type=str, help='Debug output file name')
    ap.add_argument('-w', '--output-file', type=str, help='Output tree/text file')
    ap.add_argument('-b', '--beam-size', type=int,
                    help='Override beam size for beam search decoding')
    ap.add_argument('-c', '--context-file', type=str,
                    help='Input ttree/text file with context utterances')

    ap.add_argument('seq2seq_model_file', type=str, help='Trained Seq2Seq generator model')
    ap.add_argument('da_test_file', type=str, help='Input DAs for generation')

    args = ap.parse_args(args)

    if args.debug_logfile:
        set_debug_stream(file_stream(args.debug_logfile, mode='w'))

    # load the generator
    tgen = Seq2SeqBase.load_from_file(args.seq2seq_model_file)
    if args.beam_size is not None:
        tgen.beam_size = args.beam_size

    # read input files (DAs, contexts)
    das = read_das(args.da_test_file)
    if args.context_file:
        if not tgen.use_context and not tgen.context_bleu_weight:
            log_warn('Generator is not trained to use context, ignoring context input file.')
        else:
            if args.context_file.endswith('.txt'):
                contexts = read_tokens(args.context_file)
            else:
                contexts = tokens_from_doc(read_ttrees(args.context_file),
                                           tgen.language, tgen.selector)
            das = [(context, da) for context, da in zip(contexts, das)]
    elif tgen.use_context or tgen.context_bleu_weight:
        log_warn('Generator is trained to use context. ' +
                 'Using empty contexts, expect lower performance.')
        das = [([], da) for da in das]

    # generate
    log_info('Generating...')
    gen_trees = []
    for num, da in enumerate(das, start=1):
        log_debug("\n\nTREE No. %03d" % num)
        gen_trees.append(tgen.generate_tree(da))
        if num % 100 == 0:
            log_info("Generated tree %d" % num)
    log_info(tgen.get_slot_err_stats())

    # evaluate the generated trees against golden trees (delexicalized)
    eval_doc = None
    if args.eval_file and not args.eval_file.endswith('.txt'):
        eval_doc = read_ttrees(args.eval_file)
        evaler = Evaluator()
        evaler.process_eval_doc(eval_doc, gen_trees, tgen.language, args.ref_selector,
                                args.target_selector or tgen.selector)

    # lexicalize, if required
    if args.abstr_file and tgen.lexicalizer:
        log_info('Lexicalizing...')
        tgen.lexicalize(gen_trees, args.abstr_file)

    # we won't need contexts anymore, but we do need DAs
    if tgen.use_context or tgen.context_bleu_weight:
        das = [da for _, da in das]

    # evaluate the generated & lexicalized tokens (F1 and BLEU scores)
    if args.eval_file and args.eval_file.endswith('.txt'):
        eval_tokens(das, read_tokens(args.eval_file, ref_mode=True),
                    [t.to_tok_list() for t in gen_trees])

    # write output .yaml.gz or .txt
    if args.output_file is not None:
        log_info('Writing output...')
        if args.output_file.endswith('.txt'):
            gen_toks = [t.to_tok_list() for t in gen_trees]
            postprocess_tokens(gen_toks, das)
            write_tokens(gen_toks, args.output_file)
        else:
            write_ttrees(create_ttree_doc(gen_trees, eval_doc, tgen.language,
                                          args.target_selector or tgen.selector),
                         args.output_file)


def eval_tokens(das, eval_tokens, gen_tokens):
    """Evaluate generated tokens and print out statistics."""
    postprocess_tokens(eval_tokens, das)
    postprocess_tokens(gen_tokens, das)

    evaluator = BLEUMeasure()
    for pred_sent, gold_sents in zip(gen_tokens, eval_tokens):
        evaluator.append(pred_sent, gold_sents)
    log_info("BLEU score: %.4f" % (evaluator.bleu() * 100))

    evaluator = Evaluator()
    for pred_sent, gold_sents in zip(gen_tokens, eval_tokens):
        for gold_sent in gold_sents:  # effectively an average over all gold paraphrases
            evaluator.append(gold_sent, pred_sent)

    log_info("TOKEN precision: %.4f, Recall: %.4f, F1: %.4f" % evaluator.p_r_f1(EvalTypes.TOKEN))
    log_info("Sentence length stats:\n * GOLD %s\n * PRED %s\n * DIFF %s" % evaluator.size_stats())
    log_info("Common subphrase stats:\n -- SIZE: %s\n -- ΔGLD: %s\n -- ΔPRD: %s" %
             evaluator.common_substruct_stats())


def rerank_cl_eval(args):
    ap = ArgumentParser(prog=' '.join(sys.argv[0:2]))
    ap.add_argument('-l', '--language', type=str,
                    help='Override classifier language (for t-tree input files)')
    ap.add_argument('-s', '--selector', type=str,
                    help='Override classifier selector (for t-tree input files)')
    ap.add_argument('fname_cl_model', type=str, help='Path to trained reranking classifier model')
    ap.add_argument('fname_test_da', type=str, help='Path to test DA file')
    ap.add_argument('fname_test_sent', type=str, help='Path to test trees file (must be trees!)')
    args = ap.parse_args(args)

    log_info("Loading reranking classifier...")
    rerank_cl = RerankingClassifier.load_from_file(args.fname_cl_model)
    if args.language is not None:
        rerank_cl.language = args.language
    if args.selector is not None:
        rerank_cl.selector = args.selector

    log_info("Evaluating...")
    tot_len, dist = rerank_cl.evaluate_file(args.fname_test_da, args.fname_test_sent)
    log_info("Penalty: %d, Total DAIs %d." % (dist, tot_len))


if __name__ == '__main__':

    if len(sys.argv) < 2:
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
