#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import re
import numpy as np
import tensorflow as tf
import cPickle as pickle
from itertools import izip_longest, groupby
import sys
import math
import tempfile
import shutil
import os
from functools import partial

from pytreex.core.util import file_stream

from tgen.logf import log_info, log_debug, log_warn
from tgen.futil import read_das, read_ttrees, trees_from_doc, tokens_from_doc, chunk_list, \
    read_tokens, tagged_lemmas_from_doc
from tgen.embeddings import DAEmbeddingSeq2SeqExtract, TokenEmbeddingSeq2SeqExtract, \
    TreeEmbeddingSeq2SeqExtract, ContextDAEmbeddingSeq2SeqExtract, \
    TaggedLemmasEmbeddingSeq2SeqExtract
from tgen.rnd import rnd
from tgen.planner import SentencePlanner
from tgen.tree import TreeData, TreeNode
from tgen.eval import Evaluator, SlotErrAnalyzer
from tgen.bleu import BLEUMeasure
from tgen.tfclassif import RerankingClassifier
from tgen.tf_ml import TFModel, embedding_attention_seq2seq_context
from tgen.ml import softmax
from tgen.lexicalize import Lexicalizer
import tgen.externals.seq2seq as tf06s2s


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks, from Python Itertools recipes."
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def cut_batch_into_steps(batch):
    """Take a batch (list of examples, which are lists of steps/words themselves), and slice
    it along the other dimension – return a list of steps/words, each containing a numpy array of
    items for the given step for all examples from the batch.
    """
    return np.squeeze(np.array(np.split(np.array([ex for ex in batch if ex is not None]),
                                        len(batch[0]), axis=1)), axis=2)


class Seq2SeqBase(SentencePlanner):
    """A common ancestor for the Plain and Ensemble Seq2Seq generators (decoding methods only)."""

    def __init__(self, cfg):
        super(Seq2SeqBase, self).__init__(cfg)
        # save the whole configuration for later use (save/load, construction of embedding
        # extractors)
        self.cfg = cfg

        # decoding options
        self.beam_size = cfg.get('beam_size', 1)
        self.sample_top_k = cfg.get('sample_top_k', 1)
        self.length_norm_weight = cfg.get('length_norm_weight', 0.0)
        self.context_bleu_weight = cfg.get('context_bleu_weight', 0.0)
        self.context_bleu_metric = cfg.get('context_bleu_metric', 'bleu')
        self.slot_err_stats = None

        self.classif_filter = None
        if 'classif_filter' in cfg:
            # use the specialized settings for the reranking classifier
            rerank_cfg = cfg['classif_filter']
            # plus, copy some settings from the main Seq2Seq module (so we're consistent)
            for setting in ['mode', 'use_tokens', 'embeddings_lowercase',
                            'embeddings_split_plurals', 'tb_summary_dir']:
                if setting in cfg:
                    rerank_cfg[setting] = cfg[setting]
            self.classif_filter = RerankingClassifier(rerank_cfg)
            self.misfit_penalty = cfg.get('misfit_penalty', 100)

        self.lexicalizer = None
        if 'lexicalizer' in cfg:
            # build a lexicalizer with the given settings
            lexer_cfg = cfg['lexicalizer']
            for setting in ['mode', 'language']:
                if setting in cfg:
                    lexer_cfg[setting] = cfg[setting]
            self.lexicalizer = Lexicalizer(lexer_cfg)

        self.init_slot_err_stats()

    def process_das(self, das, gold_trees=None):
        """
        Process a list of input DAs, return the corresponding trees (using the generator
        network with current parameters).

        @param das: input DAs
        @param gold_trees: (optional) gold trees against which cost is computed
        @return: generated trees as `TreeData` instances, cost if `gold_trees` are given
        """
        # encoder inputs
        enc_inputs = cut_batch_into_steps([self.da_embs.get_embeddings(da)
                                           for da in das])

        if self.beam_size > 1 and len(das) == 1:
            dec_output_ids = self._beam_search(enc_inputs, das[0])
            dec_cost = None
        else:
            dec_output_ids, dec_cost = self._greedy_decoding(enc_inputs, gold_trees)

        dec_trees = [self.tree_embs.ids_to_tree(ids) for ids in dec_output_ids.transpose()]

        # return result (trees and optionally cost)
        if dec_cost is None:
            return dec_trees
        return dec_trees, dec_cost

    def _greedy_decoding(self, enc_inputs, gold_trees):
        """Run greedy decoding with the given encoder inputs; optionally use given gold trees
        as decoder inputs for cost computation."""

        # prepare decoder inputs (either fake, or true but used just for cost computation)
        if gold_trees is None:
            empty_tree_emb = self.tree_embs.get_embeddings(TreeData())
            dec_inputs = cut_batch_into_steps([empty_tree_emb for _ in enc_inputs[0]])
        else:
            dec_inputs = cut_batch_into_steps([self.tree_embs.get_embeddings(tree)
                                               for tree in gold_trees])

        # run the decoding per se
        dec_output_ids, dec_cost = self._get_greedy_decoder_output(
            enc_inputs, dec_inputs, compute_cost=gold_trees is not None)

        return dec_output_ids, dec_cost

    def _get_greedy_decoder_output(initial_state, enc_inputs, dec_inputs, compute_cost=False):
        raise NotImplementedError()

    class DecodingPath(object):
        """A decoding path to be used in beam search."""

        __slots__ = ['stop_token_id', 'dec_inputs', 'dec_states', 'logprob', '_length']

        def __init__(self, stop_token_id, dec_inputs=[], dec_states=[], logprob=0.0, length=-1):
            self.stop_token_id = stop_token_id
            self.dec_inputs = list(dec_inputs)
            self.dec_states = list(dec_states)
            self.logprob = logprob
            self._length = length if length >= 0 else len(dec_inputs)

        def expand(self, max_variants, dec_out_probs, dec_state):
            """Expand the path with all possible outputs, updating the log probabilities.

            @param max_variants: expand to this number of variants at maximum, discard the less \
                probable ones
            @param dec_output: the decoder output scores for the current step
            @param dec_state: the decoder hidden state for the current step
            @return: an array of all possible continuations of this path
            """
            ret = []

            # select only up to max_variants most probable variants
            top_n_idx = np.argpartition(-dec_out_probs, max_variants)[:max_variants]

            for idx in top_n_idx:
                expanded = Seq2SeqGen.DecodingPath(self.stop_token_id,
                                                   self.dec_inputs, self.dec_states, self.logprob,
                                                   len(self))
                if len(self) == len(self.dec_inputs) and idx != self.stop_token_id:
                    expanded._length += 1
                expanded.logprob += np.log(dec_out_probs[idx])
                expanded.dec_inputs.append(np.array(idx, ndmin=1))
                expanded.dec_states.append(dec_state)
                ret.append(expanded)

            return ret

        def __len__(self):
            """Return decoding path length (number of decoder input tokens)."""
            return self._length

    def _beam_search(self, enc_inputs, da):
        """Run beam search decoding."""

        # true "batches" not implemented
        assert len(enc_inputs[0]) == 1

        # run greedy decoder for comparison (debugging purposes)
        log_debug("GREEDY DEC WOULD RETURN:\n" +
                  " ".join(self.tree_embs.ids_to_strings(
                      [out_tok[0] for out_tok in self._greedy_decoding(enc_inputs, None)[0]])))

        # initialize
        self._init_beam_search(enc_inputs)
        empty_tree_emb = self.tree_embs.get_embeddings(TreeData())
        dec_inputs = cut_batch_into_steps([empty_tree_emb])

        paths = [self.DecodingPath(stop_token_id=self.tree_embs.STOP, dec_inputs=[dec_inputs[0]])]

        # beam search steps
        for step in xrange(len(dec_inputs)):

            new_paths = []

            for path in paths:
                out_probs, st = self._beam_search_step(path.dec_inputs, path.dec_states)
                new_paths.extend(path.expand(self.beam_size, out_probs, st))

            def cmp_func(p, q):
                """Length-weighted comparison of two paths' logprobs."""
                return cmp(p.logprob / (len(p) ** self.length_norm_weight),
                           q.logprob / (len(q) ** self.length_norm_weight))

            paths = sorted(new_paths, cmp=cmp_func, reverse=True)[:self.beam_size]

            if all([p.dec_inputs[-1] == self.tree_embs.VOID for p in paths]):
                break  # stop decoding if we have reached the end in all paths

            log_debug(("\nBEAM SEARCH STEP %d\n" % step) +
                      "\n".join([("%f\t" % p.logprob) +
                                 " ".join(self.tree_embs.ids_to_strings([inp[0] for inp in p.dec_inputs]))
                                 for p in paths]) + "\n")

        # rerank paths by their distance to the input DA
        if self.classif_filter or self.context_bleu_weight:
            paths = self._rerank_paths(paths, da)

        # measure slot error on the top k paths
        if self.slot_err_stats:
            for path in paths[:self.sample_top_k]:
                self.slot_err_stats.append(
                    da, self.tree_embs.ids_to_strings([inp[0] for inp in path.dec_inputs]))

        # select the "best" path -- either the best, or one in top k
        if self.sample_top_k > 1:
            best_path = self._sample_path(paths[:self.sample_top_k])
        else:
            best_path = paths[0]

        # return just the best path (as token IDs)
        return np.array(best_path.dec_inputs)

    def _init_beam_search(self, enc_inputs):
        raise NotImplementedError()

    def _beam_search_step(self, dec_inputs, dec_states):
        raise NotImplementedError()

    def _rerank_paths(self, paths, da):
        """Rerank the n-best decoded paths according to the reranking classifier and/or
        BLEU against context."""

        trees = [self.tree_embs.ids_to_tree(np.array(path.dec_inputs).transpose()[0])
                 for path in paths]

        # rerank using BLEU against context if set to do so
        if self.context_bleu_weight:
            bm = BLEUMeasure(max_ngram=2)
            bleus = []
            for path, tree in zip(paths, trees):
                bm.reset()
                bm.append([(n.t_lemma, None) for n in tree.nodes[1:]], [da[0]])
                bleu = (bm.ngram_precision()
                        if self.context_bleu_metric == 'ngram_prec'
                        else bm.bleu())
                bleus.append(bleu)
                path.logprob += self.context_bleu_weight * bleu

            log_debug(("BLEU for context: %s\n\n" % " ".join([form for form, _ in da[0]])) +
                      "\n".join([("%.5f\t" % b) + " ".join([n.t_lemma for n in t.nodes[1:]])
                                 for b, t in zip(bleus, trees)]))

        # add distances to logprob so that non-fitting will be heavily penalized
        if self.classif_filter:
            self.classif_filter.init_run(da)
            fits = self.classif_filter.dist_to_cur_da(trees)
            for path, fit in zip(paths, fits):
                path.logprob -= self.misfit_penalty * fit

            log_debug(("Misfits for DA: %s\n\n" % str(da)) +
                      "\n".join([("%.5f\t" % fit) +
                                 " ".join([unicode(n.t_lemma) for n in tree.nodes[1:]])
                                 for fit, tree in zip(fits, trees)]))

        # adjust paths for length (if set to do so)
        if self.length_norm_weight:
            for path in paths:
                path.logprob /= len(path) ** self.length_norm_weight

        return sorted(paths, cmp=lambda p, q: cmp(p.logprob, q.logprob), reverse=True)

    def _sample_path(self, paths):
        """Sample one path from the top k paths, based on their probabilities."""

        # convert the logprobs to a probability distribution, proportionate to their sizes
        logprobs = [p.logprob for p in paths]
        max_logprob = max(logprobs)
        probs = [math.exp(l - max_logprob) for l in logprobs]  # discount to avoid underflow, result is unnormalized
        sum_prob = sum(probs)
        probs = [p / sum_prob for p in probs]  # normalized

        # select the path based on a draw from the uniform distribution
        draw = rnd.random()
        cum = 0.0  # building cumulative distribution function on-the-fly
        selected = -1
        for idx, prob in enumerate(probs):
            high = cum + prob
            if cum <= draw and draw < high:  # the draw has hit this index in the CDF
                selected = idx
                break
            cum = high

        return paths[selected]

    def generate_tree(self, da, gen_doc=None):
        """Generate one tree, saving it into the document provided (if applicable).

        @param da: the input DA
        @param gen_doc: the document where the tree should be saved (defaults to None)
        """
        # generate the tree
        log_debug("GENERATE TREE FOR DA: " + unicode(da))
        tree = self.process_das([da])[0]
        log_debug("RESULT: %s" % unicode(tree))
        # append the tree to a t-tree document, if requested
        if gen_doc:
            zone = self.get_target_zone(gen_doc)
            zone.ttree = tree.create_ttree()
            zone.sentence = unicode(da)
        # return the result
        return tree

    def init_slot_err_stats(self):
        """Initialize slot error statistics accumulator."""
        self.slot_err_stats = SlotErrAnalyzer()

    def get_slot_err_stats(self):
        """Return current slot error statistics, as a string."""
        return ("Slot error: %.6f (M: %d, S: %d, T: %d)" %
                (self.slot_err_stats.slot_error(), self.slot_err_stats.missing,
                 self.slot_err_stats.superfluous, self.slot_err_stats.total))

    @staticmethod
    def load_from_file(model_fname):
        """Detect correct model type (plain/ensemble) and start loading."""
        model_type = Seq2SeqGen  # default to plain generator
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            data = pickle.load(fh)
            if isinstance(data, type):
                model_type = data

        return model_type.load_from_file(model_fname)


class Seq2SeqGen(Seq2SeqBase, TFModel):
    """A plain sequence-to-sequence generator (using encoder-decoder architecture
    from TensorFlow)."""

    def __init__(self, cfg):
        """Initialize the generator, fill in the configuration."""

        Seq2SeqBase.__init__(self, cfg)
        TFModel.__init__(self, scope_name='seq2seq_gen-' + cfg.get('scope_suffix', ''))

        # extract the individual elements out of the configuration dict

        self.emb_size = cfg.get('emb_size', 50)
        self.batch_size = cfg.get('batch_size', 10)
        self.dropout_keep_prob = cfg.get('dropout_prob', 1)
        self.optimizer_type = cfg.get('optimizer_type', 'adam')

        self.passes = cfg.get('passes', 5)
        self.min_passes = cfg.get('min_passes', 1)
        self.improve_interval = cfg.get('improve_interval', 10)
        self.top_k = cfg.get('top_k', 5)
        # self.checkpoint_dir = cfg.get('checkpoint_dir', '/tmp/')  # TODO fix (not used now)
        self.use_dec_cost = cfg.get('use_dec_cost', False)

        self.alpha = cfg.get('alpha', 1e-3)
        self.alpha_decay = cfg.get('alpha_decay', 0.0)
        self.validation_size = cfg.get('validation_size', 0)
        self.validation_freq = cfg.get('validation_freq', 10)
        self.validation_use_all_refs = cfg.get('validation_use_all_refs', False)
        self.validation_delex_slots = cfg.get('validation_delex_slots', set())
        self.validation_use_train_refs = cfg.get('validation_use_train_refs', False)
        if self.validation_delex_slots:
            self.validation_delex_slots = set(self.validation_delex_slots.split(','))
        self.multiple_refs = cfg.get('multiple_refs', False)  # multiple references for validation
        self.ref_selectors = cfg.get('ref_selectors', None)  # selectors of validation trees (if in separate file)
        self.max_cores = cfg.get('max_cores')
        self.mode = cfg.get('mode', 'tokens' if cfg.get('use_tokens') else 'trees')
        self.nn_type = cfg.get('nn_type', 'emb_seq2seq')
        self.randomize = cfg.get('randomize', True)
        self.cell_type = cfg.get('cell_type', 'lstm')
        self.bleu_validation_weight = cfg.get('bleu_validation_weight', 0.0)

        self.use_context = cfg.get('use_context', False)

        # Train Summaries
        self.train_summary_dir = cfg.get('tb_summary_dir', None)
        if self.train_summary_dir:
            self.loss_summary_seq2seq = None
            self.train_summary_op = None
            self.train_summary_writer = None

    def _init_training(self, das_file, ttree_file, data_portion,
                       context_file, validation_files, lexic_files):
        """Load training data, prepare batches, build the NN.

        @param das_file: training DAs (file path)
        @param ttree_file: training t-trees (file path)
        @param data_portion: portion of the data to be actually used for training
        @param context_file: training contexts (file path)
        @param validation_files: validation file paths (or None)
        @param lexic_files: paths to lexicalization data (or None)
        """
        # read training data
        log_info('Reading DAs from ' + das_file + '...')
        das = read_das(das_file)
        trees = self._load_trees(ttree_file)
        if self.use_context:
            das = self._load_contexts(das, context_file)

        # make training data smaller if necessary
        train_size = int(round(data_portion * len(trees)))
        self.train_trees = trees[:train_size]
        self.train_das = das[:train_size]

        # get validation set (default to empty)
        self.valid_trees = []
        self.valid_das = []
        # load separate validation data files...
        if validation_files:
            self._load_valid_data(validation_files)
        # ... or save part of the training data for validation:
        elif self.validation_size > 0:
            self._cut_valid_data()  # will set train_trees, valid_trees, train_das, valid_das

        self.valid_trees_for_lexic = self.valid_trees  # store original validation data
        if self.validation_use_all_refs:  # try to use multiple references (not in lexicalizer)
            self._regroup_valid_refs()

        log_info('Using %d training, %d validation instances.' %
                 (len(self.train_das), len(self.valid_das)))

        # initialize embeddings
        if self.use_context:
            self.da_embs = ContextDAEmbeddingSeq2SeqExtract(cfg=self.cfg)
        else:
            self.da_embs = DAEmbeddingSeq2SeqExtract(cfg=self.cfg)
        if self.mode == 'tokens':
            self.tree_embs = TokenEmbeddingSeq2SeqExtract(cfg=self.cfg)
        elif self.mode == 'tagged_lemmas':
            self.tree_embs = TaggedLemmasEmbeddingSeq2SeqExtract(cfg=self.cfg)
        else:
            self.tree_embs = TreeEmbeddingSeq2SeqExtract(cfg=self.cfg)

        self.da_dict_size = self.da_embs.init_dict(self.train_das)
        self.tree_dict_size = self.tree_embs.init_dict(self.train_trees)
        self.max_tree_len = self.tree_embs.get_embeddings_shape()[0]
        self.max_da_len = self.da_embs.get_embeddings_shape()[0]

        # prepare training batches
        self.train_enc = [cut_batch_into_steps(b)
                          for b in grouper([self.da_embs.get_embeddings(da)
                                            for da in self.train_das],
                                           self.batch_size, None)]
        self.train_dec = [cut_batch_into_steps(b)
                          for b in grouper([self.tree_embs.get_embeddings(tree)
                                            for tree in self.train_trees],
                                           self.batch_size, None)]

        # train lexicalizer (store surface forms, possibly train LM)
        if self.lexicalizer:
            self.lexicalizer.train(lexic_files, self.train_trees, self.valid_trees_for_lexic)

        # train the classifier for filtering n-best lists
        if self.classif_filter:
            self.classif_filter.train(self.train_das, self.train_trees,
                                      valid_das=self.valid_das,
                                      valid_trees=self.valid_trees)

        # convert validation data to flat trees to enable F1 measuring
        if self.validation_size > 0 and self.mode in ['tokens', 'tagged_lemmas']:
            self.valid_trees = self._valid_data_to_flat_trees(self.valid_trees)

        # initialize top costs
        self.top_k_costs = [float('nan')] * self.top_k
        self.checkpoint_path = None

        # build the NN
        self._init_neural_network()

        # initialize the NN variables
        self.session.run(tf.global_variables_initializer())

    def _load_trees(self, ttree_file, selector=None):
        """Load input trees/sentences from a .yaml.gz/.pickle.gz (trees) or .txt (sentences) file."""
        log_info('Reading t-trees/sentences from ' + ttree_file + '...')
        if ttree_file.endswith('.txt'):
            if self.mode == 'trees':
                raise ValueError("Cannot read trees from a .txt file (%s)!" % ttree_file)
            return read_tokens(ttree_file)
        else:
            ttree_doc = read_ttrees(ttree_file)
            if selector is None:
                selector = self.selector
            if self.mode == 'tokens':
                return tokens_from_doc(ttree_doc, self.language, selector)
            elif self.mode == 'tagged_lemmas':
                return tagged_lemmas_from_doc(ttree_doc, self.language, selector)
            else:
                return trees_from_doc(ttree_doc, self.language, selector)

    def _load_contexts(self, das, context_file):
        """Load input context utterances from a .yaml.gz/.pickle.gz/.txt file and add them to the
        given DAs (each returned item is then a tuple of context + DA)."""
        # read contexts, combine them with corresponding DAs for easier handling
        if context_file is None:
            raise ValueError('Expected context utterances file name!')
        log_info('Reading context utterances from %s...' % context_file)
        if context_file.endswith('.txt'):
            contexts = read_tokens(context_file)
        else:
            contexts = tokens_from_doc(read_ttrees(context_file), self.language, self.selector)
        return [(context, da) for context, da in zip(contexts, das)]

    def _load_valid_data(self, valid_data_paths):
        """Load validation data from separate files (comma-separated list of files with DAs, trees,
        and optionally contexts is expected)."""
        # parse validation data file specification
        valid_data_paths = valid_data_paths.split(',')
        if len(valid_data_paths) == 3:  # with contexts (this does not determine if they're used)
            valid_das_file, valid_trees_file, valid_context_file = valid_data_paths
        else:
            valid_das_file, valid_trees_file = valid_data_paths

        # load the validation data
        log_info('Reading DAs from ' + valid_das_file + '...')
        self.valid_das = read_das(valid_das_file)
        self.valid_trees = self._load_trees(valid_trees_file, selector=self.ref_selectors)
        if self.use_context:
            self.valid_das = self._load_contexts(self.valid_das, valid_context_file)

        # reorder validation data for multiple references (see also _cut_valid_data)
        valid_size = len(self.valid_trees)
        if self.multiple_refs:
            num_refs, refs_stored = self._check_multiple_ref_type(valid_size)

            # serial: different instances next to each other, then synonymous in the same order
            if refs_stored == 'serial':
                valid_tree_chunks = [chunk for chunk in
                                     chunk_list(self.valid_trees, valid_size / num_refs)]
                self.valid_trees = [[chunk[i] for chunk in valid_tree_chunks]
                                    for i in xrange(valid_size / num_refs)]
                if len(self.valid_das) > len(self.valid_trees):
                    self.valid_das = self.valid_das[0:valid_size / num_refs]
            # parallel: synonymous instances next to each other
            elif refs_stored == 'parallel':
                self.valid_trees = [chunk for chunk in chunk_list(self.valid_trees, num_refs)]
                if len(self.valid_das) > len(self.valid_trees):
                    self.valid_das = self.valid_das[::num_refs]

        # no multiple references; make lists of size 1 to simplify working with the data
        else:
            self.valid_trees = [[tree] for tree in self.valid_trees]

    def _tokens_to_flat_trees(self, sents):
        """Use sentences (pairs token-tag) read from Treex files and convert them into flat
        trees (each token has a node right under the root, lemma is the token, formeme is 'x').
        Uses TokenEmbeddingSeq2SeqExtract conversion there and back.

        @param sents: sentences to be converted
        @return: a list of flat trees
        """
        return [self.tree_embs.ids_to_tree(self.tree_embs.get_embeddings(sent)) for sent in sents]

    def _check_multiple_ref_type(self, data_size):
        """Parse multiple references setting from the configuration file and check if the data size
        is compatible with it."""
        num_refs, refs_stored = self.multiple_refs.split(',')
        num_refs = int(num_refs)
        if data_size % num_refs != 0:
            raise Exception('Data length must be divisible by the number of references!')
        return num_refs, refs_stored

    def _regroup_valid_refs(self):
        """Group all validation trees/sentences according to the same DA (sorted and
        possibly delexicalized).
        """
        if self.use_context:  # if context is used, then train_das are da[1]
            normalized_das = [da[1].get_delexicalized(self.validation_delex_slots)
                              for da in self.valid_das]
        else:
            normalized_das = [da.get_delexicalized(self.validation_delex_slots)
                              for da in self.valid_das]
        da_groups = {}
        for trees, da in zip(self.valid_trees, normalized_das):
            da.sort()
            da_groups[da] = da_groups.get(da, [])
            da_groups[da].extend(trees)

        # use training trees as additional references if needed
        if self.validation_use_train_refs:
            if self.use_context:  # if context is used, then train_das are da[1]
                normalized_train_das = [da[1].get_delexicalized(self.validation_delex_slots)
                                        for da in self.train_das]
            else:
                normalized_train_das = [da.get_delexicalized(self.validation_delex_slots)
                                        for da in self.train_das]

            for tree, da in zip(self.train_trees, normalized_train_das):
                da.sort()
                if da in da_groups:
                    da_groups[da].append(tree)

        # deduplicate the references
        for da_group in da_groups.itervalues():
            da_group.sort()
        da_groups = {da: [sent for sent, _ in groupby(da_group)]
                     for da, da_group in da_groups.iteritems()}
        # store the references in correct order
        self.valid_trees = [da_groups[da] for da in normalized_das]

    def _cut_valid_data(self):
        """Put aside part of the training set for validation."""
        train_size = len(self.train_trees)

        # we have multiple references
        if self.multiple_refs:
            num_refs, refs_stored = self._check_multiple_ref_type(train_size)

            # data stored "serially" (all different instances next to each other, then again in the
            # same order)
            if refs_stored == 'serial':
                train_tree_chunks = [chunk for chunk in
                                     chunk_list(self.train_trees, train_size / num_refs)]
                train_da_chunks = [chunk for chunk in
                                   chunk_list(self.train_das, train_size / num_refs)]
                self.valid_trees = [[chunk[i] for chunk in train_tree_chunks]
                                    for i in xrange(train_size / num_refs - self.validation_size,
                                                    train_size / num_refs)]
                self.valid_das = train_da_chunks[0][-self.validation_size:]
                self.train_trees = sum([chunk[:-self.validation_size]
                                        for chunk in train_tree_chunks], [])
                self.train_das = sum([chunk[:-self.validation_size]
                                      for chunk in train_da_chunks], [])
            # data stored in "parallel" (all synonymous instances next to each other)
            else:
                self.valid_trees = [chunk for chunk in
                                    chunk_list(self.train_trees[-self.validation_size * num_refs:],
                                               num_refs)]
                self.valid_das = self.train_das[-self.validation_size * num_refs::num_refs]
                self.train_trees = self.train_trees[:-self.validation_size * num_refs]
                self.train_das = self.train_das[:-self.validation_size * num_refs]

        # single validation reference
        else:
            # make "reference lists" of length 1 to accommodate for functions working
            # with multiple references
            self.valid_trees = [[tree] for tree in self.train_trees[-self.validation_size:]]
            self.valid_das = self.train_das[-self.validation_size:]
            self.train_trees = self.train_trees[:-self.validation_size]
            self.train_das = self.train_das[:-self.validation_size]

    def _valid_data_to_flat_trees(self, valid_sents):
        """Convert validation data to flat trees, which are the result of `process_das` when
        `self.mode` is 'tokens' or 'tagged_lemmas'. This enables to measure F1 on the resulting
        flat trees (equals to unigram F1 on sentence tokens/lemmas&tags).

        @param valid_sents: validation set sentences (for each sentence, list of lists of \
            tokens+tags/just tokens/lemmas+tags)
        @return: the same sentences converted to flat trees \
            (see `TokenEmbeddingSeq2SeqExtract.ids_to_tree`)
        """
        # sent = list of paraphrases for a given sentence
        return [self._tokens_to_flat_trees(sent) for sent in valid_sents]

    def _init_neural_network(self):
        """Initializing the NN (building a TensorFlow graph and initializing session)."""

        # set TensorFlow random seed
        tf.set_random_seed(rnd.randint(-sys.maxint, sys.maxint))

        # create placeholders for input & output (always batch-size * 1, list of up to num. steps)
        self.enc_inputs = []
        self.enc_inputs_drop = []
        for i in xrange(self.max_da_len):
            enc_input = tf.placeholder(tf.int32, [None], name=('enc_inp-%d' % i))
            self.enc_inputs.append(enc_input)
            if self.dropout_keep_prob < 1:
                enc_input_drop = tf.nn.dropout(enc_input, self.dropout_keep_prob,
                                               name=('enc_inp-drop-%d' % i))
                self.enc_inputs_drop.append(enc_input_drop)

        self.dec_inputs = []
        for i in xrange(self.max_tree_len):
            self.dec_inputs.append(tf.placeholder(tf.int32, [None], name=('dec_inp-%d' % i)))

        # targets are just decoder inputs shifted by one (+pad with one empty spot)
        self.targets = [self.dec_inputs[i + 1] for i in xrange(len(self.dec_inputs) - 1)]
        self.targets.append(tf.placeholder(tf.int32, [None], name=('target-pad')))

        # prepare cells
        self.initial_state = tf.placeholder(tf.float32, [None, self.emb_size])
        if self.cell_type.startswith('gru'):
            self.cell = tf.contrib.rnn.GRUCell(self.emb_size)
        else:
            self.cell = tf.contrib.rnn.BasicLSTMCell(self.emb_size)

        if self.cell_type.endswith('/2'):
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * 2)

        # build the actual LSTM Seq2Seq network (for training and decoding)
        with tf.variable_scope(self.scope_name) as scope:

            rnn_func = tf06s2s.embedding_rnn_seq2seq
            if self.nn_type == 'emb_attention_seq2seq':
                rnn_func = tf06s2s.embedding_attention_seq2seq
            elif self.nn_type == 'emb_attention2_seq2seq':
                rnn_func = partial(tf06s2s.embedding_attention_seq2seq, num_heads=2)
            elif self.nn_type == 'emb_attention_seq2seq_context':
                rnn_func = embedding_attention_seq2seq_context
            elif self.nn_type == 'emb_attention2_seq2seq_context':
                rnn_func = partial(embedding_attention_seq2seq_context, num_heads=2)

            # for training: feed_previous == False, using dropout if available
            # outputs = batch_size * num_decoder_symbols ~ i.e. output logits at each steps
            # states = cell states at each steps
            self.outputs, self.states = rnn_func(
                self.enc_inputs_drop if self.enc_inputs_drop else self.enc_inputs,
                self.dec_inputs, self.cell,
                self.da_dict_size, self.tree_dict_size,
                self.emb_size,
                scope=scope)

            scope.reuse_variables()

            # for decoding: feed_previous == True
            self.dec_outputs, self.dec_states = rnn_func(
                self.enc_inputs, self.dec_inputs, self.cell,
                self.da_dict_size, self.tree_dict_size,
                self.emb_size,
                feed_previous=True, scope=scope)

        # TODO use output projection ???

        # target weights
        # TODO change to actual weights, zero after the end of tree ???
        self.cost_weights = [tf.ones_like(trg, tf.float32, name='cost_weights')
                             for trg in self.targets]

        # cost
        self.tf_cost = tf06s2s.sequence_loss(self.outputs, self.targets,
                                             self.cost_weights, self.tree_dict_size)
        self.dec_cost = tf06s2s.sequence_loss(self.dec_outputs, self.targets,
                                              self.cost_weights, self.tree_dict_size)
        if self.use_dec_cost:
            self.cost = 0.5 * (self.tf_cost + self.dec_cost)
        else:
            self.cost = self.tf_cost

        # Tensorboard summaries
        if self.train_summary_dir:
            self.loss_summary_seq2seq = tf.summary.scalar("loss_seq2seq", self.cost)
            self.train_summary_op = tf.summary.merge([self.loss_summary_seq2seq])

        # optimizer (default to Adam)
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        if self.optimizer_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        if self.optimizer_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        else:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_func = self.optimizer.minimize(self.cost)

        # initialize session
        session_config = None
        if self.max_cores:
            session_config = tf.ConfigProto(inter_op_parallelism_threads=self.max_cores,
                                            intra_op_parallelism_threads=self.max_cores)
        self.session = tf.Session(config=session_config)

        # this helps us load/save the model
        self.saver = tf.train.Saver(tf.global_variables())
        if self.train_summary_dir:  # Tensorboard summary writer
            self.train_summary_writer = tf.summary.FileWriter(
                os.path.join(self.train_summary_dir, "main_seq2seq"), self.session.graph)

    def _training_pass(self, iter_no):
        """Perform one pass through the training data (epoch).
        @param iter_no: pass number (for logging)
        """
        it_cost = 0.0
        it_learning_rate = self.alpha * np.exp(-self.alpha_decay * iter_no)
        log_info('IT %d alpha: %8.5f' % (iter_no, it_learning_rate))

        for batch_no in self.train_order:

            # feed data into the TF session:

            # initial state
            initial_state = np.zeros([self.batch_size, self.emb_size])
            feed_dict = {self.initial_state: initial_state,
                         self.learning_rate: it_learning_rate}

            # encoder inputs
            for i in xrange(len(self.train_enc[batch_no])):
                feed_dict[self.enc_inputs[i]] = self.train_enc[batch_no][i]

            # decoder inputs
            for i in xrange(len(self.train_dec[batch_no])):
                feed_dict[self.dec_inputs[i]] = self.train_dec[batch_no][i]

            # the last target output (padding, to have the same number of step as there are decoder
            # inputs) is always 'VOID' for all instances of the batch
            feed_dict[self.targets[-1]] = len(self.train_dec[batch_no][0]) * [self.tree_embs.VOID]

            # run the TF session (one optimizer step == train_func) and get the cost
            # (1st value returned is None, throw it away)
            if self.train_summary_dir:  # compute also summaries for Tensorboard
                _, cost, train_summary_op = self.session.run(
                    [self.train_func, self.cost, self.train_summary_op], feed_dict=feed_dict)
            else:
                _, cost = self.session.run([self.train_func, self.cost], feed_dict=feed_dict)
            it_cost += cost

        if self.train_summary_dir:  # Tensorboard: iteration summary
            self.train_summary_writer.add_summary(train_summary_op, iter_no)

        log_info('IT %d total cost: %8.5f' % (iter_no, cost))

    def _should_stop(self, iter_no, cur_cost):
        """Determine if the training should stop (i.e., we've run for more than self.min_passes
        and self.top_k_costs hasn't changed for more than self.improve_interval passes).

        @param iter_no: current iteration number
        @param cur_cost: current validation cost
        @return a boolean value indicating whether the training should stop (True to stop)
        """
        pos = self.top_k
        while (pos > 0 and
               (math.isnan(self.top_k_costs[pos - 1]) or
                cur_cost < self.top_k_costs[pos - 1])):
            pos -= 1

        if pos < self.top_k:
            self.top_k_change = iter_no
            self.top_k_costs.insert(pos, cur_cost)
            self.top_k_costs.pop()
            return False

        return iter_no > self.min_passes and iter_no > self.top_k_change + self.improve_interval

    def train(self, das_file, ttree_file, data_portion=1.0,
              context_file=None, validation_files=None, lexic_files=None):
        """
        The main training process – initialize and perform a specified number of
        training passes, validating every couple iterations.

        @param das_file: training data file with DAs
        @param ttree_file: training data file with output t-trees/sentences
        @param data_portion: portion of training data to be actually used, defaults to 1.0
        @param context_file: path to training file with contexts (trees/sentences)
        @param validation_files: paths to validation data (DAs, trees/sentences, possibly contexts)
        @param lexic_files: paths to lexicalization data (surface forms, possibly lexicalization \
                instruction file to train LM)
        """
        # load and prepare data and initialize the neural network
        self._init_training(das_file, ttree_file, data_portion,
                            context_file, validation_files, lexic_files)

        # do the training passes
        for iter_no in xrange(1, self.passes + 1):

            self.train_order = range(len(self.train_enc))
            if self.randomize:
                rnd.shuffle(self.train_order)

            self._training_pass(iter_no)

            # validate every couple iterations
            if self.validation_size > 0 and iter_no % self.validation_freq == 0:

                cur_train_out = self.process_das(self.train_das[:self.batch_size])
                log_info("Current train output:\n" +
                         "\n".join([" ".join(n.t_lemma for n in tree.nodes[1:])
                                    if self.mode in ['tokens', 'tagged_lemmas']
                                    else unicode(tree)
                                    for tree in cur_train_out]))

                cur_valid_out = self.process_das(self.valid_das[:self.batch_size])
                cur_cost = self._compute_valid_cost(cur_valid_out, self.valid_trees)
                log_info("Current validation output:\n" +
                         "\n".join([" ".join(n.t_lemma for n in tree.nodes[1:])
                                    if self.mode in ['tokens', 'tagged_lemmas']
                                    else unicode(tree)
                                    for tree in cur_valid_out]))
                log_info('IT %d validation cost: %5.4f' % (iter_no, cur_cost))

                # if we have the best model so far, save it as a checkpoint (overwrite previous)
                if math.isnan(self.top_k_costs[0]) or cur_cost < self.top_k_costs[0]:
                    self._save_checkpoint()

                if self._should_stop(iter_no, cur_cost):
                    log_info("Stoping criterion met.")
                    break

    def _compute_valid_cost(self, cur_valid_out, valid_trees):
        """Compute the validation set cost for the current output (interpolate negative
        BLEU and F1 scores according to `self.bleu_validation_weight`).

        @param cur_valid_out: the current system output on the validation DAs
        @param valid_trees: the gold trees for the validation DAs (one or two paraphrases)
        @return the cost, as a negative interpolation of BLEU and F1
        """
        cost = 0.0
        if self.bleu_validation_weight > 0.0:
            cost -= self.bleu_validation_weight * self._compute_bleu(cur_valid_out, valid_trees)
        if self.bleu_validation_weight < 1.0:
            cost -= ((1.0 - self.bleu_validation_weight) *
                     self._compute_f1(cur_valid_out, valid_trees))
        return cost

    def _compute_bleu(self, cur_valid_out, valid_trees):
        """Compute BLEU score of the current output on a set of validation trees. If the
        validation set is a tuple (two paraphrases), use them both for BLEU computation.

        @param cur_valid_out: the current system output on the validation DAs
        @param valid_trees: the gold trees for the validation DAs (one or two paraphrases)
        @return: BLEU score, as a float (percentage)
        """
        evaluator = BLEUMeasure()
        for pred_tree, gold_trees in zip(cur_valid_out, valid_trees):
            evaluator.append(pred_tree, gold_trees)
        return evaluator.bleu()

    def _compute_f1(self, cur_valid_out, valid_trees):
        """Compute F1 score of the current output on a set of validation trees. If the validation
        set is a tuple (two paraphrases), returns the average.

        @param cur_valid_out: the current system output on the validation DAs
        @param valid_trees: the gold trees for the validation DAs (one or two paraphrases)
        @return: (average) F1 score, as a float
        """
        evaluator = Evaluator()
        for pred_tree, gold_trees in zip(cur_valid_out, valid_trees):
            for gold_tree in gold_trees:
                evaluator.append(TreeNode(gold_tree), TreeNode(pred_tree))
        return evaluator.f1()

    def save_to_file(self, model_fname):
        """Save the generator to a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph will be stored with a \
            different extension
        """
        model_fname = self.tf_check_filename(model_fname)
        log_info("Saving generator to %s..." % model_fname)
        if self.classif_filter:
            classif_filter_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tftreecl\1', model_fname)
            self.classif_filter.save_to_file(classif_filter_fname)
        if self.lexicalizer:
            lexicalizer_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.lexic\1', model_fname)
            self.lexicalizer.save_to_file(lexicalizer_fname)

        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self.get_all_settings(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        if hasattr(self, 'checkpoint_path') and self.checkpoint_path:
            self.saver.restore(self.session, self.checkpoint_path)
            shutil.rmtree(os.path.dirname(self.checkpoint_path))
        self.saver.save(self.session, tf_session_fname)

    def get_all_settings(self):
        """Get all settings except the trained model parameters (to be stored in a pickle)."""
        data = {'cfg': self.cfg,
                'da_embs': self.da_embs,
                'tree_embs': self.tree_embs,
                'da_dict_size': self.da_dict_size,
                'tree_dict_size': self.tree_dict_size,
                'max_da_len': self.max_da_len,
                'max_tree_len': self.max_tree_len,
                'classif_filter': self.classif_filter is not None,
                'lexicalizer': self.lexicalizer is not None}
        return data

    def _save_checkpoint(self):
        """Save a checkpoint to a temporary path; set `self.checkpoint_path` to the path
        where it is saved; if called repeatedly, will always overwrite the last checkpoint."""
        if not self.checkpoint_path:
            path = tempfile.mkdtemp(suffix="", prefix="tgen-")
            self.checkpoint_path = os.path.join(path, "ckpt")
        log_info('Saving checkpoint to %s' % self.checkpoint_path)
        self.saver.save(self.session, self.checkpoint_path)

    @staticmethod
    def load_from_file(model_fname):
        """Load the generator from a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph must be stored with a \
            different extension
        """
        log_info("Loading generator from %s..." % model_fname)
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            data = pickle.load(fh)
            ret = Seq2SeqGen(cfg=data['cfg'])
            ret.load_all_settings(data)

        if ret.classif_filter:
            classif_filter_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tftreecl\1', model_fname)
            if os.path.isfile(classif_filter_fname):
                ret.classif_filter = RerankingClassifier.load_from_file(classif_filter_fname)
            else:
                log_warn("Classification filter data not found, ignoring.")
                ret.classif_filter = False

        if ret.lexicalizer:
            lexicalizer_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.lexic\1', model_fname)
            if os.path.isfile(lexicalizer_fname):
                ret.lexicalizer = Lexicalizer.load_from_file(lexicalizer_fname)
            else:
                log_warn("Lexicalizer data not found, ignoring.")
                ret.lexicalizer = None

        # re-build TF graph and restore the TF session
        tf_session_fname = os.path.abspath(re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname))
        param_dump_fname = re.sub(r'(.pickle)?(.gz)?$', '.params.gz', model_fname)
        ret._init_neural_network()
        if os.path.isfile(param_dump_fname):
            log_info('Loading params dump from %s...' % param_dump_fname)
            with file_stream(param_dump_fname, 'rb', encoding=None) as fh:
                ret.set_model_params(pickle.load(fh))
        else:
            log_info('Loading saved TF session from %s...' % tf_session_fname)
            ret.saver.restore(ret.session, tf_session_fname)

        return ret

    def _get_greedy_decoder_output(self, enc_inputs, dec_inputs, compute_cost=False):
        """Run greedy decoding with the given inputs; return decoder outputs and the cost
        (if required).

        @param enc_inputs: encoder inputs (list of token IDs)
        @param dec_inputs: decoder inputs (list of token IDs)
        @param compute_cost: if True, decoding cost is computed (the dec_inputs must be valid trees)
        @return a tuple of list of decoder outputs + decoding cost (None if not required)
        """
        initial_state = np.zeros([len(enc_inputs[0]), self.emb_size])
        feed_dict = {self.initial_state: initial_state}

        for i in xrange(len(enc_inputs)):
            feed_dict[self.enc_inputs[i]] = enc_inputs[i]

        for i in xrange(len(dec_inputs)):
            feed_dict[self.dec_inputs[i]] = dec_inputs[i]

        feed_dict[self.targets[-1]] = len(enc_inputs[0]) * [self.tree_embs.VOID]

        # run the decoding
        if not compute_cost:
            dec_outputs = self.session.run(self.dec_outputs, feed_dict=feed_dict)
            dec_cost = None
        else:
            res = self.session.run(self.dec_outputs + [self.dec_cost], feed_dict=feed_dict)
            dec_outputs = res[:-1]
            dec_cost = res[-1]

        # get the highest-scoring IDs
        dec_output_ids = np.argmax(dec_outputs, axis=2)

        return dec_output_ids, dec_cost

    def _init_beam_search(self, enc_inputs):
        """Initialize beam search for the current DA (with the given encoder inputs)."""
        # initial state
        initial_state = np.zeros([1, self.emb_size])
        self._beam_search_feed_dict = {self.initial_state: initial_state}
        # encoder inputs
        for i in xrange(len(enc_inputs)):
            self._beam_search_feed_dict[self.enc_inputs[i]] = enc_inputs[i]

    def _beam_search_step(self, dec_inputs, dec_states):
        """Run one step of beam search decoding with the given decoder inputs and
        (previous steps') outputs and states."""

        step = len(dec_states)  # find the decoder position

        # fill in all previous path data
        for i in xrange(step):
            self._beam_search_feed_dict[self.dec_inputs[i]] = dec_inputs[i]
            self._beam_search_feed_dict[self.states[i]] = dec_states[i]

        # the decoder outputs are always one step longer
        self._beam_search_feed_dict[self.dec_inputs[step]] = dec_inputs[step]

        # run one step of the decoder
        output, state = self.session.run([self.outputs[step], self.states[step]],
                                         feed_dict=self._beam_search_feed_dict)

        # softmax (normalize decoder outputs to obtain prob. distribution), assuming batches size 1
        out_probs = softmax(output[0])
        return out_probs, state

    def lexicalize(self, trees, abstr_file):
        """Lexicalize generated trees according to the given lexicalization instruction file.
        @param trees: list of generated TreeData instances (delexicalized)
        @Param abstr_file: a file containing lexicalization instructions
        @return: None
        """
        self.lexicalizer.lexicalize(trees, abstr_file)
