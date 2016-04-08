#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import re
import numpy as np
import tensorflow as tf
import cPickle as pickle
from itertools import izip_longest
import sys
import math
import tempfile
import shutil
import os

from tensorflow.models.rnn.seq2seq import embedding_rnn_seq2seq, embedding_attention_seq2seq, \
    sequence_loss
from tensorflow.models.rnn import rnn_cell

from pytreex.core.util import file_stream

from tgen.logf import log_info, log_debug, log_warn
from tgen.futil import read_das, read_ttrees, trees_from_doc, tokens_from_doc
from tgen.embeddings import DAEmbeddingSeq2SeqExtract, TokenEmbeddingSeq2SeqExtract, \
    TreeEmbeddingSeq2SeqExtract
from tgen.rnd import rnd
from tgen.planner import SentencePlanner
from tgen.tree import TreeData, TreeNode
from tgen.eval import Evaluator
from tgen.bleu import BLEUMeasure
from tgen.tfclassif import RerankingClassifier
from tgen.tf_ml import TFModel


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

        self.beam_size = cfg.get('beam_size', 1)

        self.classif_filter = None
        if 'classif_filter' in cfg:
            self.classif_filter = RerankingClassifier(cfg['classif_filter'])
            self.misfit_penalty = cfg.get('misfit_penalty', 100)

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

        # decoder inputs (either fake, or true but used just for cost computation)
        if gold_trees is None:
            empty_tree_emb = self.tree_embs.get_embeddings(TreeData())
            dec_inputs = cut_batch_into_steps([empty_tree_emb for _ in enc_inputs[0]])
        else:
            dec_inputs = cut_batch_into_steps([self.tree_embs.get_embeddings(tree)
                                               for tree in gold_trees])

        # run the decoding per se
        dec_outputs, dec_cost = self._get_greedy_decoder_output(
                enc_inputs, dec_inputs, compute_cost=gold_trees is not None)

        # convert the output back into a tree
        dec_output_ids = np.argmax(dec_outputs, axis=2)
        return dec_output_ids, dec_cost

    def _get_greedy_decoder_output(initial_state, enc_inputs, dec_inputs, compute_cost=False):
        raise NotImplementedError()

    class DecodingPath(object):
        """A decoding path to be used in beam search."""

        __slots__ = ['dec_inputs', 'dec_outputs', 'dec_states', 'logprob']

        def __init__(self, dec_inputs=[], dec_outputs=[], dec_states=[], logprob=0.0):
            self.dec_inputs = list(dec_inputs)
            self.dec_outputs = list(dec_outputs)
            self.dec_states = list(dec_states)
            self.logprob = logprob

        def expand(self, max_variants, dec_output, dec_state):
            """Expand the path with all possible outputs, updating the log probabilities.

            @param max_variants: expand to this number of variants at maximum, discard the less \
                probable ones
            @param dec_output: the decoder output scores for the current step
            @param dec_state: the decoder hidden state for the current step
            @return: an array of all possible continuations of this path
            """
            ret = []

            # softmax, assuming batches size 1
            # http://stackoverflow.com/questions/34968722/softmax-function-python
            probs = np.exp(dec_output[0]) / np.sum(np.exp(dec_output[0]), axis=0)
            # select only up to max_variants most probable variants
            top_n_idx = np.argpartition(-probs, max_variants)[:max_variants]

            for idx in top_n_idx:
                expanded = Seq2SeqGen.DecodingPath(self.dec_inputs, self.dec_outputs,
                                                   self.dec_states, self.logprob)
                expanded.logprob += np.log(probs[idx])
                expanded.dec_inputs.append(np.array(idx, ndmin=1))
                expanded.dec_outputs.append(dec_output)
                expanded.dec_states.append(dec_state)
                ret.append(expanded)

            return ret

        def __cmp__(self, other):
            """Comparing the paths according to their logprob."""
            if self.logprob < other.logprob:
                return -1
            if self.logprob > other.logprob:
                return 1
            return 0

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

        paths = [self.DecodingPath(dec_inputs=[dec_inputs[0]])]

        # beam search steps
        for step in xrange(len(dec_inputs)):

            new_paths = []

            for path in paths:
                out, st = self._beam_search_step(path.dec_inputs, path.dec_outputs, path.dec_states)
                new_paths.extend(path.expand(self.beam_size, out, st))

            paths = sorted(new_paths, reverse=True)[:self.beam_size]

            if all([p.dec_inputs[-1] == self.tree_embs.VOID for p in paths]):
                break  # stop decoding if we have reached the end in all paths

            log_debug(("\nBEAM SEARCH STEP %d\n" % step) +
                      "\n".join([("%f\t" % p.logprob) +
                                 " ".join(self.tree_embs.ids_to_strings([inp[0] for inp in p.dec_inputs]))
                                 for p in paths]) + "\n")

        # rerank paths by their distance to the input DA
        if self.classif_filter:
            paths = self._rerank_paths(paths, da)

        # return just the best path (as token IDs)
        return np.array(paths[0].dec_inputs)

    def _init_beam_search(self, enc_inputs):
        raise NotImplementedError()

    def _beam_search_step(self, dec_inputs, dec_outputs, dec_states):
        raise NotImplementedError()

    def _rerank_paths(self, paths, da):
        """Rerank the n-best decoded paths according to the reranking classifier."""

        trees = [self.tree_embs.ids_to_tree(np.array(path.dec_inputs).transpose()[0])
                 for path in paths]
        self.classif_filter.init_run(da)
        fits = self.classif_filter.dist_to_cur_da(trees)
        # add distances to logprob so that non-fitting will be heavily penalized
        for path, fit in zip(paths, fits):
            path.logprob -= self.misfit_penalty * fit
        return sorted(paths, reverse=True)

    def generate_tree(self, da, gen_doc=None):
        """Generate one tree, saving it into the document provided (if applicable).

        @param da: the input DA
        @param gen_doc: the document where the tree should be saved (defaults to None)
        """
        # generate the tree
        log_debug("GENERATE TREE FOR DA: " + unicode(da))
        tree = self.process_das([da])[0]
        log_debug("RESULT: %s" % unicode(tree))
        # if requested, append the result to the document
        if gen_doc:
            zone = self.get_target_zone(gen_doc)
            zone.ttree = tree.create_ttree()
            zone.sentence = unicode(da)
        # return the result
        return tree

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
        self.max_cores = cfg.get('max_cores')
        self.use_tokens = cfg.get('use_tokens', False)
        self.nn_type = cfg.get('nn_type', 'emb_seq2seq')
        self.randomize = cfg.get('randomize', True)
        self.cell_type = cfg.get('cell_type', 'lstm')
        self.bleu_validation_weight = cfg.get('bleu_validation_weight', 0.0)

    def _init_training(self, das_file, ttree_file, data_portion):
        """Load training data, prepare batches, build the NN.

        @param das_file: training DAs (file path)
        @param ttree_file: training t-trees (file path)
        @param data_portion: portion of the data to be actually used for training
        """
        # read input
        log_info('Reading DAs from ' + das_file + '...')
        das = read_das(das_file)
        log_info('Reading t-trees from ' + ttree_file + '...')
        ttree_doc = read_ttrees(ttree_file)
        if self.use_tokens:
            trees = tokens_from_doc(ttree_doc, self.language, self.selector)
        else:
            trees = trees_from_doc(ttree_doc, self.language, self.selector)

        # make training data smaller if necessary
        train_size = int(round(data_portion * len(trees)))
        self.train_trees = trees[:train_size]
        self.train_das = das[:train_size]

        # save part of the training data for validation:
        if self.validation_size > 0:
            # check if there are 2 copies of input DAs in the training data.
            # if so, put aside both copies of validation DAs/trees
            cut_dbl = self.train_das[train_size / 2 - 1] == self.train_das[-1]
            if cut_dbl:
                log_info('Detected duplicate DAs in training data -- ' +
                         'using both copies for validation')
            self.train_trees, self.valid_trees = self._cut_valid_data(self.train_trees, cut_dbl)
            self.train_das, self.valid_das = self._cut_valid_data(self.train_das, cut_dbl)
            if cut_dbl:
                self.valid_das = self.valid_das[0]  # the DAs are identical in both copies

        log_info('Using %d training, %d validation instances.' %
                 (len(self.train_das), self.validation_size))

        # initialize embeddings
        self.da_embs = DAEmbeddingSeq2SeqExtract(cfg=self.cfg)
        if self.use_tokens:
            self.tree_embs = TokenEmbeddingSeq2SeqExtract(cfg=self.cfg)
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

        # convert validation data to flat trees to enable F1 measuring
        if self.validation_size > 0 and self.use_tokens:
            self.valid_trees = self._valid_data_to_flat_trees(self.valid_trees)

        # train the classifier for filtering n-best lists
        if self.classif_filter:
            classif_train_trees = (self.train_trees if not self.use_tokens else
                                   self._tokens_to_flat_trees(self.train_trees))
            self.classif_filter.train(self.train_das, classif_train_trees,
                                      valid_das=self.valid_das,
                                      valid_trees=self.valid_trees)
            self.classif_filter.restore_checkpoint()  # restore the best performance on devel data

        # initialize top costs
        self.top_k_costs = [float('nan')] * self.top_k
        self.checkpoint_path = None

        # build the NN
        self._init_neural_network()

        # initialize the NN variables
        self.session.run(tf.initialize_all_variables())

    def _tokens_to_flat_trees(self, sents):
        return [self.tree_embs.ids_to_tree(self.tree_embs.get_embeddings(sent)) for sent in sents]

    def _cut_valid_data(self, insts, cut_double):
        """Put aside part of the training set  for validation.

        @param insts: original training set (DAs, tokens, or trees)
        @param cut_double: put aside both copies of the training instances (in case the training \
            data contain two copies of all input DAs -- this holds for the BAGEL set)
        @return: new training and validation sets
        """
        train_size = len(insts)
        valid = insts[-self.validation_size:]
        train = insts[:-self.validation_size]
        if cut_double:
            valid = (train[train_size / 2 - self.validation_size:train_size / 2], valid)
            train = train[:train_size / 2 - self.validation_size] + train[train_size / 2:]
        return train, valid

    def _valid_data_to_flat_trees(self, valid_sents):
        """Convert validation data to flat trees, which are the result of `process_das` when
        `self.use_tokens` is in force. This enables to measure F1 on the resulting flat trees
        (equals to unigram F1 on sentence tokens).

        @param valid_sents: validation set sentences (list of list of tokens, or a tuple \
            of two lists containing the individual paraphrases)
        @return: the same sentences converted to flat trees \
            (see `TokenEmbeddingSeq2SeqExtract.ids_to_tree`)
        """
        if isinstance(valid_sents, tuple):
            return (self._valid_data_to_flat_trees(valid_sents[0]),
                    self._valid_data_to_flat_trees(valid_sents[1]))

        return self._tokens_to_flat_trees(valid_sents)

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
            self.cell = rnn_cell.GRUCell(self.emb_size)
        else:
            self.cell = rnn_cell.BasicLSTMCell(self.emb_size)

        if self.cell_type.endswith('/2'):
            self.cell = rnn_cell.MultiRNNCell([self.cell] * 2)

        # build the actual LSTM Seq2Seq network (for training and decoding)
        with tf.variable_scope(self.scope_name) as scope:

            rnn_func = embedding_rnn_seq2seq
            if self.nn_type == 'emb_attention_seq2seq':
                rnn_func = embedding_attention_seq2seq

            # for training: feed_previous == False, using dropout if available
            # outputs = batch_size * num_decoder_symbols ~ i.e. output logits at each steps
            # states = cell states at each steps
            self.outputs, self.states = rnn_func(
                self.enc_inputs_drop if self.enc_inputs_drop else self.enc_inputs,
                self.dec_inputs, self.cell,
                self.da_dict_size, self.tree_dict_size,
                scope=scope)

            scope.reuse_variables()

            # for decoding: feed_previous == True
            self.dec_outputs, self.dec_states = rnn_func(
                self.enc_inputs, self.dec_inputs, self.cell,
                self.da_dict_size, self.tree_dict_size,
                feed_previous=True, scope=scope)

        # TODO use output projection ???

        # target weights
        # TODO change to actual weights, zero after the end of tree ???
        self.cost_weights = [tf.ones_like(trg, tf.float32, name='cost_weights')
                             for trg in self.targets]

        # cost
        self.tf_cost = sequence_loss(self.outputs, self.targets,
                                     self.cost_weights, self.tree_dict_size)
        self.dec_cost = sequence_loss(self.dec_outputs, self.targets,
                                      self.cost_weights, self.tree_dict_size)
        if self.use_dec_cost:
            self.cost = 0.5 * (self.tf_cost + self.dec_cost)
        else:
            self.cost = self.tf_cost

        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # optimizer (default to Adam)
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
        self.saver = tf.train.Saver(tf.all_variables())

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
            _, cost = self.session.run([self.train_func, self.cost], feed_dict=feed_dict)

            it_cost += cost

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

    def train(self, das_file, ttree_file, data_portion=1.0):
        """
        The main training process – initialize and perform a specified number of
        training passes, validating every couple iterations.

        @param das_file: training data file with DAs
        @param ttree_file: training data file with t-trees
        @param data_portion: portion of training data to be actually used, defaults to 1.0
        """

        # load and prepare data and initialize the neural network
        self._init_training(das_file, ttree_file, data_portion)

        # do the training passes
        for iter_no in xrange(1, self.passes + 1):

            self.train_order = range(len(self.train_enc))
            if self.randomize:
                rnd.shuffle(self.train_order)

            self._training_pass(iter_no)

            # validate every couple iterations
            if iter_no % self.validation_freq == 0 and self.validation_size > 0:

                cur_train_out = self.process_das(self.train_das[:self.batch_size])
                log_info("Current train output:\n" +
                         "\n".join([unicode(tree) for tree in cur_train_out]))

                cur_valid_out = self.process_das(self.valid_das)
                cur_cost = self._compute_valid_cost(cur_valid_out, self.valid_trees)
                log_info("Gold validation trees:\n" +
                         "\n".join([unicode(tree) for tree in self.valid_trees]))
                log_info("Current validation output:\n" +
                         "\n".join([unicode(tree) for tree in cur_valid_out]))
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
        if isinstance(valid_trees, tuple):
            valid_trees = [valid_trees_inst for valid_trees_inst in zip(*valid_trees)]
        else:
            valid_trees = [(valid_tree,) for valid_tree in valid_trees]
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
        if isinstance(valid_trees, tuple):
            return np.mean((self._compute_f1(cur_valid_out, valid_trees[0]),
                            self._compute_f1(cur_valid_out, valid_trees[1])))
        evaluator = Evaluator()
        for gold_tree, pred_tree in zip(valid_trees, cur_valid_out):
            evaluator.append(TreeNode(gold_tree), TreeNode(pred_tree))
        return evaluator.f1()

    def save_to_file(self, model_fname):
        """Save the generator to a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph will be stored with a \
            different extension
        """
        log_info("Saving generator to %s..." % model_fname)
        if self.classif_filter:
            classif_filter_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tftreecl\1', model_fname)
            self.classif_filter.save_to_file(classif_filter_fname)

        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self.get_all_settings(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        if hasattr(self, 'checkpoint_path') and self.checkpoint_path:
            shutil.copyfile(self.checkpoint_path, tf_session_fname)
        else:
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
                'classif_filter': self.classif_filter is not None}
        return data

    def _save_checkpoint(self):
        """Save a checkpoint to a temporary path; set `self.checkpoint_path` to the path
        where it is saved; if called repeatedly, will always overwrite the last checkpoint."""
        if not self.checkpoint_path:
            fh, path = tempfile.mkstemp(".ckpt", "tgen-", self.checkpoint_path)
            self.checkpoint_path = path
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

        # re-build TF graph and restore the TF session
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        ret._init_neural_network()
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

        return dec_outputs, dec_cost

    def _init_beam_search(self, enc_inputs):
        """Initialize beam search for the current DA (with the given encoder inputs)."""
        # initial state
        initial_state = np.zeros([1, self.emb_size])
        self._beam_search_feed_dict = {self.initial_state: initial_state}
        # encoder inputs
        for i in xrange(len(enc_inputs)):
            self._beam_search_feed_dict[self.enc_inputs[i]] = enc_inputs[i]

    def _beam_search_step(self, dec_inputs, dec_outputs, dec_states):
        """Run one step of beam search decoding with the given decoder inputs and
        (previous steps') outputs and states."""

        step = len(dec_outputs)  # find the decoder position

        # fill in all previous path data
        for i in xrange(step):
            self._beam_search_feed_dict[self.dec_inputs[i]] = dec_inputs[i]
            self._beam_search_feed_dict[self.outputs[i]] = dec_outputs[i]
            self._beam_search_feed_dict[self.states[i]] = dec_states[i]

        # the decoder outputs are always one step longer
        self._beam_search_feed_dict[self.dec_inputs[step]] = dec_inputs[step]

        # run one step of the decoder
        return self.session.run([self.outputs[step], self.states[step]],
                                feed_dict=self._beam_search_feed_dict)
