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

from tensorflow.models.rnn.seq2seq import embedding_rnn_seq2seq, embedding_attention_seq2seq, \
    sequence_loss
from tensorflow.models.rnn import rnn_cell

from pytreex.core.util import file_stream

from tgen.logf import log_info, log_debug
from tgen.futil import read_das, read_ttrees, trees_from_doc, tokens_from_doc
from tgen.embeddings import EmbeddingExtract
from tgen.rnd import rnd
from tgen.planner import SentencePlanner
from tgen.tree import TreeData, NodeData, TreeNode
from tgen.eval import Evaluator
from tgen.bleu import BLEUMeasure
from tgen.tfclassif import TFTreeClassifier


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


class DAEmbeddingSeq2SeqExtract(EmbeddingExtract):

    UNK_SLOT = 0
    UNK_VALUE = 1
    UNK_ACT = 2

    def __init__(self, cfg={}):
        super(DAEmbeddingSeq2SeqExtract, self).__init__()

        self.dict_act = {'UNK_ACT': self.UNK_ACT}
        self.dict_slot = {'UNK_SLOT': self.UNK_SLOT}
        self.dict_value = {'UNK_VALUE': self.UNK_VALUE}
        self.max_da_len = cfg.get('max_da_len', 10)
        self.sort = cfg.get('sort_da_emb', False)

    def init_dict(self, train_das, dict_ord=None):
        """"""
        if dict_ord is None:
            dict_ord = self.MIN_VALID

        for da in train_das:
            for dai in da:
                if dai.dat not in self.dict_act:
                    self.dict_act[dai.dat] = dict_ord
                    dict_ord += 1
                if dai.name not in self.dict_slot:
                    self.dict_slot[dai.name] = dict_ord
                    dict_ord += 1
                if dai.value not in self.dict_value:
                    self.dict_value[dai.value] = dict_ord
                    dict_ord += 1

        return dict_ord

    def get_embeddings(self, da):
        """"""
        # list the IDs of act types, slots, values
        da_emb_idxs = []
        sorted_da = da
        if hasattr(self, 'sort') and self.sort:
            sorted_da = sorted(da, cmp=lambda a, b:
                               cmp(a.dat, b.dat) or cmp(a.name, b.name) or cmp(a.value, b.value))
        for dai in sorted_da[:self.max_da_len]:
            da_emb_idxs.append(self.dict_act.get(dai.dat, self.UNK_ACT))
            da_emb_idxs.append(self.dict_slot.get(dai.name, self.UNK_SLOT))
            da_emb_idxs.append(self.dict_value.get(dai.value, self.UNK_VALUE))
        # left-pad with unknown
        padding = []
        if len(da) < self.max_da_len:
            padding = [self.UNK_ACT, self.UNK_SLOT, self.UNK_VALUE] * (self.max_da_len - len(da))
        return padding + da_emb_idxs

    def get_embeddings_shape(self):
        """Return the shape of the embedding matrix (for one object, disregarding batches)."""
        return [3 * self.max_da_len]


class TreeEmbeddingSeq2SeqExtract(EmbeddingExtract):
    # TODO try relative parents (good for non-projective, may be bad for non-local) ???

    UNK_T_LEMMA = 0
    UNK_FORMEME = 1
    BR_OPEN = 2
    BR_CLOSE = 3
    GO = 4
    STOP = 5
    VOID = 6
    MIN_VALID = 7

    def __init__(self, cfg={}):
        super(TreeEmbeddingSeq2SeqExtract, self).__init__()

        self.dict_t_lemma = {'UNK_T_LEMMA': self.UNK_T_LEMMA}
        self.dict_formeme = {'UNK_FORMEME': self.UNK_FORMEME}
        self.max_tree_len = cfg.get('max_tree_len', 25)
        self.id_to_string = {self.UNK_T_LEMMA: '<UNK_T_LEMMA>',
                             self.UNK_FORMEME: '<UNK_FORMEME>',
                             self.BR_OPEN: '<(>',
                             self.BR_CLOSE: '<)>',
                             self.GO: '<GO>',
                             self.STOP: '<STOP>',
                             self.VOID: '<>'}

    def init_dict(self, train_trees, dict_ord=None):
        """"""
        if dict_ord is None:
            dict_ord = self.MIN_VALID

        for tree in train_trees:
            for t_lemma, formeme in tree.nodes:
                if t_lemma not in self.dict_t_lemma:
                    self.dict_t_lemma[t_lemma] = dict_ord
                    self.id_to_string[dict_ord] = t_lemma
                    dict_ord += 1
                if formeme not in self.dict_formeme:
                    self.dict_formeme[formeme] = dict_ord
                    self.id_to_string[dict_ord] = formeme
                    dict_ord += 1

        return dict_ord

    def _get_subtree_embeddings(self, tree, root_idx):
        """Bracketed-style embeddings for a projective tree."""
        embs = [self.BR_OPEN]

        for left_child_idx in tree.children_idxs(root_idx, left_only=True):
            embs.extend(self._get_subtree_embeddings(tree, left_child_idx))

        embs.extend([self.dict_t_lemma.get(tree[root_idx].t_lemma, self.UNK_T_LEMMA),
                     self.dict_formeme.get(tree[root_idx].formeme, self.UNK_FORMEME)])

        for right_child_idx in tree.children_idxs(root_idx, right_only=True):
            embs.extend(self._get_subtree_embeddings(tree, right_child_idx))

        embs.append(self.BR_CLOSE)
        return embs

    def get_embeddings(self, tree):
        """Return (list of) embedding (integer) IDs for a tree."""

        # get tree embeddings recursively
        tree_emb_idxs = [self.GO] + self._get_subtree_embeddings(tree, 0) + [self.STOP]

        # right-pad with unknown
        shape = self.get_embeddings_shape()[0]
        padding = [self.VOID] * (shape - len(tree_emb_idxs))

        return tree_emb_idxs + padding

    def ids_to_strings(self, emb):
        """Given embedding IDs, return list of strings where all VOIDs at the end are truncated."""

        # skip VOIDs at the end
        i = len(emb) - 1
        while i > 0 and emb[i] == self.VOID:
            i -= 1

        # convert all IDs to their tokens
        ret = [unicode(self.id_to_string.get(tok_id, '<???>')) for tok_id in emb[:i + 1]]
        return ret

    def ids_to_tree(self, emb):
        """Rebuild a tree from the embeddings (token IDs).

        @param emb: source embeddings (token IDs)
        @return: the corresponding tree
        """

        tree = TreeData()
        tree.nodes = []  # override the technical root -- the tree will be created including the technical root
        tree.parents = []

        # build the tree recursively (start at position 2 to skip the <GO> symbol and 1st opening bracket)
        self._create_subtree(tree, -1, emb, 2)
        return tree

    def _create_subtree(self, tree, parent_idx, emb, pos):
        """Recursive subroutine used for `ids_to_tree()`, do not use otherwise.
        Solves a subtree (starting just after the opening bracket, returning a position
        just after the corresponding closing bracket).

        @param tree: the tree to work on (will be enhanced by the subtree)
        @param parent_idx: the ID of the parent for the current subtree
        @param emb: the source embeddings
        @param pos: starting position in the source embeddings
        @return: the final position used in the current subtree
        """

        if pos >= len(emb):  # avoid running out of the tree (for invalid trees)
            return pos

        node_idx = tree.create_child(parent_idx, len(tree), NodeData(None, None))
        t_lemma = None
        formeme = None

        while pos < len(emb) and emb[pos] not in [self.BR_CLOSE, self.STOP, self.VOID]:

            if emb[pos] == self.BR_OPEN:
                # recurse into subtree
                pos = self._create_subtree(tree, node_idx, emb, pos + 1)

            elif emb[pos] == self.UNK_T_LEMMA:
                if t_lemma is None:
                    t_lemma = self.id_to_string[self.UNK_T_LEMMA]
                pos += 1

            elif emb[pos] == self.UNK_FORMEME:
                if formeme is None:
                    formeme = self.id_to_string[self.UNK_FORMEME]
                pos += 1

            elif emb[pos] >= self.MIN_VALID:
                # remember the t-lemma and formeme for normal nodes
                token = self.id_to_string.get(emb[pos])
                if t_lemma is None:
                    t_lemma = token
                elif formeme is None:
                    formeme = token

                # move the node to its correct position
                # (which we now know it's at the current end of the tree)
                if node_idx != len(tree) - 1:
                    tree.move_node(node_idx, len(tree) - 1)
                    node_idx = len(tree) - 1
                pos += 1

        if pos < len(emb) and emb[pos] == self.BR_CLOSE:
            # skip this closing bracket so that we don't process it next time
            pos += 1

        # fill in the t-lemma and formeme that we've found
        if t_lemma is not None or formeme is not None:
            tree.nodes[node_idx] = NodeData(t_lemma, formeme)

        return pos

    def get_embeddings_shape(self):
        """Return the shape of the embedding matrix (for one object, disregarding batches)."""
        return [4 * self.max_tree_len + 2]


class TokenEmbeddingSeq2SeqExtract(EmbeddingExtract):
    """Extracting token emeddings from a string (array of words)."""

    VOID = 0
    GO = 1
    STOP = 2
    UNK = 3
    PLURAL_S = 4
    MIN_VALID = 5

    def __init__(self, cfg={}):
        self.max_sent_len = cfg.get('max_sent_len', 50)
        self.dict = {'UNK': self.UNK}
        self.rev_dict = {self.VOID: '<VOID>', self.GO: '<GO>',
                         self.STOP: '<STOP>', self.UNK: '<UNK>',
                         self.PLURAL_S: '<-s>'}

    def init_dict(self, train_sents, dict_ord=None):
        """Initialize embedding dictionary (word -> id)."""
        if dict_ord is None:
            dict_ord = self.MIN_VALID

        for sent in train_sents:
            for form, tag in sent:
                if tag == 'NNS' and form.endswith('s'):
                    form = form[:-1]  # TODO this is very stupid, but probably works with BAGEL
                if form not in self.dict:
                    self.dict[form] = dict_ord
                    self.rev_dict[dict_ord] = form
                    dict_ord += 1

        return dict_ord

    def get_embeddings(self, sent):
        """Get the embeddings of a sentence (list of word form/tag pairs)."""
        embs = [self.GO]
        for form, tag in sent:
            add_plural = False
            if tag == 'NNS' and form.endswith('s'):
                add_plural = True
                form = form[-1]
            embs.append(self.dict.get(form, self.UNK))
            if add_plural:
                embs.append(self.PLURAL_S)

        embs += [self.STOP]
        if len(embs) > self.max_sent_len + 2:
            embs = embs[:self.max_sent_len + 2]
        elif len(embs) < self.max_sent_len + 2:
            embs += [self.VOID] * (self.max_sent_len + 2 - len(embs))

        return embs

    def get_embeddings_shape(self):
        """Return the shape of the embedding matrix (for one object, disregarding batches)."""
        return [self.max_sent_len + 2]

    def ids_to_strings(self, emb):
        """Given embedding IDs, return list of strings where all VOIDs at the end are truncated."""

        # skip VOIDs at the end
        i = len(emb) - 1
        while i > 0 and emb[i] == self.VOID:
            i -= 1

        # convert all IDs to their tokens
        ret = [unicode(self.rev_dict.get(tok_id, '<???>')) for tok_id in emb[:i + 1]]

        return ret

    def ids_to_tree(self, emb):
        """Create a fake (flat) t-tree from token embeddings (IDs).

        @param emb: source embeddings (token IDs)
        @return: the corresponding tree
        """

        tree = TreeData()
        tokens = self.ids_to_strings(emb)

        for token in tokens:
            if token in ['<GO>', '<STOP>', '<VOID>']:
                continue
            if token == '<-s>':
                tree.nodes[-1] = NodeData(tree.nodes[-1].t_lemma + 's', 'x')
            else:
                tree.create_child(0, len(tree), NodeData(token, 'x'))

        return tree


class Seq2SeqGen(SentencePlanner):
    """A sequence-to-sequence generator (using encoder-decoder architecture from TensorFlow)."""

    def __init__(self, cfg):
        """Initialize the generator, fill in the configuration."""

        super(Seq2SeqGen, self).__init__(cfg)

        # save the whole configuration for later use (save/load, construction of embedding
        # extractors)
        self.cfg = cfg
        # extract the individual elements out of it
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

        self.beam_size = cfg.get('beam_size', 1)

        self.classif_filter = None
        if 'classif_filter' in cfg:
            self.classif_filter = TFTreeClassifier(cfg['classif_filter'])
            self.misfit_penalty = cfg.get('misfit_penalty', 100)

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
        with tf.variable_scope("seq2seq_gen") as scope:

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

        # initial state
        initial_state = np.zeros([len(enc_inputs[0]), self.emb_size])
        feed_dict = {self.initial_state: initial_state}

        for i in xrange(len(enc_inputs)):
            feed_dict[self.enc_inputs[i]] = enc_inputs[i]

        # decoder inputs (either fake, or true but used just for cost computation)
        if gold_trees is None:
            empty_tree_emb = self.tree_embs.get_embeddings(TreeData())
            dec_inputs = cut_batch_into_steps([empty_tree_emb for _ in enc_inputs[0]])
        else:
            dec_inputs = cut_batch_into_steps([self.tree_embs.get_embeddings(tree)
                                               for tree in gold_trees])
        for i in xrange(len(dec_inputs)):
            feed_dict[self.dec_inputs[i]] = dec_inputs[i]

        feed_dict[self.targets[-1]] = len(enc_inputs[0]) * [self.tree_embs.VOID]

        # run the decoding
        if gold_trees is None:
            dec_outputs = self.session.run(self.dec_outputs, feed_dict=feed_dict)
            dec_cost = None
        else:
            res = self.session.run(self.dec_outputs + [self.dec_cost], feed_dict=feed_dict)
            dec_outputs = res[:-1]
            dec_cost = res[-1]

        # convert the output back into a tree
        dec_output_ids = np.argmax(dec_outputs, axis=2)
        return dec_output_ids, dec_cost

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

        # true "batches" not implemented yet
        assert len(enc_inputs[0]) == 1

        log_debug("GREEDY DEC WOULD RETURN:\n" +
                  " ".join(self.tree_embs.ids_to_strings(
                      [out_tok[0] for out_tok in self._greedy_decoding(enc_inputs, None)[0]])))

        # initial state
        initial_state = np.zeros([1, self.emb_size])
        feed_dict = {self.initial_state: initial_state}

        for i in xrange(len(enc_inputs)):
            feed_dict[self.enc_inputs[i]] = enc_inputs[i]

        empty_tree_emb = self.tree_embs.get_embeddings(TreeData())
        dec_inputs = cut_batch_into_steps([empty_tree_emb])

        paths = [self.DecodingPath(dec_inputs=[dec_inputs[0]])]

        for step in xrange(len(dec_inputs)):

            new_paths = []

            for path in paths:

                for i in xrange(step):
                    feed_dict[self.dec_inputs[i]] = path.dec_inputs[i]
                    feed_dict[self.outputs[i]] = path.dec_outputs[i]
                    feed_dict[self.states[i]] = path.dec_states[i]

                feed_dict[self.dec_inputs[step]] = path.dec_inputs[step]
                out, st = self.session.run([self.outputs[step], self.states[step]],
                                           feed_dict=feed_dict)

                new_paths.extend(path.expand(self.beam_size, out, st))

            paths = sorted(new_paths, reverse=True)[:self.beam_size]

            if all([p.dec_inputs[-1] == self.tree_embs.VOID for p in paths]):
                break  # stop decoding if we have reached the end in all paths

            log_debug(("\nBEAM SEARCH STEP %d\n" % step) +
                      "\n".join([("%f\t" % p.logprob) +
                                 " ".join(self.tree_embs.ids_to_strings([inp[0] for inp in p.dec_inputs]))
                                 for p in paths]) + "\n")

        if self.classif_filter:  # filter out paths that are
            paths = self._filter_paths(paths, da)

        return np.array(paths[0].dec_inputs)

    def _filter_paths(self, paths, da):

        trees = [self.tree_embs.ids_to_tree(np.array(path.dec_inputs).transpose()[0])
                 for path in paths]
        self.classif_filter.init_run(da)
        fits = self.classif_filter.dist_to_cur_da(trees)
        # add distances to logprob so that non-fitting will be heavily penalized
        for path, fit in zip(paths, fits):
            path.logprob -= self.misfit_penalty * fit
        return sorted(paths, reverse=True)

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
        # TODO: better stopping criterion than just # of passes
        # look at the tagger – they're much better -- using performance on development data
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
            data = {'cfg': self.cfg,
                    'da_embs': self.da_embs,
                    'tree_embs': self.tree_embs,
                    'da_dict_size': self.da_dict_size,
                    'tree_dict_size': self.tree_dict_size,
                    'max_da_len': self.max_da_len,
                    'max_tree_len': self.max_tree_len,
                    'classif_filter': self.classif_filter is not None}
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        if self.checkpoint_path:
            shutil.copyfile(self.checkpoint_path, tf_session_fname)
        else:
            self.saver.save(self.session, tf_session_fname)

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
            ret.__dict__.update(data)

        if ret.classif_filter:
            classif_filter_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tftreecl\1', model_fname)
            ret.classif_filter = TFTreeClassifier.load_from_file(classif_filter_fname)

        # re-build TF graph and restore the TF session
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        ret._init_neural_network()
        ret.saver.restore(ret.session, tf_session_fname)

        return ret

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
