#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import re
import numpy as np
import tensorflow as tf
import cPickle as pickle
from itertools import izip_longest
import sys

from tensorflow.models.rnn.seq2seq import embedding_rnn_seq2seq, embedding_attention_seq2seq, \
    sequence_loss
from tensorflow.models.rnn import rnn_cell

from pytreex.core.util import file_stream

from tgen.logf import log_info, log_debug
from tgen.futil import read_das, read_ttrees, trees_from_doc, tokens_from_doc
from tgen.embeddings import EmbeddingExtract
from tgen.rnd import rnd
from tgen.planner import SentencePlanner
from tgen.tree import TreeData, NodeData


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
        for dai in da[:self.max_da_len]:
            da_emb_idxs.append(self.dict_slot.get(dai.dat, self.UNK_ACT))
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
    """Abstract ancestor of embedding extraction classes."""

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

        # TODO fix configuration
        self.emb_size = cfg.get('emb_size', 50)
        self.batch_size = cfg.get('batch_size', 10)
        self.passes = cfg.get('passes', 5)
        self.alpha = cfg.get('alpha', 1e-3)
        self.validation_size = cfg.get('validation_size', 0)
        self.validation_freq = cfg.get('validation_freq', 10)
        self.max_cores = cfg.get('max_cores')
        self.use_tokens = cfg.get('use_tokens', False)
        self.nn_type = cfg.get('nn_type', 'seq2seq_emb')
        self.randomize = True

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
            # check if there are 2 copies of input DAs in the training data. if so, remove both
            # copies of validation DAs from training
            cut_dbl = self.train_das[train_size / 2 - 1] == self.train_das[-1]
            if cut_dbl:
                log_info('Detected duplicate DAs in training data: removing both copies for validation')
            self.train_trees, self.valid_trees = self._cut_valid_data(self.train_trees, cut_dbl)
            self.train_das, self.valid_das = self._cut_valid_data(self.train_das, cut_dbl)

        log_info('Using %d training, %d validation instances.' %
                 (len(self.train_das), self.validation_size))

        # initialize embeddings
        self.da_embs = DAEmbeddingSeq2SeqExtract()
        if self.use_tokens:
            self.tree_embs = TokenEmbeddingSeq2SeqExtract()
        else:
            self.tree_embs = TreeEmbeddingSeq2SeqExtract()
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

        # build the NN
        self._init_neural_network()

        # initialize the NN variables
        self.session.run(tf.initialize_all_variables())

    def _cut_valid_data(self, insts, cut_double):
        """Put aside part of the training set  for validation.

        @param insts: original training set (DAs, tokens, or trees)
        @param cut_double: remove two copies of the training instances (in case the training \
            data contain two copies of all input DAs -- this holds for the BAGEL set)
        @return: new training and validation sets
        """
        train_size = len(insts)
        valid = insts[-self.validation_size:]
        train = insts[:-self.validation_size]
        if cut_double:
            train = train[:train_size / 2 - self.validation_size] + train[train_size / 2:]
        return train, valid

    def _init_neural_network(self):
        """Initializing the NN (building a TensorFlow graph and initializing session)."""

        # set TensorFlow random seed
        tf.set_random_seed(rnd.randint(-sys.maxint, sys.maxint))

        # create placeholders for input & output (always batch-size * 1, list of up to num. steps)
        self.enc_inputs = []
        for i in xrange(self.max_da_len):
            self.enc_inputs.append(tf.placeholder(tf.int32, [None], name=('enc_inp-%d' % i)))
        self.dec_inputs = []
        for i in xrange(self.max_tree_len):
            self.dec_inputs.append(tf.placeholder(tf.int32, [None], name=('dec_inp-%d' % i)))

        # targets are just decoder inputs shifted by one (+pad with one empty spot)
        self.targets = [self.dec_inputs[i + 1] for i in xrange(len(self.dec_inputs) - 1)]
        self.targets.append(tf.placeholder(tf.int32, [None], name=('target-pad')))

        # prepare building blocks
        # TODO change dimension when BasicLSTMCell is replaced
        self.initial_state = tf.placeholder(tf.float32, [None, self.emb_size])
        self.cell = rnn_cell.BasicLSTMCell(self.emb_size)

        # build the actual LSTM Seq2Seq network (for training and decoding)
        with tf.variable_scope("seq2seq_gen") as scope:

            rnn_func = embedding_rnn_seq2seq
            if self.nn_type == 'emb_attention_seq2seq':
                rnn_func = embedding_attention_seq2seq

            # for training: feed_previous == False
            # outputs = batch_size * num_decoder_symbols ~ i.e. output logits at each steps
            # states = cell states at each steps
            self.outputs, self.states = rnn_func(
                self.enc_inputs, self.dec_inputs, self.cell,
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
        self.cost = sequence_loss(self.outputs, self.targets, self.cost_weights, self.tree_dict_size)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(self.alpha)
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

        for batch_no in self.train_order:

            # feed data into the TF session:

            # initial state
            initial_state = np.zeros([self.batch_size, self.emb_size])
            feed_dict = {self.initial_state: initial_state}

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

    def process_das(self, das):
        """
        Process a list of input DAs, return the corresponding trees (using the generator
        network with current parameters).

        @param das: input DAs
        @return: generated trees as `TreeData` instances
        """

        # initial state
        initial_state = np.zeros([len(das), self.emb_size])
        feed_dict = {self.initial_state: initial_state}

        # encoder inputs
        enc_inputs = cut_batch_into_steps([self.da_embs.get_embeddings(da)
                                           for da in das])
        for i in xrange(len(enc_inputs)):
            feed_dict[self.enc_inputs[i]] = enc_inputs[i]

        # (fake) decoder inputs
        empty_tree_emb = self.tree_embs.get_embeddings(TreeData())
        dec_inputs = cut_batch_into_steps([empty_tree_emb for _ in das])
        for i in xrange(len(dec_inputs)):
            feed_dict[self.dec_inputs[i]] = dec_inputs[i]

        feed_dict[self.targets[-1]] = len(das) * [self.tree_embs.VOID]

        # run the decoding
        dec_outputs = self.session.run(self.dec_outputs, feed_dict=feed_dict)

        # convert the output back into a tree
        dec_output_ids = np.argmax(dec_outputs, axis=2)
        return [self.tree_embs.ids_to_tree(ids) for ids in dec_output_ids.transpose()]

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

            # validate every couple iteration
            if iter_no % self.validation_freq == 0:
                cur_out = self.process_das(self.train_das[:self.batch_size])
                log_info("Current train output:\n" + "\n".join([unicode(tree) for tree in cur_out]))
                if self.validation_size > 0:
                    cur_out = self.process_das(self.valid_das)
                    log_info("Current validation output:\n" + "\n".join([unicode(tree) for tree in cur_out]))

    def save_to_file(self, model_fname):
        """Save the generator to a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph will be stored with a \
            different extension
        """
        log_info("Saving generator to %s..." % model_fname)
        with file_stream(model_fname, 'wb', encoding=None) as fh:
            data = {'emb_size': self.emb_size,
                    'batch_size': self.batch_size,
                    'randomize': self.randomize,
                    'passes': self.passes,
                    'da_embs': self.da_embs,
                    'tree_embs': self.tree_embs,
                    'da_dict_size': self.da_dict_size,
                    'tree_dict_size': self.tree_dict_size,
                    'max_da_len': self.max_da_len,
                    'max_tree_len': self.max_tree_len,
                    'alpha': self.alpha,
                    'max_cores': self.max_cores,
                    'use_tokens': self.use_tokens,
                    'nn_type': self.nn_type,
                    }
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        self.saver.save(self.session, tf_session_fname)

    @staticmethod
    def load_from_file(model_fname):
        """Load the generator from a file (actually two files, one for configuration and one
        for the TensorFlow graph, which must be stored separately).

        @param model_fname: file name (for the configuration file); TF graph must be stored with a \
            different extension
        """
        log_info("Loading generator from %s..." % model_fname)
        ret = Seq2SeqGen(cfg={})
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            data = pickle.load(fh)
            ret.__dict__.update(data)

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
        tree = self.process_das([da])[0]
        log_debug("RESULT: %s" % unicode(tree))
        # if requested, append the result to the document
        if gen_doc:
            zone = self.get_target_zone(gen_doc)
            zone.ttree = tree.create_ttree()
            zone.sentence = unicode(da)
        # return the result
        return tree
