#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lexicalization functions (postprocessing generated trees).
"""

from __future__ import unicode_literals
import json
import re
import cPickle as pickle
import numpy as np
import tempfile
import time
import datetime
import shutil
import codecs
import operator
from subprocess import Popen, PIPE
import tensorflow as tf
import sys
import math
import kenlm

from tgen.tree import NodeData, TreeData
from tgen.rnd import rnd
from tgen.futil import file_stream, read_absts, smart_load_absts
from tgen.logf import log_warn, log_info, log_debug
from tgen.tf_ml import TFModel
from tgen.ml import softmax
import tgen.externals.seq2seq as tf06s2s


class FormSelect(object):
    """Interface for all surface form selector classes."""

    def __init__(self, cfg=None):
        pass

    def get_surface_form(self, sentence, pos, possible_forms):
        """Return a suitable surface form for the given position in the given sentence,
        selecting from a list of possible forms.

        @param sentence: input sentence (a list of tokens)
        @param pos: position in the sentence where the form should be selected
        @param possible_forms: list of possible surface forms for this position
        @return: the selected surface form (string)
        """
        raise NotImplementedError()


class RandomFormSelect(FormSelect):
    """A dummy form surface selector, selecting a form at random from the offered choice."""

    def __init__(self, cfg=None):
        super(RandomFormSelect, self).__init__(cfg)

    def get_surface_form(self, sentence, pos, possible_forms):
        return rnd.choice(possible_forms)


class FrequencyFormSelect(FormSelect):
    """A very simple surface form selector, picking the form that has been seen most frequently
    in the training data (or sampling along the frequency distribution in the training data).
    It disregards any context completely."""

    def __init__(self, cfg):
        super(FrequencyFormSelect, self).__init__(cfg)
        self._sample = cfg.get('form_sample', False)
        self._word_freq = None
        np.random.seed(rnd.randint(0, 2**32 - 1))

    def train(self, train_sents, valid_sents=None):
        """Train the model (memorize the frequency of the various forms seen in the training data.
        Validation data are ignored (parameter only used for compatibility).
        @param train_sents: training sentences (list of lists of tokens)
        @param valid_sents: unused
        """
        self._word_freq = {}
        for sent in train_sents:
            for tok in sent:
                tok = tok.lower()
                self._word_freq[tok] = self._word_freq.get(tok, 0) + 1

    def load_model(self, model_fname_pattern):
        model_fname = re.sub(r'(.pickle)?(.gz)?$', '.wfreq', model_fname_pattern)
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            self._word_freq = pickle.load(fh)

    def save_model(self, model_fname_pattern):
        if not self._word_freq:
            log_warn('No lexicalizer model trained, skipping saving!')
        model_fname = re.sub(r'(.pickle)?(.gz)?$', '.wfreq', model_fname_pattern)
        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self._word_freq, fh, pickle.HIGHEST_PROTOCOL)

    def get_surface_form(self, sentence, pos, possible_forms):
        scores = [self._word_freq.get(possible_form.lower(), 0) + 0.1
                  for possible_form in possible_forms]
        if self._sample:
            probs = softmax(scores)
            return np.random.choice(possible_forms, p=probs)
        max_idx, _ = max(enumerate(scores), key=operator.itemgetter(1))
        return possible_forms[max_idx]


class KenLMFormSelect(FormSelect):
    """A surface form selector based on KenLM n-gram language models, selecting according
    to language model scores (or sampling from the corresponding softmax distribution)."""

    def __init__(self, cfg):
        super(KenLMFormSelect, self).__init__(cfg)
        self._sample = cfg.get('form_sample', False)
        self._trained_model = None
        np.random.seed(rnd.randint(0, 2**32 - 1))

    def get_surface_form(self, sentence, pos, possible_forms):
        state, dummy_state = kenlm.State(), kenlm.State()
        self._lm.BeginSentenceWrite(state)
        for idx in xrange(pos):
            self._lm.BaseScore(state, sentence[idx].encode('utf-8'), state)
        best_form_idx = 0
        best_score = float('-inf')
        scores = []
        for form_idx, possible_form in enumerate(possible_forms):
            possible_form = possible_form.lower().replace(' ', '^').encode('utf-8')
            score = self._lm.BaseScore(state, possible_form, dummy_state)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_form_idx = form_idx
        if self._sample:
            probs = softmax(scores)
            return np.random.choice(possible_forms, p=probs)
        return possible_forms[best_form_idx]

    def load_model(self, model_fname_pattern):
        model_fname = re.sub(r'(.pickle)?(.gz)?$', '.kenlm.bin', model_fname_pattern)
        self._lm = kenlm.Model(model_fname)
        self._trained_model = model_fname

    def save_model(self, model_fname_pattern):
        if not self._trained_model:
            log_warn('No lexicalizer model trained, skipping saving!')

        model_fname = re.sub(r'(\.pickle)?(\.gz)?$', '.kenlm.bin', model_fname_pattern)
        shutil.copyfile(self._trained_model, model_fname)

    def train(self, train_sents, valid_sents=None):
        """Train the model, running KenLM over the training sentences.
        Validation data are ignored (parameter only used for compatibility).
        @param train_sents: training sentences (list of lists of tokens)
        @param valid_sents: unused
        """
        # create tempfile for output
        tmpfile, tmppath = tempfile.mkstemp(".kenlm.bin", "formselect-")
        # create processes
        lmplz = Popen(['lmplz', '-o', '5', '-S', '2G'], stdin=PIPE, stdout=PIPE)
        binarize = Popen(['build_binary', '/dev/stdin', tmppath], stdin=lmplz.stdout)
        # feed input data
        lmplz_stdin = codecs.getwriter('UTF-8')(lmplz.stdin)
        for sent in train_sents:
            lmplz_stdin.write(' '.join([tok.lower().replace(' ', '^') for tok in sent]) + "\n")
        lmplz_stdin.close()
        # wait for the process to complete
        binarize.wait()
        if binarize.returncode != 0:
            raise RuntimeError("LM build failed (error code: %d)" % binarize.returncode)
        # load the output into memory (extension will be filled in again)
        self.load_model(re.sub('\.kenlm\.bin$', '', tmppath))


class RNNLMFormSelect(FormSelect, TFModel):
    """A form selector based on RNN LM (with configurable properties; with the option of
    sampling from the RNN LM predictions)."""

    VOID = 0
    GO = 1
    STOP = 2
    UNK = 3
    MIN_VALID = 4

    def __init__(self, cfg):
        FormSelect.__init__(self)
        TFModel.__init__(self, scope_name='formselect-' + cfg.get('scope_suffix', ''))
        # load configuration
        self._sample = cfg.get('form_sample', False)

        self.randomize = cfg.get('randomize', True)
        self.emb_size = cfg.get('emb_size', 50)
        self.passes = cfg.get('passes', 200)
        self.alpha = cfg.get('alpha', 1)
        self.batch_size = cfg.get('batch_size', 1)
        self.max_sent_len = cfg.get('max_sent_len', 32)
        self.cell_type = cfg.get('cell_type', 'lstm')
        self.max_grad_norm = cfg.get('max_grad_norm', 100)
        self.optimizer_type = cfg.get('optimizer_type', 'adam')
        self.max_cores = cfg.get('max_cores', 4)
        self.alpha_decay = cfg.get('alpha_decay', 0.0)
        self.validation_freq = cfg.get('validation_freq', 1)
        self.min_passes = cfg.get('min_passes', self.passes / 2)
        self.vocab = {'<VOID>': self.VOID, '<GO>': self.GO,
                      '<STOP>': self.STOP, '<UNK>': self.UNK}
        self.reverse_dict = {self.VOID: '<VOID>', self.GO: '<GO>',
                             self.STOP: '<STOP>', self.UNK: '<UNK>'}
        self.vocab_size = None
        self._checkpoint_params = None
        self._checkpoint_settings = None
        np.random.seed(rnd.randint(0, 2**32 - 1))
        tf.set_random_seed(rnd.randint(-sys.maxint, sys.maxint))

    def _init_training(self, train_sents, valid_sents=None):
        """Initialize training (prepare vocabulary, prepare training batches), initialize the
        RNN.
        @param train_sents: training data (list of lists of tokens, lexicalized)
        @param valid_sents: validation data (list of lists of tokens, lexicalized)
        """
        # initialize embeddings
        dict_ord = self.MIN_VALID
        for sent in train_sents:
            for tok in sent:
                tok = tok.lower()
                if tok not in self.vocab:
                    self.vocab[tok] = dict_ord
                    self.reverse_dict[dict_ord] = tok
                    dict_ord += 1
        self.vocab_size = dict_ord
        # prepare training data
        self._train_data = []
        for sent in train_sents:
            self._train_data.append(self._sent_to_ids(sent))
        self._valid_data = []
        if valid_sents:
            for sent in valid_sents:
                self._valid_data.append(self._sent_to_ids(sent))
        self._init_neural_network()
        self.session.run(tf.global_variables_initializer())

    def _sent_to_ids(self, sent):
        """Convert tokens in a sentence to integer IDs to be used as an input to the RNN.
        Pad with "<VOID>" to maximum sentence length.
        @param sent: input sentence (list of string tokens)
        @return: list of IDs corresponding to the input tokens, padded to RNN length
        """
        ids = [self.vocab.get('<GO>')]
        ids += [self.vocab.get(tok.lower(), self.vocab.get('<UNK>')) for tok in sent]
        ids = ids[:self.max_sent_len]
        ids += [self.vocab.get('<STOP>')]
        # actually using max_sent_len + 1 (inputs exclude last step, targets exclude the 1st)
        ids += [self.vocab.get('<VOID>') for _ in xrange(self.max_sent_len - len(ids) + 1)]
        return ids

    def _train_batches(self):
        """Create batches from the input; use as iterator."""
        for batch_start in xrange(0, len(self._train_order), self.batch_size):
            sents = [self._train_data[idx]
                     for idx in self._train_order[batch_start: batch_start + self.batch_size]]
            inputs = np.array([sent[:-1] for sent in sents], dtype=np.int32)
            targets = np.array([sent[1:] for sent in sents], dtype=np.int32)
            yield inputs, targets

    def _valid_batches(self):
        for batch_start in xrange(0, len(self._valid_data), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(self._valid_data))
            sents = [self._valid_data[idx] for idx in xrange(batch_start, batch_end)]
            inputs = np.array([sent[:-1] for sent in sents], dtype=np.int32)
            targets = np.array([sent[1:] for sent in sents], dtype=np.int32)
            yield inputs, targets

    def _init_neural_network(self):
        """Initialize the RNNLM network."""

        with tf.variable_scope(self.scope_name):
            # TODO dropout
            # I/O placeholders
            self._inputs = tf.placeholder(tf.int32, [None, self.max_sent_len], name='inputs')
            self._targets = tf.placeholder(tf.int32, [None, self.max_sent_len], name='targets')

            # RNN cell type
            if self.cell_type.startswith('gru'):
                self._cell = tf.contrib.rnn.GRUCell(self.emb_size)
            else:
                self._cell = tf.contrib.rnn.BasicLSTMCell(self.emb_size)
            if re.match(r'/[0-9]$', self.cell_type):
                self._cell = tf.contrib.rnn.MultiRNNCell([self.cell] * int(self.cell_type[-1]))
            self._initial_state = self._cell.zero_state(tf.shape(self._inputs)[0], tf.float32)

            # embeddings
            emb_cell = tf.contrib.rnn.EmbeddingWrapper(self._cell, self.vocab_size)
            # RNN encoder
            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(axis=1, num_or_size_splits=self.max_sent_len, value=self._inputs)]
            outputs, states = tf.contrib.rnn.static_rnn(emb_cell, inputs, initial_state=self._initial_state)

            # output layer
            output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.emb_size])
            self._logits = (tf.matmul(output,
                                      tf.get_variable("W", [self.emb_size, self.vocab_size])) +
                            tf.get_variable("b", [self.vocab_size]))

            # cost
            targets_1d = tf.reshape(self._targets, [-1])
            self._loss = tf06s2s.sequence_loss_by_example(
                    [self._logits], [targets_1d],
                    [tf.ones_like(targets_1d, dtype=tf.float32)], self.vocab_size)
            self._cost = tf.reduce_mean(self._loss)

            # optimizer
            self._learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            if self.optimizer_type == 'sgd':
                opt = tf.train.GradientDescentOptimizer(self._learning_rate)
            if self.optimizer_type == 'adagrad':
                opt = tf.train.AdagradOptimizer(self._learning_rate)
            else:
                opt = tf.train.AdamOptimizer(self._learning_rate)

            # gradient clipping
            grads_tvars = opt.compute_gradients(self._loss, tf.trainable_variables())
            grads, _ = tf.clip_by_global_norm([g for g, _ in grads_tvars], self.max_grad_norm)
            self._train_func = opt.apply_gradients(zip(grads, [v for _, v in grads_tvars]))

        # initialize TF session
        session_config = None
        if self.max_cores:
            session_config = tf.ConfigProto(inter_op_parallelism_threads=self.max_cores,
					    intra_op_parallelism_threads=self.max_cores)
	self.session = tf.Session(config=session_config)

    def get_all_settings(self):
        return {'vocab': self.vocab,
                'reverse_dict': self.reverse_dict,
                'vocab_size': self.vocab_size}

    def train(self, train_sents, valid_sents=None):
        """Train the RNNLM on the given data (list of lists of tokens).
        @param train_sents: training data (list of lists of tokens, lexicalized)
        @param valid_sents: validation data (list of lists of tokens, lexicalized, may be None \
            if no validation should be performed)
        """
        self._init_training(train_sents, valid_sents)

        top_perp = float('nan')

        for iter_no in xrange(1, self.passes + 1):
            # preparing parameters
            iter_alpha = self.alpha * np.exp(-self.alpha_decay * iter_no)
            self._train_order = range(len(self._train_data))
            if self.randomize:
                rnd.shuffle(self._train_order)
            # training
            self._training_pass(iter_no, iter_alpha)

            # validation
            if (self.validation_freq and iter_no > self.min_passes and
                    iter_no % self.validation_freq == 0):
                perp = self._valid_perplexity()
                log_info("Perplexity: %.3f" % perp)
                # if we have the best model so far, save it as a checkpoint (overwrite previous)
                if math.isnan(top_perp) or perp < top_perp:
                    top_perp = perp
                    self._save_checkpoint()

        self._restore_checkpoint()  # restore the best parameters so far

    def _training_pass(self, pass_no, pass_alpha):
        """Run one pass over the training data (epoch).
        @param pass_no: pass number (for logging)
        @param pass_alpha: learning rate for the current pass
        """
        pass_start_time = time.time()
        pass_cost = 0
        for inputs, targets in self._train_batches():
            cost, _ = self.session.run([self._cost, self._train_func],
                                       {self._inputs: inputs,
                                        self._targets: targets,
                                        self._learning_rate: pass_alpha, })
            pass_cost += cost
        duration = str(datetime.timedelta(seconds=(time.time() - pass_start_time)))
        log_info("Pass %d: alpha %.3f, duration %s, cost %.3f" % (pass_no, pass_alpha,
                                                                  duration, pass_cost))
        return pass_cost

    def load_model(self, model_fname_pattern):
        """Load the RNNLM model from a file."""
        model_fname = re.sub(r'(.pickle)?(.gz)?$', '.rnnlm', model_fname_pattern)
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            self.load_all_settings(pickle.load(fh))
            self._init_neural_network()
            self.set_model_params(pickle.load(fh))

    def save_model(self, model_fname_pattern):
        """Save the RNNLM model to a file."""
        model_fname = re.sub(r'(.pickle)?(.gz)?$', '.rnnlm', model_fname_pattern)
        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self.get_all_settings(), fh, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.get_model_params(), fh, pickle.HIGHEST_PROTOCOL)

    def _valid_perplexity(self):
        """Compute perplexity of the RNNLM on validation data."""
        perp = 0
        n_toks = 0
        for inputs, targets in self._valid_batches():
            logits = self.session.run([self._logits], {self._inputs: inputs})[0]
            probs = softmax(logits)  # logits combine all sentences behind each other -- dimension
                                     # is (self.max_sent_len * self.batch_size, self.vocab_size)
            for tok_no in xrange(len(probs)):
                perp += np.log2(probs[tok_no, targets[tok_no / self.max_sent_len,
                                                      tok_no % self.max_sent_len]])
            n_toks += np.prod(inputs.shape)
        # perp = exp( -1/N * sum_i=1^N log p(x_i) )
        return np.exp2(- perp / float(n_toks))

    def _save_checkpoint(self):
        """Store current model parameters in memory."""
        self._checkpoint_settings = self.get_all_settings()
        self._checkpoint_params = self.get_model_params()

    def _restore_checkpoint(self):
        """Retrieve previously stored model parameters from memory (or do nothing if there are
        no stored parameters."""
        if not self._checkpoint_params:
            return
        self.load_all_settings(self._checkpoint_settings)
        self.set_model_params(self._checkpoint_params)

    def get_surface_form(self, sentence, pos, possible_forms):
        log_debug("Pos: %d, forms: %s" % (pos, unicode(", ".join(possible_forms))))
        # get unnormalized scores for the whole vocabulary
        if pos >= self.max_sent_len:  # don't use whole sentence if it's too long
            pos -= pos - self.max_sent_len + 1
            sentence = sentence[pos - self.max_sent_len + 1:]
        inputs = np.array([self._sent_to_ids(sentence)[:-1]], dtype=np.int32)
        logits = self.session.run([self._logits], {self._inputs: inputs})
        # pick out scores for possible forms
        scores = [logits[0][pos][self.vocab.get(form.lower(), self.vocab.get('<UNK>'))]
                  for form in possible_forms]
        probs = softmax(scores)
        log_debug("Vocab: %s" % unicode(", ".join([unicode(self.vocab.get(form.lower(),
                                                                          self.vocab.get('<UNK>')))
                                                   for f in possible_forms])))
        log_debug("Scores: %s, Probs: %s" % (unicode(", ".join(["%.3f" % s for s in scores])),
                                             unicode(", ".join(["%.3f" % p for p in probs]))))
        # sample from the prob. dist.
        if self._sample:
            return np.random.choice(possible_forms, p=probs)
        # get just the most probable option
        max_idx, _ = max(enumerate(probs), key=operator.itemgetter(1))
        return possible_forms[max_idx]


class Lexicalizer(object):
    """Main object controlling lexicalization, using a LM or random surface forms to replace
    slot placeholders in the outputs."""

    OR_STRING = {'cs': 'nebo', 'en': 'or'}
    AND_STRING = {'cs': 'a', 'en': 'and'}

    def __init__(self, cfg):
        """Read configuration, initialize internal buffers, create the surface form selection
        LM object (if required)."""
        self.cfg = cfg
        self.mode = cfg.get('mode', 'trees')
        self.language = cfg.get('language', 'en')
        self._sf_all = {}
        self._sf_by_formeme = {}
        self._sf_by_tag = {}
        self._lemma_for_sf = {}
        self._form_select = RandomFormSelect()
        if 'form_select_type' in cfg:
            if cfg['form_select_type'] == 'frequency':
                self._form_select = FrequencyFormSelect(cfg)
            elif cfg['form_select_type'] == 'kenlm':
                self._form_select = KenLMFormSelect(cfg)
            elif cfg['form_select_type'] == 'rnnlm':
                self._form_select = RNNLMFormSelect(cfg)

    def train(self, fnames, train_trees, valid_trees=None):
        """Train the lexicalizer (including its LM, if applicable).
        @param fnames: file names for surface forms (JSON) and training data lexicalization \
            instructions
        @param train_trees: loaded generator training data (TreeData trees/lists of lemma-tag \
            or form-tag pairs)
        """
        log_info('Training lexicalizer...')
        if not fnames:
            return
        valid_abst_fname = None
        if ',' in fnames:
            fnames = fnames.split(',')
            if len(fnames) == 3:
                surface_forms_fname, train_abst_fname, valid_abst_fname = fnames
            else:
                surface_forms_fname, train_abst_fname = fnames
        else:
            surface_forms_fname, train_abst_fname = fnames, None

        self.load_surface_forms(surface_forms_fname)
        if train_abst_fname and not isinstance(self._form_select, RandomFormSelect):
            log_info('Training lexicalization LM from training trees and %s...' % train_abst_fname)
            self._form_select.train(*self._prepare_train_toks(train_trees, train_abst_fname,
                                                              valid_trees, valid_abst_fname))

    def _first_abst(self, absts, slot):
        """Get 1st abstraction instruction for a specific slot in the list, put it back to
        the end of the list. If there is no matching abstraction instruction, return None."""
        try:
            i, abst = ((i, a) for i, a in enumerate(absts) if a.slot == slot).next()
            del absts[i]
            absts.append(abst)
            return abst
        except StopIteration:
            return None

    def _prepare_train_toks(self, train_trees, train_abstr_fname,
                            valid_trees=None, valid_abstr_fname=None):
        """Prepare training data for form selection LM. Use training trees/tagged lemmas/tokens,
        apply lexicalization instructions including surface forms, and convert the output to a
        list of lists of tokens (sentences).
        @param train_trees: main generator training data (trees, tagged lemmas, tokens)
        @param train_abstr_fname: file name for the corresponding lexicalization instructions
        @return: list of lists of LM training tokens (lexicalized)
        """
        # load abstraction file
        abstss = read_absts(train_abstr_fname)
        if valid_abstr_fname is not None:
            abstss.extend(read_absts(valid_abstr_fname))
        # concatenate training + validation data (will be handled in the same way)
        trees = list(train_trees)
        if valid_trees is not None:
            trees.extend(valid_trees)
        out_sents = []
        for tree, absts in zip(trees, abstss):
            # validation data may have more paraphrases -> treat them separately
            # (list of lists or list of TreeData's for self.mode == 'tree')
            if isinstance(tree[-1], (list, TreeData)):
                for tree_ in tree:
                    out_sents.append(self._tree_to_sentence(tree_, absts))
            # default: one paraphrase
            else:
                out_sents.append(self._tree_to_sentence(tree, absts))
        # split training/validation data
        return out_sents[:len(train_trees)], out_sents[len(train_trees):]

    def _tree_to_sentence(self, tree, absts=None):
        """Convert the given "tree" (i.e., tree, tagged lemmas, or tokens; represented as TreeData
        or a list of tuples form/lemma-tag) into a sentence (i.e., always a plain list of tokens)
        and lexicalize it if required.
        @param tree: input sentence -- a tree, tagged lemmas, or tokens (as TreeData or lists of \
                tuples)
        @param absts: list of abstraction instructions for the given sentence (not used if None)
        @return: list of lexicalized tokens in the sentence
        """
        # create the sentence as a list of tokens
        out_sent = []
        if self.mode == 'trees':
            for node in tree.nodes[1:]:
                out_sent.append(node.t_lemma if node.t_lemma is not None else '<None>')
                out_sent.append(node.formeme if node.formeme is not None else '<None>')
        else:  # tokens or tagged lemmas are stored in t_lemmas
            if isinstance(tree, TreeData):
                for node in tree.nodes[1:]:
                    out_sent.append(node.t_lemma)
            else:
                if self.mode == 'tagged_lemmas':
                    for lemma, tag in tree:
                        out_sent.append(lemma)
                        out_sent.append(tag)
                else:
                    out_sent = [form for form, _ in tree]

        # lexicalize the sentence using abstraction instructions (if needed)
        if absts:
            for idx, tok in enumerate(out_sent):
                if tok.startswith('X-'):
                    slot = tok[2:]
                    abst = self._first_abst(absts, slot)
                    form = re.sub(r'\b[0-9]+\b', '_', abst.surface_form)  # abstract numbers
                    # in tree mode, use lemmas instead of surface forms
                    if self.mode == 'trees' and slot in self._lemma_for_sf:
                        form = self._lemma_for_sf[slot].get(form, form)
                    out_sent[idx] = form

        return out_sent

    def load_surface_forms(self, surface_forms_fname):
        """Load all proper name surface forms from a file."""
        log_info('Loading surface forms from %s...' % surface_forms_fname)
        with file_stream(surface_forms_fname) as fh:
            data = json.load(fh)
        for slot, values in data.iteritems():
            sf_all = {}
            sf_formeme = {}
            sf_tag = {}
            lemma_for_sf = {}
            if slot == 'street':  # this is domain-specific: street names -> street name + number
                slot = 'address'  # TODO change this in the surface form file
            for value in values.keys():
                orig_value = value  # TODO get rid of this
                if slot == 'address':  # add street number placeholders to addresses
                    value += ' _'  # TODO change this in the surface form file
                for surface_form in values[orig_value]:
                    lemma, form, tag = surface_form.split("\t")
                    if slot == 'address':  # add street number placeholders to addresses
                        lemma += ' _'  # TODO change this in the surface form file
                        form += ' _'
                    # store the value globally + for all possible tag subsets/formemes
                    # store lemmas for formemes, forms for tags/global
                    sf_all[value] = sf_all.get(value, []) + [form]
                    sf_tag[value] = sf_tag.get(value, {})
                    sf_formeme[value] = sf_formeme.get(value, {})
                    for tag_subset in self._get_tag_subsets(tag):
                        sf_tag[value][tag_subset] = sf_tag[value].get(tag_subset, []) + [form]
                    for formeme in self._get_compatible_formemes(tag):
                        sf_formeme[value][formeme] = sf_formeme[value].get(formeme, []) + [lemma]
                    # store lemma for form (for lexicalizing training sentences with trees)
                    lemma_for_sf[form] = lemma
            self._sf_all[slot] = sf_all
            self._sf_by_formeme[slot] = sf_formeme
            self._sf_by_tag[slot] = sf_tag
            self._lemma_for_sf[slot] = lemma_for_sf

    def _get_tag_subsets(self, tag):
        """Select important tag subsets along which surface forms should be matched.
        @param tag: the tag to select subsets for
        @return: list of tag substrings important for matching appropriate forms (in order \
                of preference)
        """
        # TODO make this language-independent
        if tag[0] in ['N', 'A']:
            alt_pos = 'A' if tag[0] == 'N' else 'N'
            return [tag[0:3] + tag[4],  # NNF.4....
                    tag[0:2] + tag[4],  # NN..4....
                    tag[0] + tag[4],  # N...4....
                    alt_pos + tag[4],  # A...4...
                    tag[0:2]]  # NN
        # TODO this is greatly simplified for verbs, but probably won't be a problem
        return tag[0:2]

    def _get_compatible_formemes(self, tag):
        """For a morphological tag, get all matching formemes.
        @param tag: the tag to match formemes against
        @return: a list of formemes compatible with the tag; in order of preference
        """
        # TODO make this language-independent
        if tag.startswith('N'):
            return ['n:' + tag[4]]
        if tag[0] in ['A', 'C']:
            return ['adj:attr', 'n:' + (tag[4] if tag[4] != '-' else 'X')]
        if tag.startswith('D'):
            return ['adv']
        if tag.startswith('V'):
            return ['v:fin']
        return ['x']

    def lexicalize(self, gen_trees, abst_file):
        """Lexicalize nodes in the generated trees (which may represent trees, tokens, or tagged lemmas).
        Expects lexicalization file (and surface forms file) to be loaded in the Lexicalizer object,
        otherwise nothing will happen. The actual operation depends on the generator mode.

        @param gen_trees: list of TreeData objects representing generated trees/tokens/tagged lemmas
        @param abst_file: abstraction/delexicalization instructions file path
        @return: None
        """
        abstss = smart_load_absts(abst_file, len(gen_trees))
        for sent_no, (tree, absts) in enumerate(zip(gen_trees, abstss)):
            log_debug("Lexicalizing sentence %d: %s" % ((sent_no + 1), unicode(tree)))
            sent = self._tree_to_sentence(tree)
            log_debug(unicode(sent))
            for idx, tok in enumerate(sent):
                if tok and tok.startswith('X-'):  # we would like to lexicalize
                    slot = tok[2:]
                    # check if we have a value to substitute; if yes, do it
                    abst = self._first_abst(absts, slot)
                    if abst:
                        # tagged lemmas: one token with appropriate value
                        if self.mode == 'tagged_lemmas':
                            tag = sent[idx+1] if idx < len(sent) - 1 else None
                            val = self.get_surface_form(sent, idx, slot, abst.value, tag=tag)
                            tree.nodes[idx+1] = NodeData(t_lemma=val, formeme='x')
                        # trees: one node with appropriate value, keep formeme
                        elif self.mode == 'trees':
                            formeme = sent[idx+1] if idx < len(sent) - 1 else None
                            val = self.get_surface_form(sent, idx, slot, abst.value,
                                                        formeme=formeme)
                            tree.nodes[idx/2+1] = NodeData(t_lemma=val,
                                                           formeme=tree[idx/2+1].formeme)
                        # tokens: one token with all words from the value (postprocessed below)
                        else:
                            val = self.get_surface_form(sent, idx, slot, abst.value)
                            tree.nodes[idx+1] = NodeData(t_lemma=val, formeme='x')
                        sent[idx] = val  # save value to be used in LM next time
            # postprocess tokens (split multi-word nodes)
            if self.mode == 'tokens':
                idx = 1
                while idx < len(tree):
                    if ' ' in tree[idx].t_lemma:
                        value = tree[idx].t_lemma
                        tree.remove_node(idx)
                        for shift, tok in enumerate(value.split(' ')):
                            tree.create_child(0, idx + shift,
                                              NodeData(t_lemma=tok, formeme='x'))
                        idx += shift
                    idx += 1

    def get_surface_form(self, tree, idx, slot, value, tag=None, formeme=None):
        """Get the appropriate surface form for the given slot and value. Use morphological tag
        and/or formeme restrictions to select a matching one. Selects among matching forms using
        the current form selection module (random, RNNLM, KenLM, frequency).
        """
        non_num_value = re.sub(r'(^|\s+)([0-9]+)($|\s+)', r'\1_\3', value)

        # handle coordinated values
        value_parts = re.split(r'\s+(and|or)\s+', value)
        if len(value_parts) > 1 and non_num_value not in self._sf_all.get(slot, []):
            out_value = []
            for value_part in value_parts:
                if value_part == 'and':
                    out_value.append(self.AND_STRING.get(self.language, 'and'))
                elif value_part == 'or':
                    out_value.append(self.OR_STRING.get(self.language, 'or'))
                else:
                    out_value.append(self.get_surface_form(tree, idx,
                                                           slot, value_part, tag, formeme))
            return ' '.join(out_value)

        # abstract away from numbers
        nums = re.findall(r'(?:^|\s+)([0-9]+)(?:$|\s+)', value)
        value = non_num_value
        form = None

        # find the appropriate form (by tag, formeme, backoff to any form)
        if tag is not None:
            if slot in self._sf_by_tag and value in self._sf_by_tag[slot]:
                for tag_sub in self._get_tag_subsets(tag):
                    if tag_sub in self._sf_by_tag[slot][value]:
                        form = self._form_select.get_surface_form(
                                tree, idx, self._sf_by_tag[slot][value][tag_sub])
                        if form:
                            break

        if form is None and formeme is not None:
            formeme = re.sub(r':.*\+', ':', formeme)  # ignore prepositions/conjunctions
            formeme = re.sub(r':inf', r':fin', formeme)  # ignore finite/infinite verb distinction
            if slot in self._sf_by_formeme and value in self._sf_by_formeme[slot]:
                if formeme in self._sf_by_formeme[slot][value]:
                    form = self._form_select.get_surface_form(
                            tree, idx, self._sf_by_formeme[slot][value][formeme])

        if form is None:
            if slot in self._sf_all and value in self._sf_all[slot]:
                form = self._form_select.get_surface_form(tree, idx, self._sf_all[slot][value])

        # backoff to the actual value (no surface form replacement)
        if form is None:
            form = value

        # put numbers back
        for num in nums:
            form = re.sub(r'_', num, form, count=1)

        return form

    @staticmethod
    def load_from_file(lexicalizer_fname):
        """Load the lexicalizer model from a file (and a second file with the LM, if needed)."""
        log_info("Loading lexicalizer from %s..." % lexicalizer_fname)
        with file_stream(lexicalizer_fname, 'rb', encoding=None) as fh:
            data = pickle.load(fh)
            ret = Lexicalizer(cfg=data['cfg'])
            ret.__dict__.update(data)
            ret._form_select = ret._form_select(data['cfg'])

        if not isinstance(ret._form_select, RandomFormSelect):
            ret._form_select.load_model(lexicalizer_fname)
        return ret

    def save_to_file(self, lexicalizer_fname):
        """Save the lexicalizer model to a file (and a second file with the LM, if needed)."""
        log_info("Saving lexicalizer to %s..." % lexicalizer_fname)
        with file_stream(lexicalizer_fname, 'wb', encoding=None) as fh:
            pickle.dump(self.get_all_settings(), fh, protocol=pickle.HIGHEST_PROTOCOL)

        if not isinstance(self._form_select, RandomFormSelect):
            self._form_select.save_model(lexicalizer_fname)

    def get_all_settings(self):
        """Get all settings except the trained model parameters (to be stored in a pickle)."""
        data = {'cfg': self.cfg,
                'mode': self.mode,
                '_sf_all': self._sf_all,
                '_sf_by_formeme': self._sf_by_formeme,
                '_sf_by_tag': self._sf_by_tag,
                '_form_select': type(self._form_select)}
        return data
