#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lexicalization functions (postprocessing generated trees).
"""

from __future__ import unicode_literals
import json
import re
import kenlm  # needed only if KenLMFormSelect is used
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

from tgen.tree import NodeData, TreeData
from tgen.rnd import rnd
from tgen.futil import file_stream, read_absts
from tgen.logf import log_warn, log_info
from tgen.tf_ml import TFModel


class FormSelect(object):

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

    def __init__(self, cfg=None):
        super(RandomFormSelect, self).__init__(cfg)

    def get_surface_form(self, sentence, pos, possible_forms):
        return rnd.choice(possible_forms)


class KenLMFormSelect(FormSelect):

    def __init__(self, cfg):
        super(KenLMFormSelect, self).__init__(cfg)
        self._sample = cfg.get('form_sample', False)
        self._trained_model = None
        np.random.seed(rnd.randint(0, 2**32 - 1))

    def get_surface_form(self, sentence, pos, possible_forms):
        state, dummy_state = kenlm.State(), kenlm.State()
        self._lm.BeginSentenceWrite(state)
        for idx in xrange(pos):
            self._lm.BaseScore(state, sentence[idx], state)
        best_form_idx = 0
        best_score = float('-inf')
        scores = []
        for form_idx, possible_form in enumerate(possible_forms):
            possible_form = possible_form.lower().replace(' ', '^')
            score = self._lm.BaseScore(state, possible_form, dummy_state)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_form_idx = form_idx
        if self._sample:
            probs = np.exp(scores) / np.sum(np.exp(scores))  # softmax
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

    def train(self, train_sents):
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
        self.vocab = {'<VOID>': self.VOID, '<GO>': self.GO,
                      '<STOP>': self.STOP, '<UNK>': self.UNK}
        self.reverse_dict = {self.VOID: '<VOID>', self.GO: '<GO>',
                             self.STOP: '<STOP>', self.UNK: '<UNK>'}
        self.vocab_size = None

    def _init_training(self, train_sents):
        """Initialize training (prepare vocabulary, prepare training batches), initialize the
        RNN.
        @param train_sents: training data (list of lists of tokens, lexicalized)
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
        self._init_neural_network()
        self.session.run(tf.initialize_all_variables())

    def _sent_to_ids(self, sent):
        """Convert tokens in a sentence to integer IDs to be used as an input to the RNN.
        Pad with "<VOID>" to maximum sentence length.
        @param sent: input sentence (list of string tokens)
        @return: list of IDs corresponding to the input tokens, padded to RNN length
        """
        ids = [self.vocab.get('<GO>')]
        ids += [self.vocab.get(tok, self.vocab.get('<UNK>')) for tok in sent]
        ids = ids[:self.max_sent_len]
        ids += [self.vocab.get('<STOP>')]
        # actually using max_sent_len + 1 (inputs exclude last step, targets exclude the 1st)
        ids += [self.vocab.get('<VOID>') for _ in xrange(self.max_sent_len - len(ids) + 1)]
        return ids

    def _batches(self):
        """Create batches from the input; use as iterator."""
        for batch_start in xrange(0, len(self._train_order), self.batch_size):
            sents = [self._train_data[idx]
                     for idx in self._train_order[batch_start: batch_start + self.batch_size]]
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
                self._cell = tf.nn.rnn_cell.GRUCell(self.emb_size)
            else:
                self._cell = tf.nn.rnn_cell.BasicLSTMCell(self.emb_size)
            if re.match(r'/[0-9]$', self.cell_type):
                self._cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * int(self.cell_type[-1]))
            self._initial_state = self._cell.zero_state(tf.shape(self._inputs)[0], tf.float32)

            # embeddings
            emb_cell = tf.nn.rnn_cell.EmbeddingWrapper(self._cell, self.vocab_size)
            # RNN encoder
            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, self.max_sent_len, self._inputs)]
            outputs, states = tf.nn.rnn(emb_cell, inputs, initial_state=self._initial_state)

            # output layer
            output = tf.reshape(tf.concat(1, outputs), [-1, self.emb_size])
            self._logits = (tf.matmul(output,
                                      tf.get_variable("W", [self.emb_size, self.vocab_size])) +
                            tf.get_variable("b", [self.vocab_size]))

            # cost
            targets_1d = tf.reshape(self._targets, [-1])
            self._loss = tf.nn.seq2seq.sequence_loss_by_example(
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

    def train(self, train_sents):
        """Train the RNNLM on the given data (list of lists of tokens).
        @param train_sents: training data (list of lists of tokens, lexicalized)
        """
        self._init_training(train_sents)

        for iter_no in xrange(1, self.passes + 1):
            iter_alpha = self.alpha * np.exp(-self.alpha_decay * iter_no)
            self._train_order = range(len(self._train_data))
            if self.randomize:
                rnd.shuffle(self._train_order)
            self._training_pass(iter_no, iter_alpha)

    def _training_pass(self, pass_no, pass_alpha):
        """Run one pass over the training data (epoch).
        @param pass_no: pass number (for logging)
        @param pass_alpha: learning rate for the current pass
        """
        pass_start_time = time.time()
        pass_cost = 0
        for batch_no, (inputs, targets) in enumerate(self._batches()):
            cost, _ = self.session.run([self._cost, self._train_func],
                                       {self._inputs: inputs,
                                        self._targets: targets,
                                        self._learning_rate: pass_alpha, })
            pass_cost += cost
        duration = str(datetime.timedelta(seconds=(time.time() - pass_start_time)))
        log_info("Pass %d: duration %s, cost %.3f" % (pass_no, duration, pass_cost))
        return pass_cost

    def load_model(self, model_fname_pattern):
        """Load the RNNLM model from a file."""
        model_fname = re.sub(r'(.pickle)?(.gz)?$', '.rnnlm', model_fname_pattern)
        with file_stream(model_fname, 'rb', encoding=None) as fh:
            self.load_all_settings(pickle.load(fh))
            self.set_model_params(pickle.load(fh))

    def save_model(self, model_fname_pattern):
        """Save the RNNLM model to a file.""""
        model_fname = re.sub(r'(.pickle)?(.gz)?$', '.rnnlm', model_fname_pattern)
        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self.get_all_settings(), fh, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.get_model_params(), fh, pickle.HIGHEST_PROTOCOL)

    def get_surface_form(self, sentence, pos, possible_forms):
        # get unnormalized scores for the whole vocabulary
        logits = self.session.run([self._logits],
                                  {self._inputs: self._sent_to_ids(sentence)})
        # pick out scores for possible forms
        scores = [logits[self.vocab.get(form.lower(), self.vocab.get('<UNK>'))]
                  for form in possible_forms]
        probs = np.exp(scores) / np.sum(np.exp(scores))  # softmax
        # sample from the prob. dist.
        if self._sample:
            return np.random.choice(possible_forms, p=probs)
        # get just the most probable option
        max_idx, _ = max(enumerate(probs), key=operator.itemgetter(1))
        return possible_forms[max_idx]


class Lexicalizer(object):
    """Main object controlling lexicalization, using a LM or random surface forms to replace
    slot placeholders in the outputs."""

    def __init__(self, cfg):
        """Read configuration, initialize internal buffers, create the surface form selection
        LM object (if required)."""
        self.cfg = cfg
        self.mode = cfg.get('mode', 'trees')
        self._sf_all = {}
        self._sf_by_formeme = {}
        self._sf_by_tag = {}
        self._form_select = RandomFormSelect()
        if 'form_select_type' in cfg:
            if cfg['form_select_type'] == 'kenlm':
                self._form_select = KenLMFormSelect(cfg)
            elif cfg['form_select_type'] == 'rnnlm':
                self._form_select = RNNLMFormSelect(cfg)

    def train(self, fnames, train_trees):
        """Train the lexicalizer (including its LM, if applicable).
        @param fnames: file names for surface forms (JSON) and training data lexicalization \
            instructions
        @param train_trees: loaded generator training data (TreeData trees/lists of lemma-tag \
            or form-tag pairs)
        """
        log_info('Training lexicalizer...')
        if not fnames:
            return
        if ',' in fnames:
            surface_forms_fname, train_abst_fname = fnames.split(',')
        else:
            surface_forms_fname, train_abst_fname = fnames, None

        self.load_surface_forms(surface_forms_fname)
        if train_abst_fname and not isinstance(self._form_select, RandomFormSelect):
            log_info('Training lexicalization LM from training trees and %s...' % train_abst_fname)
            self._form_select.train(self._prepare_train_toks(train_trees, train_abst_fname))

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

    def _prepare_train_toks(self, train_trees, train_abstr_fname):
        """Prepare training data for form selection LM. Use training trees/tagged lemmas/tokens,
        apply lexicalization instructions including surface forms, and convert the output to a
        list of lists of tokens (sentences).
        @param train_trees: main generator training data (trees, tagged lemmas, tokens)
        @param train_abstr_fname: file name for the corresponding lexicalization instructions
        @return: list of lists of LM training tokens (lexicalized)
        """
        # load abstraction file
        abstss = read_absts(train_abstr_fname)
        out_sents = []
        for tree, absts in zip(train_trees, abstss):
            # create a list of tokens
            out_sent = self._tree_to_sentence(tree)
            # lexicalize the resulting sentence using abstraction instructions
            for idx, tok in enumerate(out_sent):
                if tok.startswith('X-'):
                    abst = self._first_abst(absts, tok[2:])
                    form = re.sub(r'\b[0-9]+\b', '_', abst.surface_form)  # abstract numbers
                    out_sent[idx] = form
            # store the result
            out_sents.append(out_sent)
        return out_sents

    def _tree_to_sentence(self, tree):
        """Convert the given "tree" (i.e., tree, tagged lemmas, or tokens; represented as TreeData
        or a list of tuples form/lemma-tag) into a sentence (i.e., always a plain list of tokens).
        @param tree: input sentence -- a tree, tagged lemmas, or tokens (as TreeData or lists of \
                tuples)
        @return: list of tokens in the sentence
        """
        # based on embedding types
        out_sent = []
        if self.mode == 'trees':
            for node in tree.nodes[1:]:
                out_sent.append(node.t_lemma)
                out_sent.append(node.formeme)
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
            for value in values.keys():
                for surface_form in values[value]:
                    form, tag = surface_form.split("\t")
                    if slot == 'street':  # add street number placeholders to addresses
                        value += ' _'
                        slot = 'address'
                    # store the value globally + for all possible tag subsets/formemes
                    sf_all[value] = sf_all.get(value, []) + [form]
                    sf_tag[value] = sf_tag.get(value, {})
                    sf_formeme[value] = sf_formeme.get(value, {})
                    for tag_subset in self._get_tag_subsets(tag):
                        sf_tag[value][tag_subset] = sf_tag[value].get(tag_subset, []) + [form]
                    for formeme in self._get_compatible_formemes(tag):
                        sf_formeme[value][formeme] = sf_formeme[value].get(formeme, []) + [form]
            self._sf_all[slot] = sf_all
            self._sf_by_formeme[slot] = sf_formeme
            self._sf_by_tag[slot] = sf_tag

    def _get_tag_subsets(self, tag):
        """Select important tag subsets along which surface forms should be matched.
        @param tag: the tag to select subsets for
        @return: list of tag substrings important for matching appropriate forms (in order \
                of preference)
        """
        # TODO make this language-independent
        if tag[0] in ['N', 'A']:
            return [tag[0:3] + tag[4], tag[0:2] + tag[4], tag[0:2]]
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
        @param mode: generator mode (acceptable string values: "trees"/"tokens"/"tagged_lemmas")
        @return: None
        """
        abstss = read_absts(abst_file)
        for tree, absts in zip(gen_trees, abstss):
            sent = self._tree_to_sentence(tree)
            for idx, tok in enumerate(sent):
                if tok.startswith('X-'):  # we would like to delexicalize
                    slot = tok[2:]
                    # check if we have a value to substitute; if yes, do it
                    abst = self._first_abst(absts, slot)
                    if abst:
                        # tagged lemmas: one token with appropriate value
                        if self.mode == 'tagged_lemmas':
                            tag = sent[idx+1] if idx < len(sent) - 1 else None
                            val = self._get_surface_form(sent, idx, slot, abst.value, tag=tag)
                            sent[idx] = val
                            tree[idx/2] = NodeData(t_lemma=val, formeme='x')
                        # trees: one node with appropriate value, keep formeme
                        elif self.mode == 'trees':
                            formeme = sent[idx + 1] if idx < len(sent) - 1 else None
                            val = self._get_surface_form(sent, idx, slot, abst.value,
                                                         formeme=formeme)
                            tree[idx/2] = NodeData(t_lemma=val, formeme=tree[idx/2].formeme)
                        # tokens: one token with all words from the value (postprocessed below)
                        else:
                            value = self.get_surface_form(sent, idx, slot, abst.value)
                            sent[idx] = val
                            tree[idx] = NodeData(t_lemma=val, formeme='x')
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
        and/or formeme restrictions to select a matching one. Selects randomly among matching
        forms.
        """
        # handle coordinated values
        value_parts = re.split(r'\s+(and|or)\s+', value)
        if len(value_parts) > 1:
            out_value = []
            for value_part in value_parts:
                # TODO avoid hardcoded constants for "and" and "or" (now Czech-specific)
                if value_part == 'and':
                    out_value.append('a')
                elif value_part == 'or':
                    out_value.append('nebo')
                else:
                    out_value.append(self.get_surface_form(slot, value_part, tag, formeme))
            return ' '.join(out_value)

        # abstract away from numbers
        nums = re.findall(r'(?:^|\s+)([0-9]+)(?:$|\s+)', value)
        value = re.sub(r'(^|\s+)([0-9]+)($|\s+)', r'\1_\3', value)
        form = None

        # find the appropriate form (by tag, formeme, backoff to any form)
        # TODO better selection than random
        if tag is not None:
            if slot in self._sf_by_tag and value in self._sf_by_tag[slot]:
                for tag_sub in self._get_tag_subsets(tag):
                    if tag_sub in self._sf_by_tag[slot][value]:
                        form = self._form_select.get_surface_form(
                                tree, idx, self._sf_by_tag[slot][value][tag_sub])

        if form is None and formeme is not None:
            formeme = re.sub(r':.*\+', '', formeme)  # ignore prepositions/conjunctions
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
