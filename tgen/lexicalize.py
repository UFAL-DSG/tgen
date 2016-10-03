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
import sys
import tempfile
import shutil
import codecs
from subprocess import Popen, PIPE

from tgen.tree import NodeData
from tgen.rnd import rnd
from tgen.futil import file_stream, read_absts
from tgen.logf import log_warn, log_info


class FormSelect(object):

    def __init__(self):
        pass

    def get_surface_form(self, sentence, pos, possible_forms):
        raise NotImplementedError()


class RandomFormSelect(FormSelect):

    def get_surface_form(self, sentence, pos, possible_forms):
        return rnd.choice(possible_forms)


class KenLMFormSelect(FormSelect):

    def __init__(self, cfg):
        self._sample = cfg.get('form_sample', False)
        self._trained_model = None
        np.random.seed(rnd.randint(0, 2**32 - 1))

    def get_surface_form(self, tree, pos, possible_forms):
        state, dummy_state = kenlm.State(), kenlm.State()
        self._lm.BeginSentenceWrite(state)
        for idx in xrange(pos, start=1):
            self._lm.BaseScore(state, tree[idx].t_lemma, state)
        best_form = None
        best_score = float('-inf')
        scores = []
        for possible_form in possible_forms:
            possible_form = possible_form.replace(' ', '_')
            score = self._lm.BaseScore(state, possible_form, dummy_state)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_form = possible_form
        if self._sample:
            probs = np.exp(scores) / np.sum(np.exp(scores))  # softmax
            return np.random.choice(possible_forms, p=probs)
        return best_form

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
            lmplz_stdin.write(" ".join(sent) + "\n")
        lmplz_stdin.close()
        # wait for the process to complete
        binarize.wait()
        if binarize.returncode != 0:
            raise RuntimeError("LM build failed (error code: %d)" % binarize.returncode)
        # load the output into memory (extension will be filled in again)
        self.load_model(re.sub('\.kenlm\.bin$', '', tmppath))


class Lexicalizer(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.get('mode', 'trees')
        self._sf_all = {}
        self._sf_by_formeme = {}
        self._sf_by_tag = {}
        self._form_select = RandomFormSelect()
        if 'form_select_type' in cfg:
            if cfg['form_select_type'] == 'kenlm':
                self._form_select = KenLMFormSelect(cfg)

    def train(self, fnames, train_trees):
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
        """Get 1st abstraction instruction for a specific slot in the lis, put it back to
        the end of the list"""
        i, abst = ((i, a) for i, a in enumerate(absts) if a.slot == slot).next()
        del absts[i]
        absts.append(abst)
        return abst

    def _prepare_train_toks(self, train_trees, train_abstr_fname):
        # load abstraction file
        abstss = read_absts(train_abstr_fname)
        out_sents = []
        for tree, absts in zip(train_trees, abstss):
            out_sent = []
            if self.mode == 'trees':
                for idx, node in enumerate(tree.nodes[1:]):
                    if node.t_lemma.startswith('X-'):
                        abst = self._first_abst(absts, node.t_lemma[2:])
                        out_sent.append(abst.surface_form.replace(' ', '_'))
                    else:
                        out_sent.append(node.t_lemma)
                    out_sent.append(node.formeme)
            elif self.mode == 'tagged_lemmas':
                for lemma, tag in tree:
                    if lemma.startswith('X-'):
                        abst = self._first_abst(absts, lemma[2:])
                        out_sent.append(abst.surface_form.replace(' ', '_'))
                    else:
                        out_sent.append(lemma)
                    out_sent.append(tag)
            else:  # tokens
                for tok, _ in tree:
                    if tok.startswith('X-'):
                        abst = self._first_abst(absts, tok[2:])
                        out_sent.append(abst.surface_form.replace(' ', '_'))
                    else:
                        out_sent.append(tok)
            out_sents.append(out_sent)
        return out_sents

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

    def load_lexicalization(self, abstr_file):
        """Read lexicalization from a file with "abstraction instructions" (telling which tokens are
        delexicalized to what). This just remembers the slots and values and returns them in a dict
        for each line (slot -> list of values)."""
        abstrs = []
        with file_stream(abstr_file) as fh:
            for line in fh:
                line_abstrs = {}
                for svp in filter(bool, line.strip().split("\t")):
                    m = re.match('([^=]*)=(.*):[0-9]+-[0-9]+$', svp)
                    slot = m.group(1)
                    value = re.sub(r'^[\'"]', '', m.group(2))
                    value = re.sub(r'[\'"]#?$', '', value)
                    value = re.sub(r'"#? and "', ' and ', value)
                    value = re.sub(r'_', ' ', value)

                    line_abstrs[slot] = line_abstrs.get(slot, [])
                    line_abstrs[slot].append(value)

                abstrs.append(line_abstrs)
        return abstrs

    def lexicalize(self, gen_trees, abstr_file):
        """Lexicalize nodes in the generated trees (which may represent trees, tokens, or tagged lemmas).
        Expects lexicalization file (and surface forms file) to be loaded in the Lexicalizer object,
        otherwise nothing will happen. The actual operation depends on the generator mode.

        @param gen_trees: list of TreeData objects representing generated trees/tokens/tagged lemmas
        @param abstr_file: abstraction/delexicalization instructions file path
        @param mode: generator mode (acceptable string values: "trees"/"tokens"/"tagged_lemmas")
        @return: None
        """
        abstrs = self.load_lexicalization(abstr_file)
        for tree, lex_dict in zip(gen_trees, abstrs):
            idx = 1
            while idx < len(tree):
                if tree[idx].t_lemma.startswith('X-'):
                    slot = tree[idx].t_lemma[2:]
                    # we can lexicalize
                    if slot in lex_dict:
                        # tagged lemmas: one token with appropriate value
                        if self.mode == 'tagged_lemmas':
                            val = self._get_surface_form(tree, idx, slot, lex_dict[slot][0],
                                                         tag=tree[idx+1].t_lemma)
                            tree[idx] = NodeData(t_lemma=val, formeme='x')
                        # trees: one node with appropriate value, keep formeme
                        elif self.mode == 'trees':
                            val = self._get_surface_form(tree, idx, slot, lex_dict[slot][0],
                                                         formeme=tree[idx].formeme)
                            tree[idx] = NodeData(t_lemma=val, formeme=tree[idx].formeme)
                        # tokens: multiple tokens with all words from the value
                        else:
                            value = self.get_surface_form(tree, idx, slot, lex_dict[slot][0])
                            tree.remove_node(idx)
                            for shift, tok in enumerate(value.split(' ')):
                                tree.create_child(0, idx + shift,
                                                  NodeData(t_lemma=tok, formeme='x'))
                            idx += shift
                        lex_dict[slot] = lex_dict[slot][1:] + [lex_dict[slot][0]]  # cycle values
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
        log_info("Loading lexicalizer from %s..." % lexicalizer_fname)
        with file_stream(lexicalizer_fname, 'rb', encoding=None) as fh:
            data = pickle.load(fh)
            ret = Lexicalizer(cfg=data['cfg'])
            ret.__dict__.update(data)
            ret._form_select = ret._form_select(data['cfg'])

        if not isinstance(ret._form_select, RandomFormSelect):
            ret._form_select.load_model(lexicalizer_fname)

    def save_to_file(self, lexicalizer_fname):
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
