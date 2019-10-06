#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import codecs
import json
import re
from argparse import ArgumentParser
from collections import deque, namedtuple
from itertools import islice

import numpy as np
import random

from ufal.morphodita import Tagger, Forms, TaggedLemma, TaggedLemmas, TokenRanges, Analyses, Indices

import sys
import os
sys.path.insert(0, os.path.abspath('../../'))  # add tgen main directory to modules path
from tgen.logf import log_info
from tgen.data import Abst, DAI, DA

# Start IPdb on error in interactive mode
from tgen.debug import exc_info_hook
import sys
sys.excepthook = exc_info_hook


Inst = namedtuple('Inst', ['da', 'text', 'delex_da', 'delex_text', 'abst'])


class Reader(object):

    def __init__(self, tagger_model, abst_slots):
        self._tagger = Tagger.load(tagger_model)
        self._analyzer = self._tagger.getMorpho()
        self._tokenizer = self._tagger.newTokenizer()
        self._abst_slots = set(abst_slots.split(','))

        self._forms_buf = Forms()
        self._tokens_buf = TokenRanges()
        self._analyses_buf = Analyses()
        self._indices_buf = Indices()

        self._sf_dict = {}
        self._rev_sf_dict = {}
        self._sf_max_len = 0

    def load_surface_forms(self, surface_forms_fname):
        """Load all proper name surface forms from a file."""
        with codecs.open(surface_forms_fname, 'rb', 'UTF-8') as fh:
            data = json.load(fh)
        for slot, values in data.items():
            for value in values.keys():
                for surface_form in values[value]:
                    lemma, form, tag = surface_form.split("\t")
                    form_toks = form.lower().split(" ")
                    if slot == 'street':  # add street number placeholders to addresses
                        lemma += ' _'
                        form_toks.append('_')
                    form_toks = tuple(form_toks)
                    self._sf_max_len = max((self._sf_max_len, len(form_toks)))
                    if form_toks not in self._sf_dict:
                        self._sf_dict[form_toks] = []
                    self._sf_dict[form_toks].append((lemma, tag))
                    self._rev_sf_dict[(form.lower(), lemma, tag)] = (slot, value)

    def _get_surface_form_taggedlemmas(self, forms_in):
        """Given a tokens deque, return the form & list of tagged lemmas (analyses)
        for the proper name in the list of forms at the current position, if applicable.
        If there is no proper name at the beginning of the tokens deque, return (None, None).

        @param forms_in: a deque of forms tokens
        @return: (form, tagged lemmas list) or (None, None)
        """
        for test_len in range(min(self._sf_max_len, len(forms_in)), 0, -1):
            # test the string, handle number placeholders
            full_substr = [form for form in islice(forms_in, 0, test_len)]
            test_substr = tuple(['_' if re.match(r'^[0-9]+$', form) else form.lower()
                                 for form in full_substr])
            if test_substr in self._sf_dict:
                tls = TaggedLemmas()
                nums = [num for num in full_substr if re.match(r'^[0-9]+$', num)]
                for lemma, tag in self._sf_dict[test_substr]:
                    tls.push_back(TaggedLemma())
                    for num in nums:  # replace number placeholders by actual values
                        lemma = re.sub(r'_', num, lemma, count=1)
                    tls[-1].lemma = lemma
                    tls[-1].tag = tag
                for _ in range(len(test_substr)):  # move on in the sentence
                    forms_in.popleft()
                return " ".join(full_substr), tls
        return None, None

    def analyze(self, sent):
        """Perform morphological analysis on the given sentence, preferring analyses from the
        list of surface forms. Return a list of tuples (form, lemma, tag)."""
        self._tokenizer.setText(sent)
        analyzed = []
        while self._tokenizer.nextSentence(self._forms_buf, self._tokens_buf):

            forms_in = deque(self._forms_buf)
            self._forms_buf.resize(0)
            self._analyses_buf.resize(0)  # reset previous analyses

            while forms_in:
                form, analyses = self._get_surface_form_taggedlemmas(forms_in)
                if form:
                    # our custom analysis
                    self._analyses_buf.push_back(analyses)
                else:
                    # Morphodita analysis
                    form = forms_in.popleft()
                    analyses = TaggedLemmas()
                    self._analyzer.analyze(form, 1, analyses)
                    for i in range(len(analyses)):  # shorten lemmas (must access the vector directly)
                        analyses[i].lemma = self._analyzer.rawLemma(analyses[i].lemma)
                    self._analyses_buf.push_back(analyses)

                self._forms_buf.push_back(form)

            # tag according to the given analysis
            self._tagger.tagAnalyzed(self._forms_buf, self._analyses_buf, self._indices_buf)
            analyzed.extend([(f, a[idx].lemma, a[idx].tag)
                             for (f, a, idx)
                             in zip(self._forms_buf, self._analyses_buf, self._indices_buf)])
        return analyzed

    def process_dataset(self, input_data):
        """Load DAs & sentences, obtain abstraction instructions, and store it all in member
        variables (to be used later by writing methods).
        @param input_data: path to the input JSON file with the data
        """
        # load data from JSON
        self._das = []
        self._texts = []
        with codecs.open(input_data, 'r', encoding='UTF-8') as fh:
            data = json.load(fh)
            for inst in data:
                da = DA.parse_cambridge_da(inst['da'])
                da.sort()
                self._das.append(da)
                self._texts.append(self.analyze(inst['text']))

        # delexicalize DAs and sentences
        self._create_delex_texts()
        self._create_delex_das()

        # return the result
        out = []
        for da, text, delex_da, delex_text, abst in zip(self._das, self._texts, self._delex_das, self._delex_texts, self._absts):
            out.append(Inst(da, text, delex_da, delex_text, abst))
        return out

    def _create_delex_texts(self):
        """Delexicalize texts in the buffers and save them separately in the member variables,
        along with the delexicalization instructions used for the operation."""
        self._delex_texts = []
        self._absts = []
        for text_idx, (text, da) in enumerate(zip(self._texts, self._das)):
            delex_text = []
            absts = []
            # do the delexicalization, keep track of which slots we used
            for tok_idx, (form, lemma, tag) in enumerate(text):
                # abstract away from numbers
                abst_form = re.sub(r'( |^)[0-9]+( |$)', r'\1_\2', form.lower())
                abst_lemma = re.sub(r'( |^)[0-9]+( |$)', r'\1_\2', lemma)
                # try to find if the surface form belongs to some slot
                slot, value = self._rev_sf_dict.get((abst_form, abst_lemma, tag), (None, None))
                # if we found a slot, get back the numbers
                if slot:
                    for num_match in re.finditer(r'(?: |^)([0-9]+)(?: |$)', lemma):
                        value = re.sub(r'_', num_match.group(1), value, count=1)
                # fall back to directly comparing against the DA value
                else:
                    slot = da.has_value(lemma)
                    value = lemma

                # if we found something, delexicalize it (check if the value corresponds to the DA!)
                if (slot and slot in self._abst_slots and
                        da.value_for_slot(slot) not in [None, 'none', 'dont_care'] and
                        value in da.value_for_slot(slot)):
                    delex_text.append(('X-' + slot, 'X-' + slot, tag))
                    absts.append(Abst(slot, value, form, tok_idx, tok_idx + 1))
                # otherwise keep the token as it is
                else:
                    delex_text.append((form, lemma, tag))
            # fix coordinated delexicalized values
            self._delex_fix_coords(delex_text, da, absts)
            covered_slots = set([a.slot for a in absts])
            # check and warn if we left isomething non-delexicalized
            for dai in da:
                if (dai.slot in self._abst_slots and
                        dai.value not in [None, 'none', 'dont_care'] and
                        dai.slot not in covered_slots):
                    log_info("Cannot delexicalize slot  %s  at %d:\nDA: %s\nTx: %s\n" %
                             (dai.slot,
                              text_idx,
                              str(da),
                              " ".join([form for form, _, _ in text])))
            # save the delexicalized text and the delexicalization instructions
            self._delex_texts.append(delex_text)
            self._absts.append(absts)

    def _delex_fix_coords(self, text, da, absts):
        """Fix (merge) coordinated values in delexicalized text (X-slot and X-slot -> X-slot).
        Modifies the input list directly.

        @param text: list of form-lemma-tag tokens of the delexicalized sentence
        @return: None
        """
        idx = 0
        while idx < len(absts) - 1:
            if (absts[idx].slot == absts[idx+1].slot and
                    absts[idx].end + 1 == absts[idx + 1].start and
                    re.search(r' (and|or) ', da.value_for_slot(absts[idx].slot))):
                for abst in absts[idx+2:]:
                    abst.start -= 2
                    abst.end -= 2
                absts[idx].value = da.value_for_slot(absts[idx].slot)
                del text[absts[idx].end:absts[idx + 1].end]
                del absts[idx + 1]
            idx += 1

    def _create_delex_das(self):
        """Delexicalize DAs in the buffers, save them separately."""
        out = []
        for da in self._das:
            delex_da = DA()
            for dai in da:
                delex_dai = DAI(dai.da_type, dai.slot,
                                'X-' + dai.slot
                                if (dai.value not in [None, 'none', 'dont_care'] and
                                    dai.slot in self._abst_slots)
                                else dai.value)
                delex_da.append(delex_dai)
            out.append(delex_da)
        self._delex_das = out


class Writer(object):

    def __init__(self):
        pass

    def _write_plain(self, output_file, data_items):
        with codecs.open(output_file, 'wb', encoding='UTF-8') as fh:
            for data_item in data_items:
                print(str(data_item), file=fh)

    def _write_conll(self, output_file, data_items):
        with codecs.open(output_file, 'wb', encoding='UTF-8') as fh:
            for line in data_items:
                for idx, tok in enumerate(line, start=1):
                    print("\t".join((str(idx),
                                     tok[0].replace(' ', '_'),
                                     tok[1].replace(' ', '_'),
                                     '_', tok[2], '_',
                                     '0', '_', '_', '_')),
                          file=fh)
                print("", file=fh)

    def _write_interleaved(self, output_file, data_items):
        with codecs.open(output_file, 'wb', encoding='UTF-8') as fh:
            for line in data_items:
                for _, lemma, tag in line:
                    print(lemma.replace(' ', '_'), tag, end=' ', file=fh)
                print('', file=fh)

    def write_text(self, data_file, out_format, insts, delex=False):
        """Write output sentences for the given data subrange.
        @param data_file: output file name
        @param out_format: output format ('conll' -- CoNLL-U morphology, \
            'interleaved' -- lemma/tag interleaved, 'plain' -- plain text)
        @param insts: instances to write
        @param delex: delexicalize? false by default
        """
        texts = [inst.delex_text if delex else inst.text for inst in insts]
        if out_format == 'interleaved':
            self._write_interleaved(data_file, texts)
        elif out_format == 'conll':
            self._write_conll(data_file, texts)
        else:
            self._write_plain(data_file, [" ".join([form for form, _, _ in sent])
                                          for sent in texts])

    def write_absts(self, data_file, insts):
        """Write delexicalization/abstraction instructions (for the given data subrange).
        @param data_file: output file name
        @param insts: instances to write
        """
        self._write_plain(data_file, ["\t".join([str(abst_) for abst_ in inst.abst])
                                      for inst in insts])

    def write_das(self, data_file, insts, delex=False):
        """Write DAs (for the given subrange).
        @param data_file: output file name
        @param insts: instances to write
        @param delex: delexicalize? false by default
        """
        das = [inst.delex_da if delex else inst.da for inst in insts]
        self._write_plain(data_file, das)


def convert(args):
    """Main conversion function (using command-line arguments as parsed by Argparse)."""
    log_info('Loading...')
    reader = Reader(args.tagger_model, args.abst_slots)
    reader.load_surface_forms(args.surface_forms)
    log_info('Processing input files...')
    insts = reader.process_dataset(args.input_data)
    log_info('Loaded %d data items.' % len(insts))

    # write all data groups
    # outputs: plain delex, plain lex, interleaved delex & lex, CoNLL-U delex & lex, DAs, abstrs
    writer = Writer()

    log_info('Writing %s (size: %d)...' % (args.out_prefix, len(insts)))

    writer.write_absts(args.out_prefix + '-abst.txt', insts)

    writer.write_das(args.out_prefix + '-das_l.txt', insts)
    writer.write_das(args.out_prefix + '-das.txt', insts, delex=True)

    writer.write_text(args.out_prefix + '-text_l.txt', 'plain', insts)
    writer.write_text(args.out_prefix + '-text.txt', 'plain', insts, delex=True)
    writer.write_text(args.out_prefix + '-tls_l.txt', 'interleaved', insts)
    writer.write_text(args.out_prefix + '-tls.txt', 'interleaved', insts, delex=True)
    writer.write_text(args.out_prefix + '-text_l.conll', 'conll', insts)
    writer.write_text(args.out_prefix + '-text.conll', 'conll', insts, delex=True)


if __name__ == '__main__':

    random.seed(1206)
    ap = ArgumentParser()

    ap.add_argument('tagger_model', type=str, help='MorphoDiTa tagger model')
    ap.add_argument('surface_forms', type=str, help='Input JSON with base forms')
    ap.add_argument('input_data', type=str, help='Input data JSON ({train,devel,test}.json)')
    ap.add_argument('out_prefix', help='Output files name prefix(es - when used with -s, comma-separated)')
    ap.add_argument('-a', '--abst-slots', help='List of slots to delexicalize/abstract (comma-separated)')

    args = ap.parse_args()
    convert(args)
