#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lexicalization functions (postprocessing generated trees).
"""

from __future__ import unicode_literals
import json
import re
from futil import file_stream
from tgen.tree import NodeData
from tgen.rnd import rnd


class FormSelect(object):

    def __init__(self):
        pass

    def get_surface_form(self, sentence, pos, possible_forms):
        raise NotImplementedError()


class RandomFormSelect(FormSelect):

    def get_surface_form(self, sentence, pos, possible_forms):
        return rnd.choice(possible_forms)


class KenLMFormSelect(FormSelect):

    def __init__(self, model_file=None):
        # TODO
        pass

    def get_surface_form(self, sentence, pos, possible_forms):
        # TODO
        pass


class Lexicalizer(object):

    def __init__(self, abstr_file=None, surface_forms_file=None, form_select_model=None):
        self._values = []
        self._sf_all = {}
        self._sf_by_formeme = {}
        self._sf_by_tag = {}
        if abstr_file:
            self.load_lexicalization(abstr_file)
        if surface_forms_file:
            self.load_surface_forms(surface_forms_file)
        self._form_select = RandomFormSelect()
        if form_select_model:
            # TODO
            pass

    def load_surface_forms(self, surface_forms_fname):
        """Load all proper name surface forms from a file."""
        with file_stream(surface_forms_fname) as fh:
            data = json.load(fh)
        for slot, values in data.iteritems():
            self._sf_all[slot] = {}
            self._sf_by_formeme[slot] = {}
            self._sf_by_tag[slot] = {}
            for value in values.keys():
                for surface_form in values[value]:
                    form, tag = surface_form.split("\t")
                    if slot == 'street':  # add street number placeholders to addresses
                        value += ' _'
                        slot = 'address'
                    # store the value globally + for all possible tag subsets/formemes
                    self._sf_all[value] = self._sf_all.get(value, []) + [form]
                    self._sf_by_tag[value] = self._sf_by_tag.get(value, [])
                    for tag_subset in self._get_tag_subsets(tag):
                        self._sf_by_tag[value][tag_subset] = \
                            self._sf_by_tag[value].get(tag_subset, []) + [form]
                    self._sf_by_formeme[value] = self._sf_by_formeme.get(value, [])
                    for formeme in self._get_compatible_formemes(tag):
                        self._sf_by_formeme[value][formeme] = \
                                self._sf_by_formeme[value].get(formeme, []) + [form]

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
        self._values = abstrs

    def lexicalize(self, gen_trees, mode):
        """Lexicalize nodes in the generated trees (which may represent trees, tokens, or tagged lemmas).
        Expects lexicalization file (and surface forms file) to be loaded in the Lexicalizer object,
        otherwise nothing will happen. The actual operation depends on the generator mode.

        @param gen_trees: list of TreeData objects representing generated trees/tokens/tagged lemmas
        @param mode: generator mode (acceptable string values: "trees"/"tokens"/"tagged_lemmas")
        @return: None
        """
        for tree, lex_dict in zip(gen_trees, self._values):
            idx = 0
            while idx < len(tree):
                if tree[idx].t_lemma.startswith('X-'):
                    slot = tree[idx].t_lemma[2:]
                    # we can lexicalize
                    if slot in lex_dict:
                        # tagged lemmas: one token with appropriate value
                        if mode == 'tagged_lemmas':
                            val = self._get_surface_form(tree, idx, slot, lex_dict[slot][0],
                                                         tag=tree[idx+1].t_lemma)
                            tree[idx] = NodeData(t_lemma=val, formeme='x')
                        # trees: one node with appropriate value, keep formeme
                        elif mode == 'trees':
                            val = self._get_surface_form(tree, idx, slot, lex_dict[slot][0],
                                                         formeme=tree[idx].formeme)
                            tree[idx] = NodeData(t_lemma=val, formeme=tree[idx].formeme)
                        # tokens: multiple tokens with all words from the value
                        else:
                            value = self._get_surface_form(tree, idx, slot, lex_dict[slot][0])
                            for shift, tok in enumerate(value.split(' ')):
                                tree.create_child(0, idx + shift,
                                                  NodeData(t_lemma=tok, formeme='x'))
                            tree.remove_node(idx)
                            idx += shift
                        lex_dict[slot] = lex_dict[slot][1:] + lex_dict[slot][0]  # cycle the values
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
