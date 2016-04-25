#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Various utility functions.
"""

from __future__ import unicode_literals
import cPickle as pickle
import codecs
import gzip
from io import IOBase
from codecs import StreamReader, StreamWriter
import re

from alex.components.slu.da import DialogueAct
from tree import TreeData


def file_stream(filename, mode='r', encoding='UTF-8'):
    """\
    Given a file stream or a file name, return the corresponding stream,
    handling GZip. Depending on mode, open an input or output stream.
    (A copy from pytreex.core.util to remove dependency)
    """
    # open file
    if isinstance(filename, (file, IOBase, StreamReader, StreamWriter)):
        fh = filename
    elif filename.endswith('.gz'):
        fh = gzip.open(filename, mode)
    else:
        fh = open(filename, mode)
    # support encodings
    if encoding is not None:
        if mode.startswith('r'):
            fh = codecs.getreader(encoding)(fh)
        else:
            fh = codecs.getwriter(encoding)(fh)
    return fh


def read_das(da_file):
    """Read dialogue acts from a file, one-per-line."""
    das = []
    with file_stream(da_file) as fh:
        for line in fh:
            da = DialogueAct()
            da.parse(line)
            das.append(da)
    return das


def read_ttrees(ttree_file):
    """Read t-trees from a YAML/Pickle file."""
    from pytreex.block.read.yaml import YAML as YAMLReader
    if 'pickle' in ttree_file:
        # if pickled, read just the pickle
        fh = file_stream(ttree_file, mode='rb', encoding=None)
        unpickler = pickle.Unpickler(fh)
        ttrees = unpickler.load()
        fh.close()
    else:
        # if not pickled, read YAML and save a pickle nearby
        yaml_reader = YAMLReader(scenario=None, args={})
        ttrees = yaml_reader.process_document(ttree_file)
        pickle_file = ttree_file.replace('yaml', 'pickle')
        fh = file_stream(pickle_file, mode='wb', encoding=None)
        pickle.Pickler(fh, pickle.HIGHEST_PROTOCOL).dump(ttrees)
        fh.close()
    return ttrees


def write_ttrees(ttree_doc, fname):
    """Write a t-tree Document object to a YAML file."""
    from pytreex.block.write.yaml import YAML as YAMLWriter
    writer = YAMLWriter(scenario=None, args={'to': fname})
    writer.process_document(ttree_doc)


def read_tokens(tok_file, ref_mode=False):
    """Read sentences (one per line) from a file and return them as a list of tokens
    (forms with undefined POS tags)."""
    tokens = []
    empty_lines = False
    # read all lines from the file
    with file_stream(tok_file) as fh:
        for line in fh:
            line = line.strip().split(' ')
            if not line:
                empty_lines = True
            # TODO apply Morphodita here ?
            tokens.append([(form, None) for form in line])

    # empty lines separate references from each other: regroup references by empty lines
    if ref_mode and empty_lines:
        refs = []
        cur_ref = []
        for toks in tokens:
            if not toks:
                refs.push(cur_ref)
                cur_ref = []
            cur_ref.push(toks)
        if cur_ref:
            refs.push(cur_ref)
        tokens = refs
    return tokens

def lexicalization_from_doc(abstr_file):
    """Read lexicalization from a file with "abstraction instructions" (telling which tokens are
    delexicalized to what). This just remembers the slots and values and returns them in a dict
    for each line (slot -> list of values)."""
    abstrs = []
    with file_stream(abstr_file) as fh:
        for line in fh:
            line_abstrs = {}
            for svp in line.strip().split("\t"):
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


def lexicalize_tokens(doc, language, selector, lexicalization):
    """Given lexicalization dictionaries, this lexicalizes the nodes the generated trees."""
    for bundle, lex_dict in doc.bundles, lexicalization:
        ttree = bundle.get_zone(language, selector).ttree
        for tnode in ttree:
            if tnode.t_lemma.startswith('X-'):
                slot = tnode.t_lemma[2:]
                if slot in lex_dict:
                    value = lex_dict[slot][0]
                    tnode.t_lemma = value  # lexicalize
                    lex_dict[slot] = lex_dict[slot][1:] + value  # cycle the values


def write_tokens(doc, language, selector, tok_file):
    """Write all sentences from a document into a text file."""
    with file_streem(tok_file, 'w') as fh:
        for bundle in doc.bundles:
            ttree = bundle.get_zone(language, selector).ttree
            toks = [tnode.t_lemma for tnode in ttree.get_descendants(ordered=True)]
            # TODO some nice detokenization etc.
            print >> fh, ' '.join(toks)

def chunk_list(l, n):
    """ Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def ttrees_from_doc(ttree_doc, language, selector):
    """Given a Treex document full of t-trees, return just the array of t-trees."""
    return map(lambda bundle: bundle.get_zone(language, selector).ttree,
               ttree_doc.bundles)


def trees_from_doc(ttree_doc, language, selector):
    """Given a Treex document full of t-trees, return TreeData objects for each of them."""
    return map(lambda bundle: TreeData.from_ttree(bundle.get_zone(language, selector).ttree),
               ttree_doc.bundles)


def sentences_from_doc(ttree_doc, language, selector):
    """Given a Treex document, return a list of sentences in the given language and selector."""
    return map(lambda bundle: bundle.get_zone(language, selector).sentence, ttree_doc.bundles)


def tokens_from_doc(ttree_doc, language, selector):
    """Given a Treex document, return a list of lists of tokens (word forms + tags) in the given
    language and selector."""
    atrees = map(lambda bundle: bundle.get_zone(language, selector).atree, ttree_doc.bundles)
    sents = []
    for atree in atrees:
        anodes = atree.get_descendants(ordered=True)
        sent = []
        for anode in anodes:
            form, tag = anode.form, anode.tag
            if form == 'X':
                tnodes = anode.get_referencing_nodes('a/lex.rf')
                if tnodes:
                    form = tnodes[0].t_lemma
            sent.append((form, tag))
        sents.append(sent)
    return sents


def add_bundle_text(bundle, language, selector, text):
    """Given a document bundle, add sentence text to the given language and selector."""
    zone = bundle.get_or_create_zone(language, selector)
    zone.sentence = (zone.sentence + ' ' if zone.sentence is not None else '') + text
