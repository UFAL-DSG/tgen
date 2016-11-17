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

from tree import TreeData
from data import Abst, DA


def file_stream(filename, mode='r', encoding='UTF-8'):
    """Given a file stream or a file name, return the corresponding stream,
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
            da = DA.parse(line)
            das.append(da)
    return das


def read_absts(abst_file):
    """Read abstraction/lexicalization instructions from a file, one sentence per line.
    @param abst_file: path to the file containing lexicalization instructions
    @return: list of list of Abst objects, representing the instructions
    """
    abstss = []
    with file_stream(abst_file) as fh:
        for line in fh:
            line = line.strip()
            absts = []
            if line:  # empty lines = empty abstraction instructions
                for abst_str in line.split("\t"):
                    absts.append(Abst.parse(abst_str))
            abstss.append(absts)
    return abstss


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


def create_ttree_doc(trees, base_doc, language, selector):
    """Create a t-tree document or add generated trees into existing
    document.

    @param trees: trees to add into the document
    @param base_doc: pre-existing document (where trees will be added) or None
    @param language: language of the trees to be added
    @param selector: selector of the trees to be added
    """
    if base_doc is None:
        from pytreex.core.document import Document
        base_doc = Document()
        for _ in xrange(len(trees)):
            base_doc.create_bundle()
    for tree, bundle in zip(trees, base_doc.bundles):
        zone = bundle.get_or_create_zone(language, selector)
        zone.ttree = tree.create_ttree()
    return base_doc


def read_tokens(tok_file, ref_mode=False):
    """Read sentences (one per line) from a file and return them as a list of tokens
    (forms with undefined POS tags)."""
    tokens = []
    empty_lines = False
    # read all lines from the file
    with file_stream(tok_file) as fh:
        for line in fh:
            # split to tokens + ingore consecutive spaces (no empty tokens)
            # empty line results in empty list
            line = filter(bool, line.strip().split(' '))
            if not line:
                empty_lines = True
            # TODO apply Morphodita here ?
            tokens.append([(form, None) for form in line])

    # empty lines separate references from each other: regroup references by empty lines
    if ref_mode and empty_lines:
        refs = []
        cur_ref = []
        for toks in tokens:
            if not toks:  # empty line separates references
                refs.append(cur_ref)
                cur_ref = []
            else:
                cur_ref.append(toks)
        if cur_ref:
            refs.append(cur_ref)
        tokens = refs
    return tokens


def write_tokens(doc, tok_file):
    """Write all sentences from a document into a text file."""
    with file_stream(tok_file, 'w') as fh:
        for sent in doc:
            toks = [tok for (tok, _) in sent]
            # TODO some nice detokenization etc.
            print >> fh, ' '.join(toks)


def chunk_list(l, n):
    """ Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def ttrees_from_doc(ttree_doc, language, selector):
    """Given a Treex document full of t-trees, return just the array of t-trees."""
    selectors = selector.split(',')
    return [bundle.get_zone(language, sel).ttree
            for bundle in ttree_doc.bundles
            for sel in selectors]


def trees_from_doc(ttree_doc, language, selector):
    """Given a Treex document full of t-trees, return TreeData objects for each of them."""
    selectors = selector.split(',')
    return [TreeData.from_ttree(bundle.get_zone(language, sel).ttree)
            for bundle in ttree_doc.bundles
            for sel in selectors]


def sentences_from_doc(ttree_doc, language, selector):
    """Given a Treex document, return a list of sentences in the given language and selector."""
    return map(lambda bundle: bundle.get_zone(language, selector).sentence, ttree_doc.bundles)


def tokens_from_doc(ttree_doc, language, selector):
    """Given a Treex document, return a list of lists of tokens (word forms + tags) in the given
    language and selector."""
    sents = []
    for atree in atrees_from_doc(ttree_doc, language, selector):
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


def tagged_lemmas_from_doc(ttree_doc, language, selector):
    """Given a Treex document, return a list of lists of tagged lemmas (interleaved lemmas + tags)
    in the given language and selector."""
    sents = []
    for atree in atrees_from_doc(ttree_doc, language, selector):
        anodes = atree.get_descendants(ordered=True)
        sent = []
        for anode in anodes:
            lemma, tag = anode.lemma, anode.tag
            sent.append((lemma, tag))
        sents.append(sent)
    return sents


def atrees_from_doc(ttree_doc, language, selector):
    """Given a Treex document, return a list of a-trees (surface trees)
    in the given language and selector."""
    selectors = selector.split(',')
    atrees = [bundle.get_zone(language, sel).atree
              for bundle in ttree_doc.bundles
              for sel in selectors]
    return atrees


def add_bundle_text(bundle, language, selector, text):
    """Given a document bundle, add sentence text to the given language and selector."""
    zone = bundle.get_or_create_zone(language, selector)
    zone.sentence = (zone.sentence + ' ' if zone.sentence is not None else '') + text


def postprocess_tokens(tokens, das):
    """Postprocessing for BLEU measurements and token outputs: special morphological tokens (-s,
    -ly) are removed, final punctuation added where needed."""

    def postprocess_sent(sent, final_punct):
        """Postprocess a single sentence (called for multiple references)."""
        # merge plurals and adverbial "-ly"
        for idx, (tok, pos) in enumerate(sent):
            if tok == '-ly' or tok == '-s':
                if tok == '-s' and idx > 0 and sent[idx-1][0] == 'child':  # irregular plural
                    tok = '-ren'
                if idx > 0:
                    sent[idx-1] = (sent[idx-1][0] + tok[1:], sent[idx-1][1])
                del sent[idx]
        # add final punctuation, if not present
        if sent[-1][0] not in ['?', '!', '.']:
            sent.append((final_punct, None))

    for sent, da in zip(tokens, das):
        final_punct = '?' if da[0].da_type[0] == '?' else '.'  # '?' for '?request...'
        if isinstance(sent[0], list):
            for sent_var in sent:  # multiple references
                postprocess_sent(sent_var, final_punct)
        else:
            postprocess_sent(sent, final_punct)
