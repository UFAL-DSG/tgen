#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Various utility functions.
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import filter
from builtins import range
import pickle as pickle
import gzip
import regex
import re
from io import IOBase, StringIO
from codecs import StreamReader, StreamWriter

from .tree import TreeData
from .data import Abst, DA


def file_stream(filename, mode='r', encoding='UTF-8'):
    """Given a file stream or a file name, return the corresponding stream,
    handling GZip. Depending on mode, open an input or output stream.
    (A copy from pytreex.core.util to remove dependency)
    """
    # open file
    if isinstance(filename, (IOBase, StreamReader, StreamWriter)):
        fh = filename
    elif filename.endswith('.gz'):
        fh = gzip.open(filename, mode, encoding=encoding)
    else:
        fh = open(filename, mode, encoding=encoding)
    return fh


def read_das(da_file):
    """Read dialogue acts from a file, one-per-line."""
    das = []
    with file_stream(da_file) as fh:
        for line in fh:
            da = DA.parse(line.strip())
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


def smart_load_absts(fname, num_expected=None):
    """Load lexicalization instructions in a smart way, i.e., be able to detect DA files
    or abstraction files with multi-reference mode."""
    with file_stream(fname) as fh:
        contents = fh.read()
        buf = StringIO(contents)
        # read DAs and convert them to Absts
        if not re.search(r'\t', contents):
            return [[Abst(dai.slot, dai.value) for dai in da
                     if dai.value not in [None, 'dont_care', 'dontcare']]
                    for da in read_das(buf)]
        # multi-reference mode: read all but only output Absts for 1st reference of each instance
        elif re.search(r'(\n\n|\r\n\r\n|\r\r)', contents):
            abstss = read_absts(buf)
            if num_expected is not None and len(abstss) == num_expected:
                return abstss  # if there's 1:1 length correspondence, assume 1 reference
            out_abstss = []
            ref1st = True
            for absts in abstss:
                if not absts:
                    ref1st = True
                elif ref1st:
                    out_abstss.append(absts)
                    ref1st = False
            return out_abstss
        # plain 1-reference abstraction file
        else:
            return read_absts(buf)


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


def read_trees_or_tokens(input_file, mode, language=None, selector=None, ref_mode=False):
    """Load input trees/sentences from a .yaml.gz/.pickle.gz (trees) or .txt (sentences) file."""
    if input_file.endswith('.txt'):
        if mode == 'trees':
            raise ValueError("Cannot read trees from a .txt file (%s)!" % input_file)
        return read_tokens(input_file, ref_mode)
    elif input_file.endswith('.conll'):
        if mode == 'trees':
            raise ValueError("Cannot read trees from a .conll file (%s)!" % input_file)
        return read_conll(input_file, use_lemmas=(mode == 'tagged_lemmas'))
    else:
        ttree_doc = read_ttrees(input_file)
        if selector is None or selector is None:  # for txt files, language/selector can be None, but not here
            raise ValueError("Undefined language or selector while reading from a trees file (%s)!" % input_file)
        if mode == 'tokens':
            return tokens_from_doc(ttree_doc, language, selector)
        elif mode == 'tagged_lemmas':
            return tagged_lemmas_from_doc(ttree_doc, language, selector)
        else:
            return trees_from_doc(ttree_doc, language, selector)


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
        for _ in range(len(trees)):
            base_doc.create_bundle()
    for tree, bundle in zip(trees, base_doc.bundles):
        zone = bundle.get_or_create_zone(language, selector)
        zone.ttree = tree.create_ttree()
    return base_doc


def tokenize(text):
    """Tokenize the given text (i.e., insert spaces around all tokens)"""
    toks = ' ' + text + ' '  # for easier regexes

    # enforce space around all punct
    toks = regex.sub(r'(([^\p{IsAlnum}\s\.\,−\-])\2*)', r' \1 ', toks)  # all punct (except ,-.)
    toks = regex.sub(r'([^\p{N}])([,.])([^\p{N}])', r'\1 \2 \3', toks)  # ,. & no numbers
    toks = regex.sub(r'([^\p{N}])([,.])([\p{N}])', r'\1 \2 \3', toks)  # ,. preceding numbers
    toks = regex.sub(r'([\p{N}])([,.])([^\p{N}])', r'\1 \2 \3', toks)  # ,. following numbers
    toks = regex.sub(r'(–-)([^\p{N}])', r'\1 \2', toks)  # -/– & no number following
    toks = regex.sub(r'(\p{N} *|[^ ])(-)', r'\1\2 ', toks)  # -/– & preceding number/no-space
    toks = regex.sub(r'([-−])', r' \1', toks)  # -/– : always space before

    # keep apostrophes together with words in most common contractions
    toks = regex.sub(r'([\'’´]) (s|m|d|ll|re|ve)\s', r' \1\2 ', toks)  # I 'm, I 've etc.
    toks = regex.sub(r'(n [\'’´]) (t\s)', r' \1\2 ', toks)  # do n't

    # other contractions, as implemented in Treex
    toks = regex.sub(r' ([Cc])annot\s', r' \1an not ', toks)
    toks = regex.sub(r' ([Dd]) \' ye\s', r' \1\' ye ', toks)
    toks = regex.sub(r' ([Gg])imme\s', r' \1im me ', toks)
    toks = regex.sub(r' ([Gg])onna\s', r' \1on na ', toks)
    toks = regex.sub(r' ([Gg])otta\s', r' \1ot ta ', toks)
    toks = regex.sub(r' ([Ll])emme\s', r' \1em me ', toks)
    toks = regex.sub(r' ([Mm])ore\'n\s', r' \1ore \'n ', toks)
    toks = regex.sub(r' \' ([Tt])is\s', r' \'\1 is ', toks)
    toks = regex.sub(r' \' ([Tt])was\s', r' \'\1 was ', toks)
    toks = regex.sub(r' ([Ww])anna\s', r' \1an na ', toks)

    # clean extra space
    toks = regex.sub(r'\s+', ' ', toks)
    toks = toks.strip()
    return toks


def read_tokens(tok_file, ref_mode=False, do_tokenize=False):
    """Read sentences (one per line) from a file and return them as a list of tokens
    (forms with undefined POS tags)."""
    tokens = []
    empty_lines = False
    # read all lines from the file
    with file_stream(tok_file) as fh:
        for line in fh:
            # split to tokens + ignore consecutive spaces (no empty tokens)
            # empty line results in empty list
            line = line.strip()
            if do_tokenize:
                line = tokenize(line)
            line = list(filter(bool, line.split(' ')))
            if not line:
                empty_lines = True
            # TODO apply Morphodita here ?
            tokens.append([(form, None) for form in line])

    if ref_mode:
        # empty lines separate references from each other: regroup references by empty lines
        if empty_lines:
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
        # no empty lines = single-references: just put references into length-1 arrays
        else:
            tokens = [[ref] for ref in tokens]
    return tokens


def read_conll(conll_file, use_lemmas=False):
    """Read forms or lemmas + tags from a CoNLL-U formatted file. Ignores syntax."""
    sents = []
    with file_stream(conll_file) as fh:
        cur_sent = []
        for line in fh:
            line = line.strip()
            if not line and cur_sent:
                sents.append(cur_sent)
                cur_sent = []
                continue
            _, form, lemma, _, tag, _ = line.split("\t", 5)
            cur_sent.append((lemma if use_lemmas else form, tag))

    if cur_sent:
        sents.append(cur_sent)
    return sents


def write_tokens(doc, tok_file):
    """Write all sentences from a document into a text file."""
    with file_stream(tok_file, 'w') as fh:
        for sent in doc:
            toks = [tok for (tok, _) in sent]
            # TODO some nice detokenization etc.
            print(' '.join(toks), file=fh)


def chunk_list(l, n):
    """ Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
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
    return [bundle.get_zone(language, selector).sentence for bundle in ttree_doc.bundles]


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
                if tok == '-s' and idx > 0 and sent[idx - 1][0] == 'child':  # irregular plural
                    tok = '-ren'
                if idx > 0:
                    sent[idx - 1] = (sent[idx - 1][0] + tok[1:], sent[idx - 1][1])
                del sent[idx]
        # add final punctuation, if not present
        if sent[-1][0] not in ['?', '!', '.']:
            sent.append((final_punct, None))

    for sent, da in zip(tokens, das):
        final_punct = '?' if da[0].da_type[0] == '?' else '.'  # '?' for '?request...'
        if not sent:
            return  # ignore empty sentences
        if isinstance(sent[0], list):
            for sent_var in sent:  # multiple references
                postprocess_sent(sent_var, final_punct)
        else:
            postprocess_sent(sent, final_punct)
