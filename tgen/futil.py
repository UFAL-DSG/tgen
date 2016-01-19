#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Various utility functions.
"""

from __future__ import unicode_literals
import cPickle as pickle

from alex.components.nlg.tectotpl.core.util import file_stream
from alex.components.slu.da import DialogueAct
from alex.components.nlg.tectotpl.block.read.yaml import YAML as YAMLReader
from alex.components.nlg.tectotpl.block.write.yaml import YAML as YAMLWriter
from tree import TreeData


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
    writer = YAMLWriter(scenario=None, args={'to': fname})
    writer.process_document(ttree_doc)


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
    return [[(node.form, node.tag) for node in atree.get_descendants(ordered=True)]
            for atree in atrees]


def add_bundle_text(bundle, language, selector, text):
    """Given a document bundle, add sentence text to the given language and selector."""
    zone = bundle.get_or_create_zone(language, selector)
    zone.sentence = (zone.sentence + ' ' if zone.sentence is not None else '') + text
