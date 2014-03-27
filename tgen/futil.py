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
        ttrees = pickle.load(file_stream(ttree_file, mode='rb', encoding=None))
    else:
        # if not pickled, read YAML and save a pickle nearby
        yaml_reader = YAMLReader(scenario=None, args={})
        ttrees = yaml_reader.process_document(ttree_file)
        pickle_file = ttree_file.replace('yaml', 'pickle')
        fh = file_stream(pickle_file, mode='wb', encoding=None)
        pickle.dump(ttrees, fh, pickle.HIGHEST_PROTOCOL)
    return ttrees
