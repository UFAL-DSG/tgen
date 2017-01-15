#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Convert saved TF session to a dict of params in a pickle (which is more flexible, allows usage
with a different lexicalizer).
"""

from __future__ import unicode_literals
from argparse import ArgumentParser
from tgen.seq2seq import Seq2SeqBase
from pytreex.core.util import file_stream
import cPickle as pickle
import re
from tgen.logf import log_info
from tensorflow.python.framework.ops import reset_default_graph


def convert_model(model_fname):

    reset_default_graph()

    param_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.params.gz', model_fname)
    log_info('Converting %s to %s...' % (model_fname, param_fname))
    model = Seq2SeqBase.load_from_file(model_fname)
    with file_stream(param_fname, 'wb', encoding=None) as fh:
        pickle.dump(model.get_model_params(), fh, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('model_file', type=str, nargs='+',
                    help='Path to the model')
    args = ap.parse_args()

    for model_file in args.model_file:
        convert_model(model_file)
