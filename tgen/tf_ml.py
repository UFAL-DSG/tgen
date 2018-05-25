#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorFlow Helper functions.
"""

from __future__ import unicode_literals
import os

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, control_flow_ops
from tensorflow.contrib.rnn import EmbeddingWrapper, OutputProjectionWrapper
from tensorflow.python.ops import variable_scope as vs

from tgen.logf import log_warn
import tgen.externals.seq2seq as tf06s2s


class TFModel(object):
    """An interface / methods wrapper for all TF models, used to load and save parameters."""

    def __init__(self, scope_name=None):
        """Just set variable scope name.
        @param scope_name: preferred variable scope name (may be None)
        """
        self.scope_name = scope_name

    def get_all_settings(self):
        """Get all settings except the trained model parameters (to be stored in a pickle)."""
        raise NotImplementedError()

    def load_all_settings(self, settings_dict):
        """Load all settings (except trained parameters) from a dictionary in the format provided
        by `get_all_settings`."""
        self.__dict__.update(settings_dict)

    def get_model_params(self):
        """Return the current model parameters in a dictionary (out of TensorFlow). Calls
        `var.eval()` on each model parameter.

        @return: all model parameters (variables), as numpy arrays, keyed in a dictionary under \
            their names
        """
        all_vars = tf.global_variables()
        ret = {}
        for var in all_vars:
            if not var.name.startswith(self.scope_name):  # skip variables not in my scope
                continue
            ret[var.name] = var.eval(session=self.session)
        return ret

    def set_model_params(self, vals):
        """Using a dictionary in the format returned by `get_model_params`, assign new parameter
        values.

        @param vals: a dictionary of new parameter values, as numpy arrays, keyed und their names \
            in a dictionary.
        """
        all_vars = tf.global_variables()
        for var in all_vars:
            if not var.name.startswith(self.scope_name):  # skip variables not in my scope
                continue
            if var.name in vals:
                op = var.assign(vals[var.name])
                self.session.run(op)

    def tf_check_filename(self, fname):
        """Checks if a directory is specified in the file name (otherwise newer TF versions
        would crash when saving a model).
        @param fname: The file name to be checked.
        @return: Adjusted file name (with "./" if no directory was specified)."""
        if not os.path.dirname(fname):
            log_warn("Directory not specified, using current directory: %s" % fname)
            fname = os.path.join(os.curdir, fname)
        return fname


def embedding_attention_seq2seq_context(encoder_inputs, decoder_inputs, cell,
                                        num_encoder_symbols, num_decoder_symbols,
                                        embedding_size,
                                        num_heads=1, output_projection=None,
                                        feed_previous=False, dtype=dtypes.float32,
                                        scope=None):
    """A seq2seq architecture with two encoders, one for context, one for input DA. The decoder
    uses twice the cell size. Code adapted from TensorFlow examples."""

    with vs.variable_scope(scope or "embedding_attention_seq2seq_context"):

        # split context and real inputs into separate vectors
        context_inputs = encoder_inputs[0:len(encoder_inputs) / 2]
        encoder_inputs = encoder_inputs[len(encoder_inputs) / 2:]

        # build separate encoders
        encoder_cell = EmbeddingWrapper(cell, num_encoder_symbols, embedding_size)
        with vs.variable_scope("context_rnn") as scope:
            context_outputs, context_states = tf06s2s.rnn(
                encoder_cell, context_inputs, dtype=dtype, scope=scope)
        with vs.variable_scope("input_rnn") as scope:
            encoder_outputs, encoder_states = tf06s2s.rnn(
                encoder_cell, encoder_inputs, dtype=dtype, scope=scope)

        # concatenate outputs & states
        # adding positional arguments and concatenating output, cell and hidden states
        encoder_outputs = [array_ops.concat([co, eo], axis=1, name="context-and-encoder-output")
                           for co, eo in zip(context_outputs, encoder_outputs)]
        encoder_states=[(array_ops.concat([c1, c2], axis=1), array_ops.concat([h1, h2], axis=1))
                        for (c1, h1), (c2, h2) in zip(context_states, encoder_states)]

        # calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size * 2])
                      for e in encoder_outputs]
        # added positional arguments since these swapped in some TF version
        attention_states = array_ops.concat(axis=1, values=top_states)

        # change the decoder cell to accommodate wider input
        # TODO this will work for BasicLSTMCell and GRUCell, but not for others
        cell = type(cell)(num_units=(cell.output_size * 2))

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            return tf06s2s.embedding_attention_decoder(
                decoder_inputs, encoder_states[-1], attention_states, cell,
                num_decoder_symbols, embedding_size, num_heads, output_size,
                output_projection, feed_previous)
        else:    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
            outputs1, states1 = tf06s2s.embedding_attention_decoder(
                decoder_inputs, encoder_states[-1], attention_states, cell,
                num_decoder_symbols, embedding_size, num_heads, output_size,
                output_projection, True)
            vs.get_variable_scope().reuse_variables()
            outputs2, states2 = tf06s2s.embedding_attention_decoder(
                decoder_inputs, encoder_states[-1], attention_states, cell,
                num_decoder_symbols, embedding_size, num_heads, output_size,
                output_projection, False)

            outputs = control_flow_ops.cond(feed_previous,
                                            lambda: outputs1, lambda: outputs2)
            states = control_flow_ops.cond(feed_previous,
                                           lambda: states1, lambda: states2)
            return outputs, states
