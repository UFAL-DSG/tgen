#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import numpy as np
import cPickle as pickle
import tensorflow as tf

from tgen.tree import TreeData
from tgen.seq2seq import Seq2SeqBase, Seq2SeqGen, cut_batch_into_steps
from tgen.tfclassif import RerankingClassifier

from tgen.logf import log_info
from pytreex.core.util import file_stream


class Seq2SeqEnsemble(Seq2SeqBase):
    """Ensemble Sequence-to-Sequence models (averaging outputs of networks with different random
    initialization)."""

    def __init__(self, cfg):
        super(Seq2SeqEnsemble, self).__init__(cfg)

        self.gens = []

    def build_ensemble(self, models, rerank_settings=None, rerank_params=None):
        """Build the ensemble model (build all networks and load their parameters).

        @param models: list of tuples (settings, parameter set) of all models in the ensemble
        @param rerank_settings:
        """

        for setting, parset in models:
            model = Seq2SeqGen(setting['cfg'])
            model.load_all_settings(setting)
            model._init_neural_network()
            model.set_model_params(parset)
            self.gens.append(model)

        # embedding IDs should be the same for all models, it is safe to use them directly
        self.da_embs = self.gens[0].da_embs
        self.tree_embs = self.gens[0].tree_embs

        if rerank_settings is not None:
            self.classif_filter = RerankingClassifier(cfg=rerank_settings['cfg'])
            self.classif_filter.load_all_settings(rerank_settings)
            self.classif_filter._init_neural_network()
            self.classif_filter.set_model_params(rerank_params)

    def _get_greedy_decoder_output(self, enc_inputs, dec_inputs, compute_cost=False):
        """Run greedy decoding with the given inputs; return decoder outputs and the cost
        (if required). For ensemble decoding, the gready search is implemented as a beam
        search with a beam size of 1.

        @param enc_inputs: encoder inputs (list of token IDs)
        @param dec_inputs: decoder inputs (list of token IDs)
        @param compute_cost: if True, decoding cost is computed (the dec_inputs must be valid trees)
        @return a tuple of list of decoder outputs + decoding cost (None if not required)
        """
        # TODO batches and cost computation not implemented
        assert len(enc_inputs[0]) == 1 and not compute_cost

        self._init_beam_search(enc_inputs)

        # for simplicity, this is implemented exacly like a beam search, but with a path sized one
        empty_tree_emb = self.tree_embs.get_embeddings(TreeData())
        dec_inputs = cut_batch_into_steps([empty_tree_emb])
        path = self.DecodingPath(stop_token_id=self.tree_embs.STOP, dec_inputs=[dec_inputs[0]])

        for step in xrange(len(dec_inputs)):
            out_probs, st = self._beam_search_step(path.dec_inputs, path.dec_states)
            path = path.expand(1, out_probs, st)[0]

            if path.dec_inputs[-1] == self.tree_embs.VOID:
                break  # stop decoding if we have reached the end of path

        # return just token IDs, ignore cost computation here
        return np.array(path.dec_inputs), None

    def _init_beam_search(self, enc_inputs):
        """Initialize beam search for the current DA (with the given encoder inputs)
        for all member generators."""
        for gen in self.gens:
            gen._init_beam_search(enc_inputs)

    def _beam_search_step(self, dec_inputs, dec_states):
        """Run one step of beam search decoding with the given decoder inputs and
        (previous steps') outputs and states. Outputs are averaged over all member generators,
        states are kept separately."""
        ensemble_state = []
        ensemble_output = None

        for gen_no, gen in enumerate(self.gens):
            output, state = gen._beam_search_step(dec_inputs,
                                                  [state[gen_no] for state in dec_states])
            ensemble_state.append(state)
            output = np.exp(output) / np.sum(np.exp(output))
            if ensemble_output is None:
                ensemble_output = output
            else:
                ensemble_output += output

        ensemble_output /= float(len(self.gens))

        return ensemble_output, ensemble_state

    @staticmethod
    def load_from_file(model_fname):
        """Load the whole ensemble from a file (load settings and model parameters, then build the
        ensemble network)."""
        # TODO support for lexicalizer

        log_info("Loading ensemble generator from %s..." % model_fname)

        with file_stream(model_fname, 'rb', encoding=None) as fh:
            typeid = pickle.load(fh)
            if typeid != Seq2SeqEnsemble:
                raise ValueError('Wrong type identifier in file %s' % model_fname)
            cfg = pickle.load(fh)
            ret = Seq2SeqEnsemble(cfg)
            gens_dump = pickle.load(fh)
            if 'classif_filter' in cfg:
                rerank_settings = pickle.load(fh)
                rerank_params = pickle.load(fh)
            else:
                rerank_settings = None
                rerank_params = None

        ret.build_ensemble(gens_dump, rerank_settings, rerank_params)
        return ret

    def save_to_file(self, model_fname):
        """Save the whole ensemble into a file (get all settings and parameters, dump them in a
        pickle)."""
        # TODO support for lexicalizer

        log_info("Saving generator to %s..." % model_fname)
        with file_stream(model_fname, 'wb', encoding=None) as fh:
            pickle.dump(self.__class__, fh, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.cfg, fh, protocol=pickle.HIGHEST_PROTOCOL)

            gens_dump = []
            for gen in self.gens:
                setting = gen.get_all_settings()
                parset = gen.get_model_params()
                setting['classif_filter'] = self.classif_filter is not None
                gens_dump.append((setting, parset))

            pickle.dump(gens_dump, fh, protocol=pickle.HIGHEST_PROTOCOL)

            if self.classif_filter:
                pickle.dump(self.classif_filter.get_all_settings(), fh,
                            protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.classif_filter.get_model_params(), fh,
                            protocol=pickle.HIGHEST_PROTOCOL)
