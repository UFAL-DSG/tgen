#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

config = {
          'alpha': 1e-3,
          'randomize': True,
          'num_hidden_units': 128,
          'emb_size': 50,
          'batch_size': 20,
          'optimizer_type': 'adam',
          'max_cores': 4,
          'use_tokens': True,
          'nn_type': 'emb_attention_seq2seq',
          'sort_da_emb': True,
          'cell_type': 'lstm',
          'dropout_keep_prob': 1,

          'validation_size': 10,
          'validation_freq': 1,
          'passes': 1000,
          'min_passes': 100,
          'improve_interval': 100,
          'top_k': 10,
          'bleu_validation_weight': 1,
          'beam_size': 10,
          }
