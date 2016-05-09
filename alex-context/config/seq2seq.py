#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

config = {
          'alpha': 5e-4,
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
          'use_dec_cost': False,
          'use_context': False,
          'use_div_token': False,
          'context_bleu_weight': 0,
          #'average_models': True,
          #'average_models_top_k': 3,

          'validation_size': 100,
          'validation_freq': 1,
          'multiple_refs': '3,parallel',
          'ref_selectors': 'ref0,ref1,ref2',
          'passes': 1000,
          'min_passes': 50,
          'improve_interval': 50,
          'top_k': 10,
          'bleu_validation_weight': 1,
          'beam_size': 20,
          'alpha_decay': 0, # 0.03

         'classif_filter': {
              'language': 'en',
              'selector': '',
              #'nn': '1-hot',
              'nn': 'emb',
              #'nn_shape': 'ff1',
              'nn_shape': 'rnn',
              'num_hidden_units': 128,
              #'passes': 200,
              'passes': 100,
              'min_passes': 10,
              'randomize': True,
              'batch_size': 20,
              'alpha': 1e-3,
              'emb_size': 50,
              'max_tree_len': 50,
              'validation_freq': 1,
              }
          }
