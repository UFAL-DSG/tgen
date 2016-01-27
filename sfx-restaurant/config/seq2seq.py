#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

config = {
          'alpha': 1e-3,
          'randomize': True,
          'passes': 200,
          'num_hidden_units': 128,
          'emb_size': 50,
          'batch_size': 20,
          'validation_freq': 10,
          'validation_size': 10,
          'max_cores': 4,
          'use_tokens': False,
          'nn_type': 'emb_attention_seq2seq',
          'sort_da_emb': True,
          'cell_type': 'lstm',
          }
