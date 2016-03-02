#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals


config = {'language': 'en',
          'selector': '',
          'use_tokens': True,
          #'nn': '1-hot',
          'nn': 'emb',
          #'nn_shape': 'ff1',
          'nn_shape': 'rnn',
          'num_hidden_units': 128,
          'passes': 100,
          'min_passes': 20,
          'randomize': True,
          'batch_size': 20,
          'alpha': 1e-3,
          'emb_size': 50,
          'max_tree_len': 50,
          'validation_freq': 1,
          }
