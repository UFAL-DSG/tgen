#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals


config = {'language': 'en',
          'selector': '',
          # 'nn': '1-hot',
          'nn': 'emb_prev',
          # 'nn_shape': 'ff1',
          'nn_shape': 'conv-maxpool-ff',
          'num_hidden_units': 128,
          'initialization': 'norm_sqrt',
          # 'passes': 200,
          'passes': 100,
          'randomize': True,
          'batch_size': 20,
          'alpha': 0.1,
          'emb_size': 50,
          }
