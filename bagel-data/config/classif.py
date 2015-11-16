#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals


config = {'language': 'en',
          'selector': '',
          'nn': 'emb_prev',
          # 'nn': '1-hot',
          'nn_shape': 'conv-maxpool-ff',
          'num_hidden_units': 128,
          'initialization': 'norm_sqrt',
          'passes': 200,
          'randomize': True,
          'batch_size': 10,
          'alpha': 0.1,
          'emb_size': 50,
          }
