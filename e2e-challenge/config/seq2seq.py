#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

config = {
          'language': 'en',
          'selector': '',
          'alpha': 5e-4,
          'randomize': True,
          'num_hidden_units': 128,
          'emb_size': 50,
          'batch_size': 20,
          'max_sent_len': 80,
          'optimizer_type': 'adam',
          'max_cores': 4,
          'mode': 'tokens',
          'nn_type': 'emb_attention_seq2seq_context',
          'sort_da_emb': True,
          'cell_type': 'lstm',
          'dropout_keep_prob': 1,
          'use_dec_cost': False,
          'use_context': True,
          'use_div_token': False,
          'context_bleu_weight': 3,
          'average_models': True,
          'average_models_top_k': 3,

          'validation_size': 2000,
          'validation_freq': 1,
          'validation_use_all_refs': True,
          'validation_use_train_refs': True,
          'validation_delex_slots': 'name,near',
          #'multiple_refs': '3,parallel',
          #'ref_selectors': 'ref0,ref1,ref2',
          'passes': 1,
          'min_passes': 5,
          'improve_interval': 5,
          'top_k': 3,
          'bleu_validation_weight': 1,
          'beam_size': 10,
          'alpha_decay': 0, # 0.03

         'classif_filter': {
              'language': 'cs',
              'selector': '',
              #'nn': '1-hot',
              'nn': 'emb',
              #'nn_shape': 'ff1',
              'nn_shape': 'rnn',
              'num_hidden_units': 128,
              #'passes': 200,
              'passes': 1,
              'min_passes': 5,
              'randomize': True,
              'batch_size': 20,
              'alpha': 1e-3,
              'emb_size': 50,
              'max_tree_len': 80,
              'validation_freq': 1,
              'delex_slots': 'name,area,address,phone,good_for_meal,near,food,price_range,count,price,postcode',
              },

        'lexicalizer': {
            'form_select_type': 'random',
        },
}
