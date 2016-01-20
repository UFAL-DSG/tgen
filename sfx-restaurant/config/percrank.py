#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

config = {
          'job_memory': 8,
          'averaging': True,
          'prune_feats': 1,  # pruning harms => do not use it
          'intermediate_features': ['rep-lemma', 'rep-slot', 'rep-lemma-formeme', 'not-rep-slot'],
          'features': [
                       # TREE SIZE FEATURES
                       'depth: depth',
                       'nodes-per-dai: nodes_per_dai',
                       'rep-nodes-per-rep-dai: rep_nodes_per_rep_dai',
                       'tree-size: tree_size',
                       #
                       'lemma: presence t_lemma',
                       'formeme: presence formeme',
                       'lemma-formeme: presence t_lemma formeme',
                       'formeme-numc: presence formeme num_children',
                       'lemma-formeme-numc: presence t_lemma formeme num_children',
                       'dai: dai_presence',
                       'slot: slot_presence',
                       'slot^2: combine slot slot',
                       'lemma-formeme+slot^2: combine lemma-formeme slot^2',
                       'lemma+slot^2: combine lemma slot^2',
                       # 'lemma^2: combine lemma lemma',
                       # 'lemma^2+slot: combine lemma^2 slot',
                       # 'lemma^2+dai: combine lemma^2 dai',
                       # 'lemma^2+slot^2: combine lemma^2 slot^2',
                       'lemma+dai: combine lemma dai',
                       'formeme+dai: combine formeme dai',
                       'lemma-formeme+dai: combine lemma-formeme dai',
                       'formeme-numc+dai: combine formeme-numc dai',
                       'lemma-formeme-numc+dai: combine lemma-formeme-numc dai',
                       'max-children: max_children',
                       'rep-nodes: rep_nodes',
                       'dep-lemma: dependency t_lemma',
                       'dep-formeme: dependency formeme',
                       'dep-lemma-formeme: dependency t_lemma formeme',
                       'dirdep-lemma: dir_dependency t_lemma',
                       'dirdep-formeme: dir_dependency formeme',
                       'dirdep-lemma-formeme: dir_dependency t_lemma formeme',
                       # 'dirdep-lemma+dai: combine dirdep-lemma dai',  # TW
                       'dirdep-lemma-formeme+dai: combine dirdep-lemma-formeme dai',
                       'slot-lemma: combine slot lemma',
#                        'count-lemma: count t_lemma',
#                        'count-slot: slot_count',
#                        'diff-count-lemma-slot: difference count-lemma count-slot',
                       'rep-lemma: repeated t_lemma',
                       'rep-lemma-formeme: repeated t_lemma formeme',
                       'rep-slot: slot_repeated',
                       'rep-lemma-formeme+rep-slot: combine rep-lemma-formeme rep-slot',  # TW-
                       # 'rep-lemma+rep-slot combine rep-lemma rep-slot',  # TW
                       # 'rep-lemma+slot^2 combine rep-lemma slot^2',  # TW
                       'lemma-formeme+rep-slot: combine lemma-formeme rep-slot',  # TW-
                       'not-rep-slot: set_difference slot rep-slot',
                       'rep-lemma+not-slot: combine rep-lemma not-rep-slot',
                       # 'count-slot+lemma: combine count-slot lemma',
                       # 'count-slot+formeme: combine count-slot formeme',
                       'siblings-lemma: siblings t_lemma',
                       'siblings-formeme: siblings formeme',
                       'siblings-lemma-formeme: siblings t_lemma formeme',
                       'siblings-lemma-formeme+dai: combine siblings-lemma-formeme dai',
                       'bigrams-lemma: bigrams t_lemma',
                       'bigrams-formeme: bigrams formeme',
                       'bigrams-lemma-formeme: bigrams t_lemma formeme',
                       'bigrams-lemma-formeme+dai: combine bigrams-lemma-formeme dai',
                       # 'trigrams-lemma: trigrams t_lemma',
                       # 'trigrams-formeme: trigrams formeme',
                       # 'trigrams-lemma-formeme: trigrams t_lemma formeme',
                       # 'trigrams-lemma-formeme+dai: combine trigrams-lemma-formeme dai',
                       ],
          'rival_number': 1,
          'rival_gen_strategy': [ (0, ['gen_update']), (5, ['gen_update','gen_cur_weights']), (10, ['gen_cur_weights']) ],
#          'rival_gen_strategy': ['other_inst', 'random'],
#          'rival_gen_strategy': ['gen_cur_weights', 'other_inst', 'random'],
          'diffing_trees': 'sym',  # False - sym/asym - nocommon? - nobad/onebad? - weighted?
                                   # (do not use nobad/onebad, it hurts badly)
          'binarize': False,
          'randomize': True,
          'passes': 100,
          'rival_gen_max_iter': 200,
          'rival_gen_max_defic_iter': 3,
          # 'rival_gen_beam_size': 1,
          'future_promise_weight': 0,
          'future_promise_type': 'norm_exp_children',  # default: exp_children
          'nn': False,  # 'emb_prev',  # 'emb',
          'initialization': 'norm_sqrt',
          'num_hidden_units': 128,
          'emb_size': 50,
          'alpha': 0.01,
          'nn_shape': 'conv2-ff',
          'cnn_filter_length': 3,
          'cnn_num_filters': 6,
          'normgrad': False,
          }
