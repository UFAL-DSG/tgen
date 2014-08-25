#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.linear_model.logistic import LogisticRegression


config = {
          'alpha': 0.1,
          'passes': 50,
          'features': [
#                        'lemma-count: value t_lemma',
#                        'formeme-count: value formeme',
                       'lemma-present: presence t_lemma',
                       'formeme-present: presence formeme',
                       'lemma-dai: dai_cooc t_lemma',
                       'formeme-dai: dai_cooc formeme',
                       'depth: depth',
                       'max-children: max_children',
                       'nodes-per-dai: nodes_per_dai',
                       'rep-nodes-per-rep-dai: rep_nodes_per_rep_dai',
                       'rep-nodes: rep_nodes',
                       ],
          'rival_number': 1,
          'rival_gen_strategy': ['gen_cur_weights'],
#          'rival_gen_strategy': ['other_inst', 'random'],
#          'rival_gen_strategy': ['gen_cur_weights', 'other_inst', 'random'],
          'rival_gen_max_iter': 12,
          'rival_gen_max_defic_iter': 2,
          # 'rival_gen_beam_size': 20, # actually slows it down
          }
