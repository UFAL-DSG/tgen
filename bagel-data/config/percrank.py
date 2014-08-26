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
                       'lemma: presence t_lemma',
                       'formeme: presence formeme',
                       'lemma-formeme: presence t_lemma formeme',
                       'formeme-numc: presence formeme num_children',
                       'lemma-formeme-numc: presence t_lemma formeme num_children',
                       'dai: dai_presence',
                       'lemma+dai: combine lemma dai',
                       'formeme+dai: combine formeme dai',
                       'lemma-formeme+dai: combine lemma-formeme dai',
                       'formeme-numc+dai: combine formeme-numc dai',
                       'lemma-formeme-numc+dai: combine lemma-formeme-numc dai',
                       'depth: depth',
                       'max-children: max_children',
                       'nodes-per-dai: nodes_per_dai',
                       'rep-nodes-per-rep-dai: rep_nodes_per_rep_dai',
                       'rep-nodes: rep_nodes',
                       'dep-lemma: dependency t_lemma',
                       'dep-formeme: dependency formeme',
                       'dep-lemma-formeme: dependency t_lemma formeme',
                       'dirdep-lemma: dir_dependency t_lemma',
                       'dirdep-formeme: dir_dependency formeme',
                       'dirdep-lemma-formeme: dir_dependency t_lemma formeme',
                       'dirdep-lemma-formeme+dai: combine dirdep-lemma-formeme dai',
                       'tree-size: tree_size',
                       ],
          'rival_number': 1,
          'rival_gen_strategy': ['gen_cur_weights'],
#          'rival_gen_strategy': ['other_inst', 'random'],
#          'rival_gen_strategy': ['gen_cur_weights', 'other_inst', 'random'],
          'rival_gen_max_iter': 12,
          'rival_gen_max_defic_iter': 2,
          # 'rival_gen_beam_size': 20, # actually slows it down
          }
