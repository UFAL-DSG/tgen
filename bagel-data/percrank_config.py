#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.linear_model.logistic import LogisticRegression


config = {
          'alpha': 0.01,
          'passes': 5,
          'train_cands': 10,
          'features': [
                       'lemmas: value tree t_lemma',
                       'formemes: value tree formeme',
                       'depth: depth tree',
                       ],
          }
