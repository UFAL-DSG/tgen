#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.linear_model.logistic import LogisticRegression


config = {
          'model': {
                    'class_attr': 'sel',
                    'classifier_class': LogisticRegression,
                    'classifier_params': {
                                          'penalty': ['l1'],
                                          'C': [1, 10, 100, 1000],
                                          'tol': [0.01, 0.001, 0.0001]
                                          },
                    'unfold_pattern': '^(penalty|C|tol)$'
                    },
          'features': [
                       'prob: prob',
                       'parent_formeme: value parent formeme',
                       'parent_t_lemma: value parent t_lemma',
                       'parent_isright: value parent right',
                       'parent+grandpa_formeme: value parent+grandpa formeme',
                       'parent+grandpa_t_lemma: value parent+grandpa t_lemma',
                       'parent+grandpa_isright: value parent+grandpa right',
                       'parent+siblings_formeme: value parent+siblings formeme',
                       'parent+siblings_t_lemma: value parent+siblings t_lemma',
                       'parent+siblings_isright: value parent+siblings right',
                       'parent+siblings+grandpa+uncles_formeme: value parent+siblings+grandpa+uncles formeme',
                       'parent+siblings+grandpa+uncles_t_lemma: value parent+siblings+grandpa+uncles t_lemma',
                       'parent+siblings+grandpa+uncles_isright: value parent+siblings+grandpa+uncles right',
                       'parent+siblings+grandpa+uncles_same_formeme: same_as_current parent+siblings+grandpa+uncles formeme',
                       'parent+siblings+grandpa+uncles_same_t_lemma: same_as_current parent+siblings+grandpa+uncles t_lemma',
                       'parent+siblings+grandpa+uncles_same_isright: same_as_current parent+siblings+grandpa+uncles right',
                       ],
          'attrib_order': ['prob_'],
          'attrib_types': {'prob_': 'numeric'},
          }
