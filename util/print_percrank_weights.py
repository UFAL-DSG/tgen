#!/usr/bin/env python

from tgen.rank import PerceptronRanker
import sys
import operator

if len(sys.argv[1:]) != 1:
    sys.exit('Usage: python print_percrank_weights.py <percrank.pickle.gz>')

percrank = PerceptronRanker.load_from_file(sys.argv[1])
fw = [(name, weight)
      for name, weight in zip(percrank.vectorizer.get_feature_names(), percrank.w)]
fw.sort(key=operator.itemgetter(1))
print "\n".join(['%s\t%.3f' % (name, weight) for name, weight in fw])

