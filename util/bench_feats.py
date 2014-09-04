#!/usr/bin/env python


from flect.config import Config
from tgen.features import Features
from tgen.futil import trees_from_doc, read_ttrees, read_das
import sys
import timeit
import datetime

if len(sys.argv[1:]) != 3:
    sys.exit('Usage: ./bench_feats.py features_cfg.py trees.yaml.gz das.txt')

print >> sys.stderr, 'Loading...'

cfg = Config(sys.argv[1])
trees = trees_from_doc(read_ttrees(sys.argv[2]), 'en', '')
das = read_das(sys.argv[3])

feats = Features(cfg['features'])

def test_func():
    for tree, da in zip(trees, das):
        feats.get_features(tree, {'da': da})
    

print >> sys.stderr, 'Running test...'
secs = timeit.timeit('test_func()', setup='from __main__ import test_func', number=10)
td = datetime.timedelta(seconds=secs)
print >> sys.stderr, 'Time taken: %s' % str(td)
