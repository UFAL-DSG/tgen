#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Simple statistics about types of inner nodes and leaves 
# in the training data.
#

from tgen.futil import read_ttrees, ttrees_from_doc
import sys
import codecs

if len(sys.argv[1:]) != 1:
    sys.exit('Usage: python leaf_stats.py <file.pickle>')

stats = {}

for ttree in ttrees_from_doc(read_ttrees(sys.argv[1]), 'en', ''):
    for tnode in ttree.get_descendants():
        node_id = (tnode.t_lemma, tnode.formeme)
        if node_id not in stats:
            stats[node_id] = {'leaf': 0, 'int': 0}
        if tnode.get_children():
            stats[node_id]['int'] += 1
        else:
            stats[node_id]['leaf'] += 1


out = codecs.getwriter('UTF-8')(sys.stdout)
for node_id, val in stats.iteritems():
    print >> out, node_id, "\t", val['int'], "\t", val['leaf']
