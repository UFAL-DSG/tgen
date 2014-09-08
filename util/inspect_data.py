#!/usr/bin/env python
#
# Training data inspection. Currently prints only tree sizes

from tgen.futil import read_ttrees, trees_from_doc
import sys
from getopt import getopt
from collections import defaultdict

opts, files = getopt(sys.argv[1:], 'l:s:')

if len(files) != 1:
    sys.exit('Usage: python inspect_data.py [-l lang] [-s selector] <file.yaml>')

language = 'en'
selector = ''

for opt, arg in opts:
    if opt == '-l':
        language = arg
    elif opt == '-s':
        selector = arg

trees = trees_from_doc(read_ttrees(files[0]), language, selector)

tree_sizes = defaultdict(int)
for tree in trees:
    tree_sizes[len(tree)] += 1

print "Tree sizes:\n==========="
for k, v in sorted(tree_sizes.items()):
    print k, "\t:", v
