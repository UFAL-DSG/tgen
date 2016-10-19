#!/usr/bin/env python
#
# Training data inspection.
# Prints tree sizes, DAs -> nodes/lemmas

from tgen.futil import read_ttrees, trees_from_doc, read_das
import sys
from getopt import getopt
from collections import defaultdict

opts, files = getopt(sys.argv[1:], 'l:s:')

if len(files) != 2:
    sys.exit('Usage: python inspect_data.py [-l lang] [-s selector] <trees.yaml> <das.txt>')

language = 'en'
selector = ''

for opt, arg in opts:
    if opt == '-l':
        language = arg
    elif opt == '-s':
        selector = arg

trees = trees_from_doc(read_ttrees(files[0]), language, selector)
das = read_das(files[1])


# TREE SIZES

tree_sizes = defaultdict(int)
for tree in trees:
    tree_sizes[len(tree)] += 1

print "Tree sizes:\n==========="
for k, v in sorted(tree_sizes.items()):
    print k, "\t", v

# DAS -> NODES

das_for_nodes = {}
num_occ_nodes = defaultdict(int)
for tree, da in zip(trees, das):
    for node in tree.nodes:
        num_occ_nodes[node] += 1
        if node not in das_for_nodes:
            das_for_nodes[node] = set(da.dais)
        else:
            das_for_nodes[node] &= set(da.dais)

print "\n\nDAIs for nodes:\n=========="
for node, dais in sorted(das_for_nodes.items()):
    print node.t_lemma, " ", node.formeme, "\t", num_occ_nodes[node], "\t", '&'.join([unicode(dai) for dai in dais])

# DAS -> LEMMAS

das_for_lemmas = {}
num_occ_lemmas = defaultdict(int)
num_occ_formemes = defaultdict(int)
for tree, da in zip(trees, das):
    for node in tree.nodes:
        lemma = unicode(node.t_lemma).lower()
        formeme = unicode(node.formeme)
        num_occ_lemmas[lemma] += 1
        num_occ_formemes[formeme] += 1
        if lemma not in das_for_lemmas:
            das_for_lemmas[lemma] = set(da.dais)
        else:
            das_for_lemmas[lemma] &= set(da.dais)

print "\n\nDAIs for lemmas:\n=========="
for lemma, dais in sorted(das_for_lemmas.items()):
    print lemma, "\t", num_occ_lemmas[lemma], "\t", '&'.join([unicode(dai) for dai in dais])

# DA slots -> LEMMAS

das_for_lemmas = {}
for tree, da in zip(trees, das):
    for node in tree.nodes:
        lemma = unicode(node.t_lemma).lower()
        if lemma not in das_for_lemmas:
            das_for_lemmas[lemma] = set([dai.name for dai in da.dais])
        else:
            das_for_lemmas[lemma] &= set([dai.name for dai in da.dais])

print "\n\nDA slots for lemmas:\n=========="
for lemma, dais in sorted(das_for_lemmas.items()):
    print lemma, "\t", num_occ_lemmas[lemma], "\t", ' '.join([unicode(dai) for dai in dais])


# Just lemmas, formemes
print "Lemmas, formemes"
print "Lemmas: ", len(num_occ_lemmas)
print "Formemes: ", len(num_occ_formemes)
