#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generating random t-trees.
"""

from __future__ import unicode_literals
from alex.components.nlg.tectotpl.core.node import T
from alex.components.nlg.tectotpl.core.document import Document, Bundle, Zone
from alex.components.nlg.tectotpl.block.write.yaml import YAML
import random
from collections import deque
from getopt import getopt
import sys

class RandomTTreeGenerator(object):

    T_LEMMA_DICT = ['dog', 'cat', 'bite']
    FORMEME_DICT = ['n:subj', 'n:obj', 'v:fin', 'v:inf']

    DEFAULT_PROBS = {'right': 0.7, 'child': 0.7, 'maxsize': 20.0}

    def __init__(self, language='en', selector='', probs=None, t_lemma_dict=None, formeme_dict=None):
        self.probs = probs or self.DEFAULT_PROBS
        self.t_lemma_dict = t_lemma_dict or self.T_LEMMA_DICT
        self.formeme_dict = formeme_dict or self.FORMEME_DICT
        self.language = language
        self.selector = selector

    def generate_tree(self, doc=None):
        # create a document
        if doc is None:
            doc = Document()
        bundle = doc.create_bundle()
        zone = bundle.create_zone(self.language, self.selector)
        root = zone.create_ttree()
        nodes = deque([self.__generate_child(root, right=True)])
        treesize = 1
        while nodes:
            node = nodes.popleft()
            while random.random() < self.probs['child'] * (1-treesize/self.probs['maxsize']):
                child = self.__generate_child(node, right=(random.random() < self.probs['right']))
                nodes.append(child)
                treesize += 1
        return doc

    def __generate_child(self, parent, right):
        child = parent.create_child()
        child.t_lemma = random.choice(self.t_lemma_dict)
        child.formeme = random.choice(self.formeme_dict)
        if right:
            child.shift_after_subtree(parent)
        else:
            child.shift_before_subtree(parent)
        return child


if __name__ == '__main__':
    opts, files = getopt(sys.argv[1:], 'n:')
    num_trees = 10
    for opt, arg in opts:
        if opt == '-n':
            num_trees = int(arg)
    if len(files) != 1:
        sys.exit(__doc__)

    tgen = RandomTTreeGenerator()
    doc = None
    for i in xrange(num_trees):
        doc = tgen.generate_tree(doc)
    
    writer = YAML(scenario=None, args={'to': files[0]})
    writer.process_document(doc)

