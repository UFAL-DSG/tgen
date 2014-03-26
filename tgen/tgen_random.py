#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generating random t-trees.

Usage: ./tgen_random train-das train-t-trees test-das output-t-trees
"""

from __future__ import unicode_literals
from alex.components.nlg.tectotpl.core.node import T
from alex.components.nlg.tectotpl.core.document import Document, Bundle, Zone
from alex.components.nlg.tectotpl.block.write.yaml import YAML as YAMLWriter
from alex.components.nlg.tectotpl.block.read.yaml import YAML as YAMLReader
from alex.components.slu.da import DialogueAct
from alex.components.nlg.tectotpl.core.util import file_stream
from flect.logf import log_info

import random
from collections import deque, defaultdict, Counter
from getopt import getopt
import sys
import cPickle as pickle

class RandomTTreeGenerator(object):

    MAX_TREE_SIZE = 50

    def __init__(self, language='en', selector='', probs=None):
        self.language = language
        self.selector = selector
        self.form_counts = None
        self.child_cdfs = None

    def generate_tree(self, da, doc=None):
        # create a document
        if doc is None:
            doc = Document()
        bundle = doc.create_bundle()
        zone = bundle.create_zone(self.language, self.selector)
        # creating a tree
        root = zone.create_ttree()
        cdfs = self.get_merged_cdfs(da)
        nodes = deque([self.__generate_child(root, cdfs[root.formeme])])
        treesize = 1
        while nodes and treesize < self.MAX_TREE_SIZE:
            node = nodes.popleft()
            if node.formeme not in self.child_cdfs or node.formeme not in cdfs:  # skip weirdness
                continue
            for _ in xrange(self.sample(self.child_cdfs[node.formeme])):
                child = self.__generate_child(node, cdfs[node.formeme])
                nodes.append(child)
                treesize += 1
        return doc

    def __generate_child(self, parent, cdf):
        child = parent.create_child()
        formeme, t_lemma, right = self.sample(cdf)
        child.t_lemma = t_lemma
        child.formeme = formeme
        if right:
            child.shift_after_subtree(parent)
        else:
            child.shift_before_subtree(parent)
        return child

    def train(self, da_file, t_file):
        # read training data
        log_info('Reading ' + t_file)
        if 'pickle' in t_file:
            ttrees = pickle.load(file_stream(t_file, mode='rb', encoding=None))
        else:
            yaml_reader = YAMLReader(scenario=None, args={})
            ttrees = yaml_reader.process_document(t_file)
            pickle_file = t_file.replace('yaml', 'pickle')
            fh = file_stream(pickle_file, mode='wb', encoding=None)
            pickle.dump(ttrees, fh, pickle.HIGHEST_PROTOCOL)
        log_info('Reading ' + da_file)
        das = self.read_das(da_file)
        # collect counts
        log_info('Collecting counts')
        form_counts = defaultdict(lambda: defaultdict(Counter))
        child_counts = defaultdict(Counter)
        for ttree, da in zip(ttrees.bundles, das):
            ttree = ttree.get_zone('en', '').ttree
            # counts for formeme/lemma given dai
            for dai in da:
                for tnode in ttree.get_descendants():
                    form_counts[dai][tnode.parent.formeme][(tnode.formeme, tnode.t_lemma, tnode > tnode.parent)] += 1
            # counts for number of children
            for tnode in ttree.get_descendants():
                child_counts[tnode.formeme][len(tnode.get_children())] += 1
        self.form_counts = form_counts
        self.child_cdfs = self.cdfs_from_counts(child_counts)

    def read_das(self, da_file):
        """Read dialogue acts from a file, one-per-line."""
        # TODO: move outside
        das = []
        with file_stream(da_file) as fh:
            for line in fh:
                da = DialogueAct()
                da.parse(line)
                das.append(da)
        return das

    def get_merged_cdfs(self, da):
        merged_counts = defaultdict(Counter)
        for dai in da:
            for parent_formeme in self.form_counts[dai]:
                merged_counts[parent_formeme].update(self.form_counts[dai][parent_formeme])
        return self.cdfs_from_counts(merged_counts)

    def cdfs_from_counts(self, counts):
        cdfs = {}
        for key in counts:
            tot = 0
            cdf = []
            for subkey in counts[key]:
                tot += counts[key][subkey]
                cdf.append((subkey, tot))
            cdfs[key] = cdf
        return cdfs

    def sample(self, cdf):
        max = cdf[-1][1]
        rand = random.random() * max
        for key, ubound in cdf:
            if ubound > rand:
                return key
        raise Exception('Unable to generate from CDF!')


if __name__ == '__main__':
    files = sys.argv[1:]
    if len(files) != 4:
        sys.exit(__doc__)
    fname_da_train, fname_ttrees_train, fname_da_test, fname_ttrees_out = files

    # initialize, train
    log_info('Initializing...')
    tgen = RandomTTreeGenerator()
    tgen.train(fname_da_train, fname_ttrees_train)

    # generate
    log_info('Generating...')
    doc = None
    das = tgen.read_das(fname_da_test)
    for da in das:
        doc = tgen.generate_tree(da, doc)

    # write output
    log_info('Writing output...')
    writer = YAMLWriter(scenario=None, args={'to': fname_ttrees_out})
    writer.process_document(doc)

