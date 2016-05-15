#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Selecting pairs of different output sentences for pairwise testing.
(including deduplication)
"""

from __future__ import unicode_literals
import codecs
import random
import re
import sys
from argparse import ArgumentParser
from HTMLParser import HTMLParser
from itertools import cycle

import unicodecsv as csv

from tgen.debug import exc_info_hook
sys.excepthook = exc_info_hook

class SGMParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.in_seg = False
        self.data = []

    def handle_starttag(self, tag, attrs):
        if tag == 'seg':
            self.in_seg = True

    def handle_endtag(self, tag):
        if tag == 'seg':
            self.in_seg = False

    def handle_data(self, data):
        if self.in_seg:
            self.data.append(data)


def parse_sgm_file(file_name):
    parser = SGMParser()
    with codecs.open(file_name, 'rb', 'UTF-8') as fh:
        for line in fh:
            parser.feed(line)
    parser.close()
    return parser.data


def parse_txt_file(file_name):
    data = []
    with codecs.open(file_name, 'rb', 'UTF-8') as fh:
        for line in fh:
            data.append(line.strip())
    return data


def main():

    random.seed(1206)

    ap = ArgumentParser()
    ap.add_argument('-n', '--num-samples', type=int, default=1000, help='number of samples to create')
    ap.add_argument('context_file', type=str, help='file with contexts')
    ap.add_argument('file1', type=str, help='1st file with outputs for comparison')
    ap.add_argument('file2', type=str, help='2nd file with outputs for comparison')
    ap.add_argument('out_file', type=str, help='output CSV file')

    args = ap.parse_args()

    contexts = parse_txt_file(args.context_file)
    data1 = parse_sgm_file(args.file1)
    data2 = parse_sgm_file(args.file2)

    assert(len(data1) == len(data2))
    assert(len(data1) % len(contexts) == 0)

    # deduplication
    data = []
    for context, inst1, inst2 in zip(cycle(contexts), data1, data2):
        if inst1 != inst2:
            data.append((context, inst1, inst2))

    if len(data) < args.num_samples:
        print >> sys.stderr, 'Not enough samples, generating only %d.' % len(data)

    random.shuffle(data)
    data = data[:args.num_samples]

    with codecs.open(args.out_file, 'wb', 'UTF-8)') as fh:
        writer = csv.writer(fh, delimiter=b",", lineterminator=b"\n")
        writer.writerow(('context', 'text_a', 'text_b', 'origin_a', 'origin_b'))
        for context, inst1, inst2 in data:
            if random.random() >= 0.5:
                writer.writerow((context, inst2, inst1, args.file2, args.file1))
            else:
                writer.writerow((context, inst1, inst2, args.file1, args.file2))


if __name__ == '__main__':
    main()
