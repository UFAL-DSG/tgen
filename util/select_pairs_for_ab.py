#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Selecting pairs of different output sentences for pairwise testing
(including deduplication) or creating test questions by distorting a single set
of output sentences.
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


def distort_response(response):
    response = re.sub(r'([!?,:.])( |$)', r' \1\2', response)
    response = re.sub(r'\s+', r' ', response)
    toks = response.split(' ')
    result = response
    loop = 0
    while loop < 2 or result == response:
        tok_no = random.choice(range(len(toks)))
        tok = toks[tok_no].lower() if tok_no == 0 else toks[tok_no]
        if re.match(r'^([!?,:.]|the|an?)$', tok):
            continue
        if loop >= 10:
            return None
        if random.random() < 0.3:
            toks.insert(tok_no, tok)  # duplicate token
        if random.random() < 0.3:
            del toks[tok_no]  # remove token
        if random.random() < 0.3:
            toks.insert(random.randint(0, len(toks)), tok)  # insert token at random position
        result = ' '.join(toks)
        loop += 1

    result = re.sub(r' ([!?,:.])', r'\1', result)
    result = result[0].upper() + result[1:]
    return result


def create_test_questions(contexts, data1):
    assert(len(data1) % len(contexts) == 0)

    # distorting each response somehow
    data = []
    for context, inst in zip(cycle(contexts), data1):
        distorted = distort_response(inst)
        if distorted is None:
            continue
        data.append((context, inst, distorted))
    return data


def create_real_questions(contexts, data1, data2):
    assert(len(data1) == len(data2))
    assert(len(data1) % len(contexts) == 0)

    # removing identical responses, returning the rest
    data = []
    for context, inst1, inst2 in zip(cycle(contexts), data1, data2):
        if inst1 != inst2:
            data.append((context, inst1, inst2))

    return data


def main():

    random.seed(1206)

    ap = ArgumentParser()
    ap.add_argument('-n', '--num-samples', type=int, default=1000, help='number of samples to create')
    ap.add_argument('-t', '--test-questions', action='store_true',
                    help='create test questions (uses just the 1st input file + contexts)')
    ap.add_argument('context_file', type=str, help='file with contexts')
    ap.add_argument('file1', type=str, help='1st file with outputs for comparison')
    # TODO make optional if test questions are generated
    ap.add_argument('file2', type=str, help='2nd file with outputs for comparison')
    ap.add_argument('out_file', type=str, help='output CSV file')

    args = ap.parse_args()

    contexts = parse_txt_file(args.context_file)
    data1 = parse_sgm_file(args.file1)

    if args.test_questions:
        data = create_test_questions(contexts, data1)
        header = ('context', 'text_a', 'text_b', 'origin_a', 'origin_b', '_golden', 'more_natural_gold')
    else:
        data2 = parse_sgm_file(args.file2)
        data = create_real_questions(contexts, data1, data2)
        header = ('context', 'text_a', 'text_b', 'origin_a', 'origin_b')

    if len(data) < args.num_samples:
        print >> sys.stderr, 'Not enough samples, generating only %d.' % len(data)

    random.shuffle(data)
    data = data[:args.num_samples]

    with codecs.open(args.out_file, 'wb', 'UTF-8)') as fh:
        writer = csv.writer(fh, delimiter=b",", lineterminator=b"\n")
        writer.writerow(header)
        for context, inst1, inst2 in data:
            if args.test_questions:
                if random.random() >= 0.5:
                    writer.writerow((context, inst2, inst1, args.file1 + '-distorted', args.file1,
                                     'true', 'A less than B'))
                else:
                    writer.writerow((context, inst1, inst2, args.file1, args.file1 + '-distorted',
                                     'true', 'A more than B'))
            else:
                if random.random() >= 0.5:
                    writer.writerow((context, inst2, inst1, args.file2, args.file1))
                else:
                    writer.writerow((context, inst1, inst2, args.file1, args.file2))


if __name__ == '__main__':
    main()
