#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converting the E2E Challenge dataset (http://www.macs.hw.ac.uk/InteractionLab/E2E/) to our data format.
"""

from __future__ import unicode_literals


import re
import argparse
import unicodecsv as csv
import codecs
from collections import OrderedDict

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

from tgen.data import DA
from tgen.delex import delex_sent
from tgen.futil import tokenize

from tgen.debug import exc_info_hook

# Start IPdb on error in interactive mode
sys.excepthook = exc_info_hook


def filter_abst(abst, slots_to_abstract):
    """Filter abstraction instruction to only contain slots that are actually to be abstracted."""
    return [a for a in abst if a.slot in slots_to_abstract]


def convert(args):
    """Main function – read in the CSV data and output TGEN-specific files."""

    # find out which slots should be abstracted (from command-line argument)
    slots_to_abstract = set()
    if args.abstract is not None:
        slots_to_abstract.update(re.split(r'[, ]+', args.abstract))

    # initialize storage
    conc_das = []
    das = []  # abstracted DAs
    concs = []  # concrete sentences
    texts = []  # abstracted sentences
    absts = []  # abstraction descriptions

    # statistics about different DAs
    da_keys = {}
    insts = 0

    def process_instance(da, conc):
        da.sort()
        conc_das.append(da)

        text, da, abst = delex_sent(da, tokenize(conc), slots_to_abstract, args.slot_names, repeated=True)
        text = text.lower().replace('x-', 'X-')  # lowercase all but placeholders
        da.sort()

        da_keys[unicode(da)] = da_keys.get(unicode(da), 0) + 1
        das.append(da)
        concs.append(conc)
        absts.append(abst)
        texts.append(text)

    # process the input data and store it in memory
    with open(args.in_file, 'r') as fh:
        csvread = csv.reader(fh, encoding='UTF-8')
        csvread.next()  # skip header
        for mr, text in csvread:
            da = DA.parse_diligent_da(mr)
            process_instance(da, text)
            insts += 1

        print 'Processed', insts, 'instances.'
        print '%d different DAs.' % len(da_keys)
        print '%.2f average DAIs per DA' % (sum([len(d) for d in das]) / float(len(das)))
        print 'Max DA len: %d, max text len: %d' % (max([len(da) for da in das]),
                                                    max([text.count(' ') + 1 for text in texts]))

    # for multi-ref mode, group by the same conc DA
    if args.multi_ref:
        groups = OrderedDict()
        for conc_da, da, conc, text, abst in zip(conc_das, das, concs, texts, absts):
            group = groups.get(unicode(conc_da), {})
            group['da'] = da
            group['conc_da'] = conc_da
            group['abst'] = group.get('abst', []) + [abst]
            group['conc'] = group.get('conc', []) + [conc]
            group['text'] = group.get('text', []) + [text]
            groups[unicode(conc_da)] = group

        conc_das, das, concs, texts, absts = [], [], [], [], []
        for group in groups.itervalues():
            conc_das.append(group['conc_da'])
            das.append(group['da'])
            concs.append("\n".join(group['conc']) + "\n")
            texts.append("\n".join(group['text']) + "\n")
            absts.append("\n".join(["\t".join([unicode(a) for a in absts_])
                                    for absts_ in group['abst']]) + "\n")
    else:
        # convert abstraction instruction to string (coordinate output with multi-ref mode)
        absts = ["\t".join([unicode(a) for a in absts_]) for absts_ in absts]

    with codecs.open(args.out_name + '-das.txt', 'w', 'UTF-8') as fh:
        for da in das:
            fh.write(unicode(da) + "\n")

    with codecs.open(args.out_name + '-conc_das.txt', 'w', 'UTF-8') as fh:
        for conc_da in conc_das:
            fh.write(unicode(conc_da) + "\n")

    with codecs.open(args.out_name + '-conc.txt', 'w', 'UTF-8') as fh:
        for conc in concs:
            fh.write(conc + "\n")

    with codecs.open(args.out_name + '-abst.txt', 'w', 'UTF-8') as fh:
        for abst in absts:
            fh.write(abst + "\n")

    with codecs.open(args.out_name + '-text.txt', 'w', 'UTF-8') as fh:
        for text in texts:
            fh.write(text + "\n")


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('in_file', help='Input CSV file')
    argp.add_argument('out_name', help='Output files name prefix')
    argp.add_argument('-a', '--abstract', help='Comma-separated list of slots to be abstracted')
    argp.add_argument('-m', '--multi-ref',
                      help='Multiple reference mode: relexicalize all possible references', action='store_true')
    argp.add_argument('-n', '--slot-names', help='Include slot names in delexicalized texts', action='store_true')
    args = argp.parse_args()
    convert(args)
