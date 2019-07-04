#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Converting the E2E Challenge dataset (http://www.macs.hw.ac.uk/InteractionLab/E2E/) to our data format.
"""

import re
import argparse
import pandas as pd
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

    def process_instance(conc_da, conc):
        # sort the DA using the same order as in E2E NLG data
        conc_da.dais.sort(key=lambda dai: (['name', 'eat_type', 'food', 'price_range', 'rating', 'area', 'family_friendly', 'near'].index(dai.slot), dai.value))
        conc_das.append(conc_da)

        text, da, abst = delex_sent(conc_da, tokenize(conc), slots_to_abstract, args.slot_names, repeated=True)
        text = text.lower().replace('x-', 'X-')  # lowercase all but placeholders
        da.dais.sort(key=lambda dai: (['name', 'eat_type', 'food', 'price_range', 'rating', 'area', 'family_friendly', 'near'].index(dai.slot), dai.value))

        da_keys[str(da)] = da_keys.get(str(da), 0) + 1
        das.append(da)
        concs.append(conc)
        absts.append(abst)
        texts.append(text)

    # process the input data and store it in memory
    data = pd.read_csv(args.in_file, sep=',', encoding='UTF-8')
    data['mr'] = data['mr'].fillna('')
    for inst in data.itertuples():
        da = DA.parse_diligent_da(inst.mr)
        process_instance(da, inst.ref)
        insts += 1
        if insts % 100 == 0:
            print('%d...' % insts, end='', flush=True, file=sys.stderr)

    print('Processed', insts, 'instances.', file=sys.stderr)
    print('%d different DAs.' % len(da_keys), file=sys.stderr)
    print('%.2f average DAIs per DA' % (sum([len(d) for d in das]) / float(len(das))),
          file=sys.stderr)
    print('Max DA len: %d, max text len: %d' % (max([len(da) for da in das]),
                                                max([text.count(' ') + 1 for text in texts])),
          file=sys.stderr)

    # for multi-ref mode, group by the same conc DA
    if args.multi_ref:
        groups = OrderedDict()  # keep the original order (by 1st occurrence of DA)
        for conc_da, da, conc, text, abst in zip(conc_das, das, concs, texts, absts):
            group = groups.get(str(conc_da), {})
            group['da'] = da
            group['conc_da'] = conc_da
            group['abst'] = group.get('abst', []) + [abst]
            group['conc'] = group.get('conc', []) + [conc]
            group['text'] = group.get('text', []) + [text]
            groups[str(conc_da)] = group

        conc_das, das, concs, texts, absts = [], [], [], [], []
        for group in groups.values():
            conc_das.append(group['conc_da'])
            das.append(group['da'])
            concs.append("\n".join(group['conc']) + "\n")
            texts.append("\n".join(group['text']) + "\n")
            absts.append("\n".join(["\t".join([str(a) for a in absts_])
                                    for absts_ in group['abst']]) + "\n")
    else:
        # convert abstraction instruction to string (coordinate output with multi-ref mode)
        absts = ["\t".join([str(a) for a in absts_]) for absts_ in absts]

    with codecs.open(args.out_name + '-das.txt', 'w', 'UTF-8') as fh:
        for da in das:
            fh.write(str(da) + "\n")

    with codecs.open(args.out_name + '-conc_das.txt', 'w', 'UTF-8') as fh:
        for conc_da in conc_das:
            fh.write(str(conc_da) + "\n")

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
                      help='Multiple reference mode: group by the same DA', action='store_true')
    argp.add_argument('-n', '--slot-names', help='Include slot names in delexicalized texts', action='store_true')
    args = argp.parse_args()
    convert(args)
