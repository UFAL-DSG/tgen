#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converting the Alex Context NLG data set (Dušek & Jurčíček 2016) to our data format.
"""

from __future__ import unicode_literals


import json
import re
import argparse

import sys
import os
sys.path.insert(0, os.path.abspath('../../'))  # add tgen main directory to modules path
from tgen.data import DA, Abst


def tokenize(text):
    """Tokenize the text (i.e., insert spaces around all tokens)"""
    toks = re.sub(r'([?.!;,:-]+)(?![0-9])', r' \1 ', text)  # enforce space around all punct

    # most common contractions
    toks = re.sub(r'([\'’´])(s|m|d|ll|re|ve)\s', r' \1\2 ', toks)  # I'm, I've etc.
    toks = re.sub(r'(n[\'’´]t\s)', r' \1 ', toks)  # do n't

    # other contractions, as implemented in Treex
    toks = re.sub(r' ([Cc])annot\s', r' \1an not ', toks)
    toks = re.sub(r' ([Dd])\'ye\s', r' \1\' ye ', toks)
    toks = re.sub(r' ([Gg])imme\s', r' \1im me ', toks)
    toks = re.sub(r' ([Gg])onna\s', r' \1on na ', toks)
    toks = re.sub(r' ([Gg])otta\s', r' \1ot ta ', toks)
    toks = re.sub(r' ([Ll])emme\s', r' \1em me ', toks)
    toks = re.sub(r' ([Mm])ore\'n\s', r' \1ore \'n ', toks)
    toks = re.sub(r' \'([Tt])is\s', r' \'\1 is ', toks)
    toks = re.sub(r' \'([Tt])was\s', r' \'\1 was ', toks)
    toks = re.sub(r' ([Ww])anna\s', r' \1an na ', toks)

    # clean extra space
    toks = re.sub(r'\s+', ' ', toks)
    return toks


def get_abstraction(text, conc_da, slot_names=False):
    """Get the abstraction instructions and convert the string (replace *SLOT with X).
    If slot_names is true, "X-slot_name" is used instead."""
    abstr = []
    toks = tokenize(text).split(' ')

    for dai in conc_da:
        if not dai.slot or dai.value in [None, 'none', 'dontcare', 'dont_care', 'yes', 'no']:
            continue
        slot_abst = '*' + dai.slot.upper()
        try:
            idx = toks.index(slot_abst)
            toks[idx] = 'X' + ('-' + dai.slot if slot_names else '')
            abstr.append(Abst(slot=dai.slot, value=dai.value, start=idx, end=idx + 1))
        except ValueError:
            continue

    return ' '.join(toks), "\t".join([unicode(a) for a in abstr])


def convert_abstractions(abstr_str):
    return re.sub(r'\*([A-Z_]+)', lambda m: 'X-' + m.group(1).lower(), abstr_str)


def convert_abstr_da(abstr_da):
    """Convert *SLOT to X-slot in an abstract DA."""
    for dai in abstr_da:
        if dai.value is None:
            continue
        dai.value = convert_abstractions(dai.value)
    return abstr_da


def write_part(file_name, data, part_size, repeat=1, trunc=True, separate=False):
    """Write part of the dataset to a file (if the data instances are lists, unroll them, if not,
    repeat each instance `repeat` times). Separate the lists by an empty line if `separate`
    is True (not by default). Delete the written instances from the `data` list at the end,
    if `trunc` is True (default)."""
    with open(file_name, 'w') as fh:
        for inst in data[0:part_size]:
            if isinstance(inst, list):
                for inst_part in inst:
                    fh.write(unicode(inst_part).encode('utf-8') + b"\n")
                if separate:
                    fh.write("\n")
            else:
                for _ in xrange(repeat):
                    fh.write(unicode(inst).encode('utf-8') + b"\n")
    if trunc:
        del data[0:part_size]


def convert(args):
    """Main function – read in the JSON data and output TGEN-specific files."""

    # initialize storage
    items = 0
    conc_das = [] # concrete DAs
    das = []  # abstracted DAs
    concs = []  # concrete sentences
    texts = []  # abstracted sentences
    absts = []  # abstraction descriptions
    contexts = []  # abstracted contexts
    conc_contexts = []  # lexicalized contexts

    # process the input data and store it in memory
    with open(args.in_file, 'r') as fh:
        data = json.load(fh, encoding='UTF-8')
        for item in data:
            da = convert_abstr_da(DA.parse(item['response_da']))
            context = convert_abstractions(item['context_utt'])
            context_l = item['context_utt_l']
            conc_da = DA.parse(item['response_da_l'])
            concs_ = [tokenize(s) for s in item['response_nl_l']]
            absts_ = []
            texts_ = []
            for abst_text in item['response_nl']:
                text, abst = get_abstraction(abst_text, conc_da, args.slot_names)  # convert *SLOT -> X
                absts_.append(abst)
                texts_.append(text)

            das.append(da)
            conc_das.append(conc_da)
            contexts.append(context)
            conc_contexts.append(context_l)
            concs.append(concs_)
            absts.append(absts_)
            texts.append(texts_)
            items += 1

        print 'Processed', items, 'items.'

    if args.split:
        # get file name prefixes and compute data sizes for all the parts to be split
        out_names = re.split(r'[, ]+', args.out_name)
        data_sizes = [int(part_size) for part_size in args.split.split(':')]
        assert len(out_names) == len(data_sizes)
        # compute sizes for all but the 1st part (+ round them)
        total = float(sum(data_sizes))
        remain = items
        for part_no in xrange(len(data_sizes) - 1, 0, -1):
            part_size = int(round(items * (data_sizes[part_no] / total)))
            data_sizes[part_no] = part_size
            remain -= part_size
        # put whatever remained into the 1st part
        data_sizes[0] = remain
    else:
        # use just one part -- containing all the data
        data_sizes = [items]
        out_names = [args.out_name]

    # write all data parts
    for part_size, part_name in zip(data_sizes, out_names):

        repeat_num = len(concs[0])
        if args.multi_ref and part_name in ['devel', 'test', 'dtest', 'etest']:
            repeat_num = 1

        # repeat DAs and contexts for synonymous paraphrases, unless for test data in multi-ref mode
        write_part(part_name + '-das.txt', das, part_size, repeat_num)
        write_part(part_name + '-conc_das.txt', conc_das, part_size, repeat_num)
        write_part(part_name + '-context.txt', contexts, part_size, repeat_num)
        write_part(part_name + '-conc_context.txt', conc_contexts, part_size, repeat_num)

        # write all other just once (here, each instance is a list, it will be unrolled)
        write_part(part_name + '-ref.txt', concs, part_size, trunc=False, separate=True)
        write_part(part_name + '-conc.txt', concs, part_size)
        write_part(part_name + '-abst.txt', absts, part_size)
        write_part(part_name + '-text.txt', texts, part_size)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('in_file', help='Input JSON file')
    argp.add_argument('out_name', help='Output files name prefix(es - when used with -s, comma-separated)')
    argp.add_argument('-s', '--split', help='Colon-separated sizes of splits (e.g.: 3:1:1)')
    argp.add_argument('-m', '--multi-ref',
                      help='Multiple reference mode; i.e. do not repeat DA in devel and test parts',
                      action='store_true')
    argp.add_argument('-n', '--slot-names', action='store_true',
                      help='Include slot names in the abstracted sentences.')
    args = argp.parse_args()
    convert(args)
