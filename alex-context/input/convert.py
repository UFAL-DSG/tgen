#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converting the Alex Context NLG data set (Dušek & Jurčíček 2016) to our data format.
"""

from __future__ import unicode_literals


import json
import re
import argparse

from recordclass import recordclass
from tgen.futil import tokenize

class DAI(recordclass('DAI', ['type', 'slot', 'value'])):
    """Simple representation of a single dialogue act item."""

    def __unicode__(self):
        if self.slot is None:
            return self.type + '()'
        if self.value is None:
            return self.type + '(' + self.slot + ')'
        quote = '\'' if (' ' in self.value or ':' in self.value) else ''
        return self.type + '(' + self.slot + '=' + quote + self.value + quote + ')'


class Abst(recordclass('Abst', ['slot', 'value', 'start', 'end'])):
    """Simple representation of a single abstraction instruction."""

    def __unicode__(self):
        quote = '"' if ' ' in self.value or ':' in self.value else ''
        return (self.slot + '=' + quote + self.value + quote + ':' +
                str(self.start) + '-' + str(self.end))


class DA(object):
    """Dialogue act – basically a list of DAIs."""

    def __init__(self):
        self.dais = []

    def __getitem__(self, idx):
        return self.dais[idx]

    def __setitem__(self, idx, value):
        self.dais[idx] = value

    def append(self, value):
        self.dais.append(value)

    def __unicode__(self):
        return '&'.join([unicode(dai) for dai in self.dais])

    def __str__(self):
        return unicode(self).encode('ascii', errors='xmlcharrefreplace')

    def __len__(self):
        return len(self.dais)


def parse_da(da_text):
    """Parse a DA string into DAIs (DA types, slots, and values)."""

    da = DA()

    for dai_text in da_text.split('&'):
        da_type, svp = dai_text.split('(', 1)
        svp = svp[:-1]  # remove closing paren

        if not svp:  # no slot + value (e.g. 'hello()')
            da.append(DAI(da_type, None, None))
            continue

        if '=' not in svp:  # no value (e.g. 'request(to_stop)')
            da.append(DAI(da_type, svp, None))
            continue

        slot, value = svp.split('=', 1)
        if value.startswith('"'):  # remove quotes
            value = value[1:-1]

        da.append(DAI(da_type, slot, value))

    return da


def get_abstraction(text, conc_da):
    """Get the abstraction instructions and convert the string (replace *SLOT with X)."""
    abstr = []
    toks = tokenize(text).split(' ')

    for dai in conc_da:
        slot_abst = '*' + dai.slot.upper()
        try:
            idx = toks.index(slot_abst)
            toks[idx] = 'X'
            abstr.append(Abst(dai.slot, dai.value, idx, idx + 1))
        except ValueError:
            continue

    return ' '.join(toks), abstr


def convert_abstractions(abstr_str):
    return re.sub(r'\*([A-Z_]+)', lambda m: 'X-' + m.group(1).lower(), abstr_str)


def convert_abstr_da(abstr_da):
    """Convert *SLOT to X-slot in an abstract DA."""
    for dai in abstr_da:
        if dai.value is None:
            continue
        dai.value = convert_abstractions(dai.value)
    return abstr_da


def write_part(file_name, data, part_size, repeat=1):
    """Write part of the dataset to a file (if the data instances are lists, unroll them, if not,
    repeat each instance `repeat` times). Delete the written instances from the `data` list
    at the end."""
    with open(file_name, 'w') as fh:
        for inst in data[0:part_size]:
            if isinstance(inst, list):
                for inst_part in inst:
                    fh.write(unicode(inst_part).encode('utf-8') + "\n")
            else:
                for _ in xrange(repeat):
                    fh.write(unicode(inst).encode('utf-8') + "\n")
    del data[0:part_size]


def convert(args):
    """Main function – read in the JSON data and output TGEN-specific files."""

    # initialize storage
    items = 0
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
            da = convert_abstr_da(parse_da(item['response_da']))
            context = convert_abstractions(item['context_utt'])
            context_l = item['context_utt_l']
            conc_da = parse_da(item['response_da_l'])
            concs_ = [tokenize(s) for s in item['response_nl_l']]
            absts_ = []
            texts_ = []
            for abst_text in item['response_nl']:
                text, abst = get_abstraction(abst_text, conc_da)  # convert *SLOT -> X
                absts_.append(abst)
                texts_.append(text)

            das.append(da)
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
        write_part(part_name + '-context.txt', contexts, part_size, repeat_num)
        write_part(part_name + '-conc_context.txt', conc_contexts, part_size, repeat_num)

        # write all other just once (here, each instance is a list, it will be unrolled)
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
    args = argp.parse_args()
    convert(args)
