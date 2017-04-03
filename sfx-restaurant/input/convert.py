#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converting the SFX data sets (Cambridge, Wen et al. NAACL 2015) to our data format.
"""

from __future__ import unicode_literals


import json
import re
import argparse
from math import ceil

import sys
import os
sys.path.insert(0, os.path.abspath('../../'))  # add tgen main directory to modules path
from tgen.data import Abst, DA, DAI
from tgen.delex import delex_sent


def postprocess_sent(sent):
    """Postprocess a sentence from the format used in Cambridge NN into plain English."""
    # TODO remove ?
    #sent = re.sub(r'child -s', 'children', sent)
    #sent = re.sub(r' -s', 's', sent)
    #sent = re.sub(r' -ly', 'ly', sent)
    sent = re.sub(r'\s+', ' ', sent)
    return sent

def fix_capitalization(sent):
    # TODO remove ?
    #sent = re.sub(r'( [.?!] [a-z])', lambda m: m.group(1).upper(), sent)
    #sent = re.sub(r'\b(Ok|ok|i)\b', lambda m: m.group(1).upper(), sent)
    #sent = sent[0].upper() + sent[1:]
    return sent


def relexicalize(texts, cur_abst):
    """Lexicalize given texts (list of pairs abstracted text -- abstraction instructions) based on
    the current slot values (stored in abstraction instructions)."""
    ret = []
    for text, abst in texts:
        abst.sort(key=lambda a: a.slot)
        cur_abst.sort(key=lambda a: a.slot)
        assert len(abst) == len(cur_abst)
        toks = text.split(' ')
        for a, c in zip(abst, cur_abst):
            assert a.slot == c.slot
            if a.start < 0:  # skip values that are actually not realized on the surface
                continue
            toks[a.start] = c.value
        ret.append(' '.join(toks))
    return ret


def filter_abst(abst, slots_to_abstract):
    """Filter abstraction instruction to only contain slots that are actually to be abstracted."""
    return [a for a in abst if a.slot in slots_to_abstract]


def convert(args):
    """Main function – read in the JSON data and output TGEN-specific files."""

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
    turns = 0

    def process_instance(da, conc):
        da.sort()
        conc_das.append(da)  # store the non-delexicalized version of the DA

        # delexicalize
        text, da, abst = delex_sent(da, conc, slots_to_abstract, args.slot_names)
        da.sort()  # delexicalization does not keep DAI order, need to sort again

        # store the DA
        text = fix_capitalization(text)
        conc = fix_capitalization(conc)

        da_keys[unicode(da)] = da_keys.get(unicode(da), 0) + 1
        das.append(da)
        concs.append(conc)
        absts.append(abst)
        texts.append(text)

    # process the input data and store it in memory
    with open(args.in_file, 'r') as fh:
        data = json.load(fh, encoding='UTF-8')
        for dialogue in data:
            if isinstance(dialogue, dict):
                for turn in dialogue['dial']:
                    da = DA.parse_cambridge_da(turn['S']['dact'])
                    if args.skip_hello and len(da) == 1 and da[0].da_type == 'hello':
                        continue  # skip hello() DAs
                    conc = postprocess_sent(turn['S']['ref'])
                    process_instance(da, conc)
                    turns += 1
            else:
                da = DA.parse_cambridge_da(dialogue[0])
                conc = postprocess_sent(dialogue[1])
                process_instance(da, conc)
                turns += 1

        print 'Processed', turns, 'turns.'
        print '%d different DAs.' % len(da_keys)
        print '%.2f average DAIs per DA' % (sum([len(d) for d in das]) / float(len(das)))

    if args.split:
        # get file name prefixes and compute data sizes for all the parts to be split
        out_names = re.split(r'[, ]+', args.out_name)
        data_sizes = [int(part_size) for part_size in args.split.split(':')]
        assert len(out_names) == len(data_sizes)
        # compute sizes for all but the 1st part (+ round them up, as Wen does)
        total = float(sum(data_sizes))
        remain = turns
        for part_no in xrange(len(data_sizes) - 1, 0, -1):
            part_size = int(ceil(turns * (data_sizes[part_no] / total)))
            data_sizes[part_no] = part_size
            remain -= part_size
        # put whatever remained into the 1st part
        data_sizes[0] = remain
    else:
        # use just one part -- containing all the data
        data_sizes = [turns]
        out_names = [args.out_name]

    # write all data parts
    for part_size, part_name in zip(data_sizes, out_names):

        # create multiple lexicalized references for each instance by relexicalizing sentences
        # with the same DA from the same part
        if args.multi_ref and part_name in ['devel', 'test', 'dtest', 'etest']:

            # group sentences with the same DA
            da_groups = {}
            for da, text, abst in zip(das[0:part_size], texts[0:part_size], absts[0:part_size]):
                da_groups[unicode(da)] = da_groups.get(unicode(da), [])
                da_groups[unicode(da)].append((text, filter_abst(abst, slots_to_abstract)))

            for da_str in da_groups.keys():
                seen = set()
                uniq = []
                for text, abst in da_groups[da_str]:
                    sig = text + "\n" + ' '.join([a.slot + str(a.start) for a in abst])
                    if sig not in seen:
                        seen.add(sig)
                        uniq.append((text, abst))
                da_groups[da_str] = uniq

            # relexicalize all abstract sentences for each DA
            relex = []
            for da, abst in zip(das[0:part_size], absts[0:part_size]):
                relex.append(relexicalize(da_groups[unicode(da)],
                                          filter_abst(abst, slots_to_abstract)))

            with open(part_name + '-ref.txt', 'w') as fh:
                for relex_pars in relex:
                    fh.write("\n".join(relex_pars).encode('utf-8') + "\n\n")

        with open(part_name + '-das.txt', 'w') as fh:
            for da in das[0:part_size]:
                fh.write(unicode(da).encode('utf-8') + "\n")
            del das[0:part_size]

        with open(part_name + '-conc_das.txt', 'w') as fh:
            for conc_da in conc_das[0:part_size]:
                fh.write(unicode(conc_da).encode('utf-8') + "\n")
            del conc_das[0:part_size]

        with open(part_name + '-conc.txt', 'w') as fh:
            for conc in concs[0:part_size]:
                fh.write(conc.encode('utf-8') + "\n")
            del concs[0:part_size]

        with open(part_name + '-abst.txt', 'w') as fh:
            for abst in absts[0:part_size]:
                fh.write("\t".join([unicode(a) for a in abst]).encode('utf-8') + "\n")
            del absts[0:part_size]

        with open(part_name + '-text.txt', 'w') as fh:
            for text in texts[0:part_size]:
                fh.write(text.encode('utf-8') + "\n")
            del texts[0:part_size]


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('in_file', help='Input JSON file')
    argp.add_argument('out_name', help='Output files name prefix(es - when used with -s, comma-separated)')
    argp.add_argument('-a', '--abstract', help='Comma-separated list of slots to be abstracted')
    argp.add_argument('-s', '--split', help='Colon-separated sizes of splits (e.g.: 3:1:1)')
    argp.add_argument('-m', '--multi-ref',
                      help='Multiple reference mode: relexicalize all possible references', action='store_true')
    argp.add_argument('-n', '--slot-names', help='Include slot names in delexicalized texts', action='store_true')
    argp.add_argument('-i', '--skip-hello', help='Ignore hello() DAs', action='store_true')
    args = argp.parse_args()
    convert(args)
