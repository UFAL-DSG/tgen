#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converting the SFX data sets (Cambridge, Wen et al. NAACL 2015) to our data format.
"""

from __future__ import unicode_literals


import json
import sys
import re
import argparse

from recordclass import recordclass


class DAI(recordclass('DAI', ['type', 'slot', 'value'])):
    """Simple representation of a single dialogue act item."""

    def __unicode__(self):
        if self.slot is None:
            return self.type + '()'
        if self.value is None:
            return self.type + '(' + self.slot + ')'        
        quote = '\'' if ' ' in self.value else ''
        return self.type + '(' + self.slot + '=' + quote + self.value + quote + ')'


class Abst(recordclass('Abst', ['slot', 'value', 'start', 'end'])):
    """Simple representation of a single abstraction instruction."""

    def __unicode__(self):
        quote = '\'' if ' ' in self.value else ''
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
    
    for dai_text in re.finditer(r'(\??[a-z_]+)\(([^)]*)\)', da_text):
        da_type, svps = dai_text.groups()

        if not svps:  # no slots/values (e.g. 'hello()')
            da.append(DAI(da_type, None, None))
            continue

        # we have some slots/values – split them into DAIs
        svps = re.split('(?<! ),', svps)
        for svp in svps:

            if '=' not in svp:  # no value, e.g. '?request(near)'
                da.append(DAI(da_type, svp, None))
                continue

            # we have a value
            slot, value = svp.split('=', 1)
            if re.match(r'^\'.*\'$', value):
                value = value[1:-1]
            if re.match(r'^\'', value) or re.match(r'\'$', value):
                import ipdb; ipdb.set_trace()

            da.append(DAI(da_type, slot, value))

    return da


def postprocess_sent(sent):
    """Postprocess a sentence from the format used in Cambridge NN into plain English."""
    sent = re.sub(r'child -s', 'children', sent)
    sent = re.sub(r' -s', 's', sent)
    sent = re.sub(r' -ly', 'ly', sent)
    return sent


def find_substr(needle, haystack):
    """Find a sub-list in a list of tokens.
    
    @param haystack: the longer list of tokens – the list where we should search
    @param needle: the shorter list – the list whose position is to be found
    @return: a tuple of starting and ending position of needle in the haystack, or None if not found
    """
    h = 0
    n = 0
    while True:
        if n >= len(needle):
            return h - n, h
        if h >= len(haystack):
            return None
        if haystack[h] == needle[n]:
            n += 1
            h += 1
        else:
            if n > 0:
                n = 0
            else:
                h += 1


def abstract_sent(da, conc, abst_slots):
    """Abstract the given slots in the given sentence (replace them with X).
    
    @param da: concrete DA
    @param conc: concrete sentence text
    @param abstr_slots: a set of slots to be abstracted
    @return: a tuple of the abstracted text, abstracted DA, and abstraction instructions
    """

    toks = conc.split(' ')
    absts = []
    abst_da = DA()

    if 'and their phone number is 4159212956' in conc:
        import ipdb; ipdb.set_trace()

    # find all values in the sentence, building the abstracted DA along the way
    for dai in da:
        # first, create the 'abstracted' DAI as the copy of the current DAI
        abst_da.append(DAI(dai.type, dai.slot, dai.value))
        if dai.value is None:
            continue
        val_toks = dai.value.split(' ')
        pos = find_substr(val_toks, toks)
        if pos is not None:
            # save the abstraction instruction
            absts.append(Abst(dai.slot, dai.value, pos[0], pos[1]))
            # if this is to be abstracted, replace the value in the abstracted DAI
            if dai.slot in abst_slots and dai.value != 'dont_care':
                abst_da[-1].value = 'X-' + dai.slot

    # go from the beginning of the sentence, replacing the values to be abstracted
    absts.sort(key=lambda a: a.start)
    shift = 0
    for abst in absts:
        # select only those that should actually be abstracted on the output
        if abst.slot not in abst_slots or dai.value == 'dont_care':
            continue
        # replace the text
        toks[abst.start - shift:abst.end - shift] = ['X']
        # update abstraction instruction indexes
        shift_add = abst.end - abst.start - 1
        abst.start -= shift
        abst.end = abst.start + 1
        shift += shift_add

    return ' '.join(toks), abst_da, absts


def convert(args):
    """Main function – read in the JSON data and output TGEN-specific files."""

    # find out which slots should be abstracted (from command-line argument)
    slots_to_abstract = set()
    if args.abstract is not None:
        slots_to_abstract.update(re.split(r'[, ]+', args.abstract))

    # initialize storage
    das = []  # abstracted DAs
    concs = []  # concrete sentences
    texts = []  # abstracted sentences
    absts = []  # abstraction descriptions

    with open(args.in_file, 'r') as fh:
        data = json.load(fh, encoding='UTF-8')
        turns = 0
        for dialogue in data:
            for turn in dialogue['dial']:
                da = parse_da(turn['S']['dact'])
                conc = postprocess_sent(turn['S']['ref'])
                text, da, abst = abstract_sent(da, conc, slots_to_abstract)

                das.append(da)
                concs.append(conc)
                absts.append(abst)
                texts.append(text)
                turns += 1

        print 'Processed', turns, 'turns.'

    with open(args.out_name + '-das.txt', 'w') as fh:
        for da in das:
            fh.write(unicode(da).encode('utf-8') + "\n")

    with open(args.out_name + '-conc.txt', 'w') as fh:
        for conc in concs:
            fh.write(conc.encode('utf-8') + "\n")

    with open(args.out_name + '-abst.txt', 'w') as fh:
        for abst in absts:
            fh.write("\t".join([unicode(a) for a in abst]).encode('utf-8') + "\n")

    with open(args.out_name + '-text.txt', 'w') as fh:
        for text in texts:
            fh.write(text.encode('utf-8') + "\n")

if  __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('in_file', help='Input JSON file')
    argp.add_argument('out_name', help='Output files name prefix')
    argp.add_argument('-a', '--abstract', help='Comma-separated list of slots to be abstracted')
    args = argp.parse_args()
    convert(args)
