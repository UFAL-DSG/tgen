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

    def __str__(self):
        return unicode(self).encode('ascii', errors='xmlcharrefreplace')

    def __repr__(self):
        return str(self)


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

    def __repr__(self):
        return str(self)

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
            assert not re.match(r'^\'', value) and not re.match(r'\'$', value)

            da.append(DAI(da_type, slot, value))

    return da


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


def find_substr(needle, haystack):
    """Find a sub-list in a list of tokens.

    @param haystack: the longer list of tokens – the list where we should search
    @param needle: the shorter list – the list whose position is to be found
    @return: a tuple of starting and ending position of needle in the haystack, \
            or None if not found
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


def find_substr_approx(needle, haystack):
    """Try to find a sub-list in a list of tokens using fuzzy matching (skipping some
    common prepositions and punctuation, checking for similar-length substrings)"""
    # some common 'meaningless words'
    stops = set(['and', 'or', 'in', 'of', 'the', 'to', ','])
    h = 0
    n = 0
    match_start = 0
    while True:
        n_orig = n  # remember that we skipped some stop words at beginning of needle
                    # (since we check needle position before moving on in haystack)
        # skip stop words (ignore stop words around haystack position)
        while n < len(needle) and needle[n] in stops:
            n += 1
        while n > 0 and n < len(needle) and h < len(haystack) and haystack[h] in stops:
            h += 1
        if n >= len(needle):
            return match_start, h
        if h >= len(haystack):
            return None
        # fuzzy match: one may be substring of the other, with up to 2 chars difference
        # (moderate/moderately etc.)
        if (haystack[h] == needle[n] or
            (haystack[h] != '' and
             (haystack[h] in needle[n] or needle[n] in haystack[h]) and
             abs(len(haystack[h]) - len(needle[n])) <= 2)):
            n += 1
            h += 1
        else:
            if n_orig > 0:
                n = 0
            else:
                h += 1
                match_start = h


def abstract_sent(da, conc, abst_slots, slot_names):
    """Abstract the given slots in the given sentence (replace them with X).

    @param da: concrete DA
    @param conc: concrete sentence text
    @param abstr_slots: a set of slots to be abstracted
    @return: a tuple of the abstracted text, abstracted DA, and abstraction instructions
    """

    toks = conc.split(' ')
    absts = []
    abst_da = DA()
    toks_mask = [True] * len(toks)

    # find all values in the sentence, building the abstracted DA along the way
    # search first for longer values (so that substrings don't block them)
    for dai in sorted(da,
                      key=lambda dai: len(dai.value) if dai.value is not None else 0,
                      reverse=True):
        # first, create the 'abstracted' DAI as the copy of the current DAI
        abst_da.append(DAI(dai.type, dai.slot, dai.value))
        if dai.value is None:
            continue
        # try to find the value in the sentence (first exact, then fuzzy)
        # while masking tokens of previously found values
        val_toks = dai.value.split(' ')
        pos = find_substr(val_toks, [t if m else '' for t, m in zip(toks, toks_mask)])
        if pos is None:
            pos = find_substr_approx(val_toks, [t if m else '' for t, m in zip(toks, toks_mask)])
        if pos is not None:
            # save the abstraction instruction
            absts.append(Abst(dai.slot, dai.value, pos[0], pos[1]))
            for idx in xrange(pos[0], pos[1]):  # mask found things so they're not found twice
                toks_mask[idx] = False
            # if the value is to be abstracted, replace the value in the abstracted DAI
            if dai.slot in abst_slots and dai.value != 'dont_care':
                abst_da[-1].value = 'X-' + dai.slot

    # go from the beginning of the sentence, replacing the values to be abstracted
    absts.sort(key=lambda a: a.start)
    shift = 0
    for abst in absts:
        # select only those that should actually be abstracted on the output
        if abst.slot not in abst_slots or abst.value == 'dont_care':
            continue
        # replace the text
        if slot_names:
            toks[abst.start - shift:abst.end - shift] = ['X-' + abst.slot]
        else:
            toks[abst.start - shift:abst.end - shift] = ['X']
        # update abstraction instruction indexes
        shift_add = abst.end - abst.start - 1
        abst.start -= shift
        abst.end = abst.start + 1
        shift += shift_add

    return ' '.join(toks), abst_da, absts


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
    das = []  # abstracted DAs
    concs = []  # concrete sentences
    texts = []  # abstracted sentences
    absts = []  # abstraction descriptions

    # process the input data and store it in memory
    with open(args.in_file, 'r') as fh:
        data = json.load(fh, encoding='UTF-8')
        turns = 0
        for dialogue in data:
            for turn in dialogue['dial']:
                da = parse_da(turn['S']['dact'])
                conc = postprocess_sent(turn['S']['ref'])
                text, da, abst = abstract_sent(da, conc, slots_to_abstract, args.slot_names)

                text = fix_capitalization(text)
                conc = fix_capitalization(conc)

                das.append(da)
                concs.append(conc)
                absts.append(abst)
                texts.append(text)
                turns += 1

        print 'Processed', turns, 'turns.'

    if args.split:
        # get file name prefixes and compute data sizes for all the parts to be split
        out_names = re.split(r'[, ]+', args.out_name)
        data_sizes = [int(part_size) for part_size in args.split.split(':')]
        assert len(out_names) == len(data_sizes)
        # compute sizes for all but the 1st part (+ round them)
        total = float(sum(data_sizes))
        remain = turns
        for part_no in xrange(len(data_sizes) - 1, 0, -1):
            part_size = int(round(turns * (data_sizes[part_no] / total)))
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



if  __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('in_file', help='Input JSON file')
    argp.add_argument('out_name', help='Output files name prefix(es - when used with -s, comma-separated)')
    argp.add_argument('-a', '--abstract', help='Comma-separated list of slots to be abstracted')
    argp.add_argument('-s', '--split', help='Colon-separated sizes of splits (e.g.: 3:1:1)')
    argp.add_argument('-m', '--multi-ref', help='Multiple reference mode: relexicalize all possible references', action='store_true')
    argp.add_argument('-n', '--slot-names', help='Include slot names in delexicalized texts', action='store_true')
    args = argp.parse_args()
    convert(args)
