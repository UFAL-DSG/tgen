#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Print sttatistics for A/B testing on CrowdFlower.
Prints percentage of each variant being ranked as better, plus bootstrap significance test results.
"""

from __future__ import unicode_literals
import sys
from argparse import ArgumentParser
from collections import defaultdict

import unicodecsv as csv
import numpy.random as rnd

from tgen.debug import exc_info_hook
sys.excepthook = exc_info_hook


# the color output is a recipe from the Python cookbook, #475186
def has_colours(stream):
    """Detect if the given output stream supports color codes."""
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False  # auto color only on TTYs
    try:
        import curses
        curses.setupterm()
        return curses.tigetnum("colors") > 2
    except:
        # guess false in case of error
        return False


stdout_has_colours = has_colours(sys.stdout)


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)


def printout(text, colour=WHITE):
    """Print text to sys.stdout using colors."""
    if stdout_has_colours:
        seq = "\x1b[1;%dm" % (30+colour) + text + "\x1b[0m"
        sys.stdout.write(seq)
    else:
        sys.stdout.write(text)


class Result(object):
    """This holds a general CSV line, with all CSV fields as attributes of the object."""

    def __init__(self, data, headers):
        """Initialize, storing the given CSV headers and initializing using the given data
        (in the same order as the headers).

        @param data: a list of data fields, a line as read by Python CSV module
        @param headers: a list of corresponding field names, e.g., CSV header as read by Python \
            CSV module
        """
        self.__headers = headers
        for attr, val in zip(headers, data):
            setattr(self, attr, val)

    def as_array(self):
        """Return the values as an array, in the order given by the current headers (which were
        provided upon object construction)."""
        ret = []
        for attr in self.__headers:
            ret.append(getattr(self, attr, ''))
        return ret

    def as_dict(self):
        """Return the values as a dictionary, with keys for field names and values for the
        corresponding values."""
        ret = {}
        for attr in self.__headers:
            ret[attr] = getattr(self, attr, '')
        return ret


def pairwise_bootstrap(data, iters):

    label_0 = data[0]
    label_1 = next(d for d in data if d != label_0)

    p0_better, p1_better, ties = 0, 0, 0
    for i in xrange(iters):
        sample = rnd.randint(0, len(data), len(data))
        s_p0_good = sum(1 if data[i] == label_0 else 0 for i in sample)
        s_p1_good = sum(1 if data[i] == label_1 else 0 for i in sample)

        if s_p0_good > s_p1_good:
            p0_better += 1
        elif s_p1_good > s_p0_good:
            p1_better += 1
        else:
            ties += 1

    print (('%s better: %d (%2.2f) | %s better: %d (%2.2f) | ties: %d (%2.2f)') %
           (label_0, p0_better, float(p0_better) / iters * 100,
            label_1, p1_better, float(p1_better) / iters * 100,
            ties, float(ties) / iters * 100,))


def main():

    rnd.seed(1206)

    ap = ArgumentParser()
    # TODO use more files ?
    ap.add_argument('-b', '--bootstrap-iters', type=int, default=1000)
    ap.add_argument('-e', '--example-outputs', action='store_true')
    ap.add_argument('cf_output', type=str, help='crowdflower results file')

    args = ap.parse_args()
    votes = defaultdict(int)
    res = []
    res_by_id = {}
    options = {}

    with open(args.cf_output, 'rb') as fh:
        csvread = csv.reader(fh, delimiter=b',', quotechar=b'"', encoding="UTF-8")
        headers = csvread.next()
        for row in csvread:
            row = Result(row, headers)
            if row._golden == 'true':  # skip test questions
                continue

            res_by_id[row._unit_id] = res_by_id.get(row._unit_id, {row.origin_a: 0, row.origin_b: 0})
            options[row._unit_id] = {'context': row.context,
                                     row.origin_a: row.text_a,
                                     row.origin_b: row.text_b}
            if row.more_natural == 'A less than B':
                votes[row.origin_b] += 1
                res.append(row.origin_b)
                res_by_id[row._unit_id][row.origin_b] += 1
            elif row.more_natural == 'A more than B':
                votes[row.origin_a] += 1
                res.append(row.origin_a)
                res_by_id[row._unit_id][row.origin_a] += 1

    print "Results:"
    for key, val in votes.iteritems():
        print '%s\t%d (%2.2f)' % (key, val, float(val) / len(res) * 100)

    agreement_stats = defaultdict(int)
    for unit_id, stats in res_by_id.iteritems():
        stats_key = " ".join(["%s=%d" % (key, times) for key, times in stats.iteritems()])
        agreement_stats[stats_key] += 1

    print "\nAgreement:"
    for key, val in agreement_stats.iteritems():
        print '%s\t%d (%2.2f)' % (key, val, float(val) / len(res_by_id) * 100)

    print "\nBootstrap:"
    pairwise_bootstrap(res, args.bootstrap_iters)

    if args.example_outputs:
        all_keys = next(res_by_id.itervalues()).keys()
        for good_key in all_keys:
            print "\n%s = 0:\n=================" % good_key
            for unit_id, stats in res_by_id.iteritems():
                if all([key == good_key or res_by_id[unit_id][key] == 0 for key in all_keys]):
                    printout(options[unit_id]['context'], YELLOW)
                    print " : ",
                    printout(options[unit_id][good_key], GREEN)
                    print " > ",
                    print " ".join([options[unit_id][key] for key in all_keys if key != good_key])


if __name__ == '__main__':
    main()
