#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Postprocessing for Alex Context NLG Dataset
"""

from __future__ import unicode_literals
import codecs
import re
import sys
from argparse import ArgumentParser

from tgen.debug import exc_info_hook
sys.excepthook = exc_info_hook


def process_file(file_name):

    buf = []
    with codecs.open(file_name, 'rb', 'UTF-8') as fh:
        for line in fh:
            buf.append(line)

    with codecs.open(file_name, 'wb', 'UTF-8') as fh:
        for line in buf:
            # 0:15 -> 15 minutes
            line = re.sub(r'(^| )0:([0-9]{2})(?: minutes)?( |[?!,.;:]|$)', r'\1\2 minutes\3', line)

            # number of transfers
            line = re.sub(r'(are|is) 1 transfers?', r'is 1 transfer', line)
            line = re.sub(r'1 transfers', r'1 transfer', line)
            line = re.sub(r'(are|is) ([2-9]) transfers?\b', r'are \2 transfers', line)
            line = re.sub(r'([2-9]) transfer\b', r'\1 transfers', line)
            line = re.sub(r'0 (transfers?)', r'no \1', line)

            # in the morning, in the afternoon (TODO afternoon/evening -- but it never seems to occur)
            line = re.sub(r'(,|the) pm', r'\1 afternoon', line)
            line = re.sub(r'(,|the) am', r'\1 morning', line)

            fh.write(line)


def main():

    ap = ArgumentParser()
    ap.add_argument('input_file', type=str, help='I/O file')
    args = ap.parse_args()

    process_file(args.input_file)


if __name__ == '__main__':
    main()
