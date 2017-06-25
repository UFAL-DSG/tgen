#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Postprocessing for the E2E challenge
"""

from __future__ import unicode_literals
import codecs
from regex import Regex, UNICODE, IGNORECASE
import sys
from argparse import ArgumentParser


class Detokenizer(object):
    """\
    A simple de-tokenizer class.
    """

    def __init__(self):
        """\
        Constructor (pre-compile all needed regexes).
        """
        # compile regexes
        self._currency_or_init_punct = Regex(r' ([\p{Sc}\(\[\{\¿\¡]+) ', flags=UNICODE)
        self._noprespace_punct = Regex(r' ([\,\.\?\!\:\;\\\%\}\]\)]+) ', flags=UNICODE)
        self._contract = Regex(r" (\p{Alpha}+) ' (ll|ve|re|[dsmt])(?= )", flags=UNICODE | IGNORECASE)
        self._dash_fixes = Regex(r" (\p{Alpha}+|£ [0-9]+) - (priced|star|friendly|(?:£ )?[0-9]+) ", flags=UNICODE | IGNORECASE)
        self._dash_fixes2 = Regex(r" (non) - ([\p{Alpha}-]+) ", flags=UNICODE | IGNORECASE)

    def detokenize(self, text):
        """\
        Detokenize the given text.
        """
        text = ' ' + text + ' '
        text = self._dash_fixes.sub(r' \1-\2 ', text)
        text = self._dash_fixes2.sub(r' \1-\2 ', text)
        text = self._currency_or_init_punct.sub(r' \1', text)
        text = self._noprespace_punct.sub(r'\1 ', text)
        text = self._contract.sub(r" \1'\2", text)
        text = text.strip()
        # capitalize
        if not text:
            return ''
        text = text[0].upper() + text[1:]
        return text


def process_file(input_file_name, output_file_name):
    detok = Detokenizer()
    buf = []
    with codecs.open(input_file_name, 'rb', 'UTF-8') as fh:
        for line in fh:
            buf.append(line.strip())

    with codecs.open(output_file_name, 'wb', 'UTF-8') as fh:
        for line in buf:
            fh.write(detok.detokenize(line) + "\n")


def main():

    ap = ArgumentParser(description='Postprocess TGen files for E2E challenge')
    ap.add_argument('input_file', type=str, help='input file (and output file, if in-place)')
    ap.add_argument('output_file', type=str, nargs='?', help='output file (if not in-place)')
    args = ap.parse_args()

    process_file(args.input_file, args.output_file if args.output_file else args.input_file)


if __name__ == '__main__':
    main()
