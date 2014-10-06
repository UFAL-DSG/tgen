#!/usr/bin/env python

from tgen.futil import read_ttrees
import sys

if len(sys.argv[1:]) != 1:
    sys.exit('Usage: python yaml2pickle.py <file.yaml>')
read_ttrees(sys.argv[1])
