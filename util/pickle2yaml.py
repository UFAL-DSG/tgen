#!/usr/bin/env python

from tgen.futil import read_ttrees, write_ttrees
import sys
import re

if len(sys.argv[1:]) != 1:
    sys.exit('Usage: python pickle2yaml.py <file.pickle>')
doc = read_ttrees(sys.argv[1])
out_fname = re.sub('\.pickle', '.yaml', sys.argv[1])
write_ttrees(doc, out_fname)
