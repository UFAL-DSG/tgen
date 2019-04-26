#!/usr/bin/env python
# coding=utf-8

"""
Cross-validation split of BAGEL data
------------------------------------

Will produce train-test splits for all cross-validation folds, each
in its own subdirectory.

Usage: ./cv_split.py [-f 10] [-c 2] [-d cv] all-xxx.txt all-yyy.txt

-f = number of folds (default to 10)
-c = number of chunks (alternative realizations, defaults to 2)
-d = directory prefix (defaults to "cv")

"""

from __future__ import unicode_literals

from pytreex.core.util import file_stream
from getopt import getopt
import os
import random
import sys
import re

sys.path.insert(0, os.path.abspath('../../'))  # add tgen main directory to modules path
from tgen.logf import log_warn

def write_data(dir, fname_base, fname_repl, data):
    chunk_size = len(data[0])
    for chunk_idx in xrange(chunk_size):
        fname_suff = ".%d" % chunk_idx if chunk_size > 1 else ''
        file_name = os.path.join(dir, re.sub(r'^[^-._]*', fname_repl + fname_suff, fname_base))
        print 'WRITING ' + file_name
        with file_stream(file_name, 'w') as fh:
            for chunk in data:
                print >> fh, chunk[chunk_idx],


def main(argv):

    opts, files = getopt(argv, 'f:c:d:')

    folds = 10
    chunk_size = 2
    dir_prefix = 'cv'

    for opt, arg in opts:
        if opt == '-f':
            folds = int(arg)
        elif opt == '-c':
            chunk_size = int(arg)
        elif opt == '-d':
            dir_prefix = arg

    if not files:
        sys.exit(__doc__)
    
    random.seed(1206)
    ordering = None

    for file in files:
        # read all data
        data = []
        with file_stream(file) as fh:
            chunk = []
            for line in fh:
                chunk.append(line)
                if len(chunk) == chunk_size:
                    data.append(chunk)
                    chunk = []
            if chunk:
                log_warn('Incomplete chunk at end of file %s, size %d' % (file, len(chunk)))

        if ordering is None:
            # create ordering
            ordering = range(len(data))
            random.shuffle(ordering)

            # create directories
            for fold_no in xrange(folds):
                os.mkdir(dir_prefix + "%02d" % fold_no)
            
        # output as train and test into all CV portions
        fold_size, bigger_folds = divmod(len(data), folds)
        for fold_no in xrange(folds):
            # compute test data bounds
            if fold_no < bigger_folds:
                test_lo = (fold_size + 1) * fold_no
                test_hi = (fold_size + 1) * (fold_no + 1)
            else:
                test_lo = fold_size * fold_no + bigger_folds
                test_hi = fold_size * (fold_no + 1) + bigger_folds
            # select train and test data instances
            train_data = [data[idx] for ord, idx in enumerate(ordering)
                          if ord < test_lo or ord >= test_hi]
            test_data = [data[idx] for ord, idx in enumerate(ordering)
                         if ord >= test_lo and ord < test_hi]

            # write them out to a file (replace `all' in name with train/test)
            fname_base = os.path.basename(file)
            write_data(dir_prefix + "%02d" % fold_no, fname_base, 'train', train_data)
            write_data(dir_prefix + "%02d" % fold_no, fname_base, 'test', test_data)
            

if __name__ == '__main__':
    main(sys.argv[1:])
