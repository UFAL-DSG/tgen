#!/usr/bin/env perl
#
# Learning curve -- updating the #instances to use, limiting validation to 0.2 train,
# increasing number of passes to ensure similar data exposure.
#

use strict;
use warnings;
use List::Util qw(min);


my @sizes = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7);


`cp config/seq2seq.py config/seq2seq.py.bak`;
foreach my $size (@sizes){
    my $valid_size = min(2000, int($size * 42061 * .2));
    my $passes = sprintf("%.f", 20 / $size);
    my $min_passes = sprintf("%.f", 5 / $size);

    print $size, " ", $valid_size, " ", $passes, " ", $min_passes, "\n";
    `cp config/seq2seq.py.bak config/seq2seq.py`;
    `sed "s/'passes': 20,/'passes': $passes,/" -i config/seq2seq.py`;
    `sed "s/'min_passes': 5,/'min_passes': $min_passes,/" -i config/seq2seq.py`;
    `sed "s/'validation_size': 2000,/'validation_size': $valid_size,/" -i config/seq2seq.py`;
    `make seq2seq_cv_run TRAINING_SET=training-delex_v3 RANDS=1 TRAIN_PORTION=$size`;
}
`cp config/seq2seq.py.bak config/seq2seq.py`;
