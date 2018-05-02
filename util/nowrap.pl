#!/usr/bin/env perl
#
# Shorten all lines to <width> characters, ignore the rest,
# ignore shell control sequences while doing so (to preserve colors)

use strict;
use warnings;

die("Usage: $0 <width>") if (@ARGV != 1);
my $width = shift @ARGV;

while (my $line = <>){
    $line =~ s/\r?\n$//;
    my $len = 0;
    my $lastpos = 0;
    my $start = -1;
    while ( $line =~ /(\x1B\[(?:(?:[0-9]+)(?:;[0-9]+)*)?[mKHfJ])/g){
        # TODO use (pos $line - length $1), keep previous pos => have length to add to len
        my $pos = pos $line;
        $start = $pos - length $1;
        $len += $start - $lastpos;
        $lastpos = $pos;
        # if len goes over the limit, cut off and break
        if ($len >= $width){
            last;
        }
    }
    if (!defined(pos $line)){
        $start = length $line;
        $len += $start - $lastpos;
    }
    $line = substr $line, 0, $start - $len + $width;
    print $line, "\n";
}
