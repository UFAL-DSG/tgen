#!/usr/bin/env perl

use strict;
use warnings;

my @patterns = ( 'BEST', 'NODE', 'DEP', 'NIST score ' );
my %data;
my %lines;

# collect data
while ( my $line = <> ) {
    chomp $line;
    foreach my $pattern (@patterns) {
        if ( $line =~ /$pattern/ ) {

            if ( !$lines{$pattern} ) {
                $lines{$pattern} = $line;
            }

            # extract all numbers on the line
            my @nums = $line =~ m/[0-9]+\.[0-9]+/g;
            if ( !$data{$pattern} ) {
                $data{$pattern} = [];
            }
            push @{ $data{$pattern} }, \@nums;
        }
    }
}

foreach my $pattern (@patterns) {
    
    my $values = $data{$pattern};
    
    # average data
    my $cumul = shift @$values;
    my $ctr   = 1;
    foreach my $valline (@$values) {
        for ( my $i = 0; $i < @$valline; ++$i ) {
            $cumul->[$i] += $valline->[$i];
        }
        $ctr++;
    }
    for ( my $i = 0; $i < @$cumul; ++$i ) {
        $cumul->[$i] /= $ctr;
    }

    # print out the result
    my $line = $lines{$pattern};
    my $out  = "";
    
    while ( $line =~ m/([0-9]+\.[0-9]+)/g ) {
        my $num = $1;
        my $endpos = pos $line;
        $out .= substr( $line, length($out), $endpos - length($out) - length($num) );
        $out .= sprintf( "%.4f", shift @$cumul );
    }
    print $out . "\n";
}

