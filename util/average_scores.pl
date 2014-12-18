#!/usr/bin/env perl

use strict;
use warnings;
use autodie;

my @patterns = ( 'BEST', 'NODE', 'DEP', 'NIST score ' );
my %data;
my %lines;

die("Usage: ./$0 file1.log file2.log [...]\n") if (!@ARGV);

# collect data
foreach my $file (@ARGV){

    open( my $fh, '<:utf8', $file );
    my %cur_data = ();

    while ( my $line = <$fh> ) {
        chomp $line;
        foreach my $pattern (@patterns) {
            if ( $line =~ /$pattern/ ) {

                if ( !$lines{$pattern} ) {
                    $lines{$pattern} = $line;
                }
                # extract all numbers on the line
                my @nums = $line =~ m/[0-9]+\.[0-9]+/g;
                # store them, keep only the last ones for this file
                $cur_data{$pattern} = \@nums;
            }
        }
    }

    # add to global list
    while (my ($p, $n) = each %cur_data){
        if ( !$data{$p} ) {
            $data{$p} = [];
        }
        push $data{$p}, $n;
    }
    close($fh);
}

foreach my $pattern (@patterns) {
    
    my $values = $data{$pattern};
    next if (!$values);
    
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

