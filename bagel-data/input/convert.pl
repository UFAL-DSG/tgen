#!/usr/bin/env perl
#
# Converting Bagel data set to our DA format.
#

use strict;
use warnings;
use autodie;

if (@ARGV != 3){
    die('Usage: ./convert.pl input output-das output-text');
}

open(my $in, '<:utf8', $ARGV[0]);
open(my $out_das, '>:utf8', $ARGV[1]);
open(my $out_text, '>:utf8', $ARGV[2]);


#
# TODO: musím si zapamatovat X z dialogových aktů a projektovat je do věty ve správném pořadí, 
# jinak to nebude dávat smysl
#
my $da_line = '';

while (my $line = <$in>){
    next if $line =~ /^FULL_DA/;

    if ($line =~ /^ABSTRACT_DA/){
        $line =~ s/ABSTRACT_DA = //;
        $line =~ s/,/)&inform(/g;
        $da_line = $line;
    }
    elsif ($line =~ /^->/){
        $line =~ s/-> "//;
        $line =~ s/";//;
        while ($da_line =~ m/([a-z]+)="X([0-9]?)"/g){  # number variables
            my ($slot, $xnum) = ($1, $2);
            $line =~ s/\[$slot\+X\]X/X$xnum/;
        }
        $line =~ s/\[[^\]]*\]//g;
       
        if ($line !~ m/\.\s*$/){ # add dots at every sentence
            $line =~ s/(\S)(\s*)$/$1\.$2/;
        }
        
        # remove variable numbers (for now)
        $line =~ s/X[0-9]/X/g;
        $da_line =~ s/X[0-9]/X/g;
        
        print {$out_das} $da_line;
        print {$out_text} $line;
    }
}
