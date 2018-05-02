#!/usr/bin/env perl
#
# Grep important scores from log files, print them out in a condensed format,
# with bash color codes.

use strict;
use warnings;
use autodie;
use File::Basename;
use File::stat;
use Getopt::Long;

my $USAGE = "Usage: ./$0 [--bleu-range=40:60] [--nist-range=3:6] file1.log file2.log [...]\n";

my $ranges = {
    'bleu' => '40:80',
    'nist' => '3:6',
    'meteor' => '0:1',
    'cider' => '0:10',
    'rouge' => '0:1',
};

GetOptions(
    'bleu-range|bleu|b=s' => \$ranges->{bleu},
    'nist-range|nist|n=s' => \$ranges->{nist},
    'meteor-range|meteor|m=s' => \$ranges->{meteor},
    'cider-range|cider|c=s' => \$ranges->{cider},
    'rouge_l-range|rouge-range|rouge_l|rouge|r=s' => \$ranges->{rouge},
) or die($USAGE);
die($USAGE) if ( !@ARGV );


# Filter ARGV and get just the last log file
# TODO make this an option
my $file_to_process = undef;

foreach my $file (@ARGV) {
    next if ( !-e $file );
    if ( !defined $file_to_process or ( ( stat($file_to_process) )->[9] < ( stat($file) )->[9] ) ) {
        $file_to_process = $file;
    }
}
exit() if ( !defined $file_to_process );


# Process the file
open( my $fh, '<:utf8', $file_to_process );
my ( $pr, $lists, $bleu ) = ( '', '', '' );

while ( my $line = <$fh> ) {
    chomp $line;

    $line =~ s/, / /g;
    $line =~ s/ = / /g;
    $line =~ s/: / /g;
    $line =~ s/(?<!NIST score) 0\.([0-9]{2})/ $1./g;

    # Node precision, recall, F1-measure
    if ( $line =~ /(Node precision|NODE scores|Token precision)/i ) {
        $line =~ s/.*Node p/P/i;
        $line =~ s/.*NODE scores //;
        $line =~ s/.*Token p/P/i;
        $line =~ s/F1//;
        $line =~ s/[a-z]//gi;
        $line =~ s/^\s+//;
        my ( $p, $r, $f ) = split( /\s+/, $line );
        $pr = rg( 50, 80, $p ) . "P $p" . rg( 50, 80, $r ) . "  R $r" . rg( 50, 80, $f ) . "  F $f\e[0m";
    }

    # Exact matches on open and close list
    elsif ( $line =~ /Gold tree BEST/i ) {
        $line =~ s/.*Gold tree BEST//;
        $line =~ s/[a-z]//gi;
        $line =~ s/^\s+//;
        my ( $b, $c, $a ) = split( /\s+/, $line );
        $lists = rg( 0, 40, $b ) . "B $b" . rg( 0, 40, $c ) . "  C $c" . rg( 0, 40, $a ) . "  A $a\e[0m";
    }

    elsif ( $line =~ /^(NIST|BLEU|METEOR|ROUGE_L|CIDEr) ([0-9.]+)$/){
        my ($metric, $value) = ($1, $2);
        my $letter = substr($metric, 0, 1);
        $metric =~ s/_L//;
        $metric = lc $metric;
        $bleu .= ($bleu ? "  " : "") . rg( split( /:/, $ranges->{$metric} ), $value ) . "$letter $value\e[0m";
    }
    # NIST & BLEU
    elsif ( $line =~ /^NIST/ ) {
        $line =~ s/^.*NIST/NIST/;
        $line =~ s/ for system.*$//;
        $line =~ s/[a-z]//gi;
        $line =~ s/^\s+//;
        my ( $n, $b ) = split( /\s+/, $line );
        $bleu = rg( split( /:/, $ranges->{nist} ), $n ) . "NIST $n" . rg( split( /:/, $ranges->{bleu} ), $b ) . "  BLEU $b\e[0m";
    }

    # just BLEU (for tokens setting)
    elsif ( $line =~ /BLEU score / ){
        $line =~ s/^.*BLEU score//;
        $line =~ s/\s+//g;
        my $b = sprintf "%.2f", $line;
        # leave spaces instead of NIST
        $bleu = "             " . rg( split( /:/, $ranges->{bleu} ), $b ) . "BLEU $b\e[0m";
    }
}

close($fh);

# Print the output
if (!$pr){
    print "$bleu\e[0m";
}
elsif ($lists){
    print "$pr  $lists  $bleu\e[0m";
}
else {
    print "$pr  $bleu\e[0m";
}

#
# Subs
#

# Get the bash 256 colors number given RGB (with values in the range 0-6)
sub rgb_code {
    my ( $r, $g, $b ) = @_;
    return "\e[38;5;" . ( ( 16 + ( 36 * $r ) + ( 6 * $g ) + $b ) ) . "m";
}

# Return red-green gradient rgb code
sub rg {
    my ( $t, $b, $v ) = @_;
    my $r = int( 0 + ( ( $v - $b ) / ( $t - $b ) * 6 ) );
    my $g = int( 6 - ( ( $v - $b ) / ( $t - $b ) * 6 ) );
    $r = 5 if ( $r > 5 );
    $r = 0 if ( $r < 0 );
    $g = 5 if ( $g > 5 );
    $g = 0 if ( $g < 0 );
    return rgb_code( $r, $g, 0 );
}

