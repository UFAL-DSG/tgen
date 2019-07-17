#!/usr/bin/env perl
#
# Creating text files with parallel sentences for human error annotation
#

use strict;
use warnings;
use Getopt::Long;
use List::Util qw(any shuffle);


my $num_samples = 50;
my $out_name = '';
my $unique = 0;
my $shuf = 0;
my $ordered = 0;
my $numbered = 0;

GetOptions(
    'num_samples|num|n=i' => \$num_samples,  # target number of sentences to sample
    'out_name|out|o=s' => \$out_name,  # output file name
    'unique_first|unique|u' => \$unique,  # choose samples so that contents of 1st file are unique (may cause infinite loops!)
    'shuffle|shuf|s' => \$shuf,  # shuffle the order of different files (but for 1st one)
    'ordered|ord|r' => \$ordered,  # use the original order instead of random sampling
    'line_numbers|numbered|lines|l' => \$numbered,  # print out line numbers
) or die("Argument error.");

if (!@ARGV or !$out_name){
    die("Some files and output file must be specified");
}

my @files;
foreach my $filename (@ARGV){
    push @files, read_file($filename);
    print STDERR "Read " . scalar(@{$files[-1]}) . " lines from $filename\n";
}

my $data_len = scalar(@{$files[0]});
if (any { scalar(@$_) != $data_len } @files){
    die("Not all files are the same length.");
}

open(my $fh, '>:utf8', $out_name);
srand(1206);
my %used = ();
for (my $i = 0; $i < $num_samples; ++$i){
    my $line_no = -1;
    while ($line_no == -1 or $used{$files[0]->[$line_no]}){
        $line_no = $ordered ? $line_no + 1 : int(rand($data_len));
    }
    if ($unique){  # remember which 1st file contents (typically: DAs) were used & don't choose them next time
        $used{$files[0]->[$line_no]} = 1;
    }
    my @buf = ();
    for (my $fn = 0; $fn < @files; ++$fn){
        push @buf, ($ARGV[$fn] . "\t" . $files[$fn]->[$line_no]);
    }
    if ($shuf){  # shuffle while keeping the 1st file contents (DAs) in place
        @buf = ($buf[0], shuffle @buf[1 .. $#buf]);
    }
    if ($numbered){
        print $fh ($i + 1) . "\n";
    }
    print $fh join("\n", @buf) . "\n\n";
}


sub read_file {
    my ($fname) = @_;

    open my $fh, '<:utf8', $fname;
    chomp(my @lines = <$fh>);
    close $fh;
    @lines = map { s/<[^>]*>//g; $_ } grep {'^<seg'} @lines;
    return \@lines;
}






