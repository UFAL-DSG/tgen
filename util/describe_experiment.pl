#!/usr/bin/env perl
#
# Given an experiment configuration file and several Makefile settings, this will
# create a short description
# TODO It only works if the settings in the config file are defined just once, not commented out

use strict;
use warnings;
use autodie;
use File::Basename;
use File::stat;
use File::Slurp;
use Getopt::Long;

my $USAGE = "Usage: ./$0 [-t TRAINING_SET] [-j JOBS] [-d DEBUG] [-c CV] file1.log file2.log [...]\n";

my ( $training_set, $jobs, $debug, $cv ) = ( '', '', '', '' );
GetOptions(
    'training_set|training|t=s' => \$training_set,
    'jobs|j=s'                  => \$jobs,
    'debug|d'                   => \$debug,
    'cv_runs|cv|c=s'            => \$cv,
) or die($USAGE);
die($USAGE) if ( !@ARGV );


# Gather the settings from the command arguments and config files
my ( $iters, $training_data, $gadgets, $run_setting ) = ( '', '', '', '' );
my $config_data = read_file($ARGV[0]);


# iterations
$iters = ( $config_data =~ /'passes'\s*:\s*([0-9]+)\s*,/ )[0];
$iters .= '/' . ( $config_data =~ /'rival_gen_max_iter'\s*:\s*([0-9]+)\s*,/ )[0];
$iters .= '/' . ( $config_data =~ /'rival_gen_max_defic_iter'\s*:\s*([0-9]+)\s*,/ )[0];
$iters =~ s/\/\//\/~\//;
$iters =~ s/\/$/\/~/;


# data
$training_data = ' + all data' if ( $training_set =~ /^training2/ );
$training_data = ' + half'     if ( $training_set =~ /^training1/ );
$training_data .= ' + dlimit cg' if ( $training_set =~ /dlimit$/ );
$training_data .= ' + llimit cg' if ( $training_set =~ /llimit$/ );
$training_data .= ' + delex cg'  if ( $training_set =~ /delex$/ );
$training_data .= ' + lex cg'    if ( $training_set =~ /[12]$/ );


# gadgets
if ( $config_data =~ /'diffing_trees'\s*:\s*'([^']*)'/ ) {
    $gadgets = ' + difftr ' . $1;
    $gadgets =~ s/weighted/wt/;
}

if ( $config_data =~ /'future_promise_weight'\s*:\s*([0-9.]+)\s*,/ and $1 ) {
    my $fut_weight = $1;
    $gadgets .= ' + fut:' . ( $config_data =~ /'future_promise_type'\s*:\s*'([^']*)'/ )[0] . '=' . $fut_weight;
    $gadgets =~ s/exp_children/expc/;
}


# run setting
if ($jobs) {
    $run_setting = $jobs . 'j';
}
if ($cv) {
    my @cv_runs = split /\s+/, $cv;
    $run_setting .= ' ' . scalar(@cv_runs) . 'CV';
}
if ($debug) {
    $run_setting .= ' DEBUG';
}
$run_setting =~ s/^ //;
$run_setting =~ s/ +/,/g;


# Print the output.
print "$iters$training_data$gadgets ($run_setting)";
