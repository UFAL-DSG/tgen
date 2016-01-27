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

my $USAGE = "Usage: ./$0 [-t TRAINING_SET] [-j JOBS] [-d] [-c CV] [-r] file1.log file2.log [...]\n";

my ( $training_set, $jobs, $debug, $cv, $rands, $portion ) = ( '', '', 0, '', 0, 1.0 );
GetOptions(
    'training_set|training|t=s' => \$training_set,
    'jobs|j=s'                  => \$jobs,
    'debug|d'                   => \$debug,
    'cv_runs|cv|c=s'            => \$cv,
    'rands|r'                   => \$rands,
    'train_portion|portion|p=f' => \$portion,
) or die($USAGE);
die($USAGE) if ( !@ARGV );

# Gather the settings from the command arguments and config files
my ( $iters, $training_data, $gadgets, $run_setting, $nn_shape ) = ( '', '', '', '', '' );
my $config_data = read_file( $ARGV[0] );
my $mode = 'percrank';

if ( $ARGV[0] =~ /seq2seq/ ){
    $mode = 'seq2seq';
}

# iterations
$iters = ( $config_data =~ /'passes'\s*:\s*([0-9]+)\s*,/ )[0];
if ($mode eq 'percrank'){
    $iters .= '/' . ( $config_data =~ /'rival_gen_max_iter'\s*:\s*([0-9]+)\s*,/ )[0];
    $iters .= '/' . ( $config_data =~ /'rival_gen_max_defic_iter'\s*:\s*([0-9]+)\s*,/ )[0];
}
elsif ($mode eq 'seq2seq'){
    $iters .= '/' . ( $config_data =~ /'batch_size'\s*:\s*([0-9]+)\s*,/ )[0];
    $iters .= '/' . ( $config_data =~ /'alpha'\s*:\s*([0-9eE-]+)\s*,/ )[0];
}
$iters =~ s/\/\//\/~\//;
$iters =~ s/\/$/\/~/;

# data
$training_data = ' + all' if ( $training_set =~ /^training2/ );
$training_data = ' + 1/2' if ( $training_set =~ /^training1/ );
if ( $portion < 1.0 ) {
    $training_data .= '*' . sprintf( "%.2g", $portion );
}
$training_data .= ' aall'        if ( $training_set =~ /^training[12]aall/ );
$training_data .= ' + dc'        if ( $training_set =~ /^training[12]_dc/ );
$training_data .= ' + rc'        if ( $training_set =~ /^training[12]_rc/ );
$training_data .= ' + sc'        if ( $training_set =~ /^training[12]_sc/ );
$training_data .= ' + xc'        if ( $training_set =~ /^training[12]_xc/ );
$training_data .= ' + flat'      if ( $training_set =~ /^training[12]_flat/ );
$training_data .= ' + norep'     if ( $training_set =~ /^training[12]_norep/ );
$training_data .= ' + dlimit cg' if ( $training_set =~ /dlimit$/ );
$training_data .= ' + llimit cg' if ( $training_set =~ /llimit$/ );
$training_data .= ' + delex cg'  if ( $training_set =~ /delex$/ );
$training_data .= ' + lex cg'    if ( $training_set =~ /[12]$/ );
if ( $training_set =~ /-(.*)$/ ) {
    $training_data .= ' + ' . $1 . ' cg';
}

if ( $mode eq 'percrank' ){
    # gadgets
    if ( $config_data =~ /'diffing_trees'\s*:\s*'([^']*)'/ ) {
        $gadgets = ' + dt ' . $1;
        $gadgets =~ s/weighted/wt/;
    }

    if ( $config_data =~ /'future_promise_weight'\s*:\s*([0-9.]+)\s*,/ and $1 ) {
        my $fut_weight = $1;
        $gadgets .= ' + fut:' . ( $config_data =~ /'future_promise_type'\s*:\s*'([^']*)'/ )[0] . '=' . $fut_weight;
        $gadgets =~ s/exp_children/expc/;
    }

    if ( $config_data =~ /'nn'\s*:\s*'/ ) {
        $nn_shape = ' + ' . ( $config_data =~ /'nn'\s*:\s*'([^']*)'/ )[0];
    }

    # NN shape
    if ( $config_data =~ /'nn'\s*:\s*'emb/ ) {
        $nn_shape .= '/' .  ( $config_data =~ /'nn_shape'\s*:\s*'([^']*)'/ )[0];
        $nn_shape .= ' E' . ( ( $config_data =~ /'emb_size'\s*:\s*([0-9]*)/ )[0] // 20 );
        $nn_shape .= '-N' . ( ( $config_data =~ /'num_hidden_units'\s*:\s*([0-9]*)/ )[0] // 512 );
        $nn_shape .= '-A' . ( ( $config_data =~ /'alpha'\s*:\s*([0-9.]+)/ )[0] // 0.1 );
        $nn_shape .= '-C' . ( ( $config_data =~ /'cnn_filter_length'\s*:\s*([0-9]+)/ )[0] // 3 )
            . '/' . ( ( $config_data =~ /'cnn_num_filters'\s*:\s*([0-9]+)/ )[0] // 3 );
        $nn_shape .= '-' . ( ( $config_data =~ /'initialization'\s*:\s*'([^']*)'/ )[0] // 'uniform_glorot10' );

        # NN gadgets
        $nn_shape .= ' + ngr' if ( $config_data =~ /'normgrad'\s*:\s*True/ );
    }
}
elsif ( $mode eq 'seq2seq' ){

    $nn_shape .= ' E' . ( ( $config_data =~ /'emb_size'\s*:\s*([0-9]*)/ )[0] // 50 );
    $nn_shape .= '-N' . ( ( $config_data =~ /'num_hidden_units'\s*:\s*([0-9]*)/ )[0] // 128 );

    $nn_shape .= ' +att'  if ( $config_data =~ /'nn_type'\s*:\s*'emb_attention_seq2seq'/ );
    $nn_shape .= ' +sort'  if ( $config_data =~ /'sort_da_emb'\s*:\s*True/ );
    $nn_shape .= ' ->tok'  if ( $config_data =~ /'use_tokens'\s*:\s*True/ );
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
if ($rands) {
    $run_setting .= ' RANDS';
}
$run_setting =~ s/^ //;
$run_setting =~ s/ +/,/g;

# Print the output.
print "$iters$training_data$gadgets$nn_shape ($run_setting)";
