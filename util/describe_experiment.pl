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

my $USAGE = "Usage: ./$0 [-t TRAINING_SET] [-e] [-j JOBS] [-d] [-c CV] [-r] [-k] file1.log file2.log [...]\n";

my ( $eval_data, $training_set, $jobs, $debug, $cv, $rands, $portion, $toks ) = ( 0, '', '', 0, '', 0, 1.0 );
GetOptions(
    'training_set|training|t=s' => \$training_set,
    'jobs|j=s'                  => \$jobs,
    'debug|d'                   => \$debug,
    'cv_runs|cv|c=s'            => \$cv,
    'rands|r'                   => \$rands,
    'train_portion|portion|p=f' => \$portion,
    'toks|k'                    => \$toks,
    'eval_data|eval|e'          => \$eval_data,
) or die($USAGE);
die($USAGE) if ( !@ARGV );

# Gather the settings from the command arguments and config files
my ( $data_set, $iters, $training_data, $gadgets, $run_setting, $nn_shape ) = ( '', '', '', '', '', '' );
my $config_data = read_file( $ARGV[0] );

# remove commented-out lines
$config_data =~ s/^\s*#.*$//gm;

my $mode = 'percrank';
my $classif_filter_data = '';
my $lexicalizer_data = '';

if ( $ARGV[0] =~ /seq2seq/ ){
    $mode = 'seq2seq';

    # remove classification filter data so that they do not influence reading other settings
    $classif_filter_data = ( $config_data =~ /'classif_filter'\s*:\s*{([^}]*)}/s )[0];
    $config_data =~ s/'classif_filter'\s*:\s*{[^}]*}//s;

    # do the same with lexicalizer data
    $lexicalizer_data = ( $config_data =~ /'lexicalizer'\s*:\s*{([^}]*)}/s )[0];
    $config_data =~ s/'lexicalizer'\s*:\s*{[^}]*}//s;
}

# data set (devel -- default, eval -- mark)
if ($eval_data){
    $data_set = "\e[1;31mE\e[0m ";
}

# iterations
$iters = ( $config_data =~ /'passes'\s*:\s*([0-9]+)\s*,/ )[0];
if ($mode eq 'percrank'){
    $iters .= '/' . ( $config_data =~ /'rival_gen_max_iter'\s*:\s*([0-9]+)\s*,/ )[0];
    $iters .= '/' . ( $config_data =~ /'rival_gen_max_defic_iter'\s*:\s*([0-9]+)\s*,/ )[0];
}
elsif ($mode eq 'seq2seq'){
    $iters .= '/' . ( $config_data =~ /'batch_size'\s*:\s*([0-9]+)\s*,/ )[0];
    $iters .= '/' . ( $config_data =~ /'alpha'\s*:\s*([.0-9eE-]+)\s*,/ )[0];
    if ( $config_data =~ /'alpha_decay'\s*:\s*([.0-9eE-]+)\s*,/ and $1 > 0){
        $iters .= '^' . $1;
    }
}
$iters =~ s/\/\//\/~\//;
$iters =~ s/\/$/\/~/;

if ($mode eq 'seq2seq' and $config_data =~ /'validation_size'\s*:\s*([0-9]+)\s*,/ and $1 != 0 ){
    $iters = ( ( $config_data =~ /'min_passes'\s*:\s*([0-9]+)\s*,/ )[0] // 1 ) . '-' . $iters;
    $iters .= ' V' . ( $config_data =~ /'validation_size'\s*:\s*([0-9]+)\s*,/ )[0];
    $iters .= '@' . ( ( $config_data =~ /'validation_freq'\s*:\s*([0-9]+)\s*,/ )[0] // 10);
    $iters .= ' I' . ( ( $config_data =~ /'improve_interval'\s*:\s*([0-9]+)\s*,/ )[0] // 10);
    $iters .= '@' . ( ( $config_data =~ /'top_k'\s*:\s*([0-9]+)\s*,/ )[0] // 5);
    if ( ( $config_data =~ /'bleu_validation_weight'\s*:\s*([01]\.?[0-9]*)/ ) ){
        my $val = $1;
        if ($val > 0.0){
            $iters .= ' B' . sprintf( "%.2g", $val );
        }
    }
}

# data style
$training_data = ' + all' if ( $training_set =~ /^training2/ );
$training_data = ' + 1/2' if ( $training_set =~ /^training1/ );
if ( $portion < 1.0 ) {
    $training_data .= '*' . sprintf( "%.2g", $portion );
}
$training_data .= ' aall'        if ( $training_set =~ /^training[12_]aall/ );
$training_data .= ' toks'        if ( $training_set =~ /^training[12_]toks/ or $toks );
$training_data .= ' + dc'        if ( $training_set =~ /^training[12]_dc/ );
$training_data .= ' + rc'        if ( $training_set =~ /^training[12]_rc/ );
$training_data .= ' + sc'        if ( $training_set =~ /^training[12]_sc/ );
$training_data .= ' + xc'        if ( $training_set =~ /^training[12]_xc/ );
$training_data .= ' + flat'      if ( $training_set =~ /^training[12]_flat/ );
$training_data .= ' + norep'     if ( $training_set =~ /^training[12]_norep/ );
if ($mode eq 'percrank'){
    $training_data .= ' + dlimit cg' if ( $training_set =~ /dlimit$/ );
    $training_data .= ' + llimit cg' if ( $training_set =~ /llimit$/ );
    $training_data .= ' + delex cg'  if ( $training_set =~ /delex$/ );
    $training_data .= ' + lex cg'    if ( $training_set =~ /[12]$/ );
    if ( $training_set =~ /-(.*)$/ ) {
        $training_data .= ' + ' . $1 . ' cg';
    }
}
else {
    if ( $training_set =~ /-(.*)$/ ) {
        $training_data .= ' + ' . $1;
    }
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

    $nn_shape .= ' +lc'  if ( $config_data =~ /'embeddings_lowercase'\s*:\s*True/ );

    $nn_shape .= ' E' . ( ( $config_data =~ /'emb_size'\s*:\s*([0-9]*)/ )[0] // 50 );
    $nn_shape .= '-N' . ( ( $config_data =~ /'num_hidden_units'\s*:\s*([0-9]*)/ )[0] // 128 );
    if ( $config_data =~ /'dropout_keep_prob'\s*:\s*(0\.[0-9]*)/ ){
        $nn_shape .= '-D' . ( $config_data =~ /'dropout_keep_prob'\s*:\s*(0\.[0-9]*)/ )[0];
    }
    if ( $config_data =~ /'beam_size'\s*:\s*([0-9]*)/ ){
        $nn_shape .= '-B' . ( $config_data =~ /'beam_size'\s*:\s*([0-9]*)/ )[0];
    }
    $nn_shape .= ' ' . ( ( $config_data =~ /'cell_type'\s*:\s*'([^']*)'/ )[0] // 'lstm' );

    $nn_shape .= ' +att'  if ( $config_data =~ /'nn_type'\s*:\s*'emb_attention_seq2seq(_context)?'/ );
    $nn_shape .= ' +att2'  if ( $config_data =~ /'nn_type'\s*:\s*'emb_attention2_seq2seq(_context)?'/ );
    $nn_shape .= ' +sort'  if ( $config_data =~ /'sort_da_emb'\s*:\s*True/ );
    $nn_shape .= ' +adgr'  if ( $config_data =~ /'optimizer_type'\s*:\s*'adagrad'/ );
    $nn_shape .= ' +dc'  if ( $config_data =~ /'use_dec_cost'\s*:\s*True/ );
    $nn_shape .= ' ->tok'  if ( $config_data =~ /'use_tokens'\s*:\s*True/ or $config_data =~ /'mode'\s*:\s*'tokens'/ );
    $nn_shape .= ' ->tls'  if ( $config_data =~ /'mode'\s*:\s*'tagged_lemmas'/ );

    if ( $config_data =~ /'nn_type'\s*:\s*'emb_attention2?_seq2seq_context'/ ){
        $nn_shape .= ' +C-sepenc';
    }
    elsif ( $config_data =~ /'use_context'\s*:\s*True/ and $config_data =~ /'use_div_token'\s*:\s*True/){
        $nn_shape .= ' +C-divtok';
    }
    elsif ( $config_data =~ /'use_context'\s*:\s*True/ ){
        $nn_shape .= ' +C-basic';
    }
    if ( $config_data =~ /'context_bleu_weight'\s*:\s*([0-9.e-]*)/ and $1 != 0 ){
        $nn_shape .= ' +CB-' . ( $config_data =~ /'context_bleu_weight'\s*:\s*([0-9.e-]*)/ )[0];
    }

    if ( $config_data =~ /'length_norm_weight'\s*:\s*([0-9.e-]*)/ and $1 != 0 ){
        $nn_shape .= ' +LN-' . ( $config_data =~ /'length_norm_weight'\s*:\s*([0-9.e-]*)/ )[0];
    }
    if ( $config_data =~ /'sample_top_k'\s*:\s*([0-9]*)/ and $1 > 1 ){
        $nn_shape .= ' +samp' . ( $config_data =~ /'sample_top_k'\s*:\s*([0-9]*)/ )[0];
    }

    if ( $config_data =~ /'average_models'\s*:\s*True/ ){
        $nn_shape .= ' +am';
        my $top_k = ( ( $config_data =~ /'average_models_top_k'\s*:\s*([0-9]*)/ )[0] // 0 );
        if ($top_k){
            $nn_shape .= $top_k;
        }
    }

    # classificator filter settings
    if ($classif_filter_data){
        $nn_shape .= ' +cf';
        my $cf_nn_type = (( $classif_filter_data =~ /'nn_shape'\s*:\s*'([^']*)'/ )[0] // '??' );

        $nn_shape .= '_' . $cf_nn_type . '_';

        if ( $classif_filter_data =~ /'min_passes'\s*:\s*([0-9]+)\s*,/ ){
            $nn_shape .= ( $classif_filter_data =~ /'min_passes'\s*:\s*([0-9]+)\s*,/ )[0] . '-';
        }
        $nn_shape .= (( $classif_filter_data =~ /'passes'\s*:\s*([0-9]+)\s*,/ )[0] // '~' );
        $nn_shape .= '/' . (( $classif_filter_data =~ /'batch_size'\s*:\s*([0-9]+)\s*,/ )[0] // '~' );
        $nn_shape .= '/' . (( $classif_filter_data =~ /'alpha'\s*:\s*([.0-9eE-]+)\s*,/ )[0] // '~' );
        $nn_shape .= '_';

        if ( $cf_nn_type eq 'rnn' ){
            $nn_shape .= 'E' . ( ( $classif_filter_data =~ /'emb_size'\s*:\s*([0-9]*)/ )[0] // 50 );
        }
        $nn_shape .= '-N' . ( ( $classif_filter_data =~ /'num_hidden_units'\s*:\s*([0-9]*)/ )[0] // 128 );
    }

    # lexicalizer settings
    if ($lexicalizer_data){
        $nn_shape .=  ' +lx';
        my $form_sel_type = (( $lexicalizer_data =~ /'form_select_type'\s*:\s*'([^']*)'/ )[0] // 'random' );
        my $form_sample = (( $lexicalizer_data =~ /'form_sample'\s*:\s*(False|True)/ )[0] // 'False' );
        $nn_shape .= '-' . $form_sel_type;
        if ($form_sample eq 'True'){
            $nn_shape .= '+samp';
        }
    }
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
print "$data_set$iters$training_data$gadgets$nn_shape ($run_setting)";
