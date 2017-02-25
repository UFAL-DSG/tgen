#!/usr/bin/env perl
# Code for paired bootstrap resampling to perform significance testing for BLEU. The first
# two arguments are filenames containing segment-level statistics printed by the modified
# mteval script (mteval-v13a-sig.pl). Third argument is the number of samples to use and
# fourth argument is the p-value to use for computing significance.
# The first system should be the one with the higher document-level score (only
# affects section that performs test after sampling and sample scoring).
#
# Kevin Gimpel
# 4/2/2011

if (scalar(@ARGV) != 4) {
    print "Usage: paired_bootstrap_resampling_bleu_v13a.pl system1_bleu_stats_file system2_bleu_stats_file num_samples p_value\n";
    exit;
}

$max_Ngram = 9;
$num_lines_1 = 0;
$num_lines_2 = 0;

# store the arguments
$num_samples = $ARGV[2];
$p = $ARGV[3];

# open the two stats files
open(IN_1, "<", $ARGV[0]);
open(IN_2, "<", $ARGV[1]);

my (@shortest_ref_length_array1, @shortest_ref_length_array2);
my (@match_cnt_array1, @match_cnt_array2);
my (@tst_cnt_array1, @tst_cnt_array2);
my (@ref_cnt_array1, @ref_cnt_array2);
my (@tst_info_array1, @tst_info_array2);
my (@ref_info_array1, @ref_info_array2);
my ($sc_ref_sys1, $sc_ref_sys2);

# read in the stats files
while (<IN_1>) {
    $num_lines_1++;
    $line = $_;
    chomp($line);

    my ($shortest_ref_length, @match_cnt, @tst_cnt, @ref_cnt, @tst_info, @ref_info);
    $shortest_ref_length = 0;
    # initialize variables
    for (my $j=1; $j<=$max_Ngram; $j++) {
	$match_cnt[$j] = $tst_cnt[$j] = $ref_cnt[$j] = $tst_info[$j] = $ref_info[$j] = 0;
    }

    @pieces = split /\s*\|\s*/, $line;
    $sc_ref_sys1 = $pieces[0];
    $shortest_ref_length = $pieces[1];
    push(@shortest_ref_length_array1, $shortest_ref_length);

    $match_cnt_line = $pieces[2];
    @subpieces = split / /, $match_cnt_line;
    for (my $j=1; $j<= $max_Ngram; $j++) {
	$match_cnt[$j] = @subpieces[$j - 1];
    }
    push(@match_cnt_array1, \@match_cnt);

    $tst_cnt_line = $pieces[3];
    @subpieces = split / /, $tst_cnt_line;
    for (my $j=1; $j<= $max_Ngram; $j++) {
	$tst_cnt[$j] = @subpieces[$j - 1];
    }
    push(@tst_cnt_array1, \@tst_cnt);

	$ref_cnt_line = $pieces[4];
	@subpieces = split / /, $ref_cnt_line;
	for (my $j=1; $j<= $max_Ngram; $j++) {
		$ref_cnt[$j] = @subpieces[$j - 1];
	}
	push(@ref_cnt_array1, \@ref_cnt);

	$tst_info_line = $pieces[5];
	@subpieces = split / /, $tst_info_line;
	for (my $j=1; $j<= $max_Ngram; $j++) {
		$tst_info[$j] = @subpieces[$j - 1];
	}
	push(@tst_info_array1, \@tst_info);

	$ref_info_line = $pieces[6];
	@subpieces = split / /, $ref_info_line;
	for (my $j=1; $j<= $max_Ngram; $j++) {
		$ref_info[$j] = @subpieces[$j - 1];
	}
	push(@ref_info_array1, \@ref_info);
}

while (<IN_2>) {
    $num_lines_2++;
	$line = $_;
	chomp($line);

	my ($shortest_ref_length, @match_cnt, @tst_cnt, @ref_cnt, @tst_info, @ref_info);
	$shortest_ref_length = 0;
	for (my $j=1; $j<=$max_Ngram; $j++) {
		$match_cnt[$j] = $tst_cnt[$j] = $ref_cnt[$j] = $tst_info[$j] = $ref_info[$j] = 0;
	}

	@pieces = split /\s*\|\s*/, $line;
	$sc_ref_sys2 = $pieces[0];
	$shortest_ref_length = $pieces[1];
	push(@shortest_ref_length_array2, $shortest_ref_length);

	$match_cnt_line = $pieces[2];
	@subpieces = split / /, $match_cnt_line;
	for (my $j=1; $j<= $max_Ngram; $j++) {
		$match_cnt[$j] = @subpieces[$j - 1];
	}
	push(@match_cnt_array2, \@match_cnt);

	$tst_cnt_line = $pieces[3];
	@subpieces = split / /, $tst_cnt_line;
	for (my $j=1; $j<= $max_Ngram; $j++) {
		$tst_cnt[$j] = @subpieces[$j - 1];
	}
	push(@tst_cnt_array2, \@tst_cnt);

	$ref_cnt_line = $pieces[4];
	@subpieces = split / /, $ref_cnt_line;
	for (my $j=1; $j<= $max_Ngram; $j++) {
		$ref_cnt[$j] = @subpieces[$j - 1];
	}
	push(@ref_cnt_array2, \@ref_cnt);

	$tst_info_line = $pieces[5];
	@subpieces = split / /, $tst_info_line;
	for (my $j=1; $j<= $max_Ngram; $j++) {
		$tst_info[$j] = @subpieces[$j - 1];
	}
	push(@tst_info_array2, \@tst_info);

	$ref_info_line = $pieces[6];
	@subpieces = split / /, $ref_info_line;
	for (my $j=1; $j<= $max_Ngram; $j++) {
		$ref_info[$j] = @subpieces[$j - 1];
	}
	push(@ref_info_array2, \@ref_info);
}

# close the statistics files
close IN_1;
close IN_2;

# quit if the files contained different numbers of lines
if ($num_lines_1 != $num_lines_2) {
    print "Error: two statistics files differ in number of lines. File 1: $num_lines_1 lines; File 2: $num_lines_2 lines. Quitting...\n";
    exit;
}
# if we made it here, the files must have the same number of lines, so set num_total_lines
$num_total_lines = $num_lines_1;

# now that we have the segment-level statistics, do paired bootstrap resampling

# declare data structures that will hold the document-level statistics for each sample
my ($cum_ref_length_1, @cum_match_1, @cum_tst_cnt_1, @cum_ref_cnt_1, @cum_tst_info_1, @cum_ref_info_1);
my ($cum_ref_length_2, @cum_match_2, @cum_tst_cnt_2, @cum_ref_cnt_2, @cum_tst_info_2, @cum_ref_info_2);

# declare arrays to hold the NIST and BLEU scores for each sample
my (@system1_bleu, @system1_nist, @system2_bleu, @system2_nist);

$curr_sample = 0;
while ($curr_sample < $num_samples) {
    # initialize some data structures
	$cum_ref_length_1 = 0;
	$cum_ref_length_2 = 0;
	for (my $j=1; $j<=$max_Ngram; $j++) {
	    $cum_match_1[$j] = $cum_tst_cnt_1[$j] = $cum_ref_cnt_1[$j] = $cum_tst_info_1[$j] = $cum_ref_info_1[$j] = 0;
	    $cum_match_2[$j] = $cum_tst_cnt_2[$j] = $cum_ref_cnt_2[$j] = $cum_tst_info_2[$j] = $cum_ref_info_2[$j] = 0;
	}

	my @inds;
	# first, generate num_total_lines integers between 1 and num_total_lines
	for ($i = 0; $i < $num_total_lines; $i++) {
		$r = int(rand($num_total_lines));
		push(@inds, $r);
	}

	# sort the line indices
	@sinds = sort @inds;

	# go through the sampled line indices and accumulate statistics for this
	# sample of the data
	for ($i = 0; $i < $num_total_lines; $i++) {
		$curr_ind = $sinds[$i];
		$cum_ref_length_1 += $shortest_ref_length_array1[$curr_ind];
		$cum_ref_length_2 += $shortest_ref_length_array2[$curr_ind];
		for (my $j=1; $j<=$max_Ngram; $j++) {
		    $cum_match_1[$j] += $match_cnt_array1[$curr_ind][$j];
		    $cum_tst_cnt_1[$j] += $tst_cnt_array1[$curr_ind][$j];
		    $cum_ref_cnt_1[$j] += $ref_cnt_array1[$curr_ind][$j];
		    $cum_tst_info_1[$j] += $tst_info_array1[$curr_ind][$j];
		    $cum_ref_info_1[$j] += $ref_info_array1[$curr_ind][$j];

		    $cum_match_2[$j] += $match_cnt_array2[$curr_ind][$j];
		    $cum_tst_cnt_2[$j] += $tst_cnt_array2[$curr_ind][$j];
		    $cum_ref_cnt_2[$j] += $ref_cnt_array2[$curr_ind][$j];
		    $cum_tst_info_2[$j] += $tst_info_array2[$curr_ind][$j];
		    $cum_ref_info_2[$j] += $ref_info_array2[$curr_ind][$j];
		}

	}

	# compute the NIST and BLEU scores for each system, adding the results to
	# the appropriate array
	my %DOCmt = ();
	push(@system1_nist, nist_score($sc_ref_sys1, \@cum_match_1, \@cum_tst_cnt_1, \@cum_ref_cnt_1, \@cum_tst_info_1, \@cum_ref_info_1, 'my-system', %DOCmt));
	my %DOCmt = ();
	push(@system1_bleu, bleu_score($cum_ref_length_1, \@cum_match_1, \@cum_tst_cnt_1, 'my-system', %DOCmt));
	my %DOCmt = ();
	push(@system2_nist, nist_score($sc_ref_sys2, \@cum_match_2, \@cum_tst_cnt_2, \@cum_ref_cnt_2, \@cum_tst_info_2, \@cum_ref_info_2, 'my-system', %DOCmt));
	my %DOCmt = ();
	push(@system2_bleu, bleu_score($cum_ref_length_2, \@cum_match_2, \@cum_tst_cnt_2, 'my-system', %DOCmt));

	$curr_sample++;
}

# now we've finished sampling and computing scores, so we can perform the test
# obtain the fraction of times that system 1 was better than system 2
$system_1_better_nist = 0;
for ($i = 0; $i < $num_samples; $i++) {
	if ($system1_nist[$i] > $system2_nist[$i]) {
		$system_1_better_nist++;
	}
}
$system_1_better_bleu = 0;
for ($i = 0; $i < $num_samples; $i++) {
	if ($system1_bleu[$i] > $system2_bleu[$i]) {
		$system_1_better_bleu++;
	}
}
$system_1_better_nist_fraction = $system_1_better_nist / $num_samples;
$system_1_better_bleu_fraction = $system_1_better_bleu / $num_samples;
print "System 1 NIST better: $system_1_better_nist / $num_samples = $system_1_better_nist_fraction";
if ((1 - $system_1_better_nist_fraction) < $p) {
    print " -- NIST SIGNIFICANT at p = $p";
}
print "\n";

print "System 1 BLEU better: $system_1_better_bleu / $num_samples = $system_1_better_bleu_fraction";
if ((1 - $system_1_better_bleu_fraction) < $p) {
	print " -- BLEU SIGNIFICANT at p = $p";
}
print "\n";

exit;

# function to compute BLEU score (taken from mteval-v13a.pl)
###############################################################################################################################
# Default method used to compute the BLEU score, using smoothing.
# Note that the method used can be overridden using the '--no-smoothing' command-line argument
# The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each precision score whose matching n-gram count is null
# k is 1 for the first 'n' value for which the n-gram match count is null
# For example, if the text contains:
#   - one 2-gram match
#   - and (consequently) two 1-gram matches
# the n-gram count for each individual precision score would be:
#   - n=1  =>  prec_count = 2     (two unigrams)
#   - n=2  =>  prec_count = 1     (one bigram)
#   - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
#   - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
###############################################################################################################################
sub bleu_score
{
	my ($ref_length, $matching_ngrams, $tst_ngrams, $sys, $SCOREmt) = @_;
	my $score = 0;
	my $iscore = 0;
	my $exp_len_score = 0;
	$exp_len_score = exp( min (0, 1 - $ref_length / $tst_ngrams->[ 1 ] ) ) if ( $tst_ngrams->[ 1 ] > 0 );
	my $smooth = 1;
	for ( my $j = 1; $j <= $max_Ngram; ++$j )
	{
		if ( $tst_ngrams->[ $j ] == 0 )
		{
			$iscore = 0;
		}
		elsif ( $matching_ngrams->[ $j ] == 0 )
		{
			$smooth *= 2;
			$iscore = log( 1 / ( $smooth * $tst_ngrams->[ $j ] ) );
		}
		else
		{
			$iscore = log( $matching_ngrams->[ $j ] / $tst_ngrams->[ $j ] );
		}
		$SCOREmt->{ $j }{ $sys }{ ind } = exp( $iscore );
		$score += $iscore;
		$SCOREmt->{ $j }{ $sys }{ cum } = exp( $score / $j ) * $exp_len_score;
	}
	return $SCOREmt->{ 4 }{ $sys }{ cum };
}


#################################
# function to compute NIST score (taken from mteval-v11b.pl)
sub nist_score {

    my ($nsys, $matching_ngrams, $tst_ngrams, $ref_ngrams, $tst_info, $ref_info, $sys, %SCOREmt) = @_;

    my $score = 0;
    my $iscore = 0;

    for (my $n=1; $n<=$max_Ngram; $n++) {
        $score += $tst_info->[$n]/max($tst_ngrams->[$n],1);
        $SCOREmt{$n}{$sys}{cum} = $score * nist_length_penalty($tst_ngrams->[1]/($ref_ngrams->[1]/$nsys));

        $iscore = $tst_info->[$n]/max($tst_ngrams->[$n],1);
        $SCOREmt{$n}{$sys}{ind} = $iscore * nist_length_penalty($tst_ngrams->[1]/($ref_ngrams->[1]/$nsys));
    }
    return $SCOREmt{5}{$sys}{cum};
}

#################################
# function to compute NIST length penalty (taken from mteval-v11b.pl)
sub nist_length_penalty {

    my ($ratio) = @_;
    return 1 if $ratio >= 1;
    return 0 if $ratio <= 0;
    my $ratio_x = 1.5;
    my $score_x = 0.5;
    my $beta = -log($score_x)/log($ratio_x)/log($ratio_x);
    return exp (-$beta*log($ratio)*log($ratio));
}

#################################

sub max {

    my ($max, $next);

    return unless defined ($max=pop);
    while (defined ($next=pop)) {
        $max = $next if $next > $max;
    }
    return $max;
}

#################################

sub min {

    my ($min, $next);

    return unless defined ($min=pop);
    while (defined ($next=pop)) {
        $min = $next if $next < $min;
    }
    return $min;
}
