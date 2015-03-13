#!/usr/bin/env perl

use strict;

#################################
# $Rev: 2406 $
# $LastChangedDate: 2007-01-22 14:36:52 -0500 (星期一, 22 一月 2007) $
#
# History:
#
# Jan 22, 2007,
#	NormalizeText() subroutine updated with the latest mteval-v11b script
# 	Notes from mteval-v11b:
#	version 11b -- text normalization modified:
#       take out the join digit line because it joins digits
#       when it shouldn't have
#       $norm_text =~ s/(\d)\s+(?=\d)/$1/g; #join digits
#	
#
# Jan 21, 2007, NormalizeText_Bleu() updated with the NormalizeText() method as in bleu_v1.04
# Be aware that the normalization can be wrong for non-English sentences
# http://www.cs.cmu.edu/~joy/research/compareNorm.es shows an example of using the NormalizeText() from mteval (NIST script)
# and using the NormalizeText() in bleu_v1.04
#
# version 11-a -- bug fixes:
#	the length penalty for Bleu is not calculated correctly
#	should use the closest ref len, instead of the shortest
#
# version 11 -- bug fixes:
#    * make filehandle operate in binary mode to prevent Perl from operating
#      (by default in Red Hat 9) in UTF-8
#    * fix failure on joining digits
#
# Based on the MTevalv09e. This program will just output the features which will later be used to calcualte the NIST
# 	Bleu scores using bootstrapping.
#	using Bleu style text normalization
#	By Joy, joy@cs.cmu.edu; Sep 20, 2003
#
# version 09e -- to evaluate the documents in the order as in the testing data
#
# version 09c -- bug fix (During the calculation of ngram information,
#    each ngram was being counted only once for each segment.  This has
#    been fixed so that each ngram is counted correctly in each segment.)
#
# version 09b -- text normalization modified:
#    * option flag added to preserve upper case
#    * non-ASCII characters left in place.
#
# version 09a -- text normalization modified:
#    * &quot; and &amp; converted to " and &, respectively
#    * non-ASCII characters kept together (bug fix)
#
# version 09 -- modified to accommodate sgml tag and attribute
#    names revised to conform to default SGML conventions.
#
# version 08 -- modifies the NIST metric in accordance with the
#    findings on the 2001 Chinese-English dry run corpus.  Also
#    incorporates the BLEU metric as an option and supports the
#    output of ngram detail.
#
# version 07 -- in response to the MT meeting on 28 Jan 2002 at ISI
#    Keep strings of non-ASCII characters together as one word
#    (rather than splitting them into one-character words).
#    Change length penalty so that translations that are longer than
#    the average reference translation are not penalized.
#
# version 06
#    Prevent divide-by-zero when a segment has no evaluation N-grams.
#    Correct segment index for level 3 debug output.
#
# version 05
#    improve diagnostic error messages
#
# version 04
#    tag segments
#
# version 03
#    add detailed output option (intermediate document and segment scores)
#
# version 02
#    accommodation of modified sgml tags and attributes
#
# version 01
#    same as bleu version 15, but modified to provide formal score output.
#
# original IBM version
#    Author: Kishore Papineni
#    Date: 06/10/2001
#################################

######
# Intro
#my ($date, $time) = date_time_stamp();
#print "MT evaluation run on $date at $time\n";
#print "command line:  ", $0, " ", join(" ", @ARGV), "\n";
my $usage = "\n\nUsage: $0 [-h] -r <ref_file> -s src_file -t <tst_file> [-m]\n\n".
    "Description:  This Perl script evaluates MT system performance.\n".
    "\n".
    "Required arguments:\n".
    "  -r <ref_file> is a file containing the reference translations for\n".
    "      the documents to be evaluated.\n".
    "  -s <src_file> is a file containing the source documents for which\n".
    "      translations are to be evaluated\n".
    "  -t <tst_file> is a file containing the translations to be evaluated\n".
    "\n".
    "Optional arguments:\n".    
    "  -b substitutes the original IBM BLEU score for the NIST score\n".
    "  -c preserves upper-case alphabetic characters\n".
    "  -d detailed output flag:\n".
    "         0 (default) for system-level score only\n".
    "         1 to include document-level scores\n".
    "         2 to include segment-level scores\n".
    "         3 to include ngram-level scores\n".
    "  -h prints this help message to STDOUT\n".
    "  -m normalization method:\n".
    "	      0 (default) NIST MTeval script\n".
    "	      1 Bleu MTeval script\n".
    "  -g maximum n-gram used for final Bleu/M-Bleu score\n".
    "  -i maximum n-gram used for final NIST score\n".
    "\n";

use vars qw ($opt_r $opt_s $opt_t $opt_n $opt_h $opt_d $opt_b $opt_c $opt_x $opt_m $opt_g $opt_i);
use Getopt::Std;
getopts ('r:s:t:n:d:m:g:i:hbc');
die $usage if defined($opt_h);
die "Error in command line:  ref_file not defined$usage" unless defined $opt_r;
die "Error in command line:  src_file not defined$usage" unless defined $opt_s;
die "Error in command line:  tst_file not defined$usage" unless defined $opt_t;

my $report_ngram_bleu = defined($opt_g) ? $opt_g : 4;
my $report_ngram_nist = defined($opt_i) ? $opt_i : 5;

my $max_Ngram = max($report_ngram_bleu, $report_ngram_nist);

my $detail = defined $opt_d ? $opt_d : 0;
my $preserve_case = defined $opt_c ? 1 : 0;
my $method = defined $opt_b ? "BLEU" : "NIST";
my $normalizationMethod = defined $opt_m ? $opt_m : 0;
my ($ref_file) = $opt_r;
my ($src_file) = $opt_s;
my ($tst_file) = $opt_t;

if($normalizationMethod==0){
	print "NormalizationText method: NIST $normalizationMethod\n";
}
else{
	print "NormalizationText method: Bleu $normalizationMethod\n";
}
######
# Global variables
my ($src_lang, $tgt_lang, @tst_sys, @ref_sys); # evaluation parameters
my (%tst_data, %ref_data); # the data -- with structure:  {system}{document}[segments]
my ($src_id, $ref_id, $tst_id); # unique identifiers for ref and tst translation sets
my %eval_docs;     # document information for the evaluation data set
my %ngram_info;    # the information obtained from (the last word in) the ngram

my $globalSegId = 0; # to track the segment id
print "Max_Ngram_size=",$max_Ngram,"\n";
print "Report_NIST_ngram_size=",$report_ngram_nist,"\n";
print "Report_Bleu_ngram_size=",$report_ngram_bleu,"\n";
######
# Get source document ID's
($src_id) = get_source_info ($src_file);

######
# Get reference translations
($ref_id) = get_MT_data (\%ref_data, "RefSet", $ref_file);

print STDERR "Computing n-gram info...\n";
compute_ngram_info ();
print STDERR "Done.\n";

######
# Get translations to evaluate
($tst_id) = get_MT_data (\%tst_data, "TstSet", $tst_file);

######
# Check data for completeness and correctness
check_MT_data ();

######
# Evaluate
print "RefNumber=",scalar keys %ref_data,"\n";
print "  Evaluation of $src_lang-to-$tgt_lang translation using:\n";
my $cum_seg = 0;
foreach my $doc (sort keys %eval_docs) {
    $cum_seg += @{$eval_docs{$doc}{SEGS}};
}
print "    src set \"$src_id\" (", scalar keys %eval_docs, " docs, $cum_seg segs)\n";
print "    ref set \"$ref_id\" (", scalar keys %ref_data, " refs)\n";
print "    tst set \"$tst_id\" (", scalar keys %tst_data, " systems)\n\n";

foreach my $sys (sort @tst_sys) {
    score_system ($sys);
}

exit 0;

#################################

sub get_source_info {

    my ($file) = @_;
    my ($name, $id, $src, $doc);
    my ($data, $tag, $span);
    

#read data from file
    open (FILE, $file) or die "\nUnable to open translation data file '$file'", $usage;
    $data .= $_ while <FILE>;
    close (FILE);

#get source set info
    die "\n\nFATAL INPUT ERROR:  no 'src_set' tag in src_file '$file'\n\n"
	unless ($tag, $span, $data) = extract_sgml_tag_and_span ("SrcSet", $data);

    die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
	unless ($id) = extract_sgml_tag_attribute ($name="SetID", $tag);

    die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
	unless ($src) = extract_sgml_tag_attribute ($name="SrcLang", $tag);
    die "\n\nFATAL INPUT ERROR:  $name ('$src') in file '$file' inconsistent\n"
	."                    with $name in previous input data ('$src_lang')\n\n"
	    unless (not defined $src_lang or $src eq $src_lang);
    $src_lang = $src;

#get doc info -- ID and # of segs
    $data = $span;
    while (($tag, $span, $data) = extract_sgml_tag_and_span ("Doc", $data)) {
	die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n"
	    unless ($doc) = extract_sgml_tag_attribute ($name="DocID", $tag);
	die "\n\nFATAL INPUT ERROR:  duplicate '$name' in file '$file'\n\n"
	    if defined $eval_docs{$doc};
	$span =~ s/[\s\n\r]+/ /g;  # concatenate records
	my $jseg=0, my $seg_data = $span;
	while (($tag, $span, $seg_data) = extract_sgml_tag_and_span ("Seg", $seg_data)) {
		if($normalizationMethod==1){
		    ($eval_docs{$doc}{SEGS}[$jseg++]) = NormalizeText_Bleu ($span);
		}
		else{
		    ($eval_docs{$doc}{SEGS}[$jseg++]) = NormalizeText ($span);
		}
	}
	die "\n\nFATAL INPUT ERROR:  no segments in document '$doc' in file '$file'\n\n"
	    if $jseg == 0;
    }
    die "\n\nFATAL INPUT ERROR:  no documents in file '$file'\n\n"
	unless keys %eval_docs > 0;
    return $id;
}

#################################

sub get_MT_data {

    my ($docs, $set_tag, $file) = @_;
    my ($name, $id, $src, $tgt, $sys, $doc);
    my ($tag, $span, $data);

#read data from file
    open (FILE, $file) or die "\nUnable to open translation data file '$file'", $usage;
    $data .= $_ while <FILE>;
    close (FILE);

#get tag info
    while (($tag, $span, $data) = extract_sgml_tag_and_span ($set_tag, $data)) {
	die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n" unless
	    ($id) = extract_sgml_tag_attribute ($name="SetID", $tag);

	die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n" unless
	    ($src) = extract_sgml_tag_attribute ($name="SrcLang", $tag);
	die "\n\nFATAL INPUT ERROR:  $name ('$src') in file '$file' inconsistent\n"
	    ."                    with $name of source ('$src_lang')\n\n"
		unless $src eq $src_lang;
	
	die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n" unless
	    ($tgt) = extract_sgml_tag_attribute ($name="TrgLang", $tag);
	die "\n\nFATAL INPUT ERROR:  $name ('$tgt') in file '$file' inconsistent\n"
	    ."                    with $name of the evaluation ('$tgt_lang')\n\n"
		unless (not defined $tgt_lang or $tgt eq $tgt_lang);
	$tgt_lang = $tgt;

	my $mtdata = $span;
	while (($tag, $span, $mtdata) = extract_sgml_tag_and_span ("Doc", $mtdata)) {
	    die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n" unless
		(my $sys) = extract_sgml_tag_attribute ($name="SysID", $tag);
	    
	    die "\n\nFATAL INPUT ERROR:  no tag attribute '$name' in file '$file'\n\n" unless
		$doc = extract_sgml_tag_attribute ($name="DocID", $tag);
	    
	    die "\n\nFATAL INPUT ERROR:  document '$doc' for system '$sys' in file '$file'\n"
		."                    previously loaded from file '$docs->{$sys}{$doc}{FILE}'\n\n"
		    unless (not defined $docs->{$sys}{$doc});

	    $span =~ s/[\s\n\r]+/ /g;  # concatenate records
	    my $jseg=0, my $seg_data = $span;
	    while (($tag, $span, $seg_data) = extract_sgml_tag_and_span ("Seg", $seg_data)) {
		if($normalizationMethod==1){
			($docs->{$sys}{$doc}{SEGS}[$jseg++]) = NormalizeText_Bleu ($span);
		}
		else{
			($docs->{$sys}{$doc}{SEGS}[$jseg++]) = NormalizeText ($span);
		}
	    }
	    die "\n\nFATAL INPUT ERROR:  no segments in document '$doc' in file '$file'\n\n"
		if $jseg == 0;
	    $docs->{$sys}{$doc}{FILE} = $file;
	}
    }
    return $id;
}

#################################

sub check_MT_data {

    @tst_sys = sort keys %tst_data;
    @ref_sys = sort keys %ref_data;

#every evaluation document must be represented for every system and every reference
    foreach my $doc (sort keys %eval_docs) {
	my $nseg_source = @{$eval_docs{$doc}{SEGS}};
	foreach my $sys (@tst_sys) {
	    die "\n\nFATAL ERROR:  no document '$doc' for system '$sys'\n\n"
		unless defined $tst_data{$sys}{$doc};
	    my $nseg = @{$tst_data{$sys}{$doc}{SEGS}};
	    die "\n\nFATAL ERROR:  translated documents must contain the same # of segments as the source, but\n"
		."              document '$doc' for system '$sys' contains $nseg segments, while\n"
                ."              the source document contains $nseg_source segments.\n\n"
		    unless $nseg == $nseg_source;
	}

	foreach my $sys (@ref_sys) {
	    die "\n\nFATAL ERROR:  no document '$doc' for reference '$sys'\n\n"
		unless defined $ref_data{$sys}{$doc};
	    my $nseg = @{$ref_data{$sys}{$doc}{SEGS}};
	    die "\n\nFATAL ERROR:  translated documents must contain the same # of segments as the source, but\n"
		."              document '$doc' for system '$sys' contains $nseg segments, while\n"
                ."              the source document contains $nseg_source segments.\n\n"
		    unless $nseg == $nseg_source;
	}
    }
}

#################################

sub compute_ngram_info {

    my ($ref, $doc, $seg);
    my (@wrds, $tot_wrds, %ngrams, $ngram, $mgram);
    my (%ngram_count, @tot_ngrams);

    foreach $ref (keys %ref_data) {
	foreach $doc (keys %{$ref_data{$ref}}) {
	    foreach $seg (@{$ref_data{$ref}{$doc}{SEGS}}) {
		@wrds = split /\s+/, $seg;
		$tot_wrds += @wrds;
		%ngrams = %{Words2Ngrams (@wrds)};
		foreach $ngram (keys %ngrams) {
		    $ngram_count{$ngram} += $ngrams{$ngram};
		}
	    }
	}
    }
    
    foreach $ngram (keys %ngram_count) {
	@wrds = split / /, $ngram;
	pop @wrds, $mgram = join " ", @wrds;
	$ngram_info{$ngram} = - log
	    ($mgram ? $ngram_count{$ngram}/$ngram_count{$mgram}
	            : $ngram_count{$ngram}/$tot_wrds) / log 2;
	if (defined $opt_x and $opt_x eq "ngram info") {
	    @wrds = split / /, $ngram;
	    printf "ngram info:%9.4f%6d%6d%8d%3d %s\n", $ngram_info{$ngram}, $ngram_count{$ngram},
	        $mgram ? $ngram_count{$mgram} : $tot_wrds, $tot_wrds, scalar @wrds, $ngram;
	}
    }
}

#################################

sub score_system {

    my ($sys, $ref, $doc);
    ($sys) = @_;
    my ($closest_ref_length, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info);
    my ($cum_ref_length, @cum_match, @cum_tst_cnt, @cum_ref_cnt, @cum_tst_info, @cum_ref_info);

    $cum_ref_length = 0;
    for (my $j=1; $j<=$max_Ngram; $j++) {
	$cum_match[$j] = $cum_tst_cnt[$j] = $cum_ref_cnt[$j] = $cum_tst_info[$j] = $cum_ref_info[$j] = 0;
    }
	
    foreach $doc (sort byDocIdNoCase keys %eval_docs) {
    	
	($closest_ref_length, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info) = score_document ($sys, $doc);

#output document summary score	    
	printf "$method score = %.4f for system \"$sys\" on document \"$doc\" (%d segments, %d words)\n",
            $method eq "BLEU" ?  bleu_score($closest_ref_length, $match_cnt, $tst_cnt) :
		nist_score (scalar @ref_sys, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info),
		scalar @{$tst_data{$sys}{$doc}{SEGS}}, $tst_cnt->[1]
		    if $detail >= 1;

	$cum_ref_length += $closest_ref_length;
	for (my $j=1; $j<=$max_Ngram; $j++) {
	    $cum_match[$j] += $match_cnt->[$j];
	    $cum_tst_cnt[$j] += $tst_cnt->[$j];
	    $cum_ref_cnt[$j] += $ref_cnt->[$j];
	    $cum_tst_info[$j] += $tst_info->[$j];
	    $cum_ref_info[$j] += $ref_info->[$j];
	    printf "document info: $sys $doc %d-gram %d %d %d %9.4f %9.4f\n", $j, $match_cnt->[$j],
	        $tst_cnt->[$j], $ref_cnt->[$j], $tst_info->[$j], $ref_info->[$j]
		    if (defined $opt_x and $opt_x eq "document info");
	}
    }

#output system summary score
	print "NIST metric:\n";
	my $nistScoreValue = nist_score (scalar @ref_sys, \@cum_match, \@cum_tst_cnt, \@cum_ref_cnt, \@cum_tst_info, \@cum_ref_info);
    	printf "Final NIST score = %.4f for system \"$sys\"\n", $nistScoreValue;
    	
    	print "\nBLEU metric:\n";
	my $bleuScoreValue = bleu_score($cum_ref_length, \@cum_match, \@cum_tst_cnt);
    	printf "Final BLEU score = %.4f for system \"$sys\"\n", $bleuScoreValue;
    	   
	    
}

#################################

sub score_document {

    my ($sys, $ref, $doc);
    ($sys, $doc) = @_;
    my ($closest_ref_length, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info);
    my ($cum_ref_length, @cum_match, @cum_tst_cnt, @cum_ref_cnt, @cum_tst_info, @cum_ref_info);

    $cum_ref_length = 0;
    for (my $j=1; $j<=$max_Ngram; $j++) {
	$cum_match[$j] = $cum_tst_cnt[$j] = $cum_ref_cnt[$j] = $cum_tst_info[$j] = $cum_ref_info[$j] = 0;
    }
	
#score each segment
    for (my $jseg=0; $jseg<@{$tst_data{$sys}{$doc}{SEGS}}; $jseg++) {
	my @ref_segments = ();
	foreach $ref (@ref_sys) {
	    push @ref_segments, $ref_data{$ref}{$doc}{SEGS}[$jseg];
	    printf "ref '$ref', seg %d: %s\n", $jseg+1, $ref_data{$ref}{$doc}{SEGS}[$jseg]
		if $detail >= 3;
	}
	printf "sys '$sys', seg %d: %s\n", $jseg+1, $tst_data{$sys}{$doc}{SEGS}[$jseg]
	    if $detail >= 3;
	    
	#added by Joy, joy@cs.cmu.edu
	#09/21/2003
	#for bootstrapping
	$globalSegId++;
	print "GlobalSegId=$globalSegId\n";
	print "DocId=$doc\nSegId=",$jseg+1,"\n";
	($closest_ref_length, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info) =
	    score_segment ($tst_data{$sys}{$doc}{SEGS}[$jseg], @ref_segments);

#output segment summary score	    
	printf "$method score = %.4f for system \"$sys\" on segment %d of document \"$doc\" (%d words)\n",
            $method eq "BLEU" ?  bleu_score($closest_ref_length, $match_cnt, $tst_cnt) :
		nist_score (scalar @ref_sys, $match_cnt, $tst_cnt, $ref_cnt, $tst_info, $ref_info),
		$jseg+1, $tst_cnt->[1]
		    if $detail >= 2;

	$cum_ref_length += $closest_ref_length;
	for (my $j=1; $j<=$max_Ngram; $j++) {
	    $cum_match[$j] += $match_cnt->[$j];
	    $cum_tst_cnt[$j] += $tst_cnt->[$j];
	    $cum_ref_cnt[$j] += $ref_cnt->[$j];
	    $cum_tst_info[$j] += $tst_info->[$j];
	    $cum_ref_info[$j] += $ref_info->[$j];
	}
    }
    return ($cum_ref_length, [@cum_match], [@cum_tst_cnt], [@cum_ref_cnt], [@cum_tst_info], [@cum_ref_info]);
}

#################################

sub score_segment {

    my ($tst_seg, @ref_segs) = @_;
    my (@tst_wrds, %tst_ngrams, @match_count, @tst_count, @tst_info);
    my (@ref_wrds, $ref_seg, %ref_ngrams, %ref_ngrams_max, @ref_count, @ref_info);
    my ($ngram);
    my (@nwrds_ref);
    

    for (my $j=1; $j<= $max_Ngram; $j++) {
	$match_count[$j] = $tst_count[$j] = $ref_count[$j] = $tst_info[$j] = $ref_info[$j] = 0;
    }

# get the ngram counts for the test segment
    @tst_wrds = split /\s+/, $tst_seg;
    %tst_ngrams = %{Words2Ngrams (@tst_wrds)};
    for (my $j=1; $j<=$max_Ngram; $j++) { # compute ngram counts
	$tst_count[$j]  = $j<=@tst_wrds ? (@tst_wrds - $j + 1) : 0;
    }

# get the ngram counts for the reference segments

    #output the features for this sentence, which will be used later for NIST/Bleu in the bootstrapping
    #added by Joy,joy@cs.cmu.edu
    #09/20/2003
    print "RefLen:";
    my $minDiffW=1000000000;
    my $closestRefW = 0;
    
    foreach $ref_seg (@ref_segs) {
	@ref_wrds = split /\s+/, $ref_seg;
	%ref_ngrams = %{Words2Ngrams (@ref_wrds)};
	foreach $ngram (keys %ref_ngrams) { # find the maximum # of occurrences
	    my @wrds = split / /, $ngram;
	    $ref_info[@wrds] += $ngram_info{$ngram};
	    $ref_ngrams_max{$ngram} = defined $ref_ngrams_max{$ngram} ?
		max ($ref_ngrams_max{$ngram}, $ref_ngrams{$ngram}) :
		    $ref_ngrams{$ngram};
	}
	for (my $j=1; $j<=$max_Ngram; $j++) { # update ngram counts
	    $ref_count[$j] += $j<=@ref_wrds ? (@ref_wrds - $j + 1) : 0;
	}
	
	#output the features for this sentence, which will be used later for NIST/Bleu in the bootstrapping
	#added by Joy,joy@cs.cmu.edu
	#09/20/2003
	print " ",scalar @ref_wrds;
	
	#added by Joy, joy@cs.cmu.edu
	#09/21/2003
	#to calculate the real IBM-Bleu metric, we need the closest ref_length, instead of the "shortest ref len"
	my $this_ref_len = scalar @ref_wrds;
	my $this_tst_len = $tst_count[1];
	
	if(abs($this_tst_len - $this_ref_len) <= $minDiffW){
		if(abs($this_tst_len - $this_ref_len) == $minDiffW){
			$closestRefW = $this_ref_len if($closestRefW > $this_ref_len);
		}
		else{
			$closestRefW = $this_ref_len;
		}
		
		$minDiffW = abs($this_tst_len - $this_ref_len);
	}
	
	
    }
    #added by Joy,joy@cs.cmu.edu
    #09/20/2003
    print "\n";

    # accumulate scoring stats for tst_seg ngrams that match ref_seg ngrams
    foreach $ngram (keys %tst_ngrams) {
	next unless defined $ref_ngrams_max{$ngram};
	my @wrds = split / /, $ngram;
	$tst_info[@wrds] += $ngram_info{$ngram} * min($tst_ngrams{$ngram},$ref_ngrams_max{$ngram});
	$match_count[@wrds] += my $count = min($tst_ngrams{$ngram},$ref_ngrams_max{$ngram});
	printf "%.2f info for each of $count %d-grams = '%s'\n", $ngram_info{$ngram}, scalar @wrds, $ngram
	    if $detail >= 3;
    }
    
    #output the features for this sentence, which will be used later for NIST/Bleu in the bootstrapping    
    print "ClosestRefLen $closestRefW\n";
    for (my $j=1; $j<=$max_Ngram; $j++){
    	printf "%d-gram: %d %d %.2f\n", $j,  $tst_count[$j],$match_count[$j],$tst_info[$j];
    }    
   
    print "\n";
    
    return ($closestRefW, [@match_count], [@tst_count], [@ref_count], [@tst_info], [@ref_info]);
}

#################################

sub bleu_score {

    my ($closest_ref_length, $matching_ngrams, $tst_ngrams) = @_;

    my $beta = 0.01;
    my $score = 0;
    for (my $j=1; $j<=$report_ngram_bleu; $j++) {
    	printf "$j nGram: %d / %d = %.3f\n",$matching_ngrams->[$j], $tst_ngrams->[$j], $matching_ngrams->[$j]/$tst_ngrams->[$j];
	return 0 if $matching_ngrams->[$j] == 0;
	$score += log ($matching_ngrams->[$j]/$tst_ngrams->[$j]);
    }
    my $len_score = $tst_ngrams->[1] >= $closest_ref_length ?
	0 : 1 - $closest_ref_length/$tst_ngrams->[1];
   print "ClosestRefLen=",$closest_ref_length,"\n";
   print "HypLen=",$tst_ngrams->[1],"\n";
   printf "LenPen=%.4f\n",exp($len_score);
    return exp($score/$report_ngram_bleu + $len_score);
}

#################################

sub nist_score {

    my ($nsys, $matching_ngrams, $tst_ngrams, $ref_ngrams, $tst_info, $ref_info) = @_;

    my $score = 0;
    my $lenPen = 0.0;
    for (my $n=1; $n<=$report_ngram_nist; $n++) {
	$score += $tst_info->[$n]/max($tst_ngrams->[$n],1);
	printf "%d_gramScore=%.4f %d_gramInfoSum=%.4f out of %d %d_grams\n",
		$n,
		$tst_info->[$n]/max($tst_ngrams->[$n],1),
		$n,
		$tst_info->[$n],
		max($tst_ngrams->[$n],1),
		$n;		
    }
    printf "PrecScore= %.4f\n", $score;
    print "SysLen=",$tst_ngrams->[1],"\n";
    print "RefLen=",$ref_ngrams->[1]/$nsys,"\n";
    $lenPen = nist_length_penalty($tst_ngrams->[1]/($ref_ngrams->[1]/$nsys));
    $score *= $lenPen;
    printf "LengthPenalty= %.4f\n", $lenPen;

    return $score;
}

#################################

sub Words2Ngrams { #convert a string of words to an Ngram count hash

    my %count = ();

    for (; @_; shift) {
	my ($j, $ngram, $word);
	for ($j=0; $j<$max_Ngram and defined($word=$_[$j]); $j++) {
	    $ngram .= defined $ngram ? " $word" : $word;
	    $count{$ngram}++;
	}
    }
    return {%count};
}

#################################
# updated with mteval-v11b.pl's NormalizeText() method
sub NormalizeText {
    my ($norm_text) = @_;
 
# language-independent part:
    $norm_text =~ s/<skipped>//g; # strip "skipped" tags
    $norm_text =~ s/-\n//g; # strip end-of-line hyphenation and join lines
    $norm_text =~ s/\n/ /g; # join lines
    $norm_text =~ s/&quot;/"/g;  # convert SGML tag for quote to "
    $norm_text =~ s/&amp;/&/g;   # convert SGML tag for ampersand to &
    $norm_text =~ s/&lt;/</g;    # convert SGML tag for less-than to >
    $norm_text =~ s/&gt;/>/g;    # convert SGML tag for greater-than to <
 
# language-dependent part (assuming Western languages):
    $norm_text = " $norm_text ";
    $norm_text =~ tr/[A-Z]/[a-z]/ unless $preserve_case;
    $norm_text =~ s/([\{-\~\[-\` -\&\(-\+\:-\@\/])/ $1 /g;   # tokenize punctuation
    $norm_text =~ s/([^0-9])([\.,])/$1 $2 /g; # tokenize period and comma unless preceded by a digit
    $norm_text =~ s/([\.,])([^0-9])/ $1 $2/g; # tokenize period and comma unless followed by a digit
    $norm_text =~ s/([0-9])(-)/$1 $2 /g; # tokenize dash when preceded by a digit
    $norm_text =~ s/\s+/ /g; # one space only between words
    $norm_text =~ s/^\s+//;  # no leading space
    $norm_text =~ s/\s+$//;  # no trailing space
 
    return $norm_text;
}


sub NormalizeText_v11 {
    my ($norm_text) = @_;
 
# language-independent part:
    $norm_text =~ s/<skipped>//g; # strip "skipped" tags
    $norm_text =~ s/-\n//g; # strip end-of-line hyphenation and join lines
    $norm_text =~ s/\n/ /g; # join lines
    $norm_text =~ s/(\d)\s+(?=\d)/$1/g; #join digits
    $norm_text =~ s/&quot;/"/g;  # convert SGML tag for quote to "
    $norm_text =~ s/&amp;/&/g;   # convert SGML tag for ampersand to &
    $norm_text =~ s/&lt;/</g;    # convert SGML tag for less-than to >
    $norm_text =~ s/&gt;/>/g;    # convert SGML tag for greater-than to <
 
# language-dependent part (assuming Western languages):
    $norm_text = " $norm_text ";
    $norm_text =~ tr/[A-Z]/[a-z]/ unless $preserve_case;
    $norm_text =~ s/([\{-\~\[-\` -\&\(-\+\:-\@\/])/ $1 /g;   # tokenize punctuation
    $norm_text =~ s/([^0-9])([\.,])/$1 $2 /g; # tokenize period and comma unless preceded by a digit
    $norm_text =~ s/([\.,])([^0-9])/ $1 $2/g; # tokenize period and comma unless followed by a digit
    $norm_text =~ s/([0-9])(-)/$1 $2 /g; # tokenize dash when preceded by a digit
    $norm_text =~ s/\s+/ /g; # one space only between words
    $norm_text =~ s/^\s+//;  # no leading space
    $norm_text =~ s/\s+$//;  # no trailing space
 
    return $norm_text;
}

# Some simple processing of the translations. Lowercasing is the main aspect.
# (Are we being too liberal by comparing translations after lowercasing?)
# Numbers in numeric form are rendered differently by some commercial systems, 
# so normalize the spacing conventions in numbers.
# There is no end to text normalization. For example, "1" and "one" are the same,
# aren't they? Before going ballistic, let us recall the "Keep It Simple" Principle.
sub NormalizeText_Bleu {
    my ($strPtr) = @_;

# language-independent part:
    $strPtr =~ s/^\s+//;
    $strPtr =~ s/\n/ /g; # join lines
    $strPtr =~ s/(\d)\s+(\d)/$1$2/g;  #join digits

# language-dependent part (assuming Western languages):
    $strPtr =~ tr/[A-Z]/[a-z]/ unless $preserve_case;
    $strPtr =~ s/([^A-Za-z0-9\-\'\.,])/ $1 /g; # tokenize punctuation (except for alphanumerics, "-", "'", ".", ",")
    $strPtr =~ s/([^0-9])([\.,])/$1 $2 /g; # tokenize period and comma unless preceded by a digit
    $strPtr =~ s/([\.,])([^0-9])/ $1 $2/g; # tokenize period and comma unless followed by a digit
    $strPtr =~ s/([0-9])(-)/$1 $2 /g; # tokenize dash when preceded by a digit
    $strPtr =~ s/\s+/ /g; # one space only between words
    $strPtr =~ s/^\s+//;  # no leading space
    $strPtr =~ s/\s+$//;  # no trailing space
    my $ascii = "\x20-\x7F";
    $strPtr =~ s/([^$ascii])\s+([^$ascii])/$1$2/g; # combine sequences of non-ASCII characters into single words

    
    return $strPtr;
}



#################################

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

sub date_time_stamp {

    my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime();
    my @months = qw(Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec);
    my ($date, $time);

    $time = sprintf "%2.2d:%2.2d:%2.2d", $hour, $min, $sec;
    $date = sprintf "%4.4s %3.3s %s", 1900+$year, $months[$mon], $mday;
    return ($date, $time);
}

#################################

sub extract_sgml_tag_and_span {
    
    my ($name, $data) = @_;
    
    ($data =~ m|<$name\s*([^>]*)>(.*?)</$name\s*>(.*)|si) ? ($1, $2, $3) : ();
}

#################################

sub extract_sgml_tag_attribute {

    my ($name, $data) = @_;

    ($data =~ m|$name\s*=\s*\"([^\"]*)\"|si) ? ($1) : ();
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

#################################

sub byDocIdNoCase {
	my($value1, $value2);
	
	$value1 = $a;
	$value2 = $b;
	
	$value1=~tr/a-z/A-Z/;
	$value2=~tr/a-z/A-Z/;
	
	return $value1 cmp $value2;
}
