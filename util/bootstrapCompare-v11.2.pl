#!/usr/bin/env perl
#
# This program reads two log file from "generateLog.pl" and compute the NIST/Bleu scores
# using bootstapping, then compare their differences
#
# By Joy, joy@cs.cmu.edu
#
# Oct 24, 2006
# 	Bug fix, reported by "ichael Paul" <michael.paul@atr.jp>
#	after replacing the depricated allInfo[refNum+1]=shortest_len with allInfo[refNum+1]=closest_len
#	the function reading from the log file was not updated accordingly and still uses
#	"$allInfo1[$segId][$refNum+2]=$closestLen;"
#	This can effect the outcome of the comparison method, i.e., the output
#	of an MT system with relatively short sentence length might be (wrongly)
#	judged significantly better than another system output with longer
#	translations, although the mean of the (shorter-output) MT system scores
#	is worse than the mean of the scores of the (longer-output) system.
#
#
# Apr 27, 2005 [v11.2]
#	Allow to specify ConfidenceLevel, default = 0.95 (95%)
#
# Jan 21, 2005 Bug fix, when creat sampledInfo, index should start from 1
#
# Oct 26, 2004
#	One-tail t test used for system comparison instead of the two-tail test used before
#
# Oct 25, 2004
#	Updated percentile calculation based on http://www.itl.nist.gov/div898/handbook/prc/section2/prc252.htm
#
# Oct 25, 2004
#	Fixed the bug reported by David Chiang(dchiang@umiacs.umd.edu) regarding the 
#	sort function 
# Oct 18, 2004: corrected the confusion of n-gram_size and report_n-gram_size
# 
# Sep 21, 2003
#
# Usage:
#	perl5 bootstrapSingle-v11.pl logfile1 logfile2
#
#

use lib "/afs/cs/user/joy/sharedLib";
use Statistics::Distributions;

#check parameter
if($#ARGV<1){
	print STDERR "\n-----------------------------------";
	print STDERR "\nUsage:";
	print STDERR "\n\tperl5 bootstrapCompare-v11.pl logfile1 logfile2 [ResampleTimes] [confidence level]";
	print STDERR "\n-----------------------------------\n";
	
	exit;
}



#adjustable parameter
my $B = 1000;
if($#ARGV>=2){
	$B = $ARGV[2];
}

my $confidenceLevel = 0.95;
if($#ARGV>=3){
	$confidenceLevel = $ARGV[3];
}


#global variables
@allInfo1=(); 
	#allInfo[0]=docId__segId
	#allInfo[1]=ref0_len....
	#allInfo[refNum]=ref_n_len
	#depricated #allInfo[refNum+1]=shortest_len
	#allInfo[refNum+1]=closest_len
	#allInfo[refNum+2+(n-1)*3]=n-gram count in test
	#allInfo[refNum+2+(n-1)*3+1]=n-gram match in test
	#allInfo[refNum+2+(n-1)*3+2]=n-gram info gain in test
@allInfo2=(); 


$maxNgramSize = 5;
$refNum = 4;
$segSize=0;


$report_NIST_ngram_size=5;
$report_Bleu_ngram_size=4;

open Log1,$ARGV[0];
open Log2,$ARGV[1];

print STDERR "Reading log file 1 ...\n";
while(<Log1>){
	#NIST_ngram_size=5
	#Bleu_ngram_size=4

	if(/^Max_Ngram_size=([0-9]+)/){
		$maxNgramSize = $1;
		
		$_=<Log1>;
		/^Report_NIST_ngram_size=([0-9]+)/;
		$report_NIST_ngram_size = $1;
		
		$_=<Log1>;
		/^Report_Bleu_ngram_size=([0-9]+)/;
		$report_Bleu_ngram_size = $1;
		
		
		print STDERR "Max_N=$maxNgramSize\n";
		print STDERR "Report_NIST_ngram_size=$report_NIST_ngram_size Report_Bleu_ngram_size=$report_Bleu_ngram_size\n";
	}
	
	if(/^RefNumber=([0-9]+)/){
		$refNum=$1;
		print STDERR "RefNumber=$refNum\n";
	}
	
	if(/^GlobalSegId=([0-9]+)/){	#information for one segment
		$segId=scalar $1;
		#GlobalSegId=79
		#DocId=chtb_076
		#SegId=6
		#RefLen: 28 25 28 22		
		#ClosestRefLen 22
		#1-gram: 21 15 115.94
		#2-gram: 20 6 29.02
		#3-gram: 19 4 8.15
		#4-gram: 18 2 4.46
		#5-gram: 17 1 0.00

		#store information
		$_=<Log1>;	#DocId=chtb_076
		s/^DocId=//;
		s/\x0A+//;
		s/\x0D+//;
		my $docId=$_;
		
		$_=<Log1>;	#SegId=6
		s/^SegId=//;
		s/\x0A+//;
		s/\x0D+//;
		my $segIdInDoc=$_;
		
		$allInfo1[$segId][0]=$docId."__".$segIdInDoc;
		
		#print "$segId: $docId--$segIdInDoc\n";
		#print $allInfo1[$segId][0]," in $segId\n";
		
		
		$_=<Log1>;	#RefLen: 28 25 28 22
		s/\x0A+//;
		s/\x0D+//;
		s/^RefLen:\x20+//;
		s/\x20+\Z//;
		my @refLens=split(/\x20/,$_);
		
		for(my $i=0;$i<$refNum;$i++){
			$allInfo1[$segId][$i+1]=$refLens[$i];
		}
		
		
		
		$_=<Log1>;	#ClosestRefLen 24
		s/^ClosestRefLen\x20+//;
		s/\x0A+//;
		s/\x0D+//;
		my $closestLen=$_;
		$allInfo1[$segId][$refNum+1]=$closestLen;
		
		for(my $i=1;$i<=$maxNgramSize;$i++){
			$_=<Log1>;	#1-gram: 21 15 115.94
			s/^[0-9]+-gram:\x20+//;
			s/\x0A+//;
			s/\x0D+//;
			my @values = split(/\x20/,$_);
			
			#allInfo1[refNum+2+(n-1)*3]=n-gram count in test
			$allInfo1[$segId][$refNum+2+($i-1)*3]=$values[0];
			#print "SegId=$segId I=$i Index=",$refNum+2+($i-1)*3,"\n";
			
			#allInfo1[refNum+2+(n-1)*3+1]=n-gram match in test
			$allInfo1[$segId][$refNum+2+($i-1)*3+1]=$values[1];
			#print "SegId=$segId I=$i Index=",$refNum+2+($i-1)*3+1,"\n";
			
			#allInfo1[refNum+2+(n-1)*3+2]=n-gram info gain in test
			$allInfo1[$segId][$refNum+2+($i-1)*3+2]=$values[2];
			#print "SegId=$segId I=$i Index=",$refNum+2+($i-1)*3+2,"\n";
		}	
		
	}	
}

print STDERR "Reading log file 2 ...\n";
while(<Log2>){
	#NIST_ngram_size=5
	#Bleu_ngram_size=4

	if(/^Max_Ngram_size=([0-9]+)/){
		if($maxNgramSize != $1){
			print STDERR "Max ngram size are different, quit!\n";
			exit;
		}
		
		$_=<Log2>;
		/^Report_NIST_ngram_size=([0-9]+)/;
		if($report_NIST_ngram_size!=$1){
			print STDERR "NIST ngram size are different, quit!\n";
			exit;
		}
		
		$_=<Log2>;
		/^Report_Bleu_ngram_size=([0-9]+)/;		
		if($report_Bleu_ngram_size != $1){
			print STDERR "Bleu ngram size are different, quit!\n";
			exit;
		}
		
		
		
				
		
	}
	
	if(/^RefNumber=([0-9]+)/){
		if($refNum!=$1){
			print STDERR "RefNumberare different, quit!\n";
			exit;
		}
	}
	
	if(/^GlobalSegId=([0-9]+)/){	#information for one segment
		$segId=scalar $1;
		#GlobalSegId=79
		#DocId=chtb_076
		#SegId=6
		#RefLen: 28 25 28 22		
		#ClosestRefLen 22
		#1-gram: 21 15 115.94
		#2-gram: 20 6 29.02
		#3-gram: 19 4 8.15
		#4-gram: 18 2 4.46
		#5-gram: 17 1 0.00

		#store information
		$_=<Log2>;	#DocId=chtb_076
		s/^DocId=//;
		s/\x0A+//;
		s/\x0D+//;
		my $docId=$_;
		
		$_=<Log2>;	#SegId=6
		s/^SegId=//;
		s/\x0A+//;
		s/\x0D+//;
		my $segIdInDoc=$_;
		
		$allInfo2[$segId][0]=$docId."__".$segIdInDoc;
		
		if($allInfo1[$segId][0] ne $allInfo2[$segId][0]){
			print STDERR "$segId segment are not the same.\n";
			print STDERR "From Log1 ",$allInfo1[$segId][0],"\n";
			print STDERR "From Log2 ",$allInfo2[$segId][0],"\n";
			
			exit;			
		}
		
		#print "$segId: $docId--$segIdInDoc\n";
		#print $allInfo1[$segId][0]," in $segId\n";
		
		
		$_=<Log2>;	#RefLen: 28 25 28 22
		s/\x0A+//;
		s/\x0D+//;
		s/^RefLen:\x20+//;
		s/\x20+\Z//;
		my @refLens=split(/\x20/,$_);
		
		for(my $i=0;$i<$refNum;$i++){
			$allInfo2[$segId][$i+1]=$refLens[$i];
		}
		
		
		$_=<Log2>;	#ClosestRefLen 24
		s/^ClosestRefLen\x20+//;
		s/\x0A+//;
		s/\x0D+//;
		my $closestLen=$_;
		$allInfo2[$segId][$refNum+1]=$closestLen;
		
		for(my $i=1;$i<=$maxNgramSize;$i++){
			$_=<Log2>;	#1-gram: 21 15 115.94
			s/^[0-9]+-gram:\x20+//;
			s/\x0A+//;
			s/\x0D+//;
			my @values = split(/\x20/,$_);
			
			#allInfo2[refNum+2+(n-1)*3]=n-gram count in test
			$allInfo2[$segId][$refNum+2+($i-1)*3]=$values[0];
			#print "SegId=$segId I=$i Index=",$refNum+3+($i-1)*3,"\n";
			
			#allInfo2[refNum+2+(n-1)*3+1]=n-gram match in test
			$allInfo2[$segId][$refNum+2+($i-1)*3+1]=$values[1];
			#print "SegId=$segId I=$i Index=",$refNum+3+($i-1)*3+1,"\n";
			
			#allInfo2[refNum+2+(n-1)*3+2]=n-gram info gain in test
			$allInfo2[$segId][$refNum+2+($i-1)*3+2]=$values[2];
			#print "SegId=$segId I=$i Index=",$refNum+3+($i-1)*3+2,"\n";
		}	
		
	}	
}



$segSize = $segId;
print  "Total $segSize segments read.\n";

my @nist_list1=();
my @bleu_list1=();
my @m_bleu_list1=();

my @nist_list2=();
my @bleu_list2=();
my @m_bleu_list2=();

my @nist_list_diff=();
my @bleu_list_diff=();
my @m_bleu_list_diff=();

print STDERR "Create $B Samples\n";
printf STDERR "Confidence Level at: %.2f\%\n",$confidenceLevel*100;
for(my $iteration=0;$iteration<$B;$iteration++){
	#print STDERR "Sampling for $iteration trial...\n";
	
	#sampling
	@sampledInfo1 = ();
	@sampledInfo2 = ();
	for(my $i=1;$i<=$segSize;$i++){	#generate $segSize random numbers
		my $randomNumber = rand();
		#print $randomNumber,"\n";
		$randomNumber=int($randomNumber*$segSize+1);	#should range between (1~segSize)
		#print $randomNumber,"\n";
		$sampledInfo1[$i]=$allInfo1[$randomNumber];
		$sampledInfo2[$i]=$allInfo2[$randomNumber];
	}
	
	
	#calculate the NIST/Bleu for this sample
	
	#accumulate info
	#allInfo[0]=docId__segId
	#allInfo[1]=ref0_len....
	#allInfo[refNum]=ref_n_len
	#depricated #allInfo[refNum+1]=shortest_len
	#allInfo[refNum+1]=closest_len
	#allInfo[refNum+2+(n-1)*3]=n-gram count in test
	#allInfo[refNum+2+(n-1)*3+1]=n-gram match in test
	#allInfo[refNum+2+(n-1)*3+2]=n-gram info gain in test

	
	
	
	
	
	#------------------------------------	
	#for lenPen
	$sys_len1=0;
	$ref_len_all1=0;
	
	$sys_len2=0;
	$ref_len_all2=0;
	
	$closet_ref_len1=0;
	$closet_ref_len2=0;
	
	for(my $i=1;$i<=$segSize;$i++){
		for(my $j=1;$j<=$refNum;$j++){
			$ref_len_all1 += $sampledInfo1[$i][$j];
			$ref_len_all2 += $sampledInfo2[$i][$j];
		}
		
		$sys_len1 += $sampledInfo1[$i][$refNum+2];	#unigram count
		$sys_len2 += $sampledInfo2[$i][$refNum+2];	#unigram count
		
		$closet_ref_len1 += $sampledInfo1[$i][$refNum+1]; #closest ref len
		$closet_ref_len2 += $sampledInfo2[$i][$refNum+1]; #closest ref len
	}
	#print "SysLen=$sys_len refLen=",$ref_len_all/$refNum,"\n";
	my $nist_lenPen1 = nist_length_penalty($sys_len1/($ref_len_all1/$refNum) );
	my $nist_lenPen2 = nist_length_penalty($sys_len2/($ref_len_all2/$refNum) );
	
	my $bleu_lenPen1 = bleu_length_penalty($closet_ref_len1 / $sys_len1);
	my $bleu_lenPen2 = bleu_length_penalty($closet_ref_len2 / $sys_len2);
	
	
	
	#-------------------------------------
	#for NIST
	@ngramInfoGain1=();
	@ngramCountInSys1=();	
	
	@ngramInfoGain2=();
	@ngramCountInSys2=();
	
	for(my $i=1;$i<=$segSize;$i++){			
		for(my $k=1;$k<=$report_NIST_ngram_size;$k++){
			$ngramCountInSys1[$k]+=$sampledInfo1[$i][$refNum+2+($k-1)*3];
			$ngramMatched1[$k]+=$sampledInfo1[$i][$refNum+2+($k-1)*3+1];
			$ngramInfoGain1[$k]+=$sampledInfo1[$i][$refNum+2+($k-1)*3+2];
			
			$ngramCountInSys2[$k]+=$sampledInfo2[$i][$refNum+2+($k-1)*3];
			$ngramMatched2[$k]+=$sampledInfo2[$i][$refNum+2+($k-1)*3+1];
			$ngramInfoGain2[$k]+=$sampledInfo2[$i][$refNum+2+($k-1)*3+2];
		}
	}	
	
	
	#calculate NIST score
	my $nistScore1=0.0;
	my $nistScore2=0.0;
	
	for(my $k=1;$k<=$report_NIST_ngram_size;$k++){
		$nistScore1+=$ngramInfoGain1[$k]/$ngramCountInSys1[$k];
		$nistScore2+=$ngramInfoGain2[$k]/$ngramCountInSys2[$k];
	}
	
	$nistScore1*=$nist_lenPen1;
	$nistScore2*=$nist_lenPen2;
	
	#print "NIST=$nistScore\n";
	$nist_list1[$iteration]=$nistScore1;
	$nist_list2[$iteration]=$nistScore2;
		
	$nist_list_diff[$iteration] = $nistScore1 - $nistScore2;
	
	#-----------------------------
	# for Bleu/M-Bleu
	#calculate Bleu score
	my $bleu_score1 = 0.0; 
	my $nullify1 = 0;
	my $bleu_score2 = 0.0; 
	my $nullify2 = 0;
	
	@ngramMatched1=();		
	@ngramCountInSys1=();
	@ngramPrec1=();	
	@ngramMatched2=();	
	@ngramCountInSys2=();
	@ngramPrec2=();
	
	for(my $i=1;$i<=$segSize;$i++){			
		for(my $k=1;$k<=$report_Bleu_ngram_size;$k++){
			$ngramCountInSys1[$k]+=$sampledInfo1[$i][$refNum+2+($k-1)*3];
			$ngramMatched1[$k]+=$sampledInfo1[$i][$refNum+2+($k-1)*3+1];			
			
			$ngramCountInSys2[$k]+=$sampledInfo2[$i][$refNum+2+($k-1)*3];
			$ngramMatched2[$k]+=$sampledInfo2[$i][$refNum+2+($k-1)*3+1];			
		}
	}	
	
	for(my $k=1;$k<=$report_Bleu_ngram_size;$k++){
		#print "$k-gram info=",$ngramInfoGain[$k]," outof ",$ngramCountInSys[$k],"\n";
		#print "Precision=",$ngramMatched[$k]/$ngramCountInSys[$k],"\n";
		$ngramPrec1[$k]=$ngramMatched1[$k]/$ngramCountInSys1[$k];
		$ngramPrec2[$k]=$ngramMatched2[$k]/$ngramCountInSys2[$k];
	}

	
	
	$nullify1 = 1 if @ngramPrec1 <= $report_Bleu_ngram_size;	
	$nullify2 = 1 if @ngramPrec2 <= $report_Bleu_ngram_size;

	for(my $k=1;$k<=$report_Bleu_ngram_size;$k++){
		if ($ngramPrec1[$k]) {
	    		$bleu_score1 += 1/$report_Bleu_ngram_size * log($ngramPrec1[$k]);
		}
		else {
	    		$nullify1 = 1;
	    	}
	    	
	    	if ($ngramPrec2[$k]) {
	    		$bleu_score2 += 1/$report_Bleu_ngram_size * log($ngramPrec2[$k]);
		}
		else {
	    		$nullify2 = 1;
	    	}
	}
	
	if ($nullify1) {
		$bleu_score1 = 0;
    	}
    	else {
		$bleu_score1 = exp($bleu_score1);
	}
	
	if ($nullify2) {
		$bleu_score2 = 0;
    	}
    	else {
		$bleu_score2 = exp($bleu_score2);
	}
	
	$bleu_score1*=$bleu_lenPen1;   
	$bleu_score2*=$bleu_lenPen2;   
	
	$bleu_list1[$iteration]=$bleu_score1;
	$bleu_list2[$iteration]=$bleu_score2;
	$bleu_list_diff[$iteration]=$bleu_score1-$bleu_score2;
	   

   	#calculate Modified Bleu score
   	my $modified_bleu1 = 0.0;
   	my $modified_bleu2 = 0.0;
	for(my $k=1;$k<=$report_Bleu_ngram_size;$k++){
		$modified_bleu1 += 1/$report_Bleu_ngram_size * $ngramPrec1[$k];		
		$modified_bleu2 += 1/$report_Bleu_ngram_size * $ngramPrec2[$k];		
	}
	
	$modified_bleu1*=$bleu_lenPen1; 
	$modified_bleu2*=$bleu_lenPen2; 
	
	
	#print "Modified Bleu=$modified_bleu\n";
	$m_bleu_list1[$iteration]=$modified_bleu1;
	$m_bleu_list2[$iteration]=$modified_bleu2;
		
	$m_bleu_list_diff[$iteration]=$modified_bleu1 - $modified_bleu2;
	
}

print "\n\nNIST Value:\n";
print "-----------------\n";
printf "Sys1\n";
findConfidenceInterval(\@nist_list1);
print "---\n";
printf "Sys2\n";
findConfidenceInterval(\@nist_list2);
print "---\n";
print "Diff(Sys1-Sys2):";
findConfidenceInterval(\@nist_list_diff);
print "---\n";
print "Paired t test for two systems:\n";
calcPairedTvalue(\@nist_list1,\@nist_list2);



print "\n\nBleu Value:\n";
print "-----------------\n";
printf "Sys1\n";
findConfidenceInterval(\@bleu_list1);
print "---\n";
printf "Sys2\n";
findConfidenceInterval(\@bleu_list2);
print "---\n";
print "Diff(Sys1-Sys2):";
findConfidenceInterval(\@bleu_list_diff);
print "---\n";
print "Paired t test for two systems:\n";
calcPairedTvalue(\@bleu_list1,\@bleu_list2);


print "\n\nModified Bleu Value:\n";
print "-----------------\n";
printf "Sys1\n";
findConfidenceInterval(\@m_bleu_list1);
print "---\n";
printf "Sys2\n";
findConfidenceInterval(\@m_bleu_list2);
print "---\n";
print "Diff(Sys1-Sys2):";
findConfidenceInterval(\@m_bleu_list_diff);
print "---\n";
print "Paired t test for two systems:\n";
calcPairedTvalue(\@m_bleu_list1,\@m_bleu_list2);

print "-----------------\n";


sub nist_length_penalty {

    my ($ratio) = @_;
    return 1 if $ratio >= 1;
    return 0 if $ratio <= 0;
    my $ratio_x = 1.5;
    my $score_x = 0.5;
    my $beta = -log($score_x)/log($ratio_x)/log($ratio_x);
    return exp (-$beta*log($ratio)*log($ratio));
}

sub bleu_length_penalty{
    my $r2hLen = shift;
    # penalize only if hypothesis is less than closest reference in length
    return 1 if $r2hLen < 1.0;
    return exp(1 - $r2hLen);
}

sub findConfidenceInterval{
	my @score_list = sort {$a<=>$b} @{$_[0]};	
	
	
	#updated based on http://www.itl.nist.gov/div898/handbook/prc/section2/prc252.htm
	my $lowRange = calcPercentile( (1-$confidenceLevel)/2, \@score_list);
	my $highRange = calcPercentile( 1-(1-$confidenceLevel)/2, \@score_list);

	printf "Median=%.4f ",calcPercentile(0.5, \@score_list);
	printf "[%.4f,%.4f]\n",$lowRange, $highRange;

}


sub calcPairedTvalue{
	my @values1 = @{$_[0]};	
	my @values2 = @{$_[1]};
	
	$size = $#values1+1;
	$df = $size-1;
	
	if($#values1!=$#values2){
		print STDERR "Can't perform t-test for two array with different number of values! Quit!\n";
		exit;
	}
	
	$mean1=0;
	$mean2=0;
	$sum1=0;
	$sum2=0;
	for($i=0;$i<$size;$i++){
		$sum1+=$values1[$i];
		$sum2+=$values2[$i];
	}
	
	$mean1=$sum1/$size;
	$mean2=$sum2/$size;
	
	$diffSumSq=0;
	for($i=0;$i<$size;$i++){
		$diff = $values1[$i]-$mean1-($values2[$i]-$mean2);
		
		$diffSumSq+=$diff*$diff;
	}
	
	$t=($mean1-$mean2)*sqrt($size*($size-1)/$diffSumSq);
	
	print "Degree of freedom: $df\n";
	printf "t=%.4f\n",$t;
	
	my $sys1_GT_sys2=1;
	if($t<0){
		$t=0.0-$t;
		$sys1_GT_sys2 = 0;
	}
	
	
	$p=Statistics::Distributions::tprob($df,$t);
	printf "p=%.4f\n",$p;
	#printf "Confidence of two systems are not equal: %.3f\%\n",100*(1-2*$p);	#two-tail
	print "Confidence of [";
	if($sys1_GT_sys2>0){
		print "Sys1 > Sys2";
	}
	else{
		print "Sys1 < Sys2";
	}
	printf "]: %.2f\%\n", 100*(1-$p);
}


sub showCriticalValues{
	my $df = $_[0];

	print "-------------- Critical Values ---------------\n"; 
	print " df = $df, c=99.9% p=.0005 = ", Statistics::Distributions::tdistr ($df,.0005),"\n";
	print " df = $df, c=99.8% p=0.001 = ", Statistics::Distributions::tdistr ($df,.001),"\n";
	print " df = $df, c=99.0% p=0.005 = ", Statistics::Distributions::tdistr ($df,.005),"\n";
	print " df = $df, c=98.0% p=0.010 = ", Statistics::Distributions::tdistr ($df,.01),"\n";
	print " df = $df, c=95.0% p=0.025 = ", Statistics::Distributions::tdistr ($df,.025),"\n";
	print " df = $df, c=90.0% p=0.050 = ", Statistics::Distributions::tdistr ($df,.05),"\n";


}


sub calcPercentile{
	my $p=$_[0];
	my @ranked_score_list = @{$_[1]};
	my $N=$#ranked_score_list+1;
		
	my $k=int( ($N+1)*$p);
	my $d=($N+1)*$p-$k;
	
	if($k==0){
		return $ranked_score_list[0];
	}
	
	if($k==$N){
		return $ranked_score_list[$N-1];
	}
	
	$k--;	#to turn it into the array index
	return $ranked_score_list[$k]+$d*($ranked_score_list[$k+1]-$ranked_score_list[$k]);

}
