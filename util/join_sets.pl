#!/usr/bin/env perl
#
# Joining several SGM data sets (source/reference/generated).
# Used for bootstrap comparison tests.
#
#
use Getopt::Long;

my $USAGE = "Usage: ./$0 [--num-refs=N] set1.sgm set2.sgm [...]\n";

my ( $num_refs ) = ( 1 );
GetOptions(
    'num-refs|refs|r|n=i' => \$num_refs,
) or die($USAGE);
die($USAGE) if ( !@ARGV );


my $line_num = 0;
my $doc_num  = 0;
my $ref_num  = 0;
my $set_foot = '';  # only last set footer is printed

while ( my $line = <> ) {
    if ( $line =~ /docid="test"/ ) {
        $line =~ s/docid="test"/docid="test$doc_num"/;
        $ref_num++;
        if ($ref_num == $num_refs){
            $ref_num = 0;
            $doc_num++;
        }
    }
    elsif ( $line =~ /<(src|ref|tst)set/ ) {
        next if ( $line_num != 0 );
    }
    elsif ( $line =~ /<\/(src|tst|ref)set/ ) {
        $set_foot = $line;  # store the set footer to be printed
        next;
    }
    $line_num++;
    print $line;
}

print $set_foot;
