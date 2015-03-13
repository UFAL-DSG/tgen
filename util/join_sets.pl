#!/usr/bin/env perl
#
# Joining several SGM data sets (source/reference/generated).
#
# Usage: ./join_sets.pl set1.sgm set2.sgm [...]
#

my $line_num = 0;
my $doc_num  = 0;
my $refid    = 0;

while ( my $line = <> ) {
    if ( $line =~ /docid="test"/ ) {
        $line =~ s/docid="test"/docid="test$doc_num"/;
        $doc_num++;    # if ($refid == 1);
                       #$refid = 1 - $refid;
    }
    elsif ( $line =~ /<(src|ref|tst)set/ ) {
        next if ( $line_num != 0 );
    }
    elsif ( $line =~ /<\/(src|tst|ref)set/ ) {
        next if ( $doc_num < 10 );
    }
    $line_num++;
    print $line;
}
