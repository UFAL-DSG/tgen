#!/usr/bin/env perl
#
# Converting Bagel data set to our DA format.
#

use strict;
use warnings;
use autodie;
use Getopt::Long;

use Treex::Core::Common;
use Treex::Core::Scenario;
use Treex::Core::Document;
use Treex::Core::Log;

my $abstr_slots_str = '';
if ( not GetOptions( "abstract|abstr|a=s" => \$abstr_slots_str ) or @ARGV != 5 ) {
    die('Usage: ./convert.pl input output-das output-text output-abstraction');
}
my %abstr_slots = map { $_ => 1 } split( /[, ]+/, $abstr_slots_str );

open( my $in,       '<:utf8', $ARGV[0] );
open( my $out_das,  '>:utf8', $ARGV[1] );
open( my $out_text, '>:utf8', $ARGV[2] );
open( my $out_abst, '>:utf8', $ARGV[3] );
open( my $out_conc, '>:utf8', $ARGV[4] );

Treex::Core::Log::log_set_error_level('WARN');
my $tokenizer = Treex::Core::Scenario->new(
    {
        'from_string' => 'Util::SetGlobal language=en W2A::EN::Tokenize'
    }
);
$tokenizer->start();

my $da_line   = '';
my %slot_vals = ();

while ( my $line = <$in> ) {

    next if $line =~ /^ABSTRACT_DA/;

    if ( $line =~ /^FULL_DA/ ) {
        $line =~ s/FULL_DA = //;

        # convert the data format
        $line =~ s/,(?! )/)&inform(/g;

        # get values for the individual slots
        while ( $line =~ m/inform\(([^=]*)=([^\&]*)\)/g ) {
            my ( $slot, $val ) = ( $1, $2 );
            if ( !$slot_vals{$slot} ) {
                $slot_vals{$slot} = [];
            }
            push @{ $slot_vals{$slot} }, $val;
            if ( defined( $abstr_slots{$slot} ) ) {
                $da_line .= "inform($slot=X-$slot)&";
            }
            else {
                $da_line .= "inform($slot=$val)&";
            }
        }
        $da_line =~ s/&*$//;
        $da_line .= "\n";
    }
    elsif ( $line =~ /^->/ ) {

        $line =~ s/-> "//;
        $line =~ s/";//;

        if ( $line !~ m/\.\s*$/ ) {    # add dots at the end of every sentence
            $line =~ s/(\S)(\s*)$/$1\.$2/;
        }

        # tokenize using Treex
        my @tokens = tokenize( $tokenizer, $line );
        my $sent = join( ' ', @tokens );    # re-join additional slot info in []
        $sent =~ s/ \+ /+/g;
        $sent =~ s/\[ /[/g;
        $sent =~ s/ \]/]/g;
        @tokens = split / /, $sent;
        $sent   = '';
        my $abstr    = '';
        my $in_abstr = 0;
        my $i        = 0;

        # produce output tokenized sentences along with abstraction instructions
        foreach my $token (@tokens) {

            if ( $token =~ /^\[(.*)\+.*\]/ ) {    # [slot+val] – need to abstract this

                my $slot = $1;
                my $val  = shift @{ $slot_vals{$slot} };
                push @{ $slot_vals{$slot} }, $val;    # cycle the values (in case they repeat)
                if ($in_abstr) {
                    $abstr .= $i . "\t";
                }
                $abstr .= "$slot=$val:$i-";
                $in_abstr = 1;
            }
            elsif ( $token =~ /^\[/ ) {               # [slot] or [] – just end previous abstraction
                if ($in_abstr) {
                    $abstr .= $i . "\t";
                    $in_abstr = 0;
                }
            }
            else {                                    # plain words
                $sent .= $token . ' ';
                $i++;
            }
        }
        if ($in_abstr) {
            $abstr .= $i;
            $in_abstr = 0;
        }

        $sent =~ s/\s*$//;
        $abstr =~ s/\s*$//;

        # produce de-abstracted sentence where X's are replaced by the actual values
        my $conc_sent = deabstract( $sent, $abstr );

        print {$out_das} $da_line;
        print {$out_text} $sent . "\n";
        print {$out_abst} $abstr . "\n";
        print {$out_conc} $conc_sent . "\n";

        print STDERR ".";
        $da_line   = '';
        %slot_vals = ();
    }
}
print STDERR "\n";

$tokenizer->end();

sub tokenize {
    my ( $tokenizer, $sent ) = @_;

    # create a Treex document for the sentence
    my $document = Treex::Core::Document->new();
    my $bundle   = $document->create_bundle();
    my $zone     = $bundle->create_zone( 'en', '' );
    $zone->set_sentence($sent);

    # apply tokenizer
    $tokenizer->apply_to_documents($document);

    # retrieve tokens
    my $atree = $zone->get_atree();
    return map { $_->form } $atree->get_descendants( { ordered => 1 } );
}

# produce de-abstracted sentence where X's are replaced by the actual values
sub deabstract {
    my ( $sent, $abstr ) = @_;

    my $i         = 0;
    my $conc_sent = '';
    foreach my $token ( split / /, $sent ) {

        # X = to be deabstracted
        if ( $token eq 'X' ) {
            my ($val) = ( $abstr =~ /=([^:]*):$i-[0-9]+/ );
            $val =~ s/^"//;
            $val =~ s/"$//;
            $conc_sent .= $val . ' ';
        }

        # other tokens
        else {
            $conc_sent .= $token . ' ';
        }
        $i++;
    }
    $conc_sent =~ s/\s*$//;
    return $conc_sent;
}
