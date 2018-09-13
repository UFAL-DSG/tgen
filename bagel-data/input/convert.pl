#!/usr/bin/env perl
#
# Converting BAGEL data set to our DA format.
#
# Usage: ./convert.pl [-a slot1,slot2] [-s] [-j] input \
#            output-das output-text output-abstraction output-bagel-abstracted output-concrete
#
# -a = list of slots to be abstracted (replaced with "X-slot")
# -s = skip values that are not abstracted in the ABSTRACT_DA lines in the actual BAGEL data.
# -j = join repeated values for the same slot (e.g. area=X&area=Y -> area="X and Y", works
#          only for abstracted values.

use strict;
use warnings;
use autodie;
use Getopt::Long;

use Treex::Core::Common;
use Treex::Core::Scenario;
use Treex::Core::Document;
use Treex::Core::Log;

#
# MAIN
#

my $abstr_slots_str   = '';
my $skip_unabstracted = 0;
my $join_repeats      = 0;
my $sort_insts        = 1;

if (not GetOptions(
        "abstract|abstr|a=s"       => \$abstr_slots_str,
        "skip-unabstracted|skip|s" => \$skip_unabstracted,
        "join-repeats|join|j"      => \$join_repeats,
        "sort-insts|sort!"         => \$sort_insts,
    )
    or @ARGV != 6
    )
{
    die('Usage: ./convert.pl input output-das output-text output-abstraction' .
            ' output-bagel-abstracted output-concrete');
}

my %abstr_slots = map { $_ => 1 } split( /[, ]+/, $abstr_slots_str );

open( my $in,       '<:utf8', $ARGV[0] );
open( my $out_das,  '>:utf8', $ARGV[1] );
open( my $out_text, '>:utf8', $ARGV[2] );
open( my $out_abst, '>:utf8', $ARGV[3] );
open( my $out_bagel, '>:utf8', $ARGV[4] );
open( my $out_conc, '>:utf8', $ARGV[5] );

Treex::Core::Log::log_set_error_level('WARN');
my $tokenizer = Treex::Core::Scenario->new(
    {
        'from_string' => 'Util::SetGlobal language=en W2A::EN::Tokenize'
    }
);
$tokenizer->start();

my @buf;
my @instances = ();

print STDERR "Processing";
while ( my $line = <$in> ) {

    next if $line =~ /^\s*$/;
    push @buf, $line;

    if ( $line =~ m/->/ ) {
        push @instances, process_instance(@buf);
        @buf = ();
    }

}
$tokenizer->end();

print STDERR "\nOutput...\n";

if ($sort_insts){
    @instances = sort { $a->{da} cmp $b->{da} || length $a->{text} <=> length $b->{text} } @instances;
}
foreach my $instance ( @instances ) {
    print_instance($instance);
}

#
# SUBROUTINES
#

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
    my ( $sent, $abstr, $bagel_level_only ) = @_;

    my $i         = 0;
    my $conc_sent = '';
    foreach my $token ( split / /, $sent ) {

        # X = to be deabstracted
        if ( $token eq 'X' ) {
            my ($val) = ( $abstr =~ /=([^:]*):$i-[0-9]+/ );
            if ($bagel_level_only){
                $val =~ s/"([^"]*)"#/$1/g;
                $val =~ s/"([^"]*)"/X/g;
            }
            else {
                $val =~ s/^"//;
                $val =~ s/"#?$//;
                $val =~ s/"#? and "/ and /g;
            }
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

# Return first element of an array and push it to the end of the array (cycle through elements)
sub shift_push {
    my ($arr) = @_;
    my $ret = shift @$arr;
    push @$arr, $ret;
    return $ret;
}

# Process one data instance: convert both DAs and text, return the result as a hash reference
sub process_instance {

    my ( $full_da, $abstract_da, $text ) = @_;

    # Get abstracted DA (to know what's abstracted and what isn't)
    my $abstract_vals = get_abstract_vals($abstract_da);

    # Convert DA text; abstract needed values
    my ( $da_line, $full_vals, $slot_joins ) = convert_da( $full_da, $abstract_vals );

    # Produce tokenized sentence and abstraction instructions according to abstraction settings
    my ( $sent, $abstr ) = convert_text( $text, $full_vals, $slot_joins );

    # produce de-abstracted sentence where X's are replaced by the actual values
    my $conc_sent = deabstract( $sent, $abstr );

    # produce a "BAGEL" canonical sentence
    my $bagel_sent = $text;
    $bagel_sent =~ s/";\s*\r?\n$//;
    $bagel_sent =~ s/^-> "//;
    $bagel_sent =~ s/\[[^\]]*\]//g;
    $bagel_sent =~ s/(?<!\.)$/./;
    $bagel_sent =~ s/(?<! )([.;,?!'])/ $1/g;

    # Indicate progress
    print STDERR ".";

    # return the result in a hash
    return {
        'da'   => $da_line,
        'text' => $sent,
        'abst' => $abstr,
        'bagel' => $bagel_sent,
        'conc' => $conc_sent,
    };
}

sub print_instance {
    my ($instance) = @_;

    print {$out_das} $instance->{da},    "\n";
    print {$out_text} $instance->{text}, "\n";
    print {$out_abst} $instance->{abst}, "\n";
    print {$out_bagel} $instance->{bagel}, "\n";
    print {$out_conc} $instance->{conc}, "\n";
}

# Retrieve abstract DA slot values
sub get_abstract_vals {

    my ($abstract_da) = @_;
    my %slot_vals = ();

    # prepare string format for value extraction
    $abstract_da =~ s/ABSTRACT_DA = //;
    $abstract_da =~ s/,(?! )/)&inform(/g;

    # get values for the slots
    while ( $abstract_da =~ m/inform\(([^=]*)=([^\&]*)\)/g ) {
        my ( $slot, $val ) = ( $1, $2 );
        if ( !$slot_vals{$slot} ) {
            $slot_vals{$slot} = [];
        }
        push @{ $slot_vals{$slot} }, $val;
    }

    return ( \%slot_vals );
}

# Convert the full DA line – producing the output DA with abstractions according to global settings
sub convert_da {

    my ( $full_da_in, $abstract_vals ) = @_;
    my $full_da_out = '';
    my %slot_vals   = ();
    my %slot_joins  = ();

    # basic data format conversion
    $full_da_in =~ s/FULL_DA = //;
    $full_da_in =~ s/,(?! )/)&inform(/g;

    # get values for the individual slots; abstract if required (according to settings)
    while ( $full_da_in =~ m/inform\(([^=]*)=([^\&]*)\)/g ) {
        my ( $slot, $val ) = ( $1, $2 );
        my $abstract_val = shift_push $abstract_vals->{$slot};

        $val =~ s/^"?/"/;
        $val =~ s/"?$/"/;
        if ($abstract_val !~ /^"?X[0-9]+"?$/){
            $val =~ s/^"citycentre"$/"city centre"/;
            $val .= '#';
        }
        if ( !$slot_vals{$slot} ) {
            $slot_vals{$slot} = [];
        }
        push @{ $slot_vals{$slot} }, $val;
        if ( defined( $abstr_slots{$slot} ) and ( !$skip_unabstracted or $abstract_val =~ /^"?X[0-9]+"?$/ ) ) {

            # join repeated values using "and" if join_repeats is in force
            # keep track of joined slots for text conversion
            if ( $join_repeats and @{ $slot_vals{$slot} } > 1 ) {
                my $join_val = join( ' and ', @{ $slot_vals{$slot} } );
                $slot_vals{$slot} = [$join_val];
                $slot_joins{$slot} = ( $slot_joins{$slot} // 0 ) + 1;
            }
            else {
                $full_da_out .= "inform($slot=X-$slot)&";
            }
        }
        else {
            $full_da_out .= "inform($slot=$val)&";
        }
    }
    $full_da_out =~ s/&*$//;

    return ( $full_da_out, \%slot_vals, \%slot_joins );
}

# Convert one text line (sentence), using abstraction according to global settings
# (+values from DA conversion and list of joined repeated values)
# Output abstract sentence + de-abstraction instructions
sub convert_text {

    my ( $sent, $da_vals, $slot_joins ) = @_;

    $sent =~ s/-> "//;
    $sent =~ s/";//;

    if ( $sent !~ m/\.\s*$/ ) {    # add dots at the end of every sentence
        $sent =~ s/(\S)(\s*)$/$1\.$2/;
    }

    # tokenize using Treex, re-join slot info in []
    my @tokens = tokenize( $tokenizer, $sent );
    $sent = join( ' ', @tokens );
    $sent =~ s/ \+ /+/g;
    $sent =~ s/\[ /[/g;
    $sent =~ s/ \]/]/g;
    @tokens = split / /, $sent;
    $sent = '';
    my $abstr    = '';
    my $in_abstr = 0;
    my $i        = 0;
    my $slot     = '';
    my @slots = ();

    # print STDERR "\n\n";
    # print STDERR join(" ", @tokens) . "\n\n";

    # produce output tokenized sentences along with abstraction instructions
    foreach my $token (@tokens) {

        # [slot+val] – starting abstraction
        if ( $token =~ /^\[(.*)\+.*\]/ ) {

            # joining repeated slots: just count the values, do not start new abstraction
            if ( $in_abstr and $slot_joins->{$slot} and ( $1 eq $slot ) ) {
                # print STDERR $i, "\t", $token, "\t", $slot, " J" . ($slot_joins->{$slot} // '-'), "\t", ( $in_abstr ? 'ABSTR' : '' ), "\t", $sent, "/skip dec\n";
                $slot_joins->{$slot}--;
                next;
            }
            # discontinuous repeated slots: just remember that this slot occurred already
            if ( $in_abstr and $slot_joins->{$slot} and ( $1 ne $slot ) ) {
                $slot_joins->{$slot}--;
                $abstr =~ s/\t[^\t]*$/\t/;
            }

            $slot = $1;
            my $val = shift_push $da_vals->{$slot};    # cycle the values (in case they repeat)
            if ($in_abstr) {
                $abstr .= $i . "\t";
            }
            $abstr .= "$slot=$val:$i-";
            $in_abstr = 1;
        }

        # [slot] or [] – end previous abstraction
        elsif ( $token =~ /^\[(.*)\]/ ) {

            # joining repeated slots: continue abstraction if slot does not change
            if ( $in_abstr and ( ( $1 || $slot ) eq $slot ) and $slot_joins->{$slot} ) {
                $slot = $1 || $slot;
                # print STDERR $i, "\t", $token, "\t", $slot, " J" . ($slot_joins->{$slot} // '-'), "\t", ( $in_abstr ? 'ABSTR' : '' ), "\t", $sent, "/skip join\n";
                next;
            }
            # discontinuous repeated slots: just remember that this slot occurred already
            if ( $in_abstr and ( ( $1 || $slot ) ne $slot ) and $slot_joins->{$slot} ) {
                $slot_joins->{$slot}--;
                $slot = $1 || $slot;
                $abstr =~ s/\t[^\t]*$/\t/;
                $in_abstr = 0;
                next;
            }
            $slot = $1 || $slot;
            if ($in_abstr) {
                $abstr .= $i . "\t";
                $in_abstr = 0;
            }
        }

        # plain words – print them to output
        else {

            # skip 'and X' if joining repeated slots
            if ( $in_abstr and $slot_joins->{$slot} ) {
                # print STDERR $i, "\t", $token, "\t", $slot, " J" . ($slot_joins->{$slot} // '-'), "\t", ( $in_abstr ? 'ABSTR' : '' ), "\t", $sent, "/skip plain\n";
                next;
            }
            if ( $token ne '.' and $in_abstr and defined( $abstr_slots{$slot} ) and not $skip_unabstracted ){
                if ( $in_abstr > 1 ){
                    next;
                }
                $in_abstr = 2;
                $token = 'X';
            }
            $sent .= $token . ' ';
            $i++;
        }

        # print STDERR $i, "\t", $token, "\t", $slot, " J" . ($slot_joins->{$slot} // '-'), "\t", ( $in_abstr ? 'ABSTR' : '' ), "\t", $sent, "\n";

    }
    if ($in_abstr) {
        $abstr .= $i;
        $in_abstr = 0;
    }

    $sent =~ s/\s*$//;
    $abstr =~ s/\s*$//;

    # print STDERR $sent, "\n";

    return ( $sent, $abstr );
}
