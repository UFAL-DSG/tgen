#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Postprocessing for Alex Context NLG Dataset
"""

from __future__ import unicode_literals
import pickle
import sys
from argparse import ArgumentParser
import subprocess
import re

from tgen.futil import file_stream
from tgen.debug import exc_info_hook
sys.excepthook = exc_info_hook


def process_file(args):

    # find out generator mode
    mode = 'trees'
    if args.model:
        with file_stream(args.model, 'rb', encoding=None) as fh:
            data = pickle.load(fh)
            if 'mode' in data['cfg']:
                mode = data['cfg']['mode']
            if 'use_tokens' in data['cfg']:
                mode = 'tokens' if data['cfg']['use_tokens'] else 'trees'

    # compose scenario

    scen = ['T2A::CS::VocalizePrepos',  # this surface stuff is needed for both tokens and tagged lemmas
            'T2A::CS::CapitalizeSentStart',
            'A2W::ConcatenateTokens',
            'A2W::CS::DetokenizeUsingRules',
            'A2W::CS::RemoveRepeatedTokens', ]
    if mode == 'tokens':
        scen = ['T2A::CopyTtree',
                'Util::Eval anode="$.set_form($.lemma);"', ] + scen
    elif mode == 'tagged_lemmas':
        scen = ['T2A::CopyTtree',
                'Util::Eval atree=\'my @as=$.get_descendants({ordered=>1}); ' +
                'while (my ($l, $t) = splice @as, 0, 2){ next if (!defined($t)); $l->set_tag($t->lemma); $t->remove(); }\'',
                'Misc::TagToMorphcat',
                'T2A::CS::GenerateWordforms', ] + scen
    else:
        # get the canonical CS generation scenario
        scen_dump_ps = subprocess.Popen('treex -d Scen::Synthesis::CS', shell=True, stdout=subprocess.PIPE)
        scen, _ = scen_dump_ps.communicate()
        scen = [block for block in scen.split("\n") if block and not block.startswith('#')]

        # insert our custom morphological processing block into it
        pos = next(i for i, block in enumerate(scen) if re.search(r'generateword', block, re.IGNORECASE))
        scen.insert(pos, 'Misc::GenerateWordformsFromJSON surface_forms="%s"' % args.surface_forms)

        # add grammatemes and clause number processing
        scen = ['Util::Eval tnode="$.set_functor(\\"???\\"); ' +
                '$.set_t_lemma(\\"\\") if (!defined($.t_lemma)); ' +
                '$.set_formeme(\\"x\\") if (!defined($.formeme));"',
                'T2T::AssignDefaultGrammatemes grammateme_file="%s" da_file="%s"' %
                (args.grammatemes, args.input_das),
                'Misc::RestoreCoordNodes',
                'T2T::CS2CS::MarkClauseHeads',
                'T2T::SetClauseNumber'] + scen

    scen = ['Read::YAML from="%s"' % args.input_file] + scen
    scen += ['Write::Treex',
             'Util::Eval document="$.set_path(\\"\\"); $.set_file_stem(\\"test\\");"',
             'Write::SgmMTEval to="%s" set_id=CsRest sys_id=TGEN add_header=tstset' % args.output_file]

    subprocess.call(('treex -Lcs -S%s ' % args.selector) + " ".join(scen), shell=True)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-m', '--model', type=str, help='Seq2seq model pickle file (to get mode)')
    ap.add_argument('-i', '--input-das', type=str, help='Input DA file')
    ap.add_argument('-g', '--grammatemes', type=str, help='Default grammateme file')
    ap.add_argument('-f', '--surface-forms', type=str, help='Surface forms file')
    ap.add_argument('-s', '--selector', type=str, help='Target selector')
    ap.add_argument('input_file', type=str, help='Input YAML file')
    ap.add_argument('output_file', type=str, help='Output SGM file')
    args = ap.parse_args()

    process_file(args)
