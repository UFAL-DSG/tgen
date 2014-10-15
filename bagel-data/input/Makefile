#
# TODO this is a bit outdated, should include train-test split (last 42 DAs used as test
#

SHELL=/bin/bash
ABSTRACTED='name near phone addr postcode'

%-text.yaml.gz: %-text.treex.gz
	treex -Len Read::Treex from=$< Write::YAML to=$@

%-text.treex.gz: %-text.txt %-abstr.txt
	treex -Len Read::Sentences from=$< en_analysis.scen \
	    Misc::AbstractDialogueSlots slots=$(ABSTRACTED) abstraction_file=$*-abstr.txt \
	    Write::Treex to=$@

all-abstr.txt: all-das.txt

all-text.txt: all-das.txt

all-conc.txt: all-das.txt

all-das.txt: ACL10-inform-training.txt convert.pl
	./convert.pl -a $(ABSTRACTED) $< all-{das,text,abstr,conc}.txt

%-conc.sgm: %-conc.txt
	treex -Len Read::Sentences from=$< Util::Eval document='$$.set_path(""); $$.set_file_stem("$*");' Write::SgmMTEval to=$@ set_id=BAGEL sys_id=manual add_header=refset

train-%.txt: all-%.txt
	head -n 362 $< > $@

test-%.txt: all-%.txt
	tail -n 42 $< > $@

grammatemes.tsv: train-text.treex.gz
	treex -Len Read::Treex from=$< Print::GrammatemesForTgen to=$@
