SHELL=/bin/bash
ABSTRACTED = 'name near phone addr postcode area'
CONVERT_SETTINGS = -a $(ABSTRACTED) -s

%-text.pickle.gz: %-text.yaml.gz
	../../util/yaml2pickle.py $<

%-text.yaml.gz: %-text.treex.gz
	treex -Len Read::Treex from=$< Write::YAML to=$@

%-text.treex.gz: %-text.txt %-conc.txt %-abstr.txt
	treex Read::AlignedSentences en=$< en_conc=$*-conc.txt en_analysis.scen \
		Misc::AbstractDialogueSlots slots=$(ABSTRACTED) abstraction_file=$*-abstr.txt \
	 	Write::Treex to=$@

all-abstr.txt: all-das.txt

all-text.txt: all-das.txt

all-conc.txt: all-das.txt

all-das.txt: ACL10-inform-training.txt convert.pl
	./convert.pl $(CONVERT_SETTINGS) $< all-{das,text,abstr,conc}.txt

%-conc.sgm: %-conc.txt
	treex -Len Read::Sentences from=$< Util::Eval document='$$.set_path(""); $$.set_file_stem("$*");' Write::SgmMTEval to=$@ set_id=BAGEL sys_id=manual add_header=refset

train-%.txt: all-%.txt
	head -n 362 $< > $@

test-%.txt: all-%.txt
	tail -n 42 $< > $@

grammatemes.tsv: train-text.treex.gz
	treex -Len Read::Treex from=$< Print::GrammatemesForTgen to=$@

# not using "conc" since we measure BLEU on X's anyway
# ignoring train.1-* (only using 1/2 of training data)
# (must do two passes over test data since otherwise Morce will segfault)
cv: all-das.txt all-text.txt all-abstr.txt
	./cv_split.py -f 10 -c 2 -d cv all-das.txt all-text.txt all-abstr.txt
	for CV in cv0*; do \
		treex -Len Read::Sentences from=$$CV/train.0-text.txt en_analysis.scen \
			Misc::AbstractDialogueSlots slots=$(ABSTRACTED) abstraction_file=$$CV/train.0-abstr.txt \
			Write::Treex to=$$CV/train-text.treex.gz ; \
		treex -Len Read::AlignedSentences en_ref0=$$CV/test.0-text.txt en_ref1=$$CV/test.1-text.txt \
			Util::SetGlobal selector=ref0 en_analysis.scen \
			Misc::AbstractDialogueSlots slots=$(ABSTRACTED) abstraction_file=$$CV/test.0-abstr.txt \
			Write::Treex to=$$CV/test-text.treex.gz ; \
		treex -Len Read::Treex from=$$CV/test-text.treex.gz \
			Util::SetGlobal selector=ref1 en_analysis.scen \
			Misc::AbstractDialogueSlots slots=$(ABSTRACTED) abstraction_file=$$CV/test.1-abstr.txt \
			Write::Treex ; \
		treex -Len Read::Treex from=$$CV/train-text.treex.gz Write::Treex to=$$CV/train-text.yaml.gz ; \
		../../util/yaml2pickle.py $$CV/train-text.yaml.gz ; \
		treex -Len Read::Treex from=$$CV/test-text.treex.gz Write::Treex to=$$CV/test-text.yaml.gz ; \
		../../util/yaml2pickle.py $$CV/test-text.yaml.gz ; \
		echo '<refset setid="BAGEL" srclang="any" trglang="en">' > $$CV/test-text.sgm ; \
		treex -Len Read::Sentences from="$$CV/test.0-text.txt" Write::SgmMTEval to=- set_id=BAGEL sys_id=manual_0 >> $$CV/test-text.sgm ; \
		treex -Len Read::Sentences from="$$CV/test.1-text.txt" Write::SgmMTEval to=- set_id=BAGEL sys_id=manual_1 >> $$CV/test-text.sgm ; \
		echo '</refset>' >> $$CV/test-text.sgm ; \
	done
	

