
SHELL=/bin/bash
ABSTRACTED = 'name,address,phone,near,area,price'
CONVERT_SETTINGS = -a $(ABSTRACTED)
COORD_ADJUST = # Adjusting coordinations (deleting or transforming) 

ifeq ($(COORD),stanford)
  COORD_ADJUST = Misc::DeleteCoordNodes attach_below_1st=1 delete_conj=0
endif
ifeq ($(COORD),delete)
  COORD_ADJUST = Misc::DeleteCoordNodes
endif
ifeq ($(COORD),flat)
  COORD_ADJUST = Misc::DeleteCoordNodes attach_below_1st=0 delete_conj=0
endif
ifeq ($(COORD),flat_trees)
  COORD_ADJUST = Util::Eval tnode='$$.set_parent($$.get_root);'
endif

all: train-text.pickle.gz train-das.txt train-abst.txt train-conc.txt \
   devel-text.pickle.gz devel-das.txt devel-abst.txt devel-conc.txt \
   test-text.pickle.gz test-das.txt test-abst.txt test-conc.txt \
   grammatemes.tsv

%-text.pickle.gz: %-text.yaml.gz
	../../util/yaml2pickle.py $<

%-text.yaml.gz: %-text.treex.gz
	treex -Len Read::Treex from=$< Write::YAML to=$@

%-text.treex.gz: %-text.txt %-conc.txt %-abst.txt
	treex Read::AlignedSentences en=$< en_conc=$*-conc.txt en_analysis.scen \
	    Misc::AbstractDialogueSlots slots=$(ABSTRACTED) abstraction_file=$*-abst.txt x_only=1 $(COORD_ADJUST) \
	 	Write::Treex to=$@

all-abst.txt: all-das.txt

all-text.txt: all-das.txt

all-conc.txt: all-das.txt

all-das.txt: sfxrestaurant.emnlp.json
	./convert.py $(CONVERT_SETTINGS) $< all

%-conc.sgm: %-conc.txt
	treex -Len Read::Sentences from=$< Util::Eval document='$$.set_path(""); $$.set_file_stem("$*");' Write::SgmMTEval to=$@ set_id=BAGEL sys_id=manual add_header=refset

train-%.txt: all-%.txt
	head -n 362 $< > $@

test-%.txt: all-%.txt
	tail -n 42 $< > $@

grammatemes.tsv: train-text.treex.gz
	treex -Len Read::Treex from=$< Print::GrammatemesForTgen to=$@

