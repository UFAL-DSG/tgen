
SHELL=/bin/bash
CONVERT_SETTINGS = -s 3:1:1

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
   devel-das.sgm devel-text.sgm devel-conc.sgm \
   test-text.pickle.gz test-das.txt test-abst.txt test-conc.txt \
   test-das.sgm test-text.sgm test-conc.sgm \
   grammatemes.tsv

# prevent deleting intermediate files
.SECONDARY:

%-text.pickle.gz: %-text.yaml.gz
	../../util/yaml2pickle.py $<

%-text.yaml.gz: %-text.treex.gz
	treex -Len Read::Treex from=$< Write::YAML to=$@

%-text.treex.gz: %-text.txt %-conc.txt %-abst.txt
	treex Read::AlignedSentences en=$< en_conc=$*-conc.txt Scen::Analysis::EN \
	    Misc::AbstractDialogueSlots slots=$(ABSTRACTED) abstraction_file=$*-abst.txt x_only=1 $(COORD_ADJUST) \
	 	Write::Treex to=$@

train-text.txt train-das.txt train-abst.txt train-conc.txt devel-text.txt devel-das.txt devel-abst.txt devel-conc.txt test-text.txt test-das.txt test-abst.txt test-conc.txt: dataset.json
	./convert.py $(CONVERT_SETTINGS) $< train,devel,test

train-%.txt: all-%.txt
	head -n 362 $< > $@

test-%.txt: all-%.txt
	tail -n 42 $< > $@

grammatemes.tsv: train-text.treex.gz train-das.txt
	treex -Len Read::Treex from=$< Print::GrammatemesForTgen to=$@ da_file=train-das.txt

%-conc.sgm: %-conc.txt
	treex -Len Read::Sentences from=$< Write::SgmMTEval to=- set_id=SfxRest sys_id=manual add_header=refset | sed 's/docid="[^"]*"/docid="test"/' > $@

%-text.sgm: %-text.txt
	treex -Len Read::Sentences from=$< Write::SgmMTEval to=- set_id=SfxRest sys_id=manual add_header=refset | sed 's/docid="[^"]*"/docid="test"/' > $@

%-das.sgm: %-das.txt
	treex -Len Read::Sentences from=$< Write::SgmMTEval to=- set_id=SfxRest sys_id=source add_header=srcset | sed 's/docid="[^"]*"/docid="test"/' > $@
