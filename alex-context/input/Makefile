
SHELL=/bin/bash
CONVERT_SETTINGS = -s 3:1:1 -m
ABSTRACTED = from_stop,to_stop,direction,departure_time,departure_time_rel,ampm,vehicle,line,arrival_time,duration,distance,num_transfers,alternative
TREEX_BASIC_SCEN = W2A::ResegmentSentences Scen::Analysis::EN pretokenized=1 Misc::JoinBundles

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
   train-context.pickle.gz devel-context.pickle.gz test-context.pickle.gz \
   grammatemes.tsv

# prevent deleting intermediate files
.SECONDARY:

%.pickle.gz: %.yaml.gz
	../../util/yaml2pickle.py $<

%.yaml.gz: %.treex.gz
	treex -Len Read::Treex from=$< Write::YAML to=$@

%-text.treex.gz: %-text.txt %-conc.txt %-abst.txt
	if echo "$@" | grep -E '^(devel|test)' >/dev/null; then \
	    GROUP="Misc::GroupBundles selector_suffixes=ref0,ref1,ref2"; \
        else \
	    GROUP=""; \
	fi; \
	treex Read::AlignedSentences en=$< en_conc=$*-conc.txt $(TREEX_BASIC_SCEN) \
	    Misc::AbstractDialogueSlots slots=$(ABSTRACTED) abstraction_file=$*-abst.txt x_only=1 $(COORD_ADJUST) $$GROUP \
	 	Write::Treex to=$@

%-context.treex.gz: %-context.txt
	treex -Len Read::Sentences from=$< $(TREEX_BASIC_SCEN) $(COORD_ADJUST) Write::Treex to=$@

train-text.txt train-das.txt train-abst.txt train-conc.txt devel-text.txt devel-das.txt devel-abst.txt devel-conc.txt test-text.txt test-das.txt test-abst.txt test-conc.txt train-context.txt devel-context.txt test-context.txt: dataset.json
	./convert.py $(CONVERT_SETTINGS) $< train,devel,test

grammatemes.tsv: train-text.treex.gz train-das.txt
	treex -Len Read::Treex from=$< Print::GrammatemesForTgen to=$@ da_file=train-das.txt

%-conc.sgm: %-conc.txt
	echo '<refset setid="AlexContext" srclang="any" trglang="en">' > $@
	treex -Len Read::Sentences from=$< \
	    Misc::GroupBundles selector_suffixes=ref0,ref1,ref2 \
	    Write::SgmMTEval to=- set_id=AlexContext sys_id=manual_0 selector=ref0 \
	    Write::SgmMTEval to=- set_id=AlexContext sys_id=manual_1 selector=ref1 \
	    Write::SgmMTEval to=- set_id=AlexContext sys_id=manual_2 selector=ref2 \
	    | sed 's/docid="[^"]*"/docid="test"/' >> $@
	echo '</refset>' >> $@

%-text.sgm: %-text.txt
	echo '<refset setid="AlexContext" srclang="any" trglang="en">' > $@
	treex -Len Read::Sentences from=$< \
	    Misc::GroupBundles selector_suffixes=ref0,ref1,ref2 \
	    Write::SgmMTEval to=- set_id=AlexContext sys_id=manual_0 selector=ref0 \
	    Write::SgmMTEval to=- set_id=AlexContext sys_id=manual_1 selector=ref1 \
	    Write::SgmMTEval to=- set_id=AlexContext sys_id=manual_2 selector=ref2 \
	    | sed 's/docid="[^"]*"/docid="test"/' >> $@
	echo '</refset>' >> $@

%-das.sgm: %-das.txt
	treex -Len Read::Sentences from=$< Write::SgmMTEval to=- set_id=AlexContext sys_id=source add_header=srcset | sed 's/docid="[^"]*"/docid="test"/' > $@
