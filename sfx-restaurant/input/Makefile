
SHELL=/bin/bash
CONVERT_SETTINGS = -a $(ABSTRACTED) -s 3:1:1 -i

ABSTRACTED = 'address,area,count,name,near,phone,postcode,price'
ifdef AALL
  ABSTRACTED = 'address,area,count,food,good_for_meal,goodformeal,name,near,phone,postcode,price,pricerange,price_range,type'
endif

ifdef TOKS
    CONVERT_SETTINGS = -a $(ABSTRACTED) -s 3:1:1 -m -n -i
endif
INPUT_FILE = sfxrestaurant.emnlp.json
ifdef RNNLG
    INPUT_FILE = sfxrestaurant.rnnlg.json
endif 

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

# Select which files are required: if TOKS is 1, Treex analysis is not needed (and is not run)
ifdef TOKS
all: train-das.txt train-abst.txt train-conc.txt \
   devel-das.txt devel-abst.txt devel-conc.txt devel-ref.txt \
   test-das.txt test-abst.txt test-conc.txt test-ref.txt
else
all: train-text.pickle.gz train-das.txt train-abst.txt train-conc.txt \
   devel-text.pickle.gz devel-das.txt devel-abst.txt devel-conc.txt \
   devel-das.sgm devel-text.sgm devel-conc.sgm \
   test-text.pickle.gz test-das.txt test-abst.txt test-conc.txt \
   test-das.sgm test-text.sgm test-conc.sgm \
   grammatemes.tsv
endif

# prevent deleting intermediate files
.SECONDARY:

%-text.pickle.gz: %-text.yaml.gz
	../../util/yaml2pickle.py $<

%-text.yaml.gz: %-text.treex.gz
	treex -Len Read::Treex from=$< Write::YAML to=$@

%-text.treex.gz: %-text.txt %-conc.txt %-abst.txt
	treex Read::AlignedSentences en=$< en_conc=$*-conc.txt en_analysis.scen \
	    Misc::AbstractDialogueSlots slots=$(ABSTRACTED) abstraction_file=$*-abst.txt x_only=1 $(COORD_ADJUST) \
	 	Write::Treex to=$@

# `%' using a pattern rule with multiple targets (even though we know the names and `%' stands just for `.')
# so that it is only called once for all of these files (they are all created in one go)
train-text%txt train-das%txt train-abst%txt train-conc%txt devel-text%txt devel-das%txt devel-abst%txt devel-conc%txt test-text%txt test-das%txt test-abst%txt test-conc%txt devel-ref%txt test-ref%txt: $(INPUT_FILE) 
	./convert.py $(CONVERT_SETTINGS) $< train,devel,test

grammatemes.tsv: train-text.treex.gz train-das.txt
	treex -Len Read::Treex from=$< Print::GrammatemesForTgen to=$@ da_file=train-das.txt

%-conc.sgm: %-conc.txt
	treex -Len Read::Sentences from=$< Write::SgmMTEval to=- set_id=SfxRest sys_id=manual add_header=refset | sed 's/docid="[^"]*"/docid="test"/' > $@

%-text.sgm: %-text.txt
	treex -Len Read::Sentences from=$< Write::SgmMTEval to=- set_id=SfxRest sys_id=manual add_header=refset | sed 's/docid="[^"]*"/docid="test"/' > $@

%-das.sgm: %-das.txt
	treex -Len Read::Sentences from=$< Write::SgmMTEval to=- set_id=SfxRest sys_id=source add_header=srcset | sed 's/docid="[^"]*"/docid="test"/' > $@
