
SHELL=/bin/bash
ABSTRACTED = name,area,address,phone,good_for_meal,near,food,price_range,count,price,postcode
CONVERT_SETTINGS = -s 3:1:1 -a $(ABSTRACTED) -i
TREEX_BASIC_SCEN = W2A::ResegmentSentences Misc::MoveNodesAfterResegment \
    Util::Eval anode='$$.set_tag($$.conll_pos); $$.set_deprel(undef);' \
    Scen::Analysis::CS tokenizer=none tagger=none \
    Misc::JoinBundles \
    Misc::EncodeGrammatemes

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

# data target storage (for the "push" command)
DATA_TARGET = ../data/training2

all: train-text.pickle.gz train-das.txt train-das_l.txt train-abst.txt \
    train-text.txt train-text_l.txt train-tls.txt train-tls_l.txt \
    devel-text.pickle.gz devel-das.txt devel-das_l.txt devel-abst.txt \
    devel-text.txt devel-text_l.txt devel-tls.txt devel-tls_l.txt \
    devel-das.sgm devel-text.sgm devel-conc.sgm \
    test-text.pickle.gz test-das.txt test-das_l.txt test-abst.txt \
    test-text.txt test-text_l.txt test-tls.txt test-tls_l.txt \
    test-das.sgm test-text.sgm test-conc.sgm \
    grammatemes.tsv

# prevent deleting intermediate files
.SECONDARY:

%.pickle.gz: %.yaml.gz
	../../util/yaml2pickle.py $<

%.yaml.gz: %.treex.gz
	treex -Len Read::Treex from=$< Write::YAML to=$@

%-text.treex.gz: %-text.conll
	treex -Lcs Read::CoNLLU from=$< $(TREEX_BASIC_SCEN) $(COORD_ADJUST) \
	 	Write::Treex to=$@

train-text.txt train-das.txt train-das_l.txt train-abst.txt train-text_l.txt train-text.conll train-text_l.conll train-tls.txt train-tls_l.txt devel-text.txt devel-das.txt devel-das_l.txt devel-abst.txt devel-text_l.txt devel-text.conll devel-text_l.conll devel-tls.txt devel-tls_l.txt test-text.txt test-das.txt test-das_l.txt test-abst.txt test-text_l.txt test-text.conll test-text_l.conll test-tls.txt test-tls_l.txt: czech-morfflex-pdt-160310.tagger surface_forms.json expand-das.txt expand-texts.txt
	./convert.py $(CONVERT_SETTINGS) $^ train,devel,test

grammatemes.tsv: train-text.treex.gz train-das.txt
	treex -Lcs Read::Treex from=$< Print::GrammatemesForTgen to=$@ da_file=train-das.txt

%-conc.sgm: %-text_l.txt
	echo '<refset setid="CsRest" srclang="any" trglang="cz">' > $@
	treex -Lcs Read::Sentences from=$< \
	    Write::SgmMTEval to=- set_id=CsRest sys_id=manual_0 selector= \
	    | sed 's/docid="[^"]*"/docid="test"/' >> $@
	echo '</refset>' >> $@

%-text.sgm: %-text.txt
	echo '<refset setid="CsRest" srclang="any" trglang="cz">' > $@
	treex -Lcs Read::Sentences from=$< \
	    Write::SgmMTEval to=- set_id=CsRest sys_id=manual_0 selector= \
	    | sed 's/docid="[^"]*"/docid="test"/' >> $@
	echo '</refset>' >> $@

%-das.sgm: %-das.txt
	treex -Lcs Read::Sentences from=$< Write::SgmMTEval to=- set_id=CsRest sys_id=source add_header=srcset | sed 's/docid="[^"]*"/docid="test"/' > $@

# copy data to target storage
push:
	mkdir -p $(DATA_TARGET)-delex $(DATA_TARGET)-lex
	cp train-* devel-* test-* grammatemes.tsv surface_forms.json $(DATA_TARGET)-delex
	rm $(DATA_TARGET)-delex/{*.conll,*tls*,*das_l*}
	cp train-* devel-* test-* grammatemes.tsv surface_forms.json $(DATA_TARGET)-lex
	for file in $(DATA_TARGET)-lex/*das_l*; do newname=`echo $$file | sed 's/das_l/das/'`; mv $$file $$newname; done
	rm $(DATA_TARGET)-lex/{*.conll,*tls*}

# clear all created files
clean:
	rm train-* devel-* test-* grammatemes.tsv
