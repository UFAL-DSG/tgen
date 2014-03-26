
TREEX_DIR=/home/odusek/treex


data-text.treex.gz: data-text.txt
	treex Read::Sentences from=$< $(TREEX_DIR)/devel/en2en/scenarios/en_analysis.scen Write::Treex to=$@

data-text.txt: ACL10-inform-training.txt convert.pl
	./convert.pl $< data-{das,text}.txt

data-das.txt: ACL10-inform-training.txt convert.pl
	./convert.pl $< data-{das,text}.txt
