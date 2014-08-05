#
# TODO this is a bit outdated, should include train-test split (last 42 DAs used as test
#


TREEX_DIR=/home/odusek/treex


all-text.yaml.gz: data-text.treex.gz
	treex Read::Treex from=$< Write::YAML to=$@

all-text.treex.gz: data-text.txt
	treex Read::Sentences from=$< $(TREEX_DIR)/devel/en2en/scenarios/en_analysis.scen Write::Treex to=$@

all-text.txt: ACL10-inform-training.txt convert.pl
	./convert.pl $< data-{das,text}.txt

all-das.txt: ACL10-inform-training.txt convert.pl
	./convert.pl $< data-{das,text}.txt
