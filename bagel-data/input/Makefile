#
# TODO this is a bit outdated, should include train-test split (last 42 DAs used as test
#

SHELL=/bin/bash
TREEX_DIR=/home/odusek/treex


%-text.yaml.gz: %-text.treex.gz
	treex -Len Read::Treex from=$< Write::YAML to=$@

%-text.treex.gz: %-text.txt
	treex -Len Read::Sentences from=$< $(TREEX_DIR)/devel/en2en/scenarios/en_analysis.scen Misc::JoinBundles Write::Treex to=$@

all-text.txt: ACL10-inform-training.txt convert.pl
	./convert.pl $< all-{das,text}.txt

all-das.txt: ACL10-inform-training.txt convert.pl
	./convert.pl $< all-{das,text}.txt

train-%.txt: all-%.txt
	head -n 362 $< > $@

test-%.txt: all-%.txt
	tail -n 42 $< > $@

