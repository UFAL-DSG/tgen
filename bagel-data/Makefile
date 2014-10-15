#
# Experiment management for Tgen
#

# Environment settings
SHELL = /bin/bash
TGEN = ../run_tgen.py
QSUBMIT = qsubmit --logdir '$(TRY_DIR)' --mem 8g --jobname T.$(TRY_NUM).$(RUN_NAME) $(QP)# using OB's qsubmit script
ifdef PYPY
  ACTIVATE = source ~/work/tools/virtualenv-pypy-x86_64/bin/activate
else
  ACTIVATE = echo -n
endif
DIFFTOOL = vimdiff
MTEVAL = ../util/mteval-v11b.pl
COMPARE_ITERS = 1# setting last iteration to be compared in log comparison

# Runs directories
RUNS_DIR  := runs# main directory for experiment outputs
TRY_NUM   := $(shell perl -e '$$m=0; for(<$(RUNS_DIR)/*>){/\/(\d+)_/ and $$1 > $$m and $$m=$$1;} printf "%03d", $$m+1;')# experiment number
RUN_NAME  := experiment# default name, to be overridden by targets
DATE      := $(shell date +%Y-%m-%d_%H-%M-%S)
TRY_DIR    = $(RUNS_DIR)/$(TRY_NUM)_$(DATE)_$(RUN_NAME)# experiment output directory

ifdef D # Shortcut D -> DESC
  DESC := $(D)
endif
ifdef J # Shortcut J -> JOBS
  JOBS := $(J)
endif

ifdef JOBS
  PARALLEL := -j $(JOBS)
endif

# Input data file defaults

CANDGEN_MODEL = model/candgen.pickle.gz
PERCRANK_MODEL = model/percrank.pickle.gz
PERCRANK_CONFIG = config/percrank.py
ASEARCH_CONFIG = config/asearch.py
TRAIN_PORTION = 1.0
TRAIN_DAS = data/train-das.txt
TRAIN_TREES = data/train-text.pickle.gz
TRAIN_ABSTR = data/train-abstr.txt
TRAIN_CONC = data/train-conc.sgm
TEST_DAS = data/test-das.txt
TEST_TREES = data/test-text.pickle.gz
TEST_ABSTR = data/test-abstr.txt
TEST_CONC = data/test-conc.sgm
SURFACE_SCEN = surface/en_synthesis.scen

# Help text

define HELP_MESSAGE
Tgen experiments
================

- Use 'make desc' to list experiments with descriptions.

- Use 'make percrank_train' to run a new percrank training
  experiment
  - set TRAIN_PORTION to a fraction of the training
    data to be used
  - set J/JOBS to the number of parallel jobs to be used
    in a parallel training setup

- Use 'make asearch_gen' to run A*search generation
  (set PERCRANK_MODEL to the model you want to use).

- Use 'make compare-X:Y' to compare the logs of runs X
  and Y (set COMPARE_ITERS to the number of iterations you
  want to compare).

Use QP='--console' if you want the job to run interactively instead
of submitting to the cluster.

Use ACTIVATE='source path/to/virtualenv/bin/activate' to run
within a Virtualenv instead of the default Python.

Use PYPY=1 to activate the preset PyPy Virtualenv.

endef
export HELP_MESSAGE

#
# Targets
#

# Auxiliary targets

help:
	@echo "$$HELP_MESSAGE" | egrep --color '^(\s*make.*|)'

desc:
	@ls -d $(RUNS_DIR)/* | sort | while read file; do \
		echo -ne $$file ": \t" | sed 's/runs\///;s/_/\t/;s/_/ /;s/_/\t/';  \
		cat $$file/ABOUT | tr  '\n' ' '; \
		if echo $$file | grep 'asearch_gen' >/dev/null 2>&1; then \
			score=`grep -i "Node precision" $$file/T.* $$file/log.txt 2>/dev/null | head -n 1 | sed 's/.*N[Oo][Dd][Ee] p/P/' | tr -d '\n'`; \
			echo -en "\t$$score"; \
			bleu=`grep -i '^NIST ' $$file/T.* $$file/log.txt 2>/dev/null | head -n 1 | sed 's/^.*NIST/NIST/;s/ for system.*$$//;' | tr -d '\n'`; \
			echo -en "\t$$bleu"; \
		fi; \
		echo; \
	done


printvars:
	$(foreach V, $(sort $(.VARIABLES)), $(if $(filter-out environment% default automatic, $(origin $V)), $(info $V=$($V) ($(value $V)))))

printgit:
	@git status
	@echo -e "\n*** *** ***\n"
	@git log --pretty=format:"%h - %an, %ar : %s" -1
	@echo -e "\n*** *** ***\n"
	@git diff


prepare_dir:
	# create the directory (and parallel run directory if needed)
	mkdir -p $(TRY_DIR)
	# write the description
	@if [ -n "$(DESC)" ]; then \
		echo "$(DESC)" > $(TRY_DIR)/ABOUT; \
	fi
	make printvars > $(TRY_DIR)/VARS
	make printgit > $(TRY_DIR)/GIT_VERSION

# Main targets

percrank_train: RUN_NAME := percrank_train
percrank_train: prepare_dir
percrank_train:
	# copy needed files (to ensure replication)
	cp $(CANDGEN_MODEL) $(TRY_DIR)/candgen.pickle.gz
	cp $(PERCRANK_CONFIG) $(TRY_DIR)/percrank_config.py
	cp $(TRAIN_DAS) $(TRY_DIR)/train-das.txt
	cp $(TRAIN_TREES) $(TRY_DIR)/train-text.pickle.gz
	# run the experiment
	$(QSUBMIT) '$(ACTIVATE); \
		$(TGEN) percrank_train -d $(TRY_DIR)/debug-log.txt -c $(TRY_DIR)/candgen.pickle.gz  \
		-s $(TRAIN_PORTION) $(PARALLEL) \
		$(TRY_DIR)/percrank_config.py $(TRY_DIR)/train-das.txt $(TRY_DIR)/train-text.pickle.gz \
		$(TRY_DIR)/percrank.pickle.gz' 2>&1 | tee $(TRY_DIR)/log.txt


asearch_gen: RUN_NAME := asearch_gen
asearch_gen: prepare_dir
asearch_gen:
	cp $(CANDGEN_MODEL) $(TRY_DIR)/candgen.pickle.gz
	cp $(PERCRANK_MODEL) $(TRY_DIR)/percrank.pickle.gz
	cp $(ASEARCH_CONFIG) $(TRY_DIR)/asearch_config.py
	cp $(TEST_DAS) $(TRY_DIR)/test-das.txt
	cp $(TEST_TREES) $(TRY_DIR)/test-text.pickle.gz
	# run
	$(QSUBMIT) '$(ACTIVATE); \
		$(TGEN) asearch_gen -e $(TRY_DIR)/test-text.pickle.gz -d $(TRY_DIR)/debug-log.txt \
		-w $(TRY_DIR)/out-trees.yaml.gz -c $(TRY_DIR)/asearch_config.py \
		$(TRY_DIR)/candgen.pickle.gz $(TRY_DIR)/percrank.pickle.gz $(TRY_DIR)/test-das.txt; \
		treex -Len -Sgen Read::YAML from=$(TRY_DIR)/out-trees.yaml.gz \
			Misc::DeabstractDialogueSlots abstraction_file=$(TEST_ABSTR) \
			Util::Eval document="$$.set_path(\"\"); $$.set_file_stem(\"test\");" \
			$(SURFACE_SCEN) Write::Treex Write::SgmMTEval to=$(TRY_DIR)/out-text.sgm set_id=BAGEL sys_id=TGEN add_header=tstset; \
		$(MTEVAL) -r $(TEST_CONC) -t $(TRY_DIR)/out-text.sgm' \
		2>&1 | tee $(TRY_DIR)/log.txt


compare_logs-%:
	NUM1=`echo $* | cut -d: -f1`; NUM2=`echo $* | cut -d: -f2`; \
	CUTOFF_IT=`printf "%05d" $$(( $(COMPARE_ITERS) + 1 ))` ; \
	cat $(RUNS_DIR)/$$NUM1*/debug-log.txt | sed -n -e "1,/TR $$CUTOFF_IT/p" | sed -r 's/^[A-Za-z]{3} [A-Za-z]{3} [0-9: ]{16} //' > compare.1.txt ; \
	cat $(RUNS_DIR)/$$NUM2*/debug-log.txt | sed -n -e "1,/TR $$CUTOFF_IT/p" | sed -r 's/^[A-Za-z]{3} [A-Za-z]{3} [0-9: ]{16} //' > compare.2.txt ; \
	$(DIFFTOOL) compare.1.txt compare.2.txt
