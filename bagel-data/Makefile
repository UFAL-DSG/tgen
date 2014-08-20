#
# Experiment management for Tgen
#

# Environment settings
SHELL = /bin/bash
TGEN = ../tgen/tgen.py
QSUBMIT = qsubmit --logdir '$(TRY_DIR)' --mem 8g --jobname T.$(RUN_NAME)# using OB's qsubmit script

# Runs directories
RUNS_DIR  := runs# main directory for experiment outputs
TRY_NUM   := $(shell perl -e '$$m=0; for(<$(RUNS_DIR)/*>){/\/(\d+)_/ and $$1 > $$m and $$m=$$1;} printf "%03d", $$m+1;')# experiment number
RUN_NAME  := experiment# default name, to be overridden by targets
DATE      := $(shell date +%Y-%m-%d_%H-%M-%S)
TRY_DIR    = $(RUNS_DIR)/$(TRY_NUM)_$(DATE)_$(RUN_NAME)# experiment output directory

ifdef D # Shortcut D -> DESC
  DESC := $(D)
endif

# Input data file defaults

CANDGEN_MODEL = model/candgen.pickle.gz
PERCRANK_MODEL = model/percrank.pickle.gz
PERCRANK_CONFIG = config/percrank.py
ASEARCH_CONFIG = config/asearch.py
TRAIN_PORTION = 1.0
TRAIN_DAS = data/train-das.txt
TRAIN_TREES = data/train-text.pickle.gz
TEST_DAS = data/test-das.txt
TEST_TREES = data/test-text.pickle.gz

# Help text

define HELP_MESSAGE
Tgen experiments
================

- Use 'make desc' to list experiments with descriptions.

- Use 'make percrank_train' to run a new percrank training
  experiment.

- Use 'make asearch_gen' to run A*search generation
  (set PERCRANK_MODEL to the model you want to use).

endef
export HELP_MESSAGE

#
# Targets
#

# Auxiliary targets

help:
	@echo "$$HELP_MESSAGE" | egrep --color '^(\s*make.*|)'

desc:
	ls $(RUNS_DIR)/*/ABOUT | sort | while read file; do echo -ne $$file ": \t"; cat $$file; done


printvars:
	$(foreach V, $(sort $(.VARIABLES)), $(if $(filter-out environment% default automatic, $(origin $V)), $(info $V=$($V) ($(value $V)))))


prepare_dir:
	# create the directory (and parallel run directory if needed)
	mkdir -p $(TRY_DIR)
	# write the description
	@if [ -n "$(DESC)" ]; then \
		echo "$(DESC)" > $(TRY_DIR)/ABOUT; \
	fi
	make printvars > $(TRY_DIR)/VARS

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
	$(QSUBMIT) \
	    '$(TGEN) percrank_train -d $(TRY_DIR)/debug-log.txt -c $(TRY_DIR)/candgen.pickle.gz  \
	        -s $(TRAIN_PORTION) \
	        $(TRY_DIR)/percrank_config.py $(TRY_DIR)/train-das.txt $(TRY_DIR)/train-text.pickle.gz \
	        $(TRY_DIR)/percrank.pickle.gz'


asearch_gen: RUN_NAME := asearch_gen
asearch_gen: prepare_dir
asearch_gen:
	cp $(CANDGEN_MODEL) $(TRY_DIR)/candgen.pickle.gz
	cp $(PERCRANK_MODEL) $(TRY_DIR)/percrank.pickle.gz
	cp $(ASEARCH_CONFIG) $(TRY_DIR)/asearch_config.py
	cp $(TEST_DAS) $(TRY_DIR)/test-das.txt
	cp $(TEST_TREES) $(TRY_DIR)/test-text.pickle.gz
	# run
	$(QSUBMIT) \
	    '$(TGEN) asearch_gen -e $(TRY_DIR)/test-text.pickle.gz -d $(TRY_DIR)/debug-log.txt \
	        -w $(TRY_DIR)/out-trees.yaml.gz -c $(TRY_DIR)/asearch_config.py \
	        $(TRY_DIR)/candgen.pickle.gz $(TRY_DIR)/percrank.pickle.gz $(TRY_DIR)/test-das.txt'



