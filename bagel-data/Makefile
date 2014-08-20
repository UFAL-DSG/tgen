#
# Experiment management for Tgen
#

# Environment settings
SHELL = /bin/bash
TGEN = ../tgen/tgen.py

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
PERCRANK_CONFIG = config/percrank.py
TRAIN_DAS = data/train-das.txt
TRAIN_TREES = data/train-text.pickle.gz

# Help text

define HELP_MESSAGE
Tgen experiments
================

- Use 'make desc' to list experiments with descriptions.

- Use 'make percrank_train' to run a new percrank training
  experiment.

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


# Main targets

prepare_dir:
	# create the directory (and parallel run directory if needed)
	mkdir -p $(TRY_DIR)
	# write the description
	@if [ -n "$(DESC)" ]; then \
		echo "$(DESC)" > $(TRY_DIR)/ABOUT; \
	fi

percrank_train: RUN_NAME := percrank_train
percrank_train: prepare_dir
percrank_train:
	# copy needed files (to ensure replication)
	cp $(CANDGEN_MODEL) $(TRY_DIR)/candgen.pickle.gz
	cp $(PERCRANK_CONFIG) $(TRY_DIR)/percrank_config.py
	cp $(TRAIN_DAS) $(TRY_DIR)/train-das.txt
	cp $(TRAIN_TREES) $(TRY_DIR)/train-text.pickle.gz
	# run the experiment
	$(TGEN) percrank_train -d $(TRY_DIR)/debug-log.txt -c $(TRY_DIR)/candgen.pickle.gz  \
	    $(TRY_DIR)/percrank_config.py $(TRY_DIR)/train-das.txt $(TRY_DIR)/train-text.pickle.gz \
	    $(TRY_DIR)/percrank.pickle.gz 2>&1 | tee $(TRY_DIR)/log.txt

