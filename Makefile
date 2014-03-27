
train:
	tgen/tgen.py train bagel-data/data-das.txt bagel-data/data-text.pickle.gz bagel-data/candgen-model.pickle.gz

generate:
	tgen/tgen.py generate bagel-data/candgen-model.pickle.gz bagel-data/data-das_test.txt bagel-data/random-output.yaml.gz
