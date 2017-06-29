
TGen Baseline for the E2E NLG Challenge
=======================================

To train the baseline, you need to:

1. Convert the [downloaded E2E data](http://www.macs.hw.ac.uk/InteractionLab/E2E/) into 
   a format used by TGen. This is done using the [input/convert.py](input/convert.py) script.
   
   Note that multiple references are joined for one MR in the development set, but kept
   separate for the training set.
   
   The `name` and `near` slots in the MRs are delexicalized.

```
./convert.py -a name,near -n new-data/trainset.csv train
./convert.py -a name,near -n -m new-data/devset.csv devel
```

2. Train TGen on the training set:

```
../run_tgen.py seq2seq_train config/seq2seq_config.py \
    input/train-das.txt input/train-text.txt \
    seq2seq.pickle.gz
```

