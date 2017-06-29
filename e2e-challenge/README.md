
TGen Baseline for the E2E NLG Challenge
=======================================

To train and evaluate the baseline, you need to:

1. __Convert the [downloaded E2E data](http://www.macs.hw.ac.uk/InteractionLab/E2E/) into 
   a format used by TGen.__ This is done using the [input/convert.py](input/convert.py) script.
   
   Note that multiple references are joined for one MR in the development set, but kept
   separate for the training set.
   
   The `name` and `near` slots in the MRs are delexicalized.

```
./convert.py -a name,near -n new-data/trainset.csv train
./convert.py -a name,near -n -m new-data/devset.csv devel
```

2. __Train TGen on the training set.__ 
   This uses the default configuration file, the converted data, and the default random seed.
   It will save the model into `model.pickle.gz` (and several other files starting with `model`).
   Note that we used five different random seeds (`-r s0`, `-r s1` ... `-r s4`), then picked
   the setup that was best on the development data

```
../run_tgen.py seq2seq_train config/seq2seq_config.py \
    input/train-das.txt input/train-text.txt \
    model.pickle.gz
```


3. __Generate outputs on the development set.__
   This will also perform lexicalization of the outputs.
   
```
../run_tgen.py seq2seq_gen -w outputs.txt -a input/devel-abst.txt \
    model.pickle.gz input/devel-das.txt
```

4. __Postprocess the outputs.__
   This basically amounts to a simple detokenization.
   The script changes the outputs in-place, or you can specify a target file name.
```
./postprocess/postprocess.py outputs.txt
```


Remarks
-------

Please refer to [../USAGE.md](../USAGE.md) for installation instructions.

The [Makefile](Makefile) in this directory contains a simple experiment management system,
but this assumes running on a [SGE](https://arc.liv.ac.uk/trac/SGE) computing cluster
and there are probably site-specific settings hardcoded. Please 
[contact me](https://github.com/tuetschek) if you want to use it.

