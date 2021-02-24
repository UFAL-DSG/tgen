
TGen Baseline for the E2E NLG Challenge
=======================================

To train and evaluate the baseline, you need to:

1. __Convert the [downloaded E2E data](http://www.macs.hw.ac.uk/InteractionLab/E2E/) into 
   a format used by TGen.__ This is done using the [input/convert.py](input/convert.py) script.
   
   Note that multiple references are joined for one MR in the development set, but kept
   separate for the training set (the `-m` switch). All files are plain text, one instance per line
   (except for multiple references, where instances are separated by empty lines).
   
   The `name` and `near` slots in the MRs are delexicalized. The output files are:

    * `*-abst.txt` -- lexicalization instructions (what was delexicalized at which position in
        the references, can be used to lexicalize the outputs)
    * `*-conc_das.txt` -- original, lexicalized MRs (converted to TGen's representation, 
        semantically equivalent)
    * `*-conc.txt` -- original, lexicalized reference texts
    * `*-das.txt` -- delexicalized MRs
    * `*-text.txt` -- delexicalized reference texts

   If you're using a test set file that only contains MRs and not references, use the 
   `--no-refs` switch. This won't produce `*-text.txt` and `*-conc.txt` (since they wouldn't 
   make sense anyway). You'll still have the `*-das.txt` and `*-abst.txt` files required
   for generation.

```
./convert.py -a name,near -n trainset.csv train
./convert.py -a name,near -n -m devset.csv devel
./convert.py -a name,near -n -m testset_with_refs.csv test
./convert.py -a name,near -n --no-refs testset_without_refs.csv test
```

2. __Train TGen on the training set.__ 
   This uses the default configuration file, the converted data, and the default random seed.
   It will save the model into `model.pickle.gz` (and several other files starting with `model`).
   Note that we used five different random seeds (`-r s0`, `-r s1` ... `-r s4`), then picked
   the setup that was best on the development data

```
../run_tgen.py seq2seq_train config/config.yaml \
    input/train-das.txt input/train-text.txt \
    model.pickle.gz
```
   
   The default configuration uses a small part of the training data for validation
   (early stopping if the performance goes down on that set).
   You can also opt to use the development set for validation (in that case, the
   validation set isn't “unseen” for the purposes of evaluation).
   If you want to use the development set during training, add `-v input/devel-das.txt,input/devel-text.txt` 
   to the parameters (right after `seq2seq_train`).
 

3. __Generate outputs on the development and test sets.__
   This will also perform lexicalization of the outputs.
   
```
../run_tgen.py seq2seq_gen -w outputs-dev.txt -a input/devel-abst.txt \
    model.pickle.gz input/devel-das.txt
../run_tgen.py seq2seq_gen -w outputs-test.txt -a input/test-abst.txt \
    model.pickle.gz input/test-das.txt
```

4. __Postprocess the outputs.__
   This basically amounts to a simple detokenization.
   The [script](postprocess/postprocess.py) changes the outputs in-place, or you can specify a target file name.
```
./postprocess/postprocess.py outputs-dev.txt
./postprocess/postprocess.py outputs-test.txt
```


Remarks
-------

Please refer to [../USAGE.md](../USAGE.md) for TGen installation instructions.

The [Makefile](Makefile) in this directory contains a simple experiment management system,
but this assumes running on a [SGE](https://arc.liv.ac.uk/trac/SGE) computing cluster
and there are probably site-specific settings hardcoded. Please 
[contact me](https://github.com/tuetschek) if you want to use it.

