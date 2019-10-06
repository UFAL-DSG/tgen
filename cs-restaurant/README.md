
TGen for CS Restaurant 
======================

To train and evaluate TGen on the CS Restaurant dataset, you need to:

1. __Convert the [CS Restaurant data](https://github.com/UFAL-DSG/cs_restaurant_dataset) into 
   a format used by TGen.__ This is done using the [input/convert.py](input/convert.py) script.
   Several slots (see below) are delexicalized. The output files are:

    * `*-abst.txt` -- lexicalization instructions (what was delexicalized at which position in
        the references, can be used to lexicalize the outputs)
    * `*-das.txt` -- delexicalized DAs
    * `*-das_l.txt` -- original, lexicalized DAs (converted to TGen's representation, 
        semantically equivalent)
    * `*-text.conll` -- delexicalized reference texts -- CoNLL-U format (morphology level only)
    * `*-text_l.conll` -- original, lexicalized reference texts -- CoNLL-U format (morphology level only)
    * `*-text.txt` -- delexicalized reference texts -- plain text
    * `*-text_l.txt` -- original, lexicalized reference texts -- plain text
    * `*-tls.txt` -- delexicalized reference texts -- interleaved forms/lemmas/tags
    * `*-tls_l.txt` -- original, lexicalized reference texts -- interleaved forms/lemmas/tags

    You need [MorphoDiTa](https://pypi.org/project/ufal.morphodita/) installed, and a Czech
    [tagger model](http://ufal.mff.cuni.cz/morphodita#language_models) saved in the current 
    directory (`czech-morfflex-pdt-160310.tagger`).

```
./convert.py -a name,area,address,phone,good_for_meal,near,food,price_range,count,price,postcode \
    czech-morfflex-pdt-160310.tagger surface_forms.json train.json train
./convert.py -a name,area,address,phone,good_for_meal,near,food,price_range,count,price,postcode \
    czech-morfflex-pdt-160310.tagger surface_forms.json devel.json devel
./convert.py -a name,area,address,phone,good_for_meal,near,food,price_range,count,price,postcode \
    czech-morfflex-pdt-160310.tagger surface_forms.json test.json test
```


2. __Train TGen on the training set.__ 
   This uses the default configuration file, the converted data, and the default random seed.
   It will save the model into `model.pickle.gz` (and several other files starting with `model`).
   If you want to use the development set for validation, add `-v input/devel-das.txt,input/devel-text.conll`
   as a parameter.

```
../run_tgen.py seq2seq_train config/config.yaml \
    input/train-das.txt input/train-text.conll \
    model.pickle.gz
```


3. __Generate outputs on the development set.__
   This will also perform lexicalization of the outputs.
   Change `devel` for `test` if you want to generate outputs on the test set.

```
../run_tgen.py seq2seq_gen -w outputs.txt -a input/devel-abst.txt \
    model.pickle.gz input/devel-das.txt
```


Remarks
-------

Please refer to [../USAGE.md](../USAGE.md) for TGen installation instructions.

The full configuration used [Treex](http://ufal.mff.cuni.cz/treex) for data storage,
tree-based generation, and output postprocessing. Getting Treex to install is a little 
tricky. Please [contact me](https://github.com/tuetschek) if you want to use it.

The [Makefile](Makefile) in this directory contains a simple experiment management system,
but this assumes running on a [SGE](https://arc.liv.ac.uk/trac/SGE) computing cluster
and there are site-specific settings hardcoded. 
Please [contact me](https://github.com/tuetschek) if you want to use it.

