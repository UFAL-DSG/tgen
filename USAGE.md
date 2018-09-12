TGen -- Installation and Usage
==============================

Installation
------------

TGen is written in Python (version 2.7). You can install it simply by cloning this repository, 
then installing all Python dependencies using pip:

```
git clone https://github.com/UFAL-DSG/tgen
cd tgen
pip install -r requirements.txt
```

We recommend using [virtualenv](https://virtualenv.pypa.io/) to install all the required libraries.

To replicate most of the experiments in our papers, you will also need to install 
[Treex](http://ufal.cz/treex) (including the newest version from the Git repository as 
described in Step 5 of the [Treex installation guide](http://ufal.mff.cuni.cz/treex/install.html)). 
It is, however, not needed for basic TGen functionality (without using deep syntactic trees).

### Dependencies ###

Required Python modules (installed using pip and the [requirements file](requirements.txt)):

- [enum34](https://pypi.python.org/pypi/enum34)
- [numpy](http://www.numpy.org/)
- [rpyc](https://pypi.python.org/pypi/rpyc/)
- [pudb](https://pypi.python.org/pypi/pudb)
- [recordclass](https://pypi.python.org/pypi/recordclass)
- [TensorFlow](https://www.tensorflow.org/), only version 1.0.1 is supported
- [kenlm](https://github.com/kpu/kenlm)
- [PyTreex](https://github.com/ufal/pytreex)


Optional, manual installation (Perl code):

- [Treex](http://ufal.cz/treex)

Additionally, some obsolete code depends on [Theano](http://deeplearning.net/software/theano/), 
but it is currently not used and will be probably removed in the future.

Parallel training on the cluster is using [SGE](https://arc.liv.ac.uk/trac/SGE)'s `qsub`.


Usage (seq2seq-based generator only)
------------------------------------

The main entry point `run_tgen.py`. The basic commands used for training and generating are:

```
./run_tgen.py seq2seq_train config-file.py train-das.txt train-text.txt model.pickle.gz
./run_tgen.py seq2seq_gen [-w out-text.txt] model.pickle.gz test-das.txt
```

You can run the program with `seq2seq_train -h` and `seq2seq_gen -h` to see more detailed options.

The file parameters for training are:

* `config-file.py` -- a configuration file, containing a Python dictionary with all generator
    parameters. A default configuration file can be found in every experiment directory (see [below](#experiments)).

* `train-das.txt` -- training DAs, one DA per line (see [below](#data-formats)).

* `train-text.txt` -- training natural language texts or trees (in a Treex YAML file) as example 
    outputs for the generator (see [below](#data-formats)). Text files should contain one instance per line.

* `model.pickle.gz` -- the output destination for the model. Note that several additional files 
    with different extensions will be created.

The generation mode requires the model and a list of DAs, one per line. It can write the outputs
into a text file (for direct string generation) or a Treex YAML file (for tree generation).
The files are typically further post-processed (lexicalization, tree-to-string surface 
realization).


### Data formats ###

The main data formats used by TGen are:

* __Dialogue Acts (DAs)__: The main input format into TGen are lists of triples of the shape 
    (DA type, slot/attribute, value), e.g.: _inform(food=Chinese)&inform(price=expensive)_.
    This easily maps on dialogue act representations used in various spoken dialogue systems.
    Conversion scripts are provided for several datasets (see below).
    DAs are delexicalized in a typical case.

* __Plain text__: Outputs for direct string generation. Use one output sentence per line
    (no comments/empty lines allowed). For best results, delexicalize sparse values, such as
    restaurant/landmark names, time values etc. and fill them in in a postprocessing step.

* __Trees__: Trees used for DA-to-tree generation are t-trees as produced by the [Treex NLP
    system](http://ufal.cz/treex). We use YAML serialization produced by the `Write::YAML` Treex
    block. Installing Treex is necessary for any experiments involving generating trees.


Experiments
-----------

Our own experiments on several datasets are included as subdirectories within this repository:

* [alex-context/](alex-context): our experiments on the 
    [Alex Context NLG Dataset](https://github.com/UFAL-DSG/alex_context_nlg_dataset) 
    (SIGDIAL 2016).

* [bagel-data/](bagel-data): our experiments on the 
    [BAGEL set](http://farm2.user.srcf.net/research/bagel/) (ACL 2015, 2016).

* [cs-restaurant/](cs-restaurant): generating for our 
    [Czech Restaurant NLG dataset](https://github.com/UFAL-DSG/cs_restaurant_dataset).

* [e2e-challenge/](e2e-challenge): the baseline system for the 
    [E2E NLG Challenge](http://www.macs.hw.ac.uk/InteractionLab/E2E/). There are some
    __[more detailed usage instructions](e2e-challenge/README.md)__ directly in the 
    experiment subdirectory. These also partially apply to other datasets.

* [sfx-restaurant/](sfx-restaurant): generating from the 
    [San Francisco Restaurants dataset](https://www.repository.cam.ac.uk/handle/1810/251304)
    collected by Wen et al., EMNLP 2015. There are some 
    __[specific usage instructions](sfx-restaurant/README.md)__ directly in the experiment
    subdirectory.


You need to download the dataset into the `input/` subdirectory to start your experiments. 
From there, a `convert` script (mostly `convert.py`, a Perl script `convert.pl` for BAGEL)
can create the data formats required by TGen. Settings used in our experiments are preset in the
`Makefile`, which, however, may contain site-specific code and need some tweaking to get working.

The default configuration file for each dataset is stored in the `config/seq2seq.py` file.
This is typically the baseline variant, with improved versions requiring slight configuration 
changes.

The main experiment directory always has a basic experiment management in the `Makefile`, where 
`make help` can list the main commands. Note that some of the code in the Makefiles is also 
site-specific, especially all parts related to computing grid batch job submission, and requires 
some tweaking to get working. The code in the `Makefile` also assumes that Treex is installed.

If you need help running some of the experiments, feel free to 
[contact me](http://github.com/tuetschek).


