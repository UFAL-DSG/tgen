
SF Restaurant dataset short HOWTO:
==============================
Assumptions:
- no usage of Treex
- full delexicalization
- EMNLP version of the dataset
- running commands from the checkout from this directory

Steps:
1. Download the data from [here](https://www.repository.cam.ac.uk/handle/1810/251304).

2. Save the SF Restaurant file as `input/sfxrestaurant.emnlp.json`.

3. Make sure you delete the copyright info at the beginning of the file (it's not valid JSON otherwise).

4. Run the data preparation script:
```
cd input
make all TOKS=1 AALL=1
cd ..
```

5. Move the prepared data over:
```
mkdir -p data
mv input/{train,devel,test}* data
```

6. Train the model and save it to model.pickle.gz:
```
../run_tgen.py seq2seq_train --random-seed "XXX" \ 
    config/seq2seq.py data/train-das.txt \
    data/train-text.txt model.pickle.gz
```

7. Run the model to generate outputs on the development data and save them to output.txt:
```
../run_tgen.py seq2seq_gen --eval-file data/devel-ref.txt  \
    --abstr-file data/devel-conc_das.txt \
    --output-file output.txt \
    model.pickle.gz data/devel-das.txt
```
  Instead of `devel-conc_das.txt` (non-delexicalized input DAs), you could use
  the file `data/devel-abst.txt` for lexicalization -- this should give the same result. 
  This file contains information about the position of the slot values in the gold standard
  file, but that information is ignored (just the slot values are used for lexicalization).
