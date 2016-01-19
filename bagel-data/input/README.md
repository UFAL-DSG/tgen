Preparing BAGEL data
====================

To prepare the data for cross-validation, similar to what Mairesse et al. (2010) did, run:

```
    make cv
```

By default, only one paraphrase per input DA is used (i.e., 202 training instances). To use 
both paraphrases, use:

```
    make cv FULL_TRAINING=1
```

There are various parameters to control the shape of the trees:

- `ABSTRACTED`: list of slots to be abstracted (their values will be converted to "X")
    - defaults to what Mairesse et al. (2010) did

- `CONVERT_SETTINGS`: additional conversion settings (control the `convert.pl` script)
    - `-a` -- list of slots to be abstracted (uses `$(ABSTRACTED)` by default)
    - `-s` -- "skip unabstracted", i.e. doesn't abstract what is not abstracted in the original set of 
      Mairesse et al. (2010) (this is the default)
    - `-j` -- join repeated slot values (collapse coordinations into the values)

- `COORD`: coordination/tree structure adjustments
    - `stanford` -- Stanford-style coordination
    - `delete` -- delete coordination nodes, keep just their children
    - `flat` -- flatten coordinations
    - `flat_trees` -- flatten whole trees, ignore structure completely

