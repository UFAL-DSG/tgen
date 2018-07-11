Computing significance using bootstrap
======================================

Prepare:
- You have 2 or more experiments, in different directories (e.g. `A`, `B`, `C`).
- In each directory, the system output has the same name (e.g. output.txt, so you have `A/output.txt`, `B/output.txt`, `C/output.txt`).
* You have a context and reference/ground truth file (e.g. context.txt, reference.txt). These are the same for all the experiments.
* You have an (empty) directory for logging and temporary files where you will run the experiment (e.g. `bootstrap-tmp`)
* Assuming 95% confidence level

Now run:
```
./compute_bootstrap.py --level 95 bootstrap-tmp context.txt reference.txt output.txt A B C
```

Look for stuff like this:
```
A vs B BLEU: System 1 BLEU better: 1000 / 1000 = 1 -- BLEU SIGNIFICANT at 0.95 level
B vs C BLEU: System 2 BLEU better: 723 / 1000 = 0.723
```
It will tell you the p-value (here: 1) and the significance if it's above the confidence level. The first here is significant but the 2nd is not.
