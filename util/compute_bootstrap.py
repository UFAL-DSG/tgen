#!/usr/bin/env python
# -"- coding: utf-8 -"-


from argparse import ArgumentParser
import os.path
import re
from subprocess import call
from tgen.logf import log_info

MY_PATH = os.path.dirname(os.path.abspath(__file__))


def lcall(arg_str):
    log_info(arg_str)
    return call(arg_str, shell=True)


def get_confidence(metric, lines):
    for idx, line in enumerate(lines):
        if line.startswith(metric):
            lines = lines[idx:]
            break
    for idx, line in enumerate(lines):
        if line.startswith('Confidence of [Sys1'):
            return line.strip()
    return '???'


def process_all(args):
    join_sets = os.path.join(MY_PATH, 'join_sets.pl')
    gen_log = os.path.join(MY_PATH, 'generateLog-v11.pl')
    bootstrap = os.path.join(MY_PATH, 'bootstrapCompare-v11.2.pl')

    # create the test and source files
    lcall("%s %s/s*/test-conc.sgm > %s/test-conc.sgm" %
          (join_sets, args.experiment_dirs[0], args.target_dir))
    lcall("%s %s/s*/test-das.sgm > %s/test-das.sgm" %
          (join_sets, args.experiment_dirs[0], args.target_dir))

    exp_nums = []
    for exp_dir in args.experiment_dirs:
        exp_num = int(re.search(r'(?:^|/)([0-9]+)', exp_dir).group(1))
        exp_nums.append(exp_num)
        lcall("%s %s/s*/out-text.sgm > %s/%d.sgm" % (join_sets, exp_dir, args.target_dir, exp_num))

    os.chdir(args.target_dir)
    for exp_num in exp_nums:
        lcall("%s -s test-das.sgm -r test-conc.sgm -t %d.sgm -m 1 > %d.log.txt" %
              (gen_log, exp_num, exp_num))

    for skip, exp_num1 in enumerate(exp_nums):
        for exp_num2 in exp_nums[skip + 1:]:
            # recompute only if not done already (TODO switch for this)
            if not os.path.isfile('bootstrap.%dvs%d.txt' % (exp_num1, exp_num2)):
                lcall("perl %s %s.log.txt %s.log.txt 1000 0.99 > bootstrap.%dvs%d.txt" %
                      (bootstrap, exp_num1, exp_num2, exp_num1, exp_num2))
            with open('bootstrap.%dvs%d.txt' % (exp_num1, exp_num2)) as fh:
                bootstrap_data = fh.readlines()
                print "%dvs%d BLEU: %s" % (exp_num1, exp_num2, get_confidence('Bleu', bootstrap_data))
                print "%dvs%d NIST: %s" % (exp_num1, exp_num2, get_confidence('NIST', bootstrap_data))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('target_dir', type=str, help='Target directory for bootstrap logs')
    ap.add_argument('experiment_dirs', nargs='+', type=str, help='Experiment directories to use')
    args = ap.parse_args()

    process_all(args)
