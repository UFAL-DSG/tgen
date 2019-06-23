#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from argparse import ArgumentParser
import yaml
import codecs


def recursive_update(orig, update):
    if not (isinstance(orig, dict) and isinstance(update, dict)):
        raise TypeError("Need dictionaries for recursive update.")

    for key in update:
        if isinstance(update[key], dict) and isinstance(orig.get(key), dict):
            orig[key] = recursive_update(orig[key], update[key])
        else:
            orig[key] = update[key]
    return orig


def main(config_file, values):
    with codecs.open(config_file, 'r', 'UTF-8') as fh:
        cfg = yaml.load(fh)
    cfg = recursive_update(cfg, yaml.load(values))
    with codecs.open(config_file, 'w', 'UTF-8') as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    ap = ArgumentParser(description='A simple tool to modify YAML config files from the command line')
    ap.add_argument('config_file', type=str, help='YAML config file, to be modified in-place')
    ap.add_argument('values', type=str, help='YAML values to replace in the config file')
    args = ap.parse_args()
    main(args.config_file, args.values)
