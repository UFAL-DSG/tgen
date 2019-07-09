#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import codecs
import yaml
import re

def main(args):

    iters, training_set, run_setting, nn_shape = '', '', '', ''

    with codecs.open(args.config_file, 'r', 'UTF-8') as fh:
        cfg = yaml.load(fh)

    iters = str(cfg.get('passes', '~'))
    iters += '/' + str(cfg.get('batch_size', '~'))
    iters += '/' + (('%.e' % cfg['alpha']).replace('e-0', 'e-') if 'alpha' in cfg else '~')
    if cfg.get('alpha_decay'):
        iters += '^' + str(cfg['alpha_decay'])

    if cfg.get('validation_size'):
        iters = str(cfg.get('min_passes', 1)) + '-' + iters
        iters += ' V' + str(cfg['validation_size'])
        iters += '+A' if cfg.get('validation_use_all_refs') else ''
        iters += '-O' if cfg.get('validation_no_overlap') else ''
        iters += '+T' if (cfg.get('validation_use_all_refs') and
                          cfg.get('validation_use_train_refs') and
                          not cfg.get('validation_no_overlap')) else ''
        iters += '@' + str(cfg.get('validation_freq', 10))
        iters += ' I' + str(cfg.get('improve_interval', 10))
        iters += '@' + str(cfg.get('top_k', 5))
        if cfg.get('bleu_validation_weight'):
            iters += ' B%.2g' % cfg['bleu_validation_weight']

    training_set = args.training_set
    training_set = re.sub('^training-', '', training_set)
    if args.eval_data:
        training_set = "\x1B[97;41mE\x1B[0m " + training_set
    if args.train_portion < 1.0:
        training_set += '/' + str(args.training_portion)

    # gadgets
    nn_shape += ' +lc' if cfg.get('embeddings_lowercase') else ''
    nn_shape += ' E' + str(cfg.get('emb_size', 50))
    nn_shape += '-N' + str(cfg.get('num_hidden_units', 128))
    if 'dropout_keep_prob' in cfg:
        nn_shape += '-D' + str(cfg['dropout_keep_prob'])
    if 'beam_size' in cfg:
        nn_shape += '-B' + str(cfg['beam_size'])

    nn_shape += ' ' + cfg.get('cell_type', 'lstm')

    nn_shape += ' +bidi +att'  if cfg.get('nn_type') == 'emb_bidi_attention_seq2seq' else ''
    nn_shape += ' +att'  if cfg.get('nn_type') in ['emb_attention_seq2seq', 'emb_attention_seq2seq_context'] else ''
    nn_shape += ' +att2'  if cfg.get('nn_type') in ['emb_attention2_seq2seq', 'emb_attention2_seq2seq_context'] else ''
    nn_shape += ' +sort'  if cfg.get('sort_da_emb') else ''
    nn_shape += ' +adgr'  if cfg.get('optimizer_type') == 'adagrad' else ''
    nn_shape += ' +dc'  if cfg.get('use_dec_cost') else ''
    if cfg.get('use_tokens') or cfg.get('mode') == 'tokens':
        nn_shape += ' ->tok'
    elif cfg.get('mode') == 'tagged_lemmas':
        nn_shape += ' ->tls'
    else:
        nn_shape += ' ->tre'

    if cfg.get('classif_filter'):
        nn_shape += ' +cf'
        cf_cfg = cfg['classif_filter']
        if cf_cfg.get('model') == 'e2e_patterns':
            nn_shape += '_e2e-pat'
        else:
            nn_shape += '_' + cf_cfg.get('nn_shape') + '_'
            if 'min_passes' in cf_cfg:
                nn_shape += str(cf_cfg['min_passes']) + '-'
            nn_shape += str(cf_cfg.get('passes', '~'))
            nn_shape += '/' + str(cf_cfg.get('batch_size', '~'))
            nn_shape += '/' + str(cf_cfg.get('alpha', '~'))

            nn_shape += '_'
            if 'rnn' in cf_cfg.get('nn_shape'):
                nn_shape += 'E' + str(cf_cfg.get('emb_size', 50))
            nn_shape += '-N' + str(cf_cfg.get('num_hidden_units', 128))

    if cfg.get('lexicalizer'):
        nn_shape += ' +lx'
        lx_cfg = cfg['lexicalizer']

        nn_shape += '-' + lx_cfg.get('form_select_type', 'random')
        nn_shape += '-bidi' if lx_cfg.get('form_select_type') == 'rnnlm' and lx_cfg.get('bidi') else ''
        nn_shape += '+samp' if lx_cfg.get('form_sample') else ''

    if args.cv_runs:
        num_runs = len(args.cv_runs.split())
        run_setting += ' ' + str(num_runs) + 'CV'
    if args.debug:
        run_setting += ' DEBUG'
    if args.rands:
        run_setting += ' RANDS' + str(args.rands)

    run_setting = run_setting.strip()
    run_setting = run_setting.replace(' ', ',')
    run_setting = ' (' + run_setting + ')' if run_setting else ''

    print(training_set + ' ' + iters + nn_shape + run_setting, end='')




if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('-t', '--training-set', '--training', required=True, type=str, help='Training set name')
    ap.add_argument('-d', '--debug', action='store_true', help='Are we running with debug prints?')
    ap.add_argument('-c', '--cv-runs', '--cv', type=str, help='Number of CV runs used')
    ap.add_argument('-r', '--rands', type=int, const=5, nargs='?', help='Are we using more random inits? Defaults to 5 if no value is specified')
    ap.add_argument('-p', '--train-portion', '--portion', type=float, help='Training data portion used', default=1.0)
    ap.add_argument('-e', '--eval-data', '--eval', action='store_true', help='Using evaluation data')
    ap.add_argument('config_file', type=str, help='Experiment YAML config file')

    args = ap.parse_args()
    main(args)
