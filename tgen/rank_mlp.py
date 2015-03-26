#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Candidate tree rankers (NN).

"""

from __future__ import unicode_literals
import theano
import theano.tensor as T
import numpy as np

from tgen.rnd import rnd
from tgen.rank import BasePerceptronRanker
from tgen.logf import log_debug, log_info


class FeedForwardLayer(object):
    """One feed forward layer, using Theano shared variables. Can be connected to more
    inputs, i.e., use the same weights to process different inputs."""

    def __init__(self, name, n_in, n_out, activation, init='random'):

        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.inputs = []
        self.outputs = []

        # weight initialization
        if init == 'random':
            w_init = np.reshape(np.asarray([rnd.uniform(-np.sqrt(6. / (n_in + n_out)),
                                                        np.sqrt(6. / (n_in + n_out)))
                                            for _ in xrange(n_in * n_out)]),
                                newshape=(n_in, n_out))
        elif init == 'ones':
            w_init = np.ones(shape=(n_in, n_out))
        else:
            w_init = np.zeros(shape=(n_in, n_out))

        self.w = theano.shared(value=w_init, name='w-' + self.name)
        self.b = theano.shared(value=np.zeros((n_out,)), name='b-' + self.name)

        # storing parameters
        self.params = [self.w, self.b]

    def connect(self, inputs):
        # creating output function
        lin_output = T.dot(inputs, self.w) + self.b
        output = lin_output if self.activation is None else self.activation(lin_output)
        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


class FeedForwardNN(object):
    """A Theano feed forward neural network for ranking with perceptron cost function."""

    def __init__(self, sizes, activations, initializations):

        self.layers = []
        self.params = []

        # initialize + connect layers, store them & all parameters
        x = T.fvector('x')
        x_gold = T.fvector('x_gold')
        y = x
        y_gold = x_gold
        for i, (n_in, n_out, act, init) in enumerate(zip(sizes[:-1], sizes[1:],
                                                         activations, initializations)):
            layer = FeedForwardLayer('L%d' % i, n_in, n_out, act, init)
            y = layer.connect(y)
            y_gold = layer.connect(y_gold)
            self.layers.append(layer)
            self.params.extend(layer.params)

        # prediction function
        # TODO fix it so we don't need input downcast -- what is it ???
        self.score = theano.function([x], y, allow_input_downcast=True)

        # cost function
        # TODO how to implant T.max in here? Is it needed when I still decide when the update is done?
        cost = T.sum(y - y_gold)
        self.cost = theano.function([x, x_gold], cost, allow_input_downcast=True)
        grad_cost = T.grad(cost, wrt=self.params)
        self.grad_cost = theano.function([x, x_gold], grad_cost, allow_input_downcast=True)

        # training function
        updates = []
        rate = T.fscalar('rate')
        for param, grad_param in zip(self.params, grad_cost):
            updates.append((param, param - rate * grad_param))
        self.update = theano.function([x, x_gold, rate], cost, updates=updates, allow_input_downcast=True)

    def get_param_values(self):
        vals = []
        for param in self.params:
            vals.append(param.get_value())
        return vals

    def set_param_values(self, vals):
        for param, val in zip(self.params, vals):
            param.set_value(val)


class SimpleNNRanker(BasePerceptronRanker):
    """A simple ranker using a neural network on top of the usual features; using the same
    updates as the original perceptron as far as possible."""

    def __init__(self, cfg):
        super(SimpleNNRanker, self).__init__(cfg)
        self.num_hidden_units = cfg.get('num_hidden_units', 512)
        self.initialization = cfg.get('initialization', 'random')
        self.net_type = cfg.get('nn', 'linear_perc')

    def _init_training(self, das_file, ttree_file, data_portion):
        # load data, determine number of features etc. etc.
        super(SimpleNNRanker, self)._init_training(das_file, ttree_file, data_portion)

        # multi-layer perceptron with tanh + linear layer
        if self.net_type == 'mlp':
            self.nn = FeedForwardNN([self.train_feats.shape[1], self.num_hidden_units, 1],
                                    [T.tanh, None],
                                    [self.initialization, self.initialization])
        # this works as a linear perceptron
        else:
            self.nn = FeedForwardNN([self.train_feats.shape[1], 1], [None], [self.initialization])

        self.w_after_iter = []
        self.update_weights_sum()

        log_debug('\n***\nINIT:')
        log_debug(self._feat_val_str())
        log_info('Training ...')

    def _score(self, cand_feats):
        return self.nn.score(cand_feats)[0]

    def _update_weights(self, da, good_tree, bad_tree, good_feats, bad_feats):
        if self.diffing_trees:
            good_sts, bad_sts = good_tree.diffing_trees(bad_tree, symmetric=True)
            for good_st, bad_st in zip(good_sts, bad_sts):
                good_feats = self._extract_feats(good_st, da)
                bad_feats = self._extract_feats(bad_st, da)
                subtree_w = 1
                if self.diffing_trees.endswith('weighted'):
                    subtree_w = (len(good_st) + len(bad_st)) / float(len(good_tree) + len(bad_tree))
                self.nn.update(bad_feats, good_feats, subtree_w * self.alpha)
        else:
            self.nn.update(bad_feats, good_feats, self.alpha)

    def get_weights(self):
        """Return the current neural net weights."""
        return self.nn.get_param_values()

    def set_weights(self, w):
        """Set new neural network weights."""
        self.nn.set_param_values(w)

    def set_weights_average(self, wss):
        """Set the weights as the average of the given array of weights (used in parallel training)."""
        self.nn.set_param_values(np.average(wss, axis=0))

    def store_iter_weights(self):
        """Remember the current weights to be used for averaged perceptron."""
        self.w_after_iter.append(self.nn.get_param_values())

    def set_weights_iter_average(self):
        """Average the remembered weights."""
        self.nn.set_param_values(np.average(self.w_after_iter, axis=0))

    def get_weights_sum(self):
        """Return the sum of weights (at start of current iteration) to be used to weigh future
        promise."""
        return self.w_sum

    def update_weights_sum(self):
        """Update the current weights sum figure."""
        vals = self.nn.get_param_values()
        # only use the last layer for summation (w, b)
        self.w_sum = np.sum(vals[-2]) + np.sum(vals[-1])
