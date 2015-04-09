#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Candidate tree rankers (NN).

"""

from __future__ import unicode_literals
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import numpy as np

from tgen.rnd import rnd
from tgen.rank import BasePerceptronRanker, FeaturesPerceptronRanker
from tgen.logf import log_debug, log_info


# TODO fix
# theano.config.floatX = 'float32'  # using floats instead of doubles ??

class Layer(object):

    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []

    def get_init_weights(self, init_type, shape):
        rows, cols = shape
        if init_type == 'uniform_glorot10':
            w_init = np.reshape(np.asarray([rnd.uniform(-np.sqrt(6. / (rows + cols)),
                                                        np.sqrt(6. / (rows + cols)))
                                            for _ in xrange(rows * cols)]),
                                newshape=(rows, cols))
        elif init_type == 'uniform_005':
            w_init = np.reshape(np.asarray([rnd.uniform(-0.05, 0.05)
                                            for _ in xrange(rows * cols)]),
                                newshape=(rows, cols))
        elif init_type == 'ones':
            w_init = np.ones(shape=(rows, cols))
        else:
            w_init = np.zeros(shape=(rows, cols))
        return w_init


class FeedForwardLayer(Layer):
    """One feed forward layer, using Theano shared variables. Can be connected to more
    inputs, i.e., use the same weights to process different inputs."""

    def __init__(self, name, n_in, n_out, activation, init='uniform_glorot10'):

        super(FeedForwardLayer, self).__init__(name)

        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        w_init = self.get_init_weights(init, (self.n_in, self.n_out))

        self.w = theano.shared(value=w_init, name='w-' + self.name)
        self.b = theano.shared(value=np.zeros((self.n_out,)), name='b-' + self.name)

        # storing parameters
        self.params = [self.w, self.b]

    def connect(self, inputs):
        # creating output function
        lin_output = T.dot(inputs, self.w) + self.b
        output = lin_output if self.activation is None else self.activation(lin_output)
        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


class Embedding(Layer):

    def __init__(self, name, dict_size, width, init='uniform_005'):

        super(Embedding, self).__init__(name)

        self.width = width
        self.dict_size = dict_size

        e_init = self.get_init_weights(init, (dict_size, width))
        self.e = theano.shared(value=e_init, name='e-' + self.name)

        self.params = [self.e]

    def connect(self, inputs):

        output = self.e[inputs].reshape((inputs.shape[0], self.width))
        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


class Conv1DLayer(Layer):

    def __init__(self, name, n_in,
                 filter_length, stride,
                 border_mode='valid', bias=True,
                 init='uniform_glorot10',
                 activation=None):

        super(Conv1DLayer, self).__init__(name)
        self.activation = activation
        self.n_in = n_in
        # self.num_filters = num_filters
        self.filter_length = filter_length
        self.stride = stride
        self.border_mode = border_mode

        # output length
        # = ceil(n_in - filter_length + 1)
        self.n_out = (n_in - filter_length + stride) // stride

        w_init = self.get_init_weights(init, (self.n_in, self.n_out))
        self.w = theano.shared(value=w_init, name='w-' + self.name)
        if bias:
            self.b = theano.shared(value=np.zeros((self.n_out,)), name='b-' + self.name)
            self.params = [self.w, self.b]
        else:
            self.b = None
            self.params = [self.w]

    @staticmethod
    def conv1d_mc0(inputs, filters, image_shape=None, filter_shape=None,
                   border_mode='valid', subsample=(1,)):
        """
        using conv2d with width == 1
        """
        input_mc0 = inputs.dimshuffle(0, 1, 'x', 2)
        filters_mc0 = filters.dimshuffle(0, 1, 'x', 2)
        # TODO image and filter shape are used for optimization
        conved = T.nnet.conv2d(input_mc0, filters_mc0, image_shape=None,
                               filter_shape=None, subsample=(1, subsample[0]),
                               border_mode=border_mode)
        return conved[:, :, 0, :]  # drop the unused dimension

    def connect(self, inputs):
        conved = self.convolution(inputs, self.W, subsample=(self.stride,),
                                  image_shape=(self.n_in,),
                                  filter_shape=(self.filter_length,),
                                  border_mode=self.border_mode)
        if self.b is None:
            lin_output = conved
            lin_output = conved + self.b

        if self.activation is None:
            output = lin_output
        else:
            output = self.activation(lin_output)
        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


class MaxPool1DLayer(Layer):

    def __init__(self, name, downscale_factor, ignore_border=False):

        super(MaxPool1DLayer, self).__init__(name)

        self.dowscale_factor = downscale_factor  # an integer
        self.ignore_border = ignore_border

        self.params = []  # no parameters here

    def connect(self, inputs):
        # pad one more dimension that we won't use
        input_padded = T.shape_padright(inputs, 1)
        # do the max-pooling
        pooled = downsample.max_pool_2d(input_padded, (self.downscale_factor, 1), self.ignore_border)
        # remove the padded dimension
        output = pooled[:, :, :, 0]
        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


class ConcatLayer(Layer):

    def __init__(self, name):

        super(ConcatLayer, self).__init__(name)
        self.params = []

    def connect(self, inputs):

        output = T.concatenate(inputs, axis=0)
        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


class NN(object):
    """A Theano neural network for ranking with perceptron cost function."""

    def __init__(self, layers, input_num=1, input_type=T.fvector):

        self.layers = layers
        self.params = []

        # connect layers, store them & all parameters
        x = [input_type('x' + str(i)) for i in xrange(input_num)]
        x_gold = [input_type('x' + str(i)) for i in xrange(input_num)]
        y = x
        y_gold = x_gold

        for layer in layers:
            if len(layer) == len(y):
                y = [l_part.connect(y_part) for y_part, l_part in zip(y, layer)]
                y_gold = [l_part.connect(y_gold_part) for y_gold_part, l_part in zip(y, layer)]
            elif len(layer) == 1:
                y = [layer[0].connect(y)]
                y_gold = [layer[0].connect(y_gold)]
            else:
                raise NotImplementedError("Only n-n and n-1 layer connections supported.")
            self.params.extend([l_part.params for l_part in layer])

        # prediction function
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


class SimpleNNRanker(FeaturesPerceptronRanker):
    """A simple ranker using a neural network on top of the usual features; using the same
    updates as the original perceptron as far as possible."""

    def __init__(self, cfg):
        super(SimpleNNRanker, self).__init__(cfg)
        self.num_hidden_units = cfg.get('num_hidden_units', 512)
        self.initialization = cfg.get('initialization', 'uniform_glorot10')
        self.net_type = cfg.get('nn', 'linear_perc')

    def _init_training(self, das_file, ttree_file, data_portion):
        # load data, determine number of features etc. etc.
        super(SimpleNNRanker, self)._init_training(das_file, ttree_file, data_portion)

        self._init_neural_network()

        self.w_after_iter = []
        self.update_weights_sum()

        log_debug('\n***\nINIT:')
        log_debug(self._feat_val_str())
        log_info('Training ...')

    def _score(self, cand_feats):
        return self.nn.score(cand_feats)[0]

    def _init_neural_network(self):
        # multi-layer perceptron with tanh + linear layer
        if self.net_type == 'mlp':
            self.nn = NN([[FeedForwardLayer('hidden', self.train_feats.shape[1], self.num_hidden_units,
                                            T.tanh, self.initialization)],
                          [FeedForwardLayer('output', self.num_hidden_units, 1,
                                            None, self.initialization)]])
        # linear perceptron
        else:
            self.nn = NN([[FeedForwardLayer('perc', self.train_feats.shape[1], 1,
                                            None, self.initialization)]])

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

#     def __getstate__(self):
#         state = dict(self.__dict__)
#         w = self.nn.get_param_values()
#         del state['nn']
#         state['w'] = w
#         return state
#
#     def __setstate__(self, state):
#         if 'w' in state:
#             w = state['w']
#             del state['w']
#         self.__dict__.update(state)
#         if 'w' in state:
#             self._init_neural_network()
#             self.set_weights(w)


class EmbNNRanker(BasePerceptronRanker):

    UNK_SLOT = 0
    UNK_VALUE = 1
    UNK_LEMMA = 2
    UNK_FORMEME = 3
    MIN_VALID = 4

    def __init__(self, cfg):
        super(EmbNNRanker, self).__init__(cfg)
        self.emb_size = cfg.get('emb_size', 20)

        self.dict_slot = {'UNK_SLOT': self.UNK_SLOT}
        self.dict_value = {'UNK_VALUE': self.UNK_VALUE}
        self.dict_lemma = {'UNK_LEMMA': self.UNK_LEMMA}
        self.dict_formeme = {'UNK_FORMEME': self.UNK_FORMEME}

        self.max_da_len = cfg.get('max_da_len', 10)
        self.max_tree_len = cfg.get('max_tree_len', 20)

    def _init_training(self, das_file, ttree_file, data_portion):
        super(EmbNNRanker, self)._init_training(das_file, ttree_file, data_portion)
        self._init_dict()
        self._init_neural_network()

        self.w_after_iter = []
        self.update_weights_sum()

    def _init_dict(self):
        """Initialize word -> integer dictionaries, starting from a minimum
        valid value, always adding a new integer to unknown values to prevent
        clashes among different types of inputs."""
        dict_ord = self.MIN_VALID

        for da in self.train_das:
            for dai in da:
                if dai.name not in self.dict_slot:
                    self.dict_slot[dai.name] = dict_ord
                    dict_ord += 1
                if dai.value not in self.dict_value:
                    self.dict_value[dai.value] = dict_ord
                    dict_ord += 1

        for tree in self.train_trees:
            for lemma, formeme in tree.nodes:
                if lemma not in self.dict_lemma:
                    self.dict_lemma[lemma] = dict_ord
                    dict_ord += 1
                if formeme not in self.dict_formeme:
                    self.dict_formeme[formeme] = dict_ord
                    dict_ord += 1

        self.dict_size = dict_ord

    def _extract_feats(self, tree, da):

        # DA embeddings
        da_emb_idxs = []
        for dai in da[:self.max_da_len]:
            da_emb_idxs.append(self.dict_slot.get(dai.name, self.UNK_SLOT))

        # pad with "unknown"
        for _ in xrange(len(da_emb_idxs), self.max_da_len):
            da_emb_idxs.extend([self.UNK_SLOT, self.UNK_VALUE])

        # tree embeddings
        tree_emb_idxs = []
        for parent_ord, (lemma, formeme) in zip(tree.parents[1:self.max_tree_len + 1],
                                                tree.nodes[1:self.max_tree_len + 1]):
            tree_emb_idxs.append(self.dict_lemma.get(tree.nodes[parent_ord].lemma, self.UNK_LEMMA))
            tree_emb_idxs.append(self.dict_formeme.get(formeme, self.UNK_FORMEME))
            tree_emb_idxs.append(self.dict_lemma.get(lemma, self.UNK_LEMMA))

        # pad with unknown
        for _ in xrange(len(tree_emb_idxs), self.max_tree_len):
            tree_emb_idxs.extend([self.UNK_LEMMA, self.UNK_FORMEME, self.UNK_LEMMA])

        return (da_emb_idxs, tree_emb_idxs)

    def _init_neural_network(self):
        self.nn = NN([[Embedding('emb_das', self.dict_size, self.emb_size, 'uniform_005'),
                       Embedding('emb_trees', self.dict_size, self.emb_size, 'uniform_005')],
                      [MaxPool1DLayer('mp_das', self.max_da_len),
                       MaxPool1DLayer('mp_trees', self.max_tree_len)],
                      [ConcatLayer('concat')],
                      [FeedForwardLayer('ff1', self.emb_size * 2, self.num_hidden_units,
                                        T.tanh, self.initialization)],
                      [FeedForwardLayer('ff2', self.num_hidden_units, self.num_hidden_units,
                                        T.tanh, self.initialization)],
                      [FeedForwardLayer('perc', self.num_hidden_units, 1, None, self.initialization)]])

