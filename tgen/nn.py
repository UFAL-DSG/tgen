#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural network components

"""

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import numpy as np
from tgen.rnd import rnd

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
