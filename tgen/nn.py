#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural network components

"""

import theano
import theano.compile
import theano.tensor as T
from theano.tensor.signal import downsample
import numpy as np
from tgen.rnd import rnd
import math
from numpy import int32, float32

# TODO fix
# theano.config.floatX = 'float32'  # using floats instead of doubles ??
# theano.config.profile = True
# theano.config.compute_test_value = 'warn'
# theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'

DEBUG_MODE = 0


class Layer(object):

    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []

    def get_init_weights(self, init_type, shape):
        total_size = np.prod(shape)
        dim_sum = np.sum(shape)
        if init_type == 'uniform_glorot10':
            w_init = np.reshape(np.asarray([rnd.uniform(-np.sqrt(6. / dim_sum),
                                                        np.sqrt(6. / dim_sum))
                                            for _ in xrange(total_size)]),
                                newshape=shape)
        elif init_type == 'uniform_005':
            w_init = np.reshape(np.asarray([rnd.uniform(-0.05, 0.05)
                                            for _ in xrange(total_size)]),
                                newshape=shape)
        elif init_type == 'norm_sqrt':
            w_init = np.reshape(np.asarray([rnd.gauss(0, math.sqrt(2.0 / shape[0]))
                                            for _ in xrange(total_size)]),
                                newshape=shape)
        elif init_type == 'ones':
            w_init = np.ones(shape=shape)
        else:
            w_init = np.zeros(shape=shape)
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

        output = self.e[inputs]
        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


class Conv1DLayer(Layer):

    def __init__(self, name, n_in,
                 num_filters, filter_length, stride=1,
                 border_mode='valid', bias=True, untie_bias=False,
                 init='uniform_glorot10',
                 activation=None):

        super(Conv1DLayer, self).__init__(name)
        self.activation = activation
        self.n_in = n_in  # 3D: num. positions x stack size (sub-embeddings) x embedding size
        self.num_filters = num_filters  # output "stack size" (sub-embeddings)
        self.filter_length = filter_length
        self.stride = stride
        self.border_mode = border_mode
        self.untie_bias = untie_bias

        # output shape:
        # 0] num. of positions according to convolution (1D),
        #    = ceil(n_in - filter_length + 1)
        # 1] num. of filters,
        # 2] no change in embeddings dimension
        self.n_out = ((n_in[0] - filter_length + stride) // stride, num_filters, n_in[2])

        # num. filters x stack size x num. rows x num. cols
        w_init = self.get_init_weights(init, (num_filters, n_in[1], filter_length, 1))
        self.w = theano.shared(value=w_init, name='w-' + self.name)
        if bias:
            if untie_bias:
                self.b = theano.shared(value=np.zeros(self.n_out), name='b-' + self.name)
            else:
                self.b = theano.shared(value=np.zeros(self.n_out[1:]), name='b-' + self.name)
            self.params = [self.w, self.b]
        else:
            self.b = None
            self.params = [self.w]

    @staticmethod
    def conv1d_mc0(inputs, filters, image_shape=None, filter_shape=None,
                   border_mode='valid', subsample=(1,)):
        """
        Adapted from Lasagne (https://github.com/Lasagne/Lasagne)
        """
        # dimensions: batch x words x sub-embeddings x embedding size
        # converted to: batch x stack size x num. rows x num. cols
        # (+ all filters num. cols=1, cols stride is size 1,
        #    so nothing is done with the embeddings themselves (is it??) )
        input_mc0 = inputs.dimshuffle(0, 2, 1, 3)
        # TODO image and filter shape are used for optimization
        conved = T.nnet.conv2d(input_mc0, filters, image_shape=None,
                               filter_shape=None, subsample=(subsample[0], 1),
                               border_mode=border_mode)
        return conved.dimshuffle(0, 2, 1, 3)  # shuffle the dimension back

    def connect(self, inputs):
        conved = self.conv1d_mc0(inputs, self.w, subsample=(self.stride,),
                                 image_shape=(self.n_in,),
                                 filter_shape=(self.filter_length,),
                                 border_mode=self.border_mode)
        if self.b is None:
            lin_output = conved
        else:
            if self.untie_bias:
                lin_output = conved + self.b
            else:
                lin_output = conved + self.b.dimshuffle('x', 0, 1)

        if self.activation is None:
            output = lin_output
        else:
            output = self.activation(lin_output)
        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


class MaxPool1DLayer(Layer):

    def __init__(self, name, axis=1, pooling_func=T.max):

        super(MaxPool1DLayer, self).__init__(name)

        self.pooling_func = pooling_func
        self.axis = axis

        self.params = []  # no parameters here

    def connect(self, inputs):
        output = T.max(inputs, axis=self.axis)

#         input_padded = T.shape_padright(inputs.dimshuffle(0, 2, 1), 1)
#         # do the max-pooling
#         pooled = downsample.max_pool_2d(input_padded, (self.downscale_factor, 1), False)
#         # remove the padded dimension + swap dimensions back
#         output = pooled[:, :, :, 0].dimshuffle(0, 2, 1)

        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


# TODO add ARG MAX to max pooling


class Concat(Layer):

    def __init__(self, name):

        super(Concat, self).__init__(name)
        self.params = []

    def connect(self, inputs):

        output = T.concatenate(inputs, axis=1)
        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


class DotProduct(Layer):

    def __init__(self, name):

        super(DotProduct, self).__init__(name)
        self.params = []

    def connect(self, inputs):

        output = T.batched_dot(inputs[0], inputs[1])
        self.inputs.append(inputs)
        self.outputs.append(output)
        return output


class Flatten(Layer):

    def __init__(self, name, num_dims=-1):

        super(Flatten, self).__init__(name)
        self.params = []
        self.num_dims = num_dims

    def connect(self, inputs):

        keep_dims = -self.num_dims
        if keep_dims < 0:
            keep_dims = len(inputs.shape) - keep_dims
        # keep the first keep_dims dimensions, flatten the rest
        output = inputs.reshape(T.concatenate([inputs.shape[0:keep_dims],
                                               [T.prod(inputs.shape[keep_dims:])]]),
                                ndim=(keep_dims + 1))
        return output


class NN(object):
    """A Theano neural network for ranking with perceptron cost function."""

    def __init__(self, layers, input_num=1, input_type=T.fvector, normgrad=False):

        self.layers = layers
        self.params = []
        self.normgrad = normgrad

        # connect layers, store them & all parameters
        x = [input_type('x' + str(i)) for i in xrange(input_num)]
        x_gold = [input_type('x' + str(i)) for i in xrange(input_num)]
        if input_type == T.itensor3 and len(x) == 2:
            x[0].tag.test_value = np.random.randint(0, 20, (5, 10, 2)).astype('int32')
            x_gold[0].tag.test_value = np.random.randint(0, 20, (5, 10, 2)).astype('int32')
            x[1].tag.test_value = np.random.randint(0, 20, (5, 20, 3)).astype('int32')
            x_gold[1].tag.test_value = np.random.randint(0, 20, (5, 20, 3)).astype('int32')
        y = x
        y_gold = x_gold

        for layer in layers:
            if len(layer) == len(y):
                y = [l_part.connect(y_part) for y_part, l_part in zip(y, layer)]
                y_gold = [l_part.connect(y_gold_part) for y_gold_part, l_part in zip(y_gold, layer)]
            elif len(layer) == 1:
                y = [layer[0].connect(y)]
                y_gold = [layer[0].connect(y_gold)]
            else:
                raise NotImplementedError("Only n-n and n-1 layer connections supported.")

            for l_part in layer:
                self.params.extend(l_part.params)

        # prediction function (different compilation mode for debugging and running)
        if DEBUG_MODE:
            from tgen.debug import inspect_input_dims, inspect_output_dims
            mode = theano.compile.MonitorMode(pre_func=inspect_input_dims,
                                              post_func=inspect_output_dims)  # .excluding('local_elemwise_fusion', 'inplace')
        else:
            mode = theano.compile.mode.FAST_COMPILE
        self.score = theano.function(x, y, allow_input_downcast=True,
                                     on_unused_input='warn', name='score', mode=mode)
        # print the prediction function when debugging
        if DEBUG_MODE:
            theano.printing.debugprint(self.score)

        # cost function
        # TODO how to implant T.max in here? Is it needed when I still decide when the update is done?
        cost = T.sum(y[0] - y_gold[0])  # y is a list, but should only have a length of 1 (single output)
        self.cost = theano.function(x + x_gold, cost, allow_input_downcast=True, name='cost')  # x, x_gold are lists

        grad_cost = T.grad(cost, wrt=self.params)
        # normalized gradient, if applicable (TODO fix!)
        if self.normgrad:
            grad_cost = map(lambda x: x / x.norm(2), grad_cost)

        self.grad_cost = theano.function(x + x_gold, grad_cost, allow_input_downcast=True, name='grad_cost')

        # training function
        updates = []
        rate = T.fscalar('rate')
        rate.tag.test_value = float32(0.1)
        for param, grad_param in zip(self.params, grad_cost):
            updates.append((param, param - rate * grad_param))
        self.update = theano.function(x + x_gold + [rate], [cost] + grad_cost, updates=updates, allow_input_downcast=True, name='update')

    def get_param_values(self):
        vals = []
        for param in self.params:
            vals.append(param.get_value())
        return vals

    def set_param_values(self, vals):
        for param, val in zip(self.params, vals):
            param.set_value(val)
