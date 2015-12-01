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
        self.n_in = None
        self.n_out = None

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

    def __str__(self, *args, **kwargs):
        out = self.__class__.__name__
        out += ' ' + str(self.n_in) + ' -> ' + str(self.n_out)
        return out


class Embedding(Layer):

    def __init__(self, name, dict_size, width, init='uniform_005'):

        super(Embedding, self).__init__(name)

        self.width = width
        self.dict_size = dict_size

        e_init = self.get_init_weights(init, (dict_size, width))
        self.e = theano.shared(value=e_init, name='e-' + self.name)

        self.params = [self.e]

    def connect(self, in_var, n_in=None):

        if not self.n_in:
            # compute shape
            self.n_in = n_in
            self.n_out = self.n_in + [self.width]

        # create output function
        output = self.e[in_var]
        self.inputs.append(in_var)
        self.outputs.append(output)
        return output


class Identity(Layer):

    def __init__(self, name, convert_to_float=False):
        super(Identity, self).__init__(name)
        self.name = name
        self.convert_to_float = convert_to_float
        # no parameters
        self.params = []

    def connect(self, in_var, n_in=None):

        if not self.n_in:
            self.n_in = n_in
            self.n_out = self.n_in

        self.inputs.append(in_var)
        output = in_var
        if self.convert_to_float:
            output = T.cast(output, 'float32')
        self.outputs.append(output)
        return output


class FeedForward(Layer):
    """One feed forward layer, using Theano shared variables. Can be connected to more
    inputs, i.e., use the same weights to process different inputs."""

    def __init__(self, name, num_hidden_units, activation, init='uniform_glorot10'):
        super(FeedForward, self).__init__(name)

        self.name = name
        self.num_hidden_units = num_hidden_units
        self.init = init
        self.activation = activation

    def connect(self, in_var, n_in=None):

        if not self.n_in:
            # computing shape
            self.n_in = n_in
            self.n_out = [self.num_hidden_units]

            # creating parameters
            w_init = self.get_init_weights(self.init, self.n_in + self.n_out)

            self.w = theano.shared(value=w_init, name='w-' + self.name)
            self.b = theano.shared(value=np.zeros(self.n_out), name='b-' + self.name)
            self.params = [self.w, self.b]

        # creating output function
        lin_output = T.dot(in_var, self.w) + self.b
        output = lin_output if self.activation is None else self.activation(lin_output)
        self.inputs.append(in_var)
        self.outputs.append(output)
        return output


class Conv1D(Layer):

    def __init__(self, name,
                 num_filters, filter_length, stride=1,
                 border_mode='valid', bias=True, untie_bias=False,
                 init='uniform_glorot10',
                 activation=None):

        super(Conv1D, self).__init__(name)
        self.init = init
        self.activation = activation
        self.num_filters = num_filters  # output "stack size" (sub-embeddings)
        self.filter_length = filter_length
        self.stride = stride
        self.border_mode = border_mode
        self.bias = bias
        self.untie_bias = untie_bias

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

    def connect(self, in_var, n_in=None):

        if not self.n_in:
            # assuming batches + 3D: num. positions x stack size (sub-embeddings) x embedding size
            self.n_in = n_in

            # output shape:
            # 0] num. of positions according to convolution (1D),
            #    = ceil(n_in - filter_length + 1)
            # 1] num. of filters,
            # 2] no change in embeddings dimension
            self.n_out = [(self.n_in[0] - self.filter_length + self.stride) // self.stride,
                          self.num_filters,
                          self.n_in[2]]

            # create parameters
            # num. filters x stack size x num. rows x num. cols
            w_init = self.get_init_weights(self.init,
                                           (self.num_filters, self.n_in[1], self.filter_length, 1))
            self.w = theano.shared(value=w_init, name='w-' + self.name)
            if self.bias:
                if self.untie_bias:
                    self.b = theano.shared(value=np.zeros(self.n_out), name='b-' + self.name)
                else:
                    self.b = theano.shared(value=np.zeros(self.n_out[1:]), name='b-' + self.name)
                self.params = [self.w, self.b]
            else:
                self.b = None
                self.params = [self.w]

        # create output function
        conved = self.conv1d_mc0(in_var, self.w, subsample=(self.stride,),
                                 image_shape=(self.n_in,),
                                 filter_shape=(self.filter_length,),
                                 border_mode=self.border_mode)
        if not self.bias:
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
        self.inputs.append(in_var)
        self.outputs.append(output)
        return output


class Pool1D(Layer):

    def __init__(self, name, axis=1, pooling_func=T.max):

        super(Pool1D, self).__init__(name)

        self.pooling_func = pooling_func
        self.axis = axis

        self.params = []  # no parameters here

    def connect(self, in_var, n_in=None):

        if not self.n_in:
            self.n_in = n_in
            self.n_out = [dim for a, dim in enumerate(self.n_in) if a != self.axis - 1]

        output = self.pooling_func(in_var, axis=self.axis)

        self.inputs.append(in_var)
        self.outputs.append(output)
        return output
        # TODO add ARG MAX to max pooling


class Flatten(Layer):

    def __init__(self, name, keep_dims=1):

        super(Flatten, self).__init__(name)
        self.params = []
        self.keep_dims = keep_dims

    def connect(self, in_var, n_in=None):

        # compute output dimensions
        if not self.n_in:
            self.n_in = n_in
            # NB: we actually have 1 dimension less here (batch size will be variable)
            self.n_out = self.n_in[0:self.keep_dims - 1] + [np.prod(self.n_in[self.keep_dims - 1:])]

        # keep the first keep_dims dimensions, flatten the rest
        output = in_var.reshape(T.concatenate([in_var.shape[0:self.keep_dims],
                                               [T.prod(in_var.shape[self.keep_dims:])]]),
                                ndim=(self.keep_dims + 1))
        self.inputs.append(in_var)
        self.outputs.append(output)
        return output


class Concat(Layer):

    def __init__(self, name, axis=1):

        super(Concat, self).__init__(name)
        self.params = []
        self.axis = axis

    def connect(self, in_vars, n_in=None):

        if not self.n_in:
            self.n_in = n_in
            self.n_out = self.n_in[0][:]
            # NB: we actually have 1 dimension less here (batch size will be variable)
            self.n_out[self.axis - 1] = sum(ni[self.axis - 1] for ni in self.n_in)

        output = T.concatenate(in_vars, axis=self.axis)

        self.inputs.append(in_vars)
        self.outputs.append(output)
        return output


class DotProduct(Layer):

    def __init__(self, name):

        super(DotProduct, self).__init__(name)
        self.params = []

    def connect(self, in_vars, n_in=None):

        if not self.n_in:
            # NB: we actually have 1 dimension less here (batch size will be variable)
            self.n_in = n_in
            assert len(self.n_in) == 2 and len(self.n_in[0] == 2) and len(self.n_in[1] == 2)
            self.n_out = [self.n_in[0][0], self.n_in[1][1]]

        output = T.batched_dot(in_vars)
        self.n_out = output.shape

        self.inputs.append(in_vars)
        self.outputs.append(output)
        return output


class NN(object):

    def __init__(self, layers, input_shapes, input_types=(T.fvector,), normgrad=False):

        self.layers = layers
        self.input_shapes = input_shapes
        self.input_types = input_types
        self.params = []
        self.normgrad = normgrad

    def get_param_values(self):
        vals = []
        for param in self.params:
            vals.append(param.get_value())
        return vals

    def set_param_values(self, vals):
        for param, val in zip(self.params, vals):
            param.set_value(val)

    def __str__(self, *args, **kwargs):
        out = ''
        for l_num, layer in enumerate(self.layers):
            out += str(l_num) + ': '
            out += ', '.join(str(li) for li in layer)
            out += "\n"
        return out

    def connect_layer(self, layer, y, shapes=None):

        if len(layer) == len(y):
            if shapes is not None:
                y = [l_i.connect(y_i, shape) for l_i, y_i, shape in zip(layer, y, shapes)]
            else:
                y = [l_i.connect(y_i) for l_i, y_i in zip(layer, y)]
        elif len(layer) == 1:
            y = [layer[0].connect(y, shapes)]
        else:
            raise NotImplementedError("Only n-n and n-1 layer connections supported.")

        if shapes is not None:
            shapes = [l_i.n_out for l_i in layer]
            for l_i in layer:  # remember parameters for gradient
                self.params.extend(l_i.params)
            return y, shapes

        return y


class RankNN(NN):
    """A Theano neural network for ranking with perceptron cost function."""

    def __init__(self, layers, input_shapes, input_types=(T.fvector,), normgrad=False):
        """Build the neural network.

        @param layers: The layers of the network, to be connected
        @param input_shapes: Shapes of the input, minus the 1st dimension that will be used \
            for (variable-sized) batches
        @param input_types: Theano tensor types for the input (including the batch dimension)
        @param normgrad: Use normalized gradients?
        """
        super(RankNN, self).__init__(layers, input_shapes, input_types, normgrad)

        # create variables
        x = [input_types[i]('x' + str(i)) for i in xrange(len(layers[0]))]
        x_gold = [input_types[i]('x' + str(i)) for i in xrange(len(layers[0]))]

        # TODO: make this depend on input_shapes
        # Debugging: test values
        if input_types[1] == T.itensor3 and len(x) == 2:
            if input_types[0] == T.fmatrix:
                x[0].tag.test_value = np.random.randint(0, 2, (5, 11)).astype('float32')
                x_gold[0].tag.test_value = np.random.randint(0, 2, (5, 11)).astype('float32')
            else:
                x[0].tag.test_value = np.random.randint(0, 20, (5, 10, 2)).astype('int32')
                x_gold[0].tag.test_value = np.random.randint(0, 20, (5, 10, 2)).astype('int32')
            x[1].tag.test_value = np.random.randint(0, 20, (5, 20, 3)).astype('int32')
            x_gold[1].tag.test_value = np.random.randint(0, 20, (5, 20, 3)).astype('int32')

        y = x
        y_gold = x_gold
        shapes = input_shapes

        # connect all layers
        for layer in layers:
            y, out_shapes = self.connect_layer(layer, y, shapes)
            y_gold = self.connect_layer(layer, y_gold)
            shapes = out_shapes

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


class ClassifNN(NN):
    """A Theano neural network for classification with cross-entropy cost function."""

    def __init__(self, layers, input_shapes, input_types=(T.fvector,), normgrad=False):
        """Build the neural network.

        @param layers: The layers of the network, to be connected
        @param input_shapes: Shapes of the input, minus the 1st dimension that will be used \
            for (variable-sized) batches
        @param input_types: Theano tensor types for the input (including the batch dimension)
        @param normgrad: Use normalized gradients?
        """
        super(ClassifNN, self).__init__(layers, input_shapes, input_types, normgrad)

        # create variables
        x = [input_types[i]('x' + str(i)) for i in xrange(len(layers[0]))]
        y = x
        shapes = input_shapes

        # connect all layers
        for layer in layers:
            y, shapes = self.connect_layer(layer, y, shapes)

        # prediction function (different compilation mode for debugging and running)
        if DEBUG_MODE:
            from tgen.debug import inspect_input_dims, inspect_output_dims
            mode = theano.compile.MonitorMode(pre_func=inspect_input_dims,
                                              post_func=inspect_output_dims)  # .excluding('local_elemwise_fusion', 'inplace')
        else:
            mode = theano.compile.mode.FAST_COMPILE
        self.classif = theano.function(x, y[0], allow_input_downcast=True,
                                       on_unused_input='warn', name='classif', mode=mode)
        # print the prediction function when debugging
        if DEBUG_MODE:
            theano.printing.debugprint(self.classif)

        y_gold = T.fmatrix(name='y_gold')
        # cross-entropy cost function
        # (=negative log likelihood)
        cost = -T.mean(T.sum(y_gold * T.log(y[0]) + (1 - y_gold) * T.log(1 - y[0]), axis=1))
        self.cost = theano.function(x + [y_gold], cost, allow_input_downcast=True, name='cost')  # x is a list

        grad_cost = T.grad(cost, wrt=self.params)
        # normalized gradient, if applicable (TODO fix!)
        if self.normgrad:
            grad_cost = map(lambda x: x / x.norm(2), grad_cost)

        self.grad_cost = theano.function(x + [y_gold], grad_cost, allow_input_downcast=True, name='grad_cost')

        # training function
        updates = []
        rate = T.fscalar('rate')
        rate.tag.test_value = float32(0.1)
        for param, grad_param in zip(self.params, grad_cost):
            updates.append((param, param - rate * grad_param))
        self.update = theano.function(x + [y_gold, rate], [cost] + grad_cost, updates=updates, allow_input_downcast=True, name='update')
