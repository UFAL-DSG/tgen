#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Candidate tree rankers (NN).

"""

from __future__ import unicode_literals

import theano.tensor as T
import numpy as np

from tgen.nn import FeedForward, Concat, Flatten, Pool1D, Embedding, RankNN, DotProduct, \
    Conv1D, Identity
from tgen.rank import BasePerceptronRanker, FeaturesPerceptronRanker
from tgen.logf import log_debug, log_info
from tgen.features import Features
from tgen.ml import DictVectorizer
from tgen.embeddings import TreeEmbeddingExtract, DAEmbeddingExtract


class NNRanker(BasePerceptronRanker):
    """Abstract ancestor of NN rankers."""

    def __init__(self, cfg):
        super(NNRanker, self).__init__(cfg)
        self.num_hidden_units = cfg.get('num_hidden_units', 512)
        self.init = cfg.get('initialization', 'uniform_glorot10')

    def store_iter_weights(self):
        """Remember the current weights to be used for averaged perceptron."""
        self.w_after_iter.append(self.nn.get_param_values())

    def update_weights_sum(self):
        """Update the current weights sum figure."""
        vals = self.nn.get_param_values()
        # only use the last layer for summation (w, b)
        self.w_sum = np.sum(vals[-2]) + np.sum(vals[-1])

    def get_weights_sum(self):
        """Return the sum of weights (at start of current iteration) to be used to weigh future
        promise."""
        return self.w_sum

    def get_weights(self):
        """Return the current neural net weights."""
        return self.nn.get_param_values()

    def set_weights(self, w):
        """Set new neural network weights."""
        self.nn.set_param_values(w)

    def set_weights_average(self, wss):
        """Set the weights as the average of the given array of weights (used in parallel training)."""
        self.nn.set_param_values(np.average(wss, axis=0))

    def set_weights_iter_average(self):
        """Average the remembered weights."""
        self.nn.set_param_values(np.average(self.w_after_iter, axis=0))

    def _update_weights(self, good, bad):
        """Update NN weights, given a good and a bad data instance (DA, tree, features)."""
        if self.diffing_trees:
            good_sts, bad_sts = good.tree.diffing_trees(bad.tree, symmetric=True)
            for good_st, bad_st in zip(good_sts, bad_sts):
                good_feats = self._extract_feats(good_st, good.da)
                bad_feats = self._extract_feats(bad_st, bad.da)
                subtree_w = 1
                if self.diffing_trees.endswith('weighted'):
                    subtree_w = (len(good_st) + len(bad_st)) / float(len(good.tree) + len(bad.tree))
                self._update_nn(bad_feats, good_feats, subtree_w * self.alpha)
        else:
            self._update_nn(bad.feats, good.feats, self.alpha)

    def _update_nn(self, bad_feats, good_feats, rate):
        """Direct call to NN weights update."""
        self.nn.update(bad_feats, good_feats, rate)

    def _ff_layers(self, name, num_layers, perc_layer=False):
        ret = []
        for i in xrange(num_layers):
            ret.append([FeedForward(name + str(i + 1), self.num_hidden_units, T.tanh, self.init)])
        if perc_layer:
            ret.append([FeedForward('perc', 1, None, self.init)])
        return ret


class SimpleNNRanker(FeaturesPerceptronRanker, NNRanker):
    """A simple ranker using a neural network on top of the usual features; using the same
    updates as the original perceptron as far as possible."""

    def __init__(self, cfg):
        super(SimpleNNRanker, self).__init__(cfg)
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
        # TODO make number of layers configurable
        if self.net_type == 'mlp':
            layers = self._ff_layers('ff', 3, perc_layer=True)
        # linear perceptron
        else:
            layers = self._ff_layers('ff', 0, perc_layer=True)

        num_features = len(self.vectorizer.get_feature_names())
        self.nn = RankNN(layers, [num_features], (T.fmatrix,), normgrad=False)


class EmbNNRanker(NNRanker):
    """A ranker using MR and tree embeddings in a NN."""

    def __init__(self, cfg):
        super(EmbNNRanker, self).__init__(cfg)
        self.emb_size = cfg.get('emb_size', 20)
        self.nn_shape = cfg.get('nn_shape', 'ff')
        self.normgrad = cfg.get('normgrad', False)

        self.cnn_num_filters = cfg.get('cnn_num_filters', 3)
        self.cnn_filter_length = cfg.get('cnn_filter_length', 3)

        # 'emb' = embeddings for both, 'emb_trees' = embeddings for tree only, 1-hot DA
        # 'emb_tree', 'emb_prev' = tree-only embeddings
        self.da_embs = cfg.get('nn', 'emb') == 'emb'

        self.tree_embs = TreeEmbeddingExtract(cfg)

        if self.da_embs:
            self.da_embs = DAEmbeddingExtract(cfg)
        else:
            self.da_feats = Features(['dat: dat_presence', 'svp: svp_presence'])
            self.vectorizer = None

    def _init_training(self, das_file, ttree_file, data_portion):
        super(EmbNNRanker, self)._init_training(das_file, ttree_file, data_portion)
        self._init_dict()
        self._init_neural_network()

        self.train_feats = [self._extract_feats(tree, da)
                            for tree, da in zip(self.train_trees, self.train_das)]

        self.w_after_iter = []
        self.update_weights_sum()

    def _init_dict(self):
        """Initialize word -> integer dictionaries, starting from a minimum
        valid value, always adding a new integer to unknown values to prevent
        clashes among different types of inputs."""

        # avoid dictionary clashes between DAs and tree embeddings
        # – remember current highest index number
        dict_ord = None

        # DA embeddings
        if self.da_embs:
            dict_ord = self.da_embs.init_dict(self.train_das)

        # DA one-hot representation
        else:
            X = []
            for da, tree in zip(self.train_das, self.train_trees):
                X.append(self.da_feats.get_features(tree, {'da': da}))

            self.vectorizer = DictVectorizer(sparse=False, binarize_numeric=True)
            self.vectorizer.fit(X)

        # tree embeddings
        # remember last dictionary key to initialize embeddings with enough rows
        self.dict_size = self.tree_embs.init_dict(self.train_trees, dict_ord)

    def _score(self, cand_embs):
        return self.nn.score([cand_embs[0]], [cand_embs[1]])[0]

    def _extract_feats(self, tree, da):
        """Extract DA and tree embeddings (return as a pair)."""
        if self.da_embs:
            # DA embeddings
            da_repr = self.da_embs.get_embeddings(da)
        else:
            # DA one-hot representation
            da_repr = self.vectorizer.transform([self.da_feats.get_features(tree, {'da': da})])[0]

        # tree embeddings
        tree_emb_idxs = self.tree_embs.get_embeddings(tree)

        return (da_repr, tree_emb_idxs)

    def _init_neural_network(self):
        # initial layer – tree embeddings & DA 1-hot or embeddings
        # input shapes don't contain the batch dimension, but the input Theano types do!
        if self.da_embs:
            input_shapes = (self.da_embs.get_embeddings_shape(),
                            self.tree_embs.get_embeddings_shape())
            input_types = (T.itensor3, T.itensor3)
            layers = [[Embedding('emb_da', self.dict_size, self.emb_size, 'uniform_005'),
                       Embedding('emb_tree', self.dict_size, self.emb_size, 'uniform_005')]]
        else:
            input_shapes = ([len(self.vectorizer.get_feature_names())],
                            self.tree_embs.get_embeddings_shape())
            input_types = (T.fmatrix, T.itensor3)
            layers = [[Identity('id_da'),
                       Embedding('emb_tree', self.dict_size, self.emb_size, 'uniform_005')]]

        # plain feed-forward networks
        if self.nn_shape.startswith('ff'):

            layers += [[Flatten('flat_da'), Flatten('flat_tree')], [Concat('concat')]]
            num_ff_layers = 2
            if self.nn_shape[-1] in ['3', '4']:
                num_ff_layers = int(self.nn_shape[-1])
            layers += self._ff_layers('ff', num_ff_layers, perc_layer=True)

        # convolution with or without max/avg-pooling
        elif self.nn_shape.startswith('conv'):

            num_conv_layers = 2 if self.nn_shape.startswith('conv2') else 1
            pooling = None
            if 'maxpool' in self.nn_shape:
                pooling = T.max
            elif 'avgpool' in self.nn_shape:
                pooling = T.mean

            if self.da_embs:
                da_layers = self._conv_layers('conv_da', num_conv_layers, pooling=pooling)
            else:
                da_layers = self._id_layers('id_da',
                                            num_conv_layers + (1 if pooling is not None else 0))
            tree_layers = self._conv_layers('conv_tree', num_conv_layers, pooling=pooling)

            for da_layer, tree_layer in zip(da_layers, tree_layers):
                layers.append([da_layer[0], tree_layer[0]])
            layers += [[Flatten('flat_da'), Flatten('flat_tree')], [Concat('concat')]]
            layers += self._ff_layers('ff', 2, perc_layer=True)

        # max-pooling without convolution
        elif 'maxpool-ff' in self.nn_shape:
            layers += [[Pool1D('mp_da') if self.da_embs else Identity('id_da'),
                        Pool1D('mp_trees')]
                       [Concat('concat')], [Flatten('flat')]]
            layers += self._ff_layers('ff', 2, perc_layer=True),

        # dot-product FF network
        elif 'dot' in self.nn_shape:
            # with max or average pooling
            if 'maxpool' in self.nn_shape or 'avgpool' in self.nn_shape:
                pooling = T.mean if 'avgpool' in self.nn_shape else T.max
                layers += [[Pool1D('mp_da', pooling_func=pooling)
                            if self.da_embs else Identity('id_da'),
                            Pool1D('mp_tree', pooling_func=pooling)]]
            layers += [[Flatten('flat_da') if self.da_embs else Identity('id_da'),
                        Flatten('flat_tree')]]

            num_ff_layers = int(self.nn_shape[-1]) if self.nn_shape[-1] in ['2', '3', '4'] else 1
            for da_layer, tree_layer in zip(self._ff_layers('ff_da', num_ff_layers),
                                            self._ff_layers('ff_tree', num_ff_layers)):
                layers.append([da_layer[0], tree_layer[0]])
            layers.append([DotProduct('dot')])

        # input: batch * word * sub-embeddings
        self.nn = RankNN(layers, input_shapes, input_types, self.normgrad)
        log_info("Network shape:\n\n" + str(self.nn))

    def _conv_layers(self, name, num_layers=1, pooling=None):
        ret = []
        for i in xrange(num_layers):
            ret.append([Conv1D(name + str(i + 1),
                               filter_length=self.cnn_filter_length,
                               num_filters=self.cnn_num_filters,
                               init=self.init, activation=T.tanh)])
        if pooling is not None:
            ret.append([Pool1D(name + str(i + 1) + 'pool', pooling_func=pooling)])
        return ret

    def _id_layers(self, name, num_layers):
        ret = []
        for i in xrange(num_layers):
            ret.append([Identity(name + str(i + 1))])
        return ret

    def _update_nn(self, bad_feats, good_feats, rate):
        """Changing the NN update call to support arrays of parameters."""
        # TODO: this is just adding another dimension to fit the parallelized scoring
        # (even if updates are not parallelized). Make it nicer.
        bad_feats = ([bad_feats[0]], [bad_feats[1]])
        good_feats = ([good_feats[0]], [good_feats[1]])

        cost_gcost = self.nn.update(*(bad_feats + good_feats + (rate,)))
        log_debug('Cost:' + str(cost_gcost[0]))
        param_vals = [param.get_value() for param in self.nn.params]
        log_debug('Param norms : ' + str(self._l2s(param_vals)))
        log_debug('Gparam norms: ' + str(self._l2s(cost_gcost[1:])))

    def _embs_to_str(self):
        out = ""
        da_emb = self.nn.layers[0][0].e.get_value()
        tree_emb = self.nn.layers[0][1].e.get_value()
        for idx, emb in enumerate(da_emb):
            for key, val in self.dict_slot.items():
                if val == idx:
                    out += key + ',' + ','.join([("%f" % d) for d in emb]) + "\n"
            for key, val in self.dict_value.items():
                if val == idx:
                    out += key + ',' + ','.join([("%f" % d) for d in emb]) + "\n"
        for idx, emb in enumerate(tree_emb):
            for key, val in self.dict_t_lemma.items():
                if val == idx:
                    out += str(key) + ',' + ','.join([("%f" % d) for d in emb]) + "\n"
            for key, val in self.dict_formeme.items():
                if val == idx:
                    out += str(key) + ',' + ','.join([("%f" % d) for d in emb]) + "\n"
        return out

    def _l2s(self, params):
        """Compute L2-norm of all members of the given list."""
        return [np.linalg.norm(param) for param in params]

    def store_iter_weights(self):
        """Remember the current weights to be used for averaged perceptron."""
        # fh = open('embs.txt', 'a')
        # print >> fh, '---', self._embs_to_str()
        # fh.close()
        self.w_after_iter.append(self.nn.get_param_values())

    def score_all(self, trees, da):
        cand_embs = [self._extract_feats(tree, da) for tree in trees]
        score = self.nn.score([emb[0] for emb in cand_embs], [emb[1] for emb in cand_embs])
        return np.atleast_1d(score[0])
