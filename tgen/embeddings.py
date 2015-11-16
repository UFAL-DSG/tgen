#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extracting embeddings from DAs and trees (basically dictionaries with indexes).
"""

from __future__ import unicode_literals


class EmbeddingExtract(object):
    """Abstract ancestor of embedding extraction classes."""

    MIN_VALID = 4

    def __init__(self):
        pass

    def init_dict(self, train_data, dict_ord=None):
        """Initialize embedding dictionary (word -> id)."""
        raise NotImplementedError()

    def get_embeddings(self, data):
        """Get the embeddings of a data instance."""
        raise NotImplementedError()


class DAEmbeddingExtract(EmbeddingExtract):
    """Extracting embeddings for dialogue acts (currently, just slot-value list)."""

    UNK_SLOT = 0
    UNK_VALUE = 1

    def __init__(self, cfg):
        super(DAEmbeddingExtract, self).__init__()

        self.dict_slot = {'UNK_SLOT': self.UNK_SLOT}
        self.dict_value = {'UNK_VALUE': self.UNK_VALUE}
        self.max_da_len = cfg.get('max_da_len', 10)

    def init_dict(self, train_das, dict_ord=None):
        """Initialize dictionary, given training DAs (store slots and values,
        assign them IDs).

        @param train_das: training DAs
        @param dict_ord: lowest ID to be assigned (if None, it is initialized to MIN_VALID)
        @return: the highest ID assigned + 1 (the current lowest available ID)
        """

        if dict_ord is None:
            dict_ord = self.MIN_VALID

        for da in train_das:
            for dai in da:
                if dai.name not in self.dict_slot:
                    self.dict_slot[dai.name] = dict_ord
                    dict_ord += 1
                if dai.value not in self.dict_value:
                    self.dict_value[dai.value] = dict_ord
                    dict_ord += 1

        return dict_ord

    def get_embeddings(self, da):
        """Get the embeddings (index IDs) for a dialogue act."""
        # DA embeddings (slot - value; size == 2x self.max_da_len)
        da_emb_idxs = []
        for dai in da[:self.max_da_len]:
            da_emb_idxs.append([self.dict_slot.get(dai.name, self.UNK_SLOT),
                                self.dict_value.get(dai.value, self.UNK_VALUE)])
        # pad with "unknown"
        for _ in xrange(len(da_emb_idxs), self.max_da_len):
            da_emb_idxs.append([self.UNK_SLOT, self.UNK_VALUE])

        return da_emb_idxs


class TreeEmbeddingExtract(EmbeddingExtract):
    """Extracting embeddings for trees (either parent_lemma-formeme-child_lemma, or
    with the addition of previous_lemma)."""

    UNK_T_LEMMA = 2
    UNK_FORMEME = 3

    def __init__(self, cfg):
        super(TreeEmbeddingExtract, self).__init__()

        self.dict_t_lemma = {'UNK_T_LEMMA': self.UNK_T_LEMMA}
        self.dict_formeme = {'UNK_FORMEME': self.UNK_FORMEME}
        self.max_tree_len = cfg.get('max_tree_len', 20)
        # 'emb_prev' = embeddings incl. prev node at each step
        self.prev_node_emb = cfg.get('nn', 'emb') == 'emb_prev'

    def init_dict(self, train_trees, dict_ord=None):
        """Initialize dictionary, given training trees (store t-lemmas and formemes,
        assign them IDs).

        @param train_das: training DAs
        @param dict_ord: lowest ID to be assigned (if None, it is initialized to MIN_VALID)
        @return: the highest ID assigned + 1 (the current lowest available ID)
        """
        if dict_ord is None:
            dict_ord = self.MIN_VALID

        for tree in train_trees:
            for t_lemma, formeme in tree.nodes:
                if t_lemma not in self.dict_t_lemma:
                    self.dict_t_lemma[t_lemma] = dict_ord
                    dict_ord += 1
                if formeme not in self.dict_formeme:
                    self.dict_formeme[formeme] = dict_ord
                    dict_ord += 1

        return dict_ord

    def get_embeddings(self, tree):
        """Get the embeddings (index IDs) for a tree."""

        # parent_lemma - formeme - lemma, + adding previous_lemma for emb_prev
        tree_emb_idxs = []
        for pos in xrange(1, min(self.max_tree_len + 1, len(tree))):
            t_lemma, formeme = tree.nodes[pos]
            parent_ord = tree.parents[pos]
            node_emb_idxs = [self.dict_t_lemma.get(tree.nodes[parent_ord].t_lemma, self.UNK_T_LEMMA),
                             self.dict_formeme.get(formeme, self.UNK_FORMEME),
                             self.dict_t_lemma.get(t_lemma, self.UNK_T_LEMMA)]
            if self.prev_node_emb:
                node_emb_idxs.append(self.dict_t_lemma.get(tree.nodes[pos - 1].t_lemma,
                                                           self.UNK_T_LEMMA))
            tree_emb_idxs.append(node_emb_idxs)

        # pad with unknown values (except for last lemma in case of emb_prev)
        for pos in xrange(len(tree) - 1, self.max_tree_len):
            node_emb_idxs = [self.UNK_T_LEMMA, self.UNK_FORMEME, self.UNK_T_LEMMA]
            if self.prev_node_emb:
                node_emb_idxs.append(self.dict_t_lemma.get(tree.nodes[-1].t_lemma, self.UNK_T_LEMMA)
                                     if pos == len(tree) - 1
                                     else self.UNK_T_LEMMA)
            tree_emb_idxs.append(node_emb_idxs)

        return tree_emb_idxs
