#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parallel training for Perceptron ranker (using Qsub & RPyC).

@todo: Local training on multiple cores should not be hard to implement.
"""

from __future__ import unicode_literals
from copy import deepcopy
from collections import deque, namedtuple
import sys
from time import sleep

import numpy as np
from rpyc import Service, connect, async
from rpyc.utils.server import ThreadPoolServer

from flect.cluster import Job

from rank import PerceptronRanker


class ParallelTrainingPercRank(PerceptronRanker, Service):

    def __init__(self, cfg):
        super(ParallelTrainingPercRank, self).__init__(cfg)
        super(Service, self).__init__()
        self.work_dir = cfg['work_dir']  # TODO possibly change this so that it is a separate param
        self.jobs_number = cfg.get('jobs_number', 10)
        self.data_portions = cfg.get('data_portions', self.jobs_number)
        self.job_memory = cfg.get('job_memory', 4)
        self.port = cfg.get('port', 25125)
        self.poll_interval = cfg.get('poll_interval', 1)
        self.server = None
        self.jobs = None
        self.pending_requests = None
        self.services = None
        self.free_services = None
        self.results = None

    def train(self, das_file, ttree_file, data_portion=1.0):
        """(Head) run the training, start and manage workers."""
        # initialize myself
        self._init_training(das_file, ttree_file, data_portion)
        # run server to process registering clients
        self.server = ThreadPoolServer(service=ParallelTrainingPercRank,
                                       nbThreads=1,
                                       port=self.port)
        self.services = set()
        self.free_services = deque()
        self.pending_requests = set()
        self.jobs = []
        # spawn training jobs
        for j in xrange(self.jobs_number):
            job = Job(code='run_worker(%s, %d)',
                      header='from tgen.parallel_percrank_train import run_worker',
                      name="PRT%02d" % j,
                      work_dir=self.work_dir)
            job.submit(self.job_memory)
            self.jobs.append(job)
        # wait for free services / assign computation
        try:
            for iter_no in xrange(1, self.passes + 1):
                cur_portion = 0
                results = [None] * self.data_portions
                while cur_portion < self.data_portions:
                    # check if some of the pending computations have finished
                    for conn, req_portion, req in list(self.pending_requests):
                        if req.ready:
                            if req.error:
                                raise Exception('Request computed with error: IT %d PORTION %d, WORKER %s:%d' %
                                                (iter_no, req_portion, conn.host, conn.port))
                            results[req_portion] = req.value
                            self.pending_requests.remove(conn, req_portion, req)
                            self.free_services.append(conn)
                    # check for free services and assign new computation
                    while cur_portion < self.data_portions and self.free_services:
                        conn = self.free_services.popleft()
                        train_func = async(conn.root.training_iter)
                        req = train_func(self.w, iter_no, *self._get_portion_bounds(cur_portion))
                        self.pending_requests.add((conn, cur_portion, req))
                        cur_portion += 1
                    # sleep for a while
                    sleep(self.poll_interval)
                # now gather the results and take an average, set it as new w
                self.w = np.average(results, axis=0)
            # kill all jobs
        finally:
            for job in self.jobs:
                job.delete()

    def _get_portion_bounds(self, portion_no):
        # TODO
        raise NotImplementedError()

    def exposed_register_worker(self, host, port):
        """(Head) register a worker with this head."""
        # initiate connection in the other direction
        conn = connect(host, port)
        # initialize the remote server (with training data etc.)
        conn.root.init_training(self)
        # add it to the list of running services
        self.services.add(conn)
        self.free_services.append(conn)

    def exposed_init_training(self, other):
        """(Worker) Just deep-copy all necessary attributes from the head instance."""
        self.w = deepcopy(other.w)
        self.feats = deepcopy(other.feats)
        self.vectorizer = deepcopy(other.vectorizer)
        self.normalizer = deepcopy(other.normalizer)
        self.alpha = deepcopy(other.alpha)
        self.passes = deepcopy(other.passes)
        self.rival_number = deepcopy(other.rival_number)
        self.language = deepcopy(other.rival_number)
        self.selector = deepcopy(other.selector)
        self.rival_gen_strategy = deepcopy(other.rival_gen_strategy)
        self.rival_gen_max_iter = deepcopy(other.rival_gen_max_iter)
        self.rival_gen_max_defic_iter = deepcopy(other.rival_gen_max_defic_iter)
        self.rival_gen_beam_size = deepcopy(other.rival_gen_beam_size)
        self.candgen_model = deepcopy(other.candgen_model)
        self.feats = deepcopy(other.feats)
        self.train_trees = deepcopy(other.train_trees)
        self.train_feats = deepcopy(other.train_feats)
        self.train_sents = deepcopy(other.train_sents)
        self.train_das = deepcopy(other.train_das)
        self.asearch_planner = deepcopy(other.asearch_planner)
        self.sampling_planner = deepcopy(other.sampling_planner)
        self.candgen = deepcopy(other.candgen)

    def exposed_training_iter(self, w, iter_no, data_offset, data_len):
        """(Worker) Run one iteration on a part of the training data."""
        # import current feature weights
        self.w = w
        # save rest of the training data to temporary variables, set just the
        # required portion for computation
        all_train_das = self.train_das
        self.train_das = self.train_das[data_offset:data_offset + data_len]
        all_train_trees = self.train_trees
        self.train_trees = self.train_trees[data_offset:data_offset + data_len]
        all_train_feats = self.train_feats
        self.train_feats = self.train_feats[data_offset:data_offset + data_len]
        all_train_sents = self.train_sents
        self.train_sents = self.train_sents[data_offset:data_offset + data_len]
        # do the actual computation (update w)
        self._training_iter(iter_no)
        # return the rest of the training data to member variables
        self.train_das = all_train_das
        self.train_trees = all_train_trees
        self.train_feats = all_train_feats
        self.train_sents = all_train_sents
        # return the result of the computation
        return self.w


def run_worker(head_host, head_port):
    # start the server
    server = ThreadPoolServer(service=ParallelTrainingPercRank, nbThreads=1)
    server.start()
    # notify main about this server
    conn = connect(head_host, head_port)
    conn.root.register_worker(server.host, server.port)
    conn.close()
    # now serve until we're killed


if __name__ == '__main__':
    try:
        host = sys.argv[1]
        port = int(sys.argv[2])
    except:
        sys.exit('Usage: ' + sys.argv[0] + ' <head-address> <head-port>')
    run_worker(host, port)
