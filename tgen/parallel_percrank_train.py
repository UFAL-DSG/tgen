#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parallel training for Perceptron ranker (using Qsub & RPyC).

@todo: Local training on multiple cores should not be hard to implement.
"""

from __future__ import unicode_literals
from collections import deque, namedtuple
import sys
from threading import Thread
import socket
import cPickle as pickle
import time
import datetime

import numpy as np
from rpyc import Service, connect, async
from rpyc.utils.server import ThreadPoolServer

from flect.cluster import Job

from rank import PerceptronRanker
from tgen.logf import log_info, set_debug_stream, log_debug
from alex.components.nlg.tectotpl.core.util import file_stream


class ServiceConn(namedtuple('ServiceConn', ['host', 'port', 'conn'])):
    """This stores a connection along with its address."""
    pass


def get_worker_registrar_for(head):
    """Return a class that will handle worker registration for the given head."""

    # create a dump of the head to be passed to workers
    head_dump = pickle.dumps(head.get_plain_percrank(), protocol=pickle.HIGHEST_PROTOCOL)

    class WorkerRegistrarService(Service):

        def exposed_register_worker(self, host, port):
            """Register a worker with my head, initialize it."""
            # initiate connection in the other direction
            log_info('Worker %s:%d connected, initializing training.' % (host, port))
            conn = connect(host, port)
            # initialize the remote server (with training data etc.)
            conn.root.init_training(head_dump)
            # add it to the list of running services
            sc = ServiceConn(host, port, conn)
            head.services.add(sc)
            head.free_services.append(sc)
            log_info('Worker %s:%d initialized.' % (host, port))

    return WorkerRegistrarService


class ParallelPerceptronRanker(PerceptronRanker):

    DEFAULT_PORT = 25125

    def __init__(self, cfg, work_dir):
        # initialize base class
        super(ParallelPerceptronRanker, self).__init__(cfg)
        # initialize myself
        self.work_dir = work_dir
        self.jobs_number = cfg.get('jobs_number', 10)
        self.data_portions = cfg.get('data_portions', self.jobs_number)
        self.job_memory = cfg.get('job_memory', 4)
        self.port = cfg.get('port', self.DEFAULT_PORT)
        self.host = socket.getfqdn()
        self.poll_interval = cfg.get('poll_interval', 1)
        # this will be needed when running
        self.server = None
        self.server_thread = None
        self.jobs = None
        self.pending_requests = None
        self.services = None
        self.free_services = None
        self.results = None

    def train(self, das_file, ttree_file, data_portion=1.0):
        """Run parallel perceptron training, start and manage workers."""
        # initialize myself
        log_info('Initializing...')
        self._init_training(das_file, ttree_file, data_portion)
        # run server to process registering clients
        self._init_server()
        # spawn training jobs
        log_info('Spawning jobs...')
        for j in xrange(self.jobs_number):
            job = Job(header='from tgen.parallel_percrank_train import run_worker',
                      code=('run_worker("%s", %d, "%s")' %
                            (self.host, self.port, "PRT%02d.debug-out.txt" % j)),
                      name="PRT%02d" % j,
                      work_dir=self.work_dir)
            job.submit(self.job_memory)
            self.jobs.append(job)
        # run the training iterations
        try:
            for iter_no in xrange(1, self.passes + 1):

                log_info('Iteration %d...' % iter_no)
                log_debug('\n***\nTR%05d:' % iter_no)

                iter_start_time = time.clock()
                cur_portion = 0
                results = [None] * self.data_portions
                w_dump = pickle.dumps(self.w, protocol=pickle.HIGHEST_PROTOCOL)
                # wait for free services / assign computation
                while cur_portion < self.data_portions or self.pending_requests:
                    # check if some of the pending computations have finished
                    for sc, req_portion, req in list(self.pending_requests):
                        if req.ready:
                            log_info('Retrieved finished request %d / %d' % (iter_no, req_portion))
                            if req.error:
                                log_info('Error found on request: IT %d PORTION %d, WORKER %s:%d' %
                                         (iter_no, req_portion, sc.host, sc.port))
                            results[req_portion] = pickle.loads(req.value)
                            self.pending_requests.remove((sc, req_portion, req))
                            self.free_services.append(sc)
                    # check for free services and assign new computation
                    while cur_portion < self.data_portions and self.free_services:
                        sc = self.free_services.popleft()
                        log_info('Assigning request %d / %d to %s:%d' %
                                 (iter_no, cur_portion, sc.host, sc.port))
                        train_func = async(sc.conn.root.training_iter)
                        req = train_func(w_dump, iter_no, *self._get_portion_bounds(cur_portion))
                        self.pending_requests.add((sc, cur_portion, req))
                        cur_portion += 1
                    # sleep for a while
                    time.sleep(self.poll_interval)
                # now gather the results and take an average, set it as new w
                self.w = np.average(results, axis=0)

                iter_end_time = time.clock()
                log_info(' * Duration: %s' % str(datetime.timedelta(seconds=(iter_end_time - iter_start_time))))
                log_debug(self._feat_val_str(self.w), '\n***')
        # kill all jobs
        finally:
            for job in self.jobs:
                job.delete()

    def _init_server(self):
        """Initializes a server that registers new workers."""
        registrar_class = get_worker_registrar_for(self)
        self.server = ThreadPoolServer(service=registrar_class, nbThreads=1, port=self.port)
        self.services = set()
        self.free_services = deque()
        self.pending_requests = set()
        self.jobs = []
        self.server_thread = Thread(target=self.server.start)
        self.server_thread.setDaemon(True)
        self.server_thread.start()

    def _get_portion_bounds(self, portion_no):
        """(Head) return the offset and size of the specified portion of the training
        data to be sent to a worker.

        @param portion_no: the number of the portion whose bounds should be computed
        @rtype: tuple
        @return: offset and size of the desired training data portion
        """
        portion_size, bigger_portions = divmod(len(self.train_trees), self.data_portions)
        if portion_no < bigger_portions:
            return (portion_size + 1) * portion_no, portion_size + 1
        else:
            return portion_size * portion_no + bigger_portions, portion_size
        raise NotImplementedError()

    def get_plain_percrank(self):
        percrank = PerceptronRanker(cfg=None)  # initialize with 'empty' configuration
        # copy all necessary data from the head
        percrank.w = self.w
        percrank.feats = self.feats
        percrank.vectorizer = self.vectorizer
        percrank.normalizer = self.normalizer
        percrank.alpha = self.alpha
        percrank.passes = self.passes
        percrank.rival_number = self.rival_number
        percrank.language = self.rival_number
        percrank.selector = self.selector
        percrank.rival_gen_strategy = self.rival_gen_strategy
        percrank.rival_gen_max_iter = self.rival_gen_max_iter
        percrank.rival_gen_max_defic_iter = self.rival_gen_max_defic_iter
        percrank.rival_gen_beam_size = self.rival_gen_beam_size
        percrank.candgen_model = self.candgen_model
        percrank.train_trees = self.train_trees
        percrank.train_feats = self.train_feats
        percrank.train_sents = self.train_sents
        percrank.train_das = self.train_das
        percrank.asearch_planner = self.asearch_planner
        percrank.sampling_planner = self.sampling_planner
        percrank.candgen = self.candgen
        percrank.lists_analyzer = self.lists_analyzer
        percrank.evaluator = self.evaluator
        return percrank


class PercRankTrainingService(Service):

    def __init__(self, conn_ref):
        super(PercRankTrainingService, self).__init__(conn_ref)
        self.percrank = None

    def exposed_init_training(self, head_percrank):
        """(Worker) Just deep-copy all necessary attributes from the head instance."""
        log_info('Initializing training...')
        self.percrank = pickle.loads(head_percrank)
        log_info('Training initialized.')

    def exposed_training_iter(self, w, iter_no, data_offset, data_len):
        """(Worker) Run one iteration on a part of the training data."""
        log_info('Training iteration %d with data portion %d + %d' %
                 (iter_no, data_offset, data_len))
        # import current feature weights
        percrank = self.percrank
        percrank.w = pickle.loads(w)
        # save rest of the training data to temporary variables, set just the
        # required portion for computation
        all_train_das = percrank.train_das
        percrank.train_das = percrank.train_das[data_offset:data_offset + data_len]
        all_train_trees = percrank.train_trees
        percrank.train_trees = percrank.train_trees[data_offset:data_offset + data_len]
        all_train_feats = percrank.train_feats
        percrank.train_feats = percrank.train_feats[data_offset:data_offset + data_len]
        all_train_sents = percrank.train_sents
        percrank.train_sents = percrank.train_sents[data_offset:data_offset + data_len]
        # do the actual computation (update w)
        percrank._training_iter(iter_no)
        # return the rest of the training data to member variables
        percrank.train_das = all_train_das
        percrank.train_trees = all_train_trees
        percrank.train_feats = all_train_feats
        percrank.train_sents = all_train_sents
        # return the result of the computation
        log_info('Training iteration %d / %d / %d done.' % (iter_no, data_offset, data_len))
        return pickle.dumps(percrank.w, pickle.HIGHEST_PROTOCOL)


def run_worker(head_host, head_port, debug_out=None):
    # setup debugging output, if applicable
    if debug_out is not None:
        set_debug_stream(file_stream(debug_out, mode='w'))
    # start the server (in the background)
    log_info('Creating worker server...')
    server = ThreadPoolServer(service=PercRankTrainingService, nbThreads=1)
    server_thread = Thread(target=server.start)
    server_thread.start()
    my_host = socket.getfqdn()
    log_info('Worker server created at %s:%d. Connecting to head at %s:%d...' %
             (my_host, server.port, head_host, head_port))
    # notify main about this server
    conn = connect(head_host, head_port)
    conn.root.register_worker(my_host, server.port)
    conn.close()
    log_info('Worker is registered with the head.')
    # now serve until we're killed (the server thread will continue to run)
    server_thread.join()


if __name__ == '__main__':
    try:
        host = sys.argv[1]
        port = int(sys.argv[2])
    except:
        sys.exit('Usage: ' + sys.argv[0] + ' <head-address> <head-port>')
    run_worker(host, port)
