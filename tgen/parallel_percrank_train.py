#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parallel training for Perceptron ranker (using Qsub & RPyC).

When run as main, this file will start a worker and register with the address given
in command-line parameters.

Usage: ./parallel_percrank_train.py <head-address> <head-port>

@todo: Local training on multiple cores should not be hard to implement.
"""

from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range
from collections import deque, namedtuple
import sys
from threading import Thread
import socket
import pickle as pickle
import time
import datetime
import os
import tempfile
import numpy as np

from rpyc import Service, connect, async_
from rpyc.utils.server import ThreadPoolServer

from tgen.futil import file_stream

from .logf import log_info, set_debug_stream, log_debug
from tgen.logf import log_warn, is_debug_stream
from tgen.rnd import rnd
from tgen.rank import Ranker, PerceptronRanker
from tgen.cluster import Job


class ServiceConn(namedtuple('ServiceConn', ['host', 'port', 'conn'])):
    """This stores a connection along with its address."""
    pass


def dump_ranker(ranker, work_dir):

    fh = tempfile.NamedTemporaryFile(suffix='.pickle', prefix='rdump-', dir=work_dir, delete=False)
    # we are storing training features separately in NumPy format since including them in
    # the pickle may lead to a crash for very large feature matrices
    # (see https://github.com/numpy/numpy/issues/2396).
    train_feats = ranker.train_feats
    ranker.train_feats = None
    pickle.dump(ranker, fh, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(fh, train_feats)
    fh.close()
    return fh.name


def load_ranker(dump):

    with open(dump, 'rb') as fh:
        ranker = pickle.load(fh)
        train_feats = np.load(fh)
        ranker.train_feats = train_feats
    return ranker


def get_worker_registrar_for(head):
    """Return a class that will handle worker registration for the given head."""

    # create a dump of the head to be passed to workers
    log_info('Saving ranker init state...')
    tstart = time.time()
    ranker_dump_path = dump_ranker(head.loc_ranker, head.work_dir)
    log_info('Ranker init state saved in %s, it took %f secs.' % (ranker_dump_path,
                                                                  time.time() - tstart))

    class WorkerRegistrarService(Service):

        def exposed_register_worker(self, host, port):
            """Register a worker with my head, initialize it."""
            # initiate connection in the other direction
            log_info('Worker %s:%d connected, initializing training.' % (host, port))
            conn = connect(host, port, config={'allow_pickle': True})
            # initialize the remote server (with training data etc.)
            init_func = async_(conn.root.init_training)
            req = init_func(ranker_dump_path)
            # add it to the list of running services
            sc = ServiceConn(host, port, conn)
            head.services.add(sc)
            head.pending_requests.add((sc, None, req))
            log_info('Worker %s:%d initialized.' % (host, port))

    return WorkerRegistrarService, ranker_dump_path


class ParallelRanker(Ranker):
    """This is used to train rankers in parallel (supports any ranker class)."""

    DEFAULT_PORT = 25125

    def __init__(self, cfg, work_dir, experiment_id=None, ranker_class=PerceptronRanker):
        # initialize base class
        super(ParallelRanker, self).__init__()
        # initialize myself
        self.work_dir = work_dir
        self.jobs_number = cfg.get('jobs_number', 10)
        self.data_portions = cfg.get('data_portions', self.jobs_number)
        self.job_memory = cfg.get('job_memory', 4)
        self.port = cfg.get('port', self.DEFAULT_PORT)
        self.host = socket.getfqdn()
        self.poll_interval = cfg.get('poll_interval', 1)
        self.experiment_id = experiment_id if experiment_id is not None else ''
        # this will be needed when running
        self.server = None
        self.server_thread = None
        self.jobs = None
        self.pending_requests = None
        self.services = None
        self.free_services = None
        self.results = None
        # create a local ranker instance that will be copied to all parallel workers
        # and will be used to average weights after each iteration
        self.loc_ranker = ranker_class(cfg)

    def train(self, das_file, ttree_file, data_portion=1.0):
        """Run parallel perceptron training, start and manage workers."""
        # initialize the ranker instance
        log_info('Initializing...')
        self.loc_ranker._init_training(das_file, ttree_file, data_portion)
        # run server to process registering clients
        self._init_server()
        # spawn training jobs
        log_info('Spawning jobs...')
        host_short, _ = self.host.split('.', 1)  # short host name for job names
        for j in range(self.jobs_number):
            # set up debugging logfile only if we have it on the head
            debug_logfile = ('"PRT%02d.debug-out.txt.gz"' % j) if is_debug_stream() else 'None'
            job = Job(header='from tgen.parallel_percrank_train import run_worker',
                      code=('run_worker("%s", %d, %s)' %
                            (self.host, self.port, debug_logfile)),
                      name=self.experiment_id + ("PRT%02d-%s-%d" % (j, host_short, self.port)),
                      work_dir=self.work_dir)
            job.submit(self.job_memory)
            self.jobs.append(job)
        # run the training passes
        try:
            for iter_no in range(1, self.loc_ranker.passes + 1):

                log_info('Pass %d...' % iter_no)
                log_debug('\n***\nTR%05d:' % iter_no)

                iter_start_time = time.time()
                cur_portion = 0
                results = [None] * self.data_portions
                w_dump = pickle.dumps(self.loc_ranker.get_weights(), protocol=pickle.HIGHEST_PROTOCOL)
                rnd_seeds = [rnd.random() for _ in range(self.data_portions)]
                # wait for free services / assign computation
                while cur_portion < self.data_portions or self.pending_requests:
                    log_debug('Starting loop over services.')

                    # check if some of the pending computations have finished
                    for sc, req_portion, req in list(self.pending_requests):
                        res = self._check_pending_request(iter_no, sc, req_portion, req)
                        if res:
                            results[req_portion] = res

                    # check for free services and assign new computation
                    while cur_portion < self.data_portions and self.free_services:
                        log_debug('Assigning request %d' % cur_portion)
                        sc = self.free_services.popleft()
                        log_info('Assigning request %d / %d to %s:%d' %
                                 (iter_no, cur_portion, sc.host, sc.port))
                        train_func = async_(sc.conn.root.training_pass)
                        req = train_func(w_dump, iter_no, rnd_seeds[cur_portion],
                                         * self._get_portion_bounds(cur_portion))
                        self.pending_requests.add((sc, cur_portion, req))
                        cur_portion += 1
                        log_debug('Assigned %d' % cur_portion)
                    # sleep for a while
                    log_debug('Sleeping.')
                    time.sleep(self.poll_interval)

                # delete the temporary ranker dump when the 1st iteration is complete
                if self.ranker_dump_path:
                    log_info('Removing temporary ranker dump at %s.' % self.ranker_dump_path)
                    os.remove(self.ranker_dump_path)
                    self.ranker_dump_path = None

                # gather/average the diagnostic statistics
                self.loc_ranker.set_diagnostics_average([d for _, d in results])

                # take an average of weights; set it as new w
                self.loc_ranker.set_weights_average([w for w, _ in results])
                self.loc_ranker.store_iter_weights()  # store a copy of w for averaged perceptron

                # print statistics
                log_debug(self.loc_ranker._feat_val_str(), '\n***')
                self.loc_ranker._print_pass_stats(iter_no, datetime.timedelta(seconds=(time.time() - iter_start_time)))

            # after all passes: average weights if set to do so
            if self.loc_ranker.averaging is True:
                self.loc_ranker.set_weights_iter_average()
        # kill all jobs
        finally:
            for job in self.jobs:
                job.delete()

    def _check_pending_request(self, iter_no, sc, req_portion, req):
        """Check whether the given request has finished (i.e., job is loaded or job has
        processed the given data portion.

        If the request is finished, the worker that processed it is moved to the pool
        of free services.

        @param iter_no: current iteration number (for logging)
        @param sc: a ServiceConn object that stores the worker connection parameters
        @param req_portion: current data portion number (is None for jobs loading)
        @param req: the request itself

        @return: the value returned by the finished data processing request, or None \
            (for loading requests or unfinished requests)
        """
        result = None
        if req_portion is not None:
            log_debug('Checking %d' % req_portion)

        # checking if the request has finished
        if req.ready:
            # loading requests -- do nothing (just logging)
            if req_portion is None:
                if req.error:
                    log_info('Error loading on %s:%d' % (sc.host, sc.port))
                else:
                    log_info('Worker %s:%d finished loading.' % (sc.host, sc.port))
            # data processing request -- retrieve the value
            else:
                log_debug('Ready %d' % req_portion)
                log_info('Retrieved finished request %d / %d' % (iter_no, req_portion))
                if req.error:
                    log_info('Error found on request: IT %d PORTION %d, WORKER %s:%d' %
                             (iter_no, req_portion, sc.host, sc.port))
                result = pickle.loads(req.value)

            # add the worker to the pool of free services (both loading and data processing requests)
            self.pending_requests.remove((sc, req_portion, req))
            self.free_services.append(sc)

        if req_portion is not None:
            log_debug('Done with %d' % req_portion)
        return result

    def _init_server(self):
        """Initializes a server that registers new workers."""
        registrar_class, ranker_dump_path = get_worker_registrar_for(self)
        n_tries = 0
        self.server = None
        last_error = None
        while self.server is None and n_tries < 10:
            try:
                n_tries += 1
                self.server = ThreadPoolServer(service=registrar_class, nbThreads=1, port=self.port)
            except socket.error as e:
                log_warn('Port %d in use, trying to use a higher port...' % self.port)
                self.port += 1
                last_error = e
        if self.server is None:
            if last_error is not None:
                raise last_error
            raise Exception('Could not initialize server')
        self.services = set()
        self.free_services = deque()
        self.pending_requests = set()
        self.jobs = []
        self.server_thread = Thread(target=self.server.start)
        self.server_thread.setDaemon(True)
        self.server_thread.start()
        self.ranker_dump_path = ranker_dump_path

    def _get_portion_bounds(self, portion_no):
        """(Head) return the offset and size of the specified portion of the training
        data to be sent to a worker.

        @param portion_no: the number of the portion whose bounds should be computed
        @rtype: tuple
        @return: offset and size of the desired training data portion
        """
        portion_size, bigger_portions = divmod(len(self.loc_ranker.train_trees), self.data_portions)
        if portion_no < bigger_portions:
            return (portion_size + 1) * portion_no, portion_size + 1
        else:
            return portion_size * portion_no + bigger_portions, portion_size
        raise NotImplementedError()

    def save_to_file(self, model_fname):
        """Saving just the "plain" perceptron ranker model to a file; discarding all the
        parallel stuff that can't be stored in a pickle anyway."""
        self.loc_ranker.save_to_file(model_fname)


class RankerTrainingService(Service):

    def __init__(self, conn_ref):
        super(RankerTrainingService, self).__init__(conn_ref)
        self.ranker_inst = None

    def exposed_init_training(self, head_ranker_path):
        """(Worker) Just deep-copy all necessary attributes from the head instance."""
        tstart = time.time()
        log_info('Initializing training...')
        self.ranker_inst = load_ranker(head_ranker_path)
        log_info('Training initialized. Time taken: %f secs.' % (time.time() - tstart))

    def exposed_training_pass(self, w, pass_no, rnd_seed, data_offset, data_len):
        """(Worker) Run one pass over a part of the training data.
        @param w: initial perceptron weights (pickled)
        @param pass_no: pass number (for logging purposes)
        @param rnd_seed: random generator seed for shuffling training examples
        @param data_offset: training data portion start
        @param data_len: training data portion size
        @return: updated perceptron weights after passing the selected data portion (pickled)
        """
        log_info('Training pass %d with data portion %d + %d' %
                 (pass_no, data_offset, data_len))
        # use the local ranker instance
        ranker = self.ranker_inst
        # import current feature weights
        tstart = time.time()
        ranker.set_weights(pickle.loads(w))
        log_info('Weights loading: %f secs.' % (time.time() - tstart))
        # save rest of the training data to temporary variables, set just the
        # required portion for computation
        all_train_das = ranker.train_das
        ranker.train_das = ranker.train_das[data_offset:data_offset + data_len]
        all_train_trees = ranker.train_trees
        ranker.train_trees = ranker.train_trees[data_offset:data_offset + data_len]
        all_train_feats = ranker.train_feats
        ranker.train_feats = ranker.train_feats[data_offset:data_offset + data_len]
        all_train_sents = ranker.train_sents
        ranker.train_sents = ranker.train_sents[data_offset:data_offset + data_len]
        all_train_order = ranker.train_order
        ranker.train_order = list(range(len(ranker.train_trees)))
        if ranker.randomize:
            rnd.seed(rnd_seed)
            rnd.shuffle(ranker.train_order)
        # do the actual computation (update w)
        ranker._training_pass(pass_no)
        # return the rest of the training data to member variables
        ranker.train_das = all_train_das
        ranker.train_trees = all_train_trees
        ranker.train_feats = all_train_feats
        ranker.train_sents = all_train_sents
        ranker.train_order = all_train_order
        # return the result of the computation
        log_info('Training pass %d / %d / %d done.' % (pass_no, data_offset, data_len))
        tstart = time.time()
        dump = pickle.dumps((ranker.get_weights(), ranker.get_diagnostics()), pickle.HIGHEST_PROTOCOL)
        log_info('Weights saving: %f secs.' % (time.time() - tstart))
        return dump


def run_worker(head_host, head_port, debug_out=None):
    # setup debugging output, if applicable
    if debug_out is not None:
        set_debug_stream(file_stream(debug_out, mode='w'))
    # start the server (in the background)
    log_info('Creating worker server...')
    server = ThreadPoolServer(service=RankerTrainingService, nbThreads=1)
    server_thread = Thread(target=server.start)
    server_thread.start()
    my_host = socket.getfqdn()
    log_info('Worker server created at %s:%d. Connecting to head at %s:%d...' %
             (my_host, server.port, head_host, head_port))
    # notify main about this server
    conn = connect(head_host, head_port, config={'allow_pickle': True})
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
