#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""


from __future__ import unicode_literals
from threading import Thread
import socket
import cPickle as pickle
import time
import os
from collections import deque
import shutil
import re
import sys

from rpyc import Service, connect, async
from rpyc.utils.server import ThreadPoolServer

from flect.cluster import Job
from pytreex.core.util import file_stream

from logf import log_info, set_debug_stream, log_debug
from tgen.logf import log_warn, is_debug_stream
from tgen.rnd import rnd
from tgen.parallel_percrank_train import ServiceConn
from tgen.seq2seq import Seq2SeqGen


def get_worker_registrar_for(head):
    """Return a class that will handle worker registration for the given head."""

    class WorkerRegistrarService(Service):

        def exposed_register_worker(self, host, port):
            """Register a worker with my head, initialize it."""
            # initiate connection in the other direction
            log_info('Worker %s:%d connected, initializing training.' % (host, port))
            conn = connect(host, port, config={'allow_pickle': True})
            # initialize the remote server (with training data etc.)
            init_func = async(conn.root.init_training)
            req = init_func(pickle.dumps(head.cfg, pickle.HIGHEST_PROTOCOL))
            # add it to the list of running services
            sc = ServiceConn(host, port, conn)
            head.services.add(sc)
            head.pending_requests.add((sc, None, req))
            log_info('Worker %s:%d initialized.' % (host, port))

    return WorkerRegistrarService


class ParallelSeq2SeqTraining(object):
    """TODO"""

    DEFAULT_PORT = 25125
    TEMPFILE_NAME = 'seq2seq_temp_dump.pickle.gz'

    def __init__(self, cfg, work_dir, experiment_id=None):
        # initialize base class
        super(ParallelSeq2SeqTraining, self).__init__()
        # store config
        self.cfg = cfg
        # initialize myself
        self.work_dir = work_dir
        self.jobs_number = cfg.get('jobs_number', 10)
        self.job_memory = cfg.get('job_memory', 8)
        self.port = cfg.get('port', self.DEFAULT_PORT)
        self.host = socket.getfqdn()
        self.poll_interval = cfg.get('poll_interval', 1)
        self.average_models = cfg.get('average_models', False)
        self.experiment_id = experiment_id if experiment_id is not None else ''
        # this will be needed when running
        self.server = None
        self.server_thread = None
        self.jobs = None
        self.pending_requests = None
        self.services = None
        self.free_services = None
        self.results = None
        # this is needed for saving the model
        self.model_temp_path = None

    def train(self, das_file, ttree_file, data_portion=1.0):
        """Run parallel perceptron training, start and manage workers."""
        # initialize the ranker instance
        log_info('Initializing...')
        # run server to process registering clients
        self._init_server()
        # spawn training jobs
        log_info('Spawning jobs...')
        host_short, _ = self.host.split('.', 1)  # short host name for job names
        for j in xrange(self.jobs_number):
            # set up debugging logfile only if we have it on the head
            debug_logfile = ('"PRT%02d.debug-out.txt.gz"' % j) if is_debug_stream() else 'None'
            job = Job(header='from tgen.parallel_seq2seq_train import run_training',
                      code=('run_training("%s", %d, %s)' %
                            (self.host, self.port, debug_logfile)),
                      name=self.experiment_id + ("PRT%02d-%s-%d" % (j, host_short, self.port)),
                      work_dir=self.work_dir)
            job.submit(self.job_memory)
            self.jobs.append(job)

        # run the training passes
        try:
            cur_assign = 0
            results = [None] * self.jobs_number
            rnd_seeds = [rnd.random() for _ in xrange(self.jobs_number)]

            # assign training and wait for it to finish
            while cur_assign < self.jobs_number or self.pending_requests:
                log_debug('Starting loop over services.')

                # check if some of the pending computations have finished
                for sc, job_no, req in list(self.pending_requests):
                    res = self._check_pending_request(sc, job_no, req)
                    if res and res is not None:
                        results[job_no] = res, sc

                # check for free services and assign new computation
                while cur_assign < self.jobs_number and self.free_services:
                    log_debug('Assigning request %d' % cur_assign)
                    sc = self.free_services.popleft()
                    log_info('Assigning request %d to %s:%d' % (cur_assign, sc.host, sc.port))
                    train_func = async(sc.conn.root.train)
                    req = train_func(rnd_seeds[cur_assign],
                                     os.path.relpath(das_file, self.work_dir),
                                     os.path.relpath(ttree_file, self.work_dir),
                                     data_portion)
                    self.pending_requests.add((sc, cur_assign, req))
                    cur_assign += 1
                    log_debug('Assigned %d' % cur_assign)

                # sleep for a while
                log_debug('Sleeping.')
                time.sleep(self.poll_interval)

            log_info("Results:\n" + "\n".join("%.5f %s:%d" % (cost, sc.host, sc.port)
                                              for cost, sc in results))

            self.model_temp_path = os.path.join(self.work_dir, self.TEMPFILE_NAME)
            # average the computed models
            if self.average_models:
                log_info('Averaging models...')
                avg_model = self.get_averaged_model(results)
                log_info('Saving the averaged model temporarily to %s...' % self.model_temp_path)
                avg_model.save_to_file(os.path.relpath(self.model_temp_path, self.work_dir))
            # select the best result on devel data + save it
            else:
                best_cost, best_sc = min(results, key=lambda res: res[0])
                log_info('Best cost: %f (computed at %s:%d).' % (best_cost, best_sc.host, best_sc.port))
                log_info('Saving best generator temporarily to %s...' % self.model_temp_path)
                best_sc.conn.root.save_model(os.path.relpath(self.model_temp_path, self.work_dir))

        # kill all jobs
        finally:
            for job in self.jobs:
                job.delete()

    def _check_pending_request(self, sc, job_no, req):
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
        if job_no is not None:
            log_debug('Checking %d' % job_no)

        # checking if the request has finished
        if req.ready:
            if job_no is not None:
                log_debug('Ready %d' % job_no)
                log_info('Retrieved finished request %d' % job_no)
            if req.error:
                log_info('Error found on request: job #%d, worker %s:%d' %
                         (job_no if job_no is not None else -1, sc.host, sc.port))
            result = req.value

            # remove from list of pending requests
            # TODO return to pool of free requests (but needs to store the results somewhere)
            self.pending_requests.remove((sc, job_no, req))
            if job_no is None:
                self.free_services.append(sc)

        return result

    def _init_server(self):
        """Initializes a server that registers new workers."""
        registrar_class = get_worker_registrar_for(self)
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

    def save_to_file(self, model_fname):
        """This will actually just move the best generator (which is saved in a temporary file)
        to the final location."""
        log_info('Moving generator to %s...' % model_fname)
        orig_model_fname = self.model_temp_path
        shutil.move(orig_model_fname, model_fname)
        orig_tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', orig_model_fname)
        tf_session_fname = re.sub(r'(.pickle)?(.gz)?$', '.tfsess', model_fname)
        shutil.move(orig_tf_session_fname, tf_session_fname)

        # move the reranking classifier model files as well, if they exist
        orig_clfilter_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tftreecl\1', orig_model_fname)
        orig_clfilter_tf_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tfsess', orig_clfilter_fname)

        if os.path.isfile(orig_clfilter_fname) and os.path.isfile(orig_clfilter_tf_fname):
            clfilter_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tftreecl\1', model_fname)
            clfilter_tf_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tfsess', clfilter_fname)
            shutil.move(orig_clfilter_fname, clfilter_fname)
            shutil.move(orig_clfilter_tf_fname, clfilter_tf_fname)

    def get_averaged_model(self, results):
        """Retrieve parameters of all models in results, average them, and save a new model with
        averaged parameters. Using a plain arithmetic average, no weighting.

        @param results: the computation results -- list of pairs (cost, ServiceConn)
        @return: the averaged model, as a Seq2SeqGen object
        """
        # get all models' parameters and average them
        avg_params = None
        for _, sc in results:
            if avg_params is None:
                avg_params = pickle.loads(sc.conn.root.get_model_params())
            else:
                cur_params = pickle.loads(sc.conn.root.get_model_params())
                for name, val in cur_params.iteritems():
                    avg_params[name] += val
        for name in avg_params.iterkeys():
            avg_params[name] /= float(len(results))

        # save a random model
        results[0][1].conn.root.save_model(os.path.relpath(self.model_temp_path, self.work_dir))
        # read it locally
        model = Seq2SeqGen.load_from_file(self.model_temp_path)
        model.set_model_params(avg_params)
        return model


class Seq2SeqTrainingService(Service):

    def __init__(self, conn_ref):
        super(Seq2SeqTrainingService, self).__init__(conn_ref)
        self.seq2seq = None

    def exposed_init_training(self, cfg):
        """Create the Seq2SeqGen object."""
        cfg = pickle.loads(cfg)
        tstart = time.time()
        log_info('Initializing training...')
        self.seq2seq = Seq2SeqGen(cfg)
        log_info('Training initialized. Time taken: %f secs.' % (time.time() - tstart))

    def exposed_train(self, rnd_seed, das_file, ttree_file, data_portion):
        """Run the whole training.
        """
        rnd.seed(rnd_seed)
        log_info('Random seed: %f' % rnd_seed)
        tstart = time.time()
        log_info('Starting training...')
        self.seq2seq.train(das_file, ttree_file, data_portion)
        log_info('Training finished -- time taken: %f secs.' % (time.time() - tstart))
        top_cost = self.seq2seq.top_k_costs[0]
        log_info('Best cost: %f' % top_cost)
        return top_cost

    def exposed_save_model(self, model_fname):
        """Save the model to the given file (must be given relative to the worker's working
        directory!).
        @param model_fname: target path where to save the model (relative to worker's \
                working directory)
        """
        self.seq2seq.save_to_file(model_fname)

    def exposed_get_model_params(self):
        """Retrieve all parameters of the worker's local model (as a dictionary)
        @return: all model parameters in a dictionary -- keys are names, values are numpy arrays
        """
        p_dump = pickle.dumps(self.seq2seq.get_model_params(), protocol=pickle.HIGHEST_PROTOCOL)
        return p_dump


def run_training(head_host, head_port, debug_out=None):
    # setup debugging output, if applicable
    if debug_out is not None:
        set_debug_stream(file_stream(debug_out, mode='w'))
    # start the server (in the background)
    log_info('Creating training server...')
    server = ThreadPoolServer(service=Seq2SeqTrainingService, nbThreads=1)
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
    run_training(host, port)
