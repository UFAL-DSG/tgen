#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""


from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
from threading import Thread
import socket
import pickle as pickle
import time
import os
from collections import deque
import shutil
import re
import sys
import hashlib

from rpyc import Service, connect, async_
from rpyc.utils.server import ThreadPoolServer

from tgen.futil import file_stream
from tgen.logf import log_info, set_debug_stream, log_debug
from tgen.logf import log_warn, is_debug_stream
from tgen.rnd import rnd
from tgen.parallel_percrank_train import ServiceConn
from tgen.seq2seq import Seq2SeqGen
from tgen.seq2seq_ensemble import Seq2SeqEnsemble
from tgen.cluster import Job


def get_worker_registrar_for(head):
    """Return a class that will handle worker registration for the given head."""

    class WorkerRegistrarService(Service):
        """An RPyC service to register workers with a head."""

        def exposed_register_worker(self, host, port):
            """Register a worker with my head, initialize it."""
            # initiate connection in the other direction
            log_info('Worker %s:%d connected, initializing training.' % (host, port))
            conn = connect(host, port, config={'allow_pickle': True})
            # initialize the remote server (with training data etc.)
            init_func = async_(conn.root.init_training)
            # add unique 'scope suffix' so that the models don't clash in ensembles
            head.cfg['scope_suffix'] = hashlib.md5("%s:%d" % (host, port)).hexdigest()
            req = init_func(pickle.dumps(head.cfg, pickle.HIGHEST_PROTOCOL))
            # add it to the list of running services
            sc = ServiceConn(host, port, conn)
            head.services.add(sc)
            head.pending_requests.add((sc, None, req))
            log_info('Worker %s:%d initialized.' % (host, port))

    return WorkerRegistrarService


class ParallelSeq2SeqTraining(object):
    """Main (head) that handles parallel Seq2Seq generator training, submitting training jobs and
    collecting their results"""

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
        self.queue_settings = cfg.get('queue_settings')
        self.host = socket.getfqdn()
        self.poll_interval = cfg.get('poll_interval', 1)
        self.average_models = cfg.get('average_models', False)
        self.average_models_top_k = cfg.get('average_models_top_k', 0)
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

    def train(self, das_file, ttree_file, data_portion=1.0, context_file=None, validation_files=None):
        """Run parallel perceptron training, start and manage workers."""
        # initialize the ranker instance
        log_info('Initializing...')
        # run server to process registering clients
        self._init_server()
        # spawn training jobs
        log_info('Spawning jobs...')
        host_short, _ = self.host.split('.', 1)  # short host name for job names
        for j in range(self.jobs_number):
            # set up debugging logfile only if we have it on the head
            debug_logfile = ('"PRT%02d.debug-out.txt.gz"' % j) if is_debug_stream() else 'None'
            job = Job(header='from tgen.parallel_seq2seq_train import run_training',
                      code=('run_training("%s", %d, %s)' %
                            (self.host, self.port, debug_logfile)),
                      name=self.experiment_id + ("PRT%02d-%s-%d" % (j, host_short, self.port)),
                      work_dir=self.work_dir)
            job.submit(memory=self.job_memory, queue=self.queue_settings)
            self.jobs.append(job)

        # run the training passes
        try:
            cur_assign = 0
            results = [None] * self.jobs_number
            rnd_seeds = [rnd.random() for _ in range(self.jobs_number)]

            # assign training and wait for it to finish
            while cur_assign < self.jobs_number or self.pending_requests:
                log_debug('Starting loop over services.')

                # check if some of the pending computations have finished
                for sc, job_no, req in list(self.pending_requests):
                    res = self._check_pending_request(sc, job_no, req)
                    if res is not None:
                        results[job_no] = res, sc

                # check for free services and assign new computation
                while cur_assign < self.jobs_number and self.free_services:
                    log_debug('Assigning request %d' % cur_assign)
                    sc = self.free_services.popleft()
                    log_info('Assigning request %d to %s:%d' % (cur_assign, sc.host, sc.port))
                    if validation_files is not None:
                        validation_files = ','.join([os.path.relpath(f, self.work_dir)
                                                     for f in validation_files.split(',')])
                    train_func = async_(sc.conn.root.train)
                    req = train_func(rnd_seeds[cur_assign],
                                     os.path.relpath(das_file, self.work_dir),
                                     os.path.relpath(ttree_file, self.work_dir),
                                     data_portion,
                                     os.path.relpath(context_file, self.work_dir)
                                     if context_file else None,
                                     validation_files)
                    self.pending_requests.add((sc, cur_assign, req))
                    cur_assign += 1
                    log_debug('Assigned %d' % cur_assign)

                # sleep for a while
                log_debug('Sleeping.')
                time.sleep(self.poll_interval)

            log_info("Results:\n" + "\n".join("%.5f %s:%d" % (cost, sc.host, sc.port)
                                              for cost, sc in results))

            self.model_temp_path = os.path.join(self.work_dir, self.TEMPFILE_NAME)
            results.sort(key=lambda res: res[0])
            # average the computed models
            if self.average_models:
                log_info('Creating ensemble models...')
                # use only top k if required
                results_for_ensemble = (results[:self.average_models_top_k]
                                        if self.average_models_top_k > 0
                                        else results)
                ensemble_model = self.build_ensemble_model(results_for_ensemble)
                log_info('Saving the ensemble model temporarily to %s...' % self.model_temp_path)
                ensemble_model.save_to_file(self.model_temp_path)
            # select the best result on devel data + save it
            else:
                best_cost, best_sc = results[0]
                log_info('Best cost: %f (computed at %s:%d).' % (best_cost, best_sc.host, best_sc.port))
                log_info('Saving best generator temporarily to %s...' % self.model_temp_path)
                # use relative path (working directory of worker jobs is different)
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
        @param job_no: current job number (is None for jobs loading)
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
        if os.path.isfile(orig_tf_session_fname):
            shutil.move(orig_tf_session_fname, tf_session_fname)

        # move the reranking classifier model files as well, if they exist
        orig_clfilter_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tftreecl\1', orig_model_fname)
        orig_clfilter_tf_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tfsess', orig_clfilter_fname)

        if os.path.isfile(orig_clfilter_fname) and os.path.isfile(orig_clfilter_tf_fname):
            clfilter_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tftreecl\1', model_fname)
            clfilter_tf_fname = re.sub(r'((.pickle)?(.gz)?)$', r'.tfsess', clfilter_fname)
            shutil.move(orig_clfilter_fname, clfilter_fname)
            shutil.move(orig_clfilter_tf_fname, clfilter_tf_fname)

    def build_ensemble_model(self, results):
        """Load the models computed by the individual jobs and compose them into a single
        ensemble model.

        @param results: list of tuples (cost, ServiceConn object), where cost is not used"""
        ensemble = Seq2SeqEnsemble(self.cfg)
        models = []
        for _, sc in results:
            models.append((pickle.loads(sc.conn.root.get_all_settings()),
                           pickle.loads(sc.conn.root.get_model_params())))

        rerank_settings = results[0][1].conn.root.get_rerank_settings()
        if rerank_settings is not None:
            rerank_settings = pickle.loads(rerank_settings)
        rerank_params = results[0][1].conn.root.get_rerank_params()
        if rerank_params is not None:
            rerank_params = pickle.loads(rerank_params)

        ensemble.build_ensemble(models, rerank_settings, rerank_params)
        return ensemble


class Seq2SeqTrainingService(Service):
    """RPyC Worker class for a job training a Seq2Seq generator."""

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

    def exposed_train(self, rnd_seed, das_file, ttree_file, data_portion, context_file, validation_files):
        """Run the whole training.
        """
        rnd.seed(rnd_seed)
        log_info('Random seed: %f' % rnd_seed)
        tstart = time.time()
        log_info('Starting training...')
        self.seq2seq.train(das_file, ttree_file, data_portion, context_file, validation_files)
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
        @return: model parameters in a pickled dictionary -- keys are names, values are numpy arrays
        """
        p_dump = pickle.dumps(self.seq2seq.get_model_params(), protocol=pickle.HIGHEST_PROTOCOL)
        return p_dump

    def exposed_get_all_settings(self):
        """Call `get_all_settings` on the worker and return the result as a pickle."""
        settings = pickle.dumps(self.seq2seq.get_all_settings(), protocol=pickle.HIGHEST_PROTOCOL)
        return settings

    def exposed_get_rerank_params(self):
        """Call `get_model_params` on the worker's reranker and return the result as a pickle."""
        if not self.seq2seq.classif_filter:
            return None
        p_dump = pickle.dumps(self.seq2seq.classif_filter.get_model_params(),
                              protocol=pickle.HIGHEST_PROTOCOL)
        return p_dump

    def exposed_get_rerank_settings(self):
        """Call `get_all_settings` on the worker's reranker and return the result as a pickle."""
        if not self.seq2seq.classif_filter:
            return None
        settings = pickle.dumps(self.seq2seq.classif_filter.get_all_settings(),
                                protocol=pickle.HIGHEST_PROTOCOL)
        return settings


def run_training(head_host, head_port, debug_out=None):
    """Main worker training routine (creates the Seq2SeqTrainingService and connects it to the
    head.

    @param head_host: hostname of the head
    @param head_port: head port number
    @param debug_out: path to the debugging output file (debug output discarded if None)
    """
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
