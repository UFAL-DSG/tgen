#!/usr/bin/env python
# coding=utf-8


from __future__ import unicode_literals
import os
import commands
import string
import random
import codecs
import sys
import re
import time
import collections
from tgen.logf import log_warn

"""\
Interface for running any Python code as a job on the cluster
(using the qsub/qstat/qacct commands).

Tested with Sun Grid Engine.

This code is copied from https://github.com/UFAL-DSG/flect
to remove an otherwise unneccessary dependency.
"""

__author__ = "Ondřej Dušek"
__date__ = "2012"


def first(condition_function, sequence, default=None):
    """\
    Return first item in sequence where condition_function(item) == True,
    or None if no such item exists.
    """
    for item in sequence:
        if condition_function(item):
            return item
    return default


class Job(object):
    """\
    This represents a piece of code as a job on the cluster, holds
    information about the job and is able to retrieve job metadata.

    The most important method is submit(), which submits the given
    piece of code to the cluster.

    Important attributes (some may be set in the constructor or
    at job submission, but all may be set between construction and
    launch):
    ------------------------------------------------------------------
    name      -- job name on the cluster (and the name of the created
                 Python script, default will be generated if not set)
    code      -- the Python code to be run (needs to have imports and
                 sys.path set properly)
    header    -- the header of the created Python script (may contain
                 imports etc.)
    memory    -- the amount of memory to reserve for this job on the
                 cluster
    cores     -- the number of cores needed for this job
    work_dir  -- the working directory where the job script will be
                 created and run (will be created on launch)
    dependencies-list of Jobs this job depends on (must be submitted
                 before submitting this job)
    queue     -- queue setting for SGE

    In addition, the following values may be queried for each job
    at runtime or later:
    ------------------------------------------------------------------
    submitted -- True if the job has been submitted to the cluster.
    state     -- current job state ('qw' = queued, 'r' = running, 'f'
                 = finished, only if the job was submitted)
    host      -- the machine where the job is running (short name)
    jobid     -- the numeric id of the job in the cluster (NB: type is
                 string!)
    report    -- job report using the qacct command (dictionary,
                 available only after the job has finished)
    exit_status- numeric job exit status (if the job is finished)
    """

    # default job header
    DEFAULT_HEADER = """\
#!/usr/bin/env python
# coding=utf8
from __future__ import unicode_literals
"""
    # job state 'FINISHED' symbol
    FINISH = 'f'
    # job name prefix
    NAME_PREFIX = 'pyjob_'
    # job directory prefix
    DIR_PREFIX = '_clrun-'
    # legal chars for generated job names
    JOBNAME_LEGAL_CHARS = string.ascii_letters + string.digits
    # default number of cores
    DEFAULT_CORES = 1
    # default memory size (in GBs)
    DEFAULT_MEMORY = 4
    # only 1 job status query per second
    TIME_QUERY_DELAY = 1
    # qsub multicore command
    QSUB_MULTICORE_CMD = '-pe smp {0}'
    # qsub memory command
    QSUB_MEMORY_CMD = '-hard -l mem_free={0} -l act_mem_free={0}' + \
                      ' -l h_vmem={0}'
    # qsub queue command
    QSUB_QUEUE_CMD = '-q "{0}"'
    # job status polling delay for wait() in seconds
    TIME_POLL_DELAY = 60

    def __init__(self, code=None, header=DEFAULT_HEADER,
                 name=None, work_dir=None, dependencies=None):
        """\
        Constructor. May provide some running options --
        the desired Python code to be run, the headers of the resulting
        script (default provided), the job name and working directory.
        All of these options can be set later via the corresponding
        attributes.
        """
        self.header = header
        self.code = code
        self.memory = self.DEFAULT_MEMORY
        self.cores = self.DEFAULT_CORES
        self.queue = None
        self.__jobid = None
        self.__host = None
        self.__state = None
        self.__report = None
        self.__state_last_query = time.time()
        self.__dependencies = []
        if dependencies is not None:
            self.add_dependency(dependencies)
        self.__name = name if name is not None else self.__generate_name()
        self.submitted = False
        self.work_dir = (work_dir
                         if work_dir is not None
                         else self.__get_work_dir())

    def submit(self, memory=None, cores=None, work_dir=None, queue=None):
        """\
        Submit the job to the cluster. Override the pre-set memory and
        cores defaults if necessary.
        The job code, header and working directory must be set in advance.
        All jobs on which this job is dependent must already be submitted!
        """
        if cores is not None:
            self.cores = cores
        if memory is not None:
            self.memory = memory
        if queue is not None:
            self.queue = queue
        # create working directory if necessary
        if not os.path.isdir(self.work_dir):
            os.mkdir(self.work_dir)
        cwd = os.getcwdu()
        os.chdir(self.work_dir)
        # create the script
        script_fh = codecs.open(self.name + '.py', 'w', 'UTF-8')
        print >> script_fh, self.get_script_text()
        script_fh.close()
        # submit the script
        command = 'qsub ' + self.__get_resource_requests() + \
                  ' ' + self.__get_dependency_string() + \
                  ' -V -cwd -j y -S ' + sys.executable + ' ' + \
                  self.name + '.py'
        output = self.__try_command(command)
        self.__jobid = re.search('([0-9]+)', output).group(0)
        self.submitted = True
        os.chdir(cwd)

    @property
    def state(self):
        """\
        Retrieve information about current job state. Will also
        retrieve the host this job is running on and store it in
        the __host variable, if applicable.
        """
        # job hasn't been submitted -- no point in retrieving state
        if not self.submitted:
            return None
        # state caching
        if time.time() < self.__state_last_query + self.TIME_QUERY_DELAY:
            return self.__state
        self.__state_last_query = time.time()
        # actually retrieve the state
        state, host = self.__get_job_state()
        self.__state = state
        if state != self.FINISH:
            self.__host = host
        return self.__state

    @property
    def report(self):
        """\
        Access to qacct report. Please note that running the qacct command
        takes a few seconds, so the first access to the report is rather
        slow.
        """
        # no stats until the job has finished
        if not self.submitted or self.state != self.FINISH:
            return None
        # the report is retrieved only once
        if self.__report is None:
            # try to retrieve the qacct report
            output = self.__try_command('qacct -j ' + self.jobid)
            self.__report = {}
            for line in output.split("\n")[1:]:
                key, val = re.split(r'\s+', line, 1)
                self.__report[key] = val.strip()
        return self.__report

    @property
    def exit_status(self):
        """\
        Retrieve the exit status of the job via the qacct report.
        Throws an exception the job is still running and the exit status
        is not known.
        """
        report = self.report
        if report is None:
            raise RuntimeError('Job' + self.jobid +
                               ' is probably still running')
        return int(report['exit_status'])

    def wait(self, poll_delay=None):
        """\
        Waits for the job to finish. Will raise an exception if the
        job did not finish successfully. The poll_delay variable controls
        how often the job state is checked.
        """
        poll_delay = poll_delay if poll_delay else self.TIME_POLL_DELAY
        while self.state != self.FINISH:
            time.sleep(poll_delay)
        if self.exit_status != 0:
            raise RuntimeError('Job ' + self.name + ' (' + self.jobid +
                               ') did not finish successfully.')

    def add_dependency(self, dependency):
        """\
        Adds a dependency on the given Job(s).
        """
        if isinstance(dependency, Job) or isinstance(dependency, basestring):
            self.__dependencies.append(dependency)
        elif isinstance(dependency, int):
            self.__dependencies.append(str(dependency))
        elif isinstance(dependency, collections.Iterable):
            for dep_elem in dependency:
                self.add_dependency(dep_elem)
        else:
            raise ValueError('Unknown dependency type!')

    def remove_dependency(self, dependency):
        """\
        Removes the given Job(s) from the dependencies list.
        """
        # single element removed
        if isinstance(dependency, (Job, basestring, int)):
            if isinstance(dependency, int):
                jobid = str(dependency)
            else:
                jobid = dependency
            rem = first(lambda d: d == jobid, self.__dependencies)
            if rem is not None:
                self.__dependencies.remove(rem)
            else:
                raise ValueError('Cannot find dependency!')
        elif isinstance(dependency, collections.Iterable):
            for dep_elem in dependency:
                self.remove_dependency(dep_elem)
        else:
            raise ValueError('Unknown dependency type!')

    def delete(self):
        """Delete this job."""
        if self.submitted:
            try:
                self.__try_command('qdel ' + self.jobid)
            except RuntimeError as e:
                log_warn('Could not delete job: ' + str(e))

    @property
    def host(self):
        """\
        Retrieve information about the host this job is/was
        running on.
        """
        # no point if the job has not been submitted
        if not self.submitted:
            return None
        # return a cached value
        if self.__host is not None:
            return self.__host
        # try to get state and return the stored value
        self.state()
        return self.__host

    @property
    def name(self):
        """\
        Return the job name.
        """
        return self.__name

    @property
    def jobid(self):
        """\
        Return the job id.
        """
        return self.__jobid

    def get_script_text(self):
        """\
        Join headers and code to create a meaningful Python script.
        """
        text = self.header
        text += "\ndef main():\n"
        text += re.sub('^', '    ', self.code, 0, re.MULTILINE) + "\n\n"
        text += "if __name__ == '__main__':\n    main()\n"
        return text

    def __generate_name(self):
        """\
        Generate a job name
        """
        return self.NAME_PREFIX + \
            ''.join([random.choice(self.JOBNAME_LEGAL_CHARS)
                     for _ in xrange(5)])

    def __get_work_dir(self):
        """\
        Generate a valid working directory name
        """
        num = 1
        workdir = None
        while workdir is None or os.path.exists(workdir):
            workdir = (os.getcwdu() + os.path.sep + self.DIR_PREFIX +
                       self.name + '-' + str(num).zfill(3))
            num += 1
        return workdir

    def __get_resource_requests(self):
        """\
        Generate qsub resource requests based on the memory and core setting.
        """
        res = self.QSUB_MEMORY_CMD.format(str(self.memory) + 'G')
        if self.cores > 1:
            res = self.QSUB_MULTICORE_CMD.format(self.cores) + ' ' + res
        if self.queue is not None:
            res += ' ' + self.QSUB_QUEUE_CMD.format(self.queue)
        return res

    def __get_dependency_string(self):
        """\
        Generate qsub dependency string based on the list of dependencies.
        """
        if self.__dependencies:
            if not all([dep.submitted if isinstance(dep, Job) else True
                        for dep in self.__dependencies]):
                raise RuntimeError('Job has unsubmitted dependencies!')
            return '-hold_jid ' + ','.join([dep.jobid if isinstance(dep, Job)
                                            else dep
                                            for dep in self.__dependencies])
        return ''

    def __try_command(self, cmd):
        """\
        Try to run a command and return its output. If the command fails,
        throw a RuntimeError.
        """
        status, output = commands.getstatusoutput(cmd)
        if status != 0:
            raise RuntimeError('Command \'' + cmd + '\' failed. Status: ' +
                               str(status) + ', Output: ' + output)
        return output

    def __get_job_state(self):
        """\
        Parse the qstat command and try to retrieve the current job
        state and the machine it is running on.
        """
        # get state of job assuming it is in the queue
        output = self.__try_command('qstat')
        # get the relevant line of the qstat output
        output = first(lambda line: re.search(self.jobid, line),
                       output.split("\n"))
        # job does not exist anymore
        if output is None:
            return self.FINISH, None
        # parse the correct line:
        fields = re.split(r'\s+', output)
        state, host = fields[4], fields[7]
        host = re.sub(r'.*@([^.]+)\..*', r'\1', host)
        return state, host

    def __eq__(self, other):
        """\
        Comparison: based on ids or reference if ids are None.
        """
        if self.__jobid is not None and other.__jobid is not None:
            return self.__jobid == other.__jobid
        return self == other

    def __str__(self):
        """\
        String representation returns the attribute name and type.
        """
        return (self.__class__.__name__ + ': ' +
                self.name + ' (' + self.work_dir + ')')
