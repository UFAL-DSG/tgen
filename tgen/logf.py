#!/usr/bin/env python
# coding=utf-8


"""
Logging functions.
"""

from __future__ import unicode_literals
from __future__ import print_function

__author__ = "Ondřej Dušek"
__date__ = "2014"

import sys
import codecs
from time import asctime


debug_stream = None
log_stream = sys.stderr


def log_info(message):
    "Print an information message"
    print(asctime(), 'INFO:', message, file=log_stream)
    sys.stderr.flush()


def log_warn(message):
    "Print a warning message"
    print(asctime(), 'WARN:', message, file=log_stream)
    sys.stderr.flush()


def log_debug(*args):
    "Print debug message(s)."
    if not debug_stream:
        return
    print(asctime(), end=' ', file=debug_stream)
    for arg in args:
        print(arg, end=' ', file=debug_stream)
    print(file=debug_stream)
    debug_stream.flush()


def set_debug_stream(stream):
    global debug_stream
    debug_stream = stream


def is_debug_stream():
    """Return True if there is a debug stream (debug logfile) set up."""
    global debug_stream
    return debug_stream is not None
