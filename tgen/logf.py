#!/usr/bin/env python
# coding=utf-8


"""
Logging functions.
"""

from __future__ import unicode_literals

__author__ = "Ondřej Dušek"
__date__ = "2014"

import sys
import codecs
from time import asctime


debug_stream = None
log_stream = codecs.getwriter('utf-8')(sys.stderr)


def log_info(message):
    "Print an information message"
    print >> log_stream, asctime(), 'INFO:', message
    sys.stderr.flush()


def log_warn(message):
    "Print a warning message"
    print >> log_stream, asctime(), 'WARN:', message
    sys.stderr.flush()


def log_debug(*args):
    "Print debug message(s)."
    if not debug_stream:
        return
    print >> debug_stream, asctime(),
    for arg in args:
        print >> debug_stream, arg,
    print >> debug_stream
    debug_stream.flush()


def set_debug_stream(stream):
    global debug_stream
    debug_stream = stream


def is_debug_stream():
    """Return True if there is a debug stream (debug logfile) set up."""
    global debug_stream
    return debug_stream is not None
