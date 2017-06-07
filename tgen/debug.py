#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Various debugging utilities.

Sources:

Starting IPdb automatically on error:
- http://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
- https://github.com/ticcky/code101/blob/master/pdb_on_error.py

Inspecting Theano functions:
- http://www.deeplearning.net/software/theano/tutorial/debug_faq.html
"""

import sys


def exc_info_hook(exc_type, value, tb):
    """An exception hook that starts IPdb automatically on error if in interactive mode."""

    if hasattr(sys, 'ps1') or not sys.stderr.isatty() or exc_type == KeyboardInterrupt:
        # we are in interactive mode, we don't have a tty-like
        # device,, or the user triggered a KeyboardInterrupt,
        # so we call the default hook
        sys.__excepthook__(exc_type, value, tb)
    else:
        import traceback
        # import ipdb
        import pudb
        # we are NOT in interactive mode, print the exception
        traceback.print_exception(exc_type, value, tb)
        print
        raw_input("Press any key to start debugging...")
        # then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        # ipdb.post_mortem(tb)  # more modern
        pudb.post_mortem(tb)


def inspect_inputs(i, node, fn):
    """Inspecting Theano function graph in MonitorMode: print inputs."""
    print i, node, "IN:", [inp[0] for inp in fn.inputs],


def inspect_outputs(i, node, fn):
    """Inspecting Theano function graph in MonitorMode: print outputs."""
    print "OUT:", [output[0] for output in fn.outputs]


def inspect_input_dims(i, node, fn):
    """Inspecting Theano function graph in MonitorMode: print inputs' dimensions."""
    print i, node, "IN:", [inp[0].shape if inp[0].shape else 'scalar/' + str(inp[0]) for inp in fn.inputs],


def inspect_output_dims(i, node, fn):
    """Inspecting Theano function graph in MonitorMode: print outputs' dimensions."""
    print "OUT:", [output[0].shape if output[0].shape else 'scalar/' + str(output[0]) for output in fn.outputs]
