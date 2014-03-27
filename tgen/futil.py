#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Various utility functions.
"""

from __future__ import unicode_literals

from alex.components.nlg.tectotpl.core.util import file_stream
from alex.components.slu.da import DialogueAct


def read_das(da_file):
    """Read dialogue acts from a file, one-per-line."""
    das = []
    with file_stream(da_file) as fh:
        for line in fh:
            da = DialogueAct()
            da.parse(line)
            das.append(da)
    return das
