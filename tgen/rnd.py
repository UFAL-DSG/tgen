#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
My own random number generator instance which is used instead of the system one to prevent
Theano from interfering with it.

"""

from random import Random

rnd = Random()
rnd.seed(1206)
