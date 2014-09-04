#!/usr/bin/env python

from alex.utils.cache import lru_cache

class T(object):
    def __init__(self):
        self.param = 1
        
    @lru_cache(maxsize=100)
    def func(self, x):
        return x + 1 + self.param

a = T()
print a.func(1)
print a.func(1)
print a.func.hits, a.func.misses
a.param = 2
print a.func(1)
print a.func.hits, a.func.misses
