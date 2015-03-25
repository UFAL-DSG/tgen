# ##

# import functools
# import inspect
# import traceback
# import sys
#
# def monitor(f):
#     @functools.wraps(f)
#     def decorated(*args, **kwargs):
#         print >> sys.stderr, "*** RND {}: {}, {}".format(f.__name__, args, kwargs)
#         # traceback.print_stack()
#         # Do whatever you want: increment some counter, etc.
#         return f(*args, **kwargs)
#     return decorated

from random import Random

# for name in dir(random):
#     f = getattr(random, name)
#     if inspect.isfunction(f) or inspect.ismethod(f) and not name.startswith('_'):
#         log_info('Wrapping ' + name)
#         setattr(random, name, monitor(f))

# ##


rnd = Random()
rnd.seed(1206)
