#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load Python code as configuration files. A copy from Alex (http://github.com/UFAL-DSG/alex)
"""

from builtins import object
from importlib import import_module
import os
import os.path
import sys
import tempfile
import codecs
import yaml

config = None


def _expand_file_var(text, path):
    # This method has clear limitations, since it ignores the whole Python
    # syntax.
    return text.replace('__file__', "'{p}'".format(p=path))


def load_as_module(path, force=False):
    """Loads a file pointed to by `path' as a Python module with minimal impact
    on the global program environment.  The file name should end in '.py'.

    Arguments:
        path -- path towards the file
        force -- whether to load the file even if its name does not end in
                 '.py'

    Returns the loaded module object.

    """
    do_delete_temp = False
    if not path.endswith('.py'):
        if force:
            happy = False
            while not happy:
                temp_fd, temp_path = tempfile.mkstemp(suffix='.py')
                dirname, basename = os.path.split(temp_path)
                modname = basename[:-3]
                if modname not in sys.modules:
                    happy = True
            temp_file = os.fdopen(temp_fd, 'wb')
            temp_file.write(_expand_file_var(open(path, 'rb').read(), path))
            temp_file.close()
            path = temp_path
            do_delete_temp = True
        else:
            raise ValueError(("Path `{path}' should be loaded as module but "
                              "does not end in '.py' and `force' wasn't set.")
                             .format(path=path))
    else:
        dirname, basename = os.path.split(path)
        modname = basename[:-3]
    sys.path.insert(0, dirname)
    mod = import_module(modname)
    sys.path.pop(0)
    if do_delete_temp:
        os.unlink(temp_path)
    del sys.modules[modname]
    return mod


class Config(object):
    """
    Config handles configuration data necessary for all the components
    in Alex. It is implemented using a dictionary so that any component can use
    arbitrarily structured configuration data.

    When the configuration file is loaded, several automatic transformations
    are applied:

        1. '{cfg_abs_path}' as a substring of atomic attributes is replaced by
            an absolute path of the configuration files.  This can be used to
            make the configuration file independent of the location of programs
            using the configuration file.

    """
    # TODO: Enable setting requirements on the configuration variables and
    # checking that they are met (i.e., 2 things:
    #   - requirements = property(get_reqs, set_reqs)
    #   - def check_requirements_are_met(self)

    def __init__(self, file_name=None, config={}):
        self.config = config

        if file_name:
            self.load(file_name)

    def get(self, i, default=None):
        return self.config.get(i, default)

    def __delitem__(self, i):
        del self.config[i]

    def __len__(self):
        return len(self.config)

    def __getitem__(self, i):
        return self.config[i]

    def __setitem__(self, key, val):
        self.config[key] = val

    def __iter__(self):
        for i in self.config:
            yield i

    def contains(self, *path):
        """Check if configuration contains given keys (= path in config tree)."""
        curr = self.config
        for path_part in path:
            if path_part in curr:
                curr = curr[path_part]
            else:
                return False

        return True

    def load(self, file_name):
        if file_name.endswith('.yaml'):
            with codecs.open(file_name, 'r', encoding='UTF-8') as fh:
                self.config = yaml.load(fh, Loader=yaml.FullLoader)
        else:
            # pylint: disable-msg=E0602
            global config
            # config = None
            self.config = config = load_as_module(file_name, force=True).config
            # execfile(file_name, globals())
            # assert config is not None
            # self.config = config

            cfg_abs_dirname = os.path.dirname(os.path.abspath(file_name))
            self.config_replace('{cfg_abs_path}', cfg_abs_dirname)
