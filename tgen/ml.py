#!/usr/bin/env python
# coding=utf-8


from __future__ import unicode_literals
from .externals import six
from array import array
from operator import itemgetter
import numpy as np
import inspect
import warnings
from numbers import Number
from bisect import bisect_left

"""
Some helper modules from Scikit-learn 0.14.1, stripped of any SciPy usage
(so that we can use PyPy). Also other general, ML-related stuff.
"""


def softmax(scores):
    """Compute the softmax of the given scores, avoiding overflow of the exponential.
    See http://stackoverflow.com/questions/34968722/softmax-function-python.
    TODO use TF's softmax?
    """
    discounted_exps = np.exp(scores - np.max(scores, axis=0))
    return discounted_exps / np.sum(discounted_exps, axis=0)


# sklearn.base
def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params: dict
        The dictionary to pretty print

    offset: int
        The offset in characters to add at the begin of each line.

    printer:
        The function to convert entries to strings, typically
        the builtin str or repr

    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


# sklearn.utils.validation
def _num_samples(x):
    """Return number of samples in array-like x."""
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        raise TypeError("Expected sequence or array-like, got %r" % x)
    return x.shape[0] if hasattr(x, 'shape') else len(x)


# sklearn.utils.validation
def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Array contains NaN or infinity.")


# sklearn.utils.validation
def check_arrays(*arrays, **options):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.
    By default lists and tuples are converted to numpy arrays.

    It is possible to enforce certain properties, such as dtype, continguity
    and sparse matrix format (if a sparse matrix is passed).

    Converting lists to arrays can be disabled by setting ``allow_lists=True``.
    Lists can then contain arbitrary objects and are not checked for dtype,
    finiteness or anything else but length. Arrays are still checked
    and possibly converted.


    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
        Python lists or tuples occurring in arrays are converted to 1D numpy
        arrays, unless allow_lists is specified.

    sparse_format : 'csr', 'csc' or 'dense', None by default
        If not None, any scipy.sparse matrix is converted to
        Compressed Sparse Rows or Compressed Sparse Columns representations.
        If 'dense', an error is raised when a sparse array is
        passed.

    copy : boolean, False by default
        If copy is True, ensure that returned arrays are copies of the original
        (if not already converted to another format earlier in the process).

    check_ccontiguous : boolean, False by default
        Check that the arrays are C contiguous

    dtype : a numpy dtype instance, None by default
        Enforce a specific dtype.

    allow_lists : bool
        Allow lists of arbitrary objects as input, just check their length.
        Disables
    """
    sparse_format = options.pop('sparse_format', None)
    if sparse_format not in (None, 'csr', 'csc', 'dense'):
        raise ValueError('Unexpected sparse format: %r' % sparse_format)
    copy = options.pop('copy', False)
    check_ccontiguous = options.pop('check_ccontiguous', False)
    dtype = options.pop('dtype', None)
    allow_lists = options.pop('allow_lists', False)
    if options:
        raise TypeError("Unexpected keyword arguments: %r" % options.keys())

    if len(arrays) == 0:
        return None

    n_samples = _num_samples(arrays[0])

    checked_arrays = []
    for array in arrays:
        array_orig = array
        if array is None:
            # special case: ignore optional y=None kwarg pattern
            checked_arrays.append(array)
            continue
        size = _num_samples(array)

        if size != n_samples:
            raise ValueError("Found array with dim %d. Expected %d"
                             % (size, n_samples))

        if not allow_lists or hasattr(array, "shape"):
            if check_ccontiguous:
                array = np.ascontiguousarray(array, dtype=dtype)
            else:
                array = np.asarray(array, dtype=dtype)
            _assert_all_finite(array)

        if copy and array is array_orig:
            array = array.copy()
        checked_arrays.append(array)

    return checked_arrays


# sklearn.utils.validation
def warn_if_not_float(X, estimator='This algorithm'):
    """Warning utility function to check that data type is floating point.

    Returns True if a warning was raised (i.e. the input is not float) and
    False otherwise, for easier input validation.
    """
    if not isinstance(estimator, six.string_types):
        estimator = estimator.__class__.__name__
    if X.dtype.kind != 'f':
        warnings.warn("%s assumes floating point values as input, "
                      "got %s" % (estimator, X.dtype))
        return True
    return False


# sklearn.preprocessing.data
def _mean_and_std(X, axis=0, with_mean=True, with_std=True):
    """Compute mean and std deviation for centering, scaling.

    Zero valued std components are reset to 1.0 to avoid NaNs when scaling.
    """
    X = np.asarray(X)
    Xr = np.rollaxis(X, axis)

    if with_mean:
        mean_ = Xr.mean(axis=0)
    else:
        mean_ = None

    if with_std:
        std_ = Xr.std(axis=0)
        if isinstance(std_, np.ndarray):
            std_[std_ == 0.0] = 1.0
        elif std_ == 0.:
            std_ = 1.
    else:
        std_ = None

    return mean_, std_


# sklearn.utils.fixes
# little danse to see if np.copy has an 'order' keyword argument
if 'order' in inspect.getargspec(np.copy)[0]:
    def safe_copy(X):
        # Copy, but keep the order
        return np.copy(X, order='K')
else:
    # Before an 'order' argument was introduced, numpy wouldn't muck with
    # the ordering
    safe_copy = np.copy


# sklearn.utils.validation
def array2d(X, dtype=None, order=None, copy=False, force_all_finite=True):
    """Returns at least 2-d array with data from X"""
    X_2d = np.asarray(np.atleast_2d(X), dtype=dtype, order=order)
    if force_all_finite:
        _assert_all_finite(X_2d)
    if X is X_2d and copy:
        X_2d = safe_copy(X_2d)
    return X_2d


# sklearn.utils.validation
def _atleast2d_or_sparse(X, dtype, order, copy, force_all_finite):
    X = array2d(X, dtype=dtype, order=order, copy=copy,
                force_all_finite=force_all_finite)
    if force_all_finite:
        _assert_all_finite(X)
    return X


# sklearn.utils.validation
def atleast2d_or_csr(X, dtype=None, order=None, copy=False,
                     force_all_finite=True):
    """Like numpy.atleast_2d, but converts sparse matrices to CSR format

    Also, converts np.matrix to np.ndarray.
    """
    return _atleast2d_or_sparse(X, dtype, order, copy, force_all_finite)


# sklearn.base
class BaseEstimator(object):
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their __init__ as explicit keyword
    arguments (no *args, **kwargs).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        try:
            # fetch the constructor or the original constructor before
            # deprecation wrapping if any
            init = getattr(cls.__init__, 'deprecated_original', cls.__init__)

            # introspect the constructor arguments to find the model parameters
            # to represent
            args, varargs, kw, default = inspect.getargspec(init)
            if not varargs is None:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s doesn't follow this convention."
                                   % (cls,))
            # Remove 'self'
            # XXX: This is going to fail if the init is a staticmethod, but
            # who would do this?
            args.pop(0)
        except TypeError:
            # No explicit __init__
            args = []
        args.sort()
        return args

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The former have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in six.iteritems(params):
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if not name in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if not key in valid_params:
                    raise ValueError('Invalid parameter %s ' 'for estimator %s'
                                     % (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name),),)

    def __str__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name,
                           _pprint(self.get_params(deep=True),
                                   offset=len(class_name), printer=str,),)


# sklearn.base
class TransformerMixin(object):
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


# sklearn.preprocessing.data
class StandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using the
    `transform` method.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual feature do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    Parameters
    ----------
    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default is True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    Attributes
    ----------
    `mean_` : array of floats with shape [n_features]
        The mean value for each feature in the training set.

    `std_` : array of floats with shape [n_features]
        The standard deviation for each feature in the training set.

    See also
    --------
    :func:`sklearn.preprocessing.scale` to perform centering and
    scaling without using the ``Transformer`` object oriented API

    :class:`sklearn.decomposition.RandomizedPCA` with `whiten=True`
    to further remove the linear correlation across features.
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like or CSR matrix with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        X = check_arrays(X, copy=self.copy, sparse_format="csr")[0]
        if warn_if_not_float(X, estimator=self):
            X = X.astype(np.float)
        self.mean_, self.std_ = _mean_and_std(
            X, axis=0, with_mean=self.with_mean, with_std=self.with_std)
        return self

    def transform(self, X, y=None, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        copy = copy if copy is not None else self.copy
        X = check_arrays(X, copy=copy, sparse_format="csr")[0]
        if warn_if_not_float(X, estimator=self):
            X = X.astype(np.float)
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.std_
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        copy = copy if copy is not None else self.copy
        X = np.asarray(X)
        if copy:
            X = X.copy()
        if self.with_std:
            X *= self.std_
        if self.with_mean:
            X += self.mean_
        return X


class DictVectorizer(BaseEstimator, TransformerMixin):
    """Transforms lists of feature-value mappings to vectors.

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    Parameters
    ----------
    dtype : callable, optional
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator: string, optional
        Separator string used when constructing new features for one-hot
        coding.
    sparse: boolean, optional.
        Whether transform should produce scipy.sparse matrices.
        True by default.

    Attributes
    ----------
    `feature_names_` : list
        A list of length n_features containing the feature names (e.g., "f=ham"
        and "f=spam").

    `vocabulary_` : dict
        A dictionary mapping feature names to feature indices.

    Examples
    --------
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> v = DictVectorizer(sparse=False)
    >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    >>> X = v.fit_transform(D)
    >>> X
    array([[ 2.,  0.,  1.],
           [ 0.,  1.,  3.]])
    >>> v.inverse_transform(X) == \
        [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]
    True
    >>> v.transform({'foo': 4, 'unseen_feature': 3})
    array([[ 0.,  0.,  4.]])

    See also
    --------
    DictVectorizer : performs vectorization in a similar as this class,
      but using a hash table instead of only a hash function.
    sklearn.preprocessing.OneHotEncoder : handles nominal/categorical features
      encoded as columns of integers.
    """

    def __init__(self, dtype=np.float64, separator="=", sparse=False, binarize_numeric=False):
        self.dtype = dtype
        self.separator = separator
        self.binarize_numeric = binarize_numeric

    def fit(self, X, y=None):
        """Learn a list of feature name -> indices mappings.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        self
        """
        binarize_numeric = self.binarize_numeric
        self.num_bounds_ = {}

        # examine values of numeric features, get low and high bound + number of distinct values
        if binarize_numeric:
            numeric_feats = {}
            for x in X:
                for f, v in six.iteritems(x):
                    if isinstance(v, Number):
                        if f not in numeric_feats:
                            lo, hi, vals = v, v, set([v])
                        else:
                            lo, hi, vals = numeric_feats[f]
                            if lo > v:
                                lo = v
                            if hi < v:
                                hi = v
                            if isinstance(vals, set):
                                if len(vals) >= 4:
                                    vals = None
                                else:
                                    vals.add(v)
                        numeric_feats[f] = (lo, hi, vals)

            # compute boundary values (only if there are more than 4 distinct values)
            for f, (lo, hi, vals) in numeric_feats.iteritems():
                if vals is None:
                    avg = (lo + hi) / 2
                    self.num_bounds_[f] = [(lo + avg) / 2, avg, (hi + avg) / 2]

        # collect all the possible feature names
        feature_names = set()
        for x in X:
            for f, v in six.iteritems(x):
                if isinstance(v, six.string_types):
                    f = "%s%s%s" % (f, self.separator, v)
                elif binarize_numeric:
                    if f in self.num_bounds_:
                        # feature name is the number of the interval given by the bounds
                        f = "%s%sInt%d" % (f, self.separator, bisect_left(self.num_bounds_[f], v))
                    else:
                        # just a few distinct values (max. 4), make a feature out of each one of them
                        f = "%s%s%f" % (f, self.separator, float(v))
                feature_names.add(f)

        # sort the feature names to define the mapping
        feature_names = sorted(feature_names)
        self.vocabulary_ = dict((f, i) for i, f in enumerate(feature_names))
        self.feature_names_ = feature_names

        return self

    def fit_transform(self, X, y=None):
        """Learn a list of feature name -> indices mappings and transform X.

        Like fit(X) followed by transform(X).

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.

        Notes
        -----
        Because this method requires two passes over X, it materializes X in
        memory.
        """
        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        """Transform feature->value dicts to array or sparse matrix.

        Named features not encountered during fit or fit_transform will be
        silently ignored.

        Parameters
        ----------
        X : Mapping or iterable over Mappings, length = n_samples
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        Xa : {array, sparse matrix}
            Feature vectors; always 2-d.
        """
        dtype = self.dtype
        vocab = self.vocabulary_
        num_bounds = self.num_bounds_
        binarize_numeric = self.binarize_numeric

        Xa = np.zeros((len(X), len(vocab)), dtype=dtype)

        for i, x in enumerate(X):
            for f, v in six.iteritems(x):
                if isinstance(v, six.string_types):
                    f = "%s%s%s" % (f, self.separator, v)
                    v = 1
                elif binarize_numeric:
                    if f in num_bounds:
                        f = "%s%sInt%d" % (f, self.separator, bisect_left(num_bounds[f], v))
                    else:
                        f = "%s%s%f" % (f, self.separator, float(v))
                    v = 1
                try:
                    Xa[i, vocab[f]] = dtype(v)
                except KeyError:
                    pass

        return Xa

    def get_feature_names(self):
        """Returns a list of feature names, ordered by their indices.

        If one-of-K coding is applied to categorical features, this will
        include the constructed feature names but not the original ones.
        """
        return self.feature_names_

    def __setstate__(self, state):
        """Backwards compatibility on loading."""
        if 'num_bounds_' not in state:
            state['num_bounds_'] = {}
            state['binarize_numeric'] = False
        self.__dict__ = state
