import os
import numpy as np
import joblib
import warnings
from joblib import Memory
from scipy.sparse import vstack
from scipy.sparse import csc_matrix
from scipy.sparse import issparse, isspmatrix_csr

from sklearn.datasets import make_classification
from sklearn.datasets import load_svmlight_file


def maybe_cache(func):
    """A decorator to decide whether or not to cache dataset."""
    if 'JOBLIB_CACHE_DIR' in os.environ:
        cachedir = os.environ['JOBLIB_CACHE_DIR']
        memory = Memory(cachedir=cachedir, verbose=0)
        # print("Loading dataset from cached memory in {}.".format(cachedir))

        func_wrapper = memory.cache(func)
    else:
        print("Environment variable JOBLIB_CACHE_DIR is not defined.")

        def func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
    return func_wrapper


def dist_coord_selection(rank, world_size, max_index, n_coords=None):
    if not n_coords:
        n_coords = max_index
    shuffled_index = np.random.permutation(np.arange(max_index))[:n_coords]
    rank_coords = shuffled_index[rank::world_size]
    return rank_coords


@maybe_cache
def dist_row_read_one(rank, world_size, n_blob, n_features, filename, zero_based, length=10000, datapoints=0, verbose=1):
    X_train, y_train = [], []
    rank_coords = []
    total_data_loaded = 0
    offset = 0

    while True:
        X, y = load_svmlight_file(filename, n_features=n_features, offset=offset,
                                  zero_based=zero_based, length=length)

        if (total_data_loaded + X.shape[0] >= n_blob):
            X, y = X[:n_blob - total_data_loaded], y[:n_blob - total_data_loaded]

        file_data_ids = dist_coord_selection(rank, world_size, X.shape[0])
        X_train.append(X[file_data_ids])
        y_train.append(y[file_data_ids])

        rank_coords += [total_data_loaded + i for i in file_data_ids]
        total_data_loaded += X.shape[0]
        if total_data_loaded >= n_blob:
            break
        
        if verbose >= 3:
            print("Rank {:2}: {} : {:.3f}".format(rank, total_data_loaded,  total_data_loaded / n_blob))
        offset += length

    X_train = vstack(X_train)
    y_train = np.concatenate(y_train)
    return X_train, y_train, rank_coords


@maybe_cache
def dist_col_read_one(rank, world_size, n_blob, n_features, filename, zero_based, length=100, datapoints=0, verbose=1):
    rank_coords = dist_coord_selection(rank, world_size, n_features, datapoints)

    offset = 0
    X_train, y_train = None, None
    while True:
        X, y = load_svmlight_file(filename, n_features=n_features, offset=offset,
                                  zero_based=zero_based, length=length)

        if X_train is None:
            X_train, y_train = X[:, rank_coords], y
        elif n_blob - X_train.shape[0] > X.shape[0]:
            X_train = vstack((X_train, X[:, rank_coords]))
            y_train = np.concatenate((y_train, y))
        else:
            X, y = X[:n_blob - X_train.shape[0]], y[:n_blob - X_train.shape[0]]
            X_train = vstack((X_train, X[:, rank_coords]))
            y_train = np.concatenate((y_train, y))
            break
        if verbose >= 3:
            print("Rank {:2}: {} : {:.3f}".format(rank, X_train.shape[0],  X_train.shape[0] / n_blob))

        offset += length

    return X_train, y_train, rank_coords


@maybe_cache
def dist_col_read(rank, world_size, n_blob, n_features, filenames, verbose=1):
    """Read svmlight files in a distributed way.

    Split the dataset by columns.

    Parameters
    ----------
    rank : {int}
        Rank of current process
    world_size : {int}
        Total number of processes.
    n_blob : {int}
        Maximum number of data point used in this network
    n_features : {int}
        Number of features in the datset
    filenames : {list}
        List of paths to the files

    Returns
    -------
    X_train : {sparse matrix}
        Feature matrix split by columns.
    y_train : {array}
        Labels of the dataset
    rank_coords :
        Indices of the coordinates used in this rank.
    """
    rank_coords = dist_coord_selection(rank, world_size, n_features)

    X_train, y_train = None, None
    for filename in filenames:
        if verbose >= 3:
            print("Rank {:3} is loading {}".format(rank, filename))
        X, y = load_svmlight_file(filename, n_features=n_features, length=-1)

        if X_train is None:
            X_train, y_train = X[:, rank_coords], y
        elif n_blob - X_train.shape[0] > X.shape[0]:
            X_train = vstack((X_train, X[:, rank_coords]))
            y_train = np.concatenate((y_train, y))
        else:
            X, y = X[:n_blob - X_train.shape[0]], y[:n_blob - X_train.shape[0]]
            X_train = vstack((X_train, X[:, rank_coords]))
            y_train = np.concatenate((y_train, y))
            break
    return X_train, y_train, rank_coords


@maybe_cache
def dist_row_read(rank, world_size, n_blob, n_features, filenames, verbose=1):
    """Read svmlight files in a distributed way.

    Split the dataset by rows.

    Parameters
    ----------
    rank : {int}
        Rank of current process
    world_size : {int}
        Total number of processes.
    n_blob : {int}
        Maximum number of data point used in this network
    n_features : {int}
        Number of features in the datset
    filenames : {list}
        List of paths to the files

    Returns
    -------
    X_train : {sparse matrix}
        Feature matrix split by columns.
    y_train : {array}
        Labels of the dataset
    rank_coords :
        Indices of the coordinates used in this rank.
    """
    X_train, y_train = [], []
    rank_coords = []
    total_data_loaded = 0

    for filename in filenames:
        if verbose >= 3:
            print("Rank {:3} is loading {}".format(rank, filename))
        X, y = load_svmlight_file(filename, n_features=n_features, length=-1)

        if (total_data_loaded + X.shape[0] >= n_blob):
            X, y = X[:n_blob - total_data_loaded], y[:n_blob - total_data_loaded]

        file_data_ids = dist_coord_selection(rank, world_size, X.shape[0])
        X_train.append(X[file_data_ids])
        y_train.append(y[file_data_ids])

        rank_coords += [total_data_loaded + i for i in file_data_ids]
        total_data_loaded += X.shape[0]
        if total_data_loaded >= n_blob:
            break

    X_train = vstack(X_train)
    y_train = np.concatenate(y_train)
    return X_train, y_train, rank_coords


class Epsilon(object):
    @staticmethod
    def dist_read(rank, world_size, file, percent=0.1, split_by='samples', random_state=42, datapoints=None, verbose=1):
        np.random.seed(random_state)
        n_blob = int(400000 * percent)
        n_features = 2000
        zero_based = False
        if split_by == 'features':
            A_block, b, features_in_partition = dist_col_read_one(rank, world_size, n_blob, n_features, file,
                                                                  zero_based, length=100*1024**2, datapoints=datapoints)
        elif split_by == 'samples':
            A_block, b, features_in_partition = dist_row_read_one(rank, world_size, n_blob, n_features, file,
                                                                  zero_based, length=100*1024**2, datapoints=datapoints)
        else:
            raise NotImplementedError

        return A_block, b, features_in_partition, n_blob, n_features


class URL(object):
    @staticmethod
    def dist_read(rank, world_size, file, percent=0.1, split_by='features', random_state=42, datapoints=None, verbose=1):
        np.random.seed(random_state)
        n_blob = int(2396130 * percent)
        n_features = 3231961
        zero_based = False
        if split_by == 'features':
            A_block, b, features_in_partition = dist_col_read_one(rank, world_size, n_blob, n_features, file,
                                                                  zero_based, length=100*1024**2, datapoints=datapoints)
        elif split_by == 'samples':
            A_block, b, features_in_partition = dist_row_read_one(rank, world_size, n_blob, n_features, file,
                                                                  zero_based, length=100*1024**2, datapoints=datapoints)
        else:
            raise NotImplementedError

        return A_block, b, features_in_partition, 2396130, 3231961


class Webspam(object):
    @staticmethod
    def dist_read(rank, world_size, file, percent=0.1, split_by='features', random_state=42, datapoints=None, verbose=1):
        np.random.seed(random_state)
        n_blob = int(350000 * percent)
        n_features = 16609143
        zero_based = False
        memory_persize = 100 * 1024 ** 2
        if split_by == 'features':
            A_block, b, features_in_partition = dist_col_read_one(
                rank, world_size, n_blob, n_features, file,
                zero_based, length=memory_persize, datapoints=datapoints)
        elif split_by == 'samples':
            A_block, b, features_in_partition = dist_row_read_one(
                rank, world_size, n_blob, n_features, file,
                zero_based, length=memory_persize, datapoints=datapoints)
            A_block = A_block.T
        else:
            raise NotImplementedError
        return A_block, b, features_in_partition, 350000, 16609143


class RCV1Test(object):
    @staticmethod
    def dist_read(rank, world_size, file, percent=0.1, split_by='features', random_state=42, datapoints=None, verbose=1):
        np.random.seed(random_state)
        n_blob = int(677399 * percent)
        n_features = 47236
        zero_based = False
        memory_persize = 100 * 1024 ** 2
        if split_by == 'features':
            A_block, b, features_in_partition = dist_col_read_one(
                rank, world_size, n_blob, n_features, file,
                zero_based, length=memory_persize, datapoints=datapoints)
        elif split_by == 'samples':
            A_block, b, features_in_partition = dist_row_read_one(
                rank, world_size, n_blob, n_features, file,
                zero_based, length=memory_persize, datapoints=datapoints)
            A_block = A_block.T
        else:
            raise NotImplementedError
        return A_block, b, features_in_partition, 677399, 47236


def rank_indices(n, rank, world_size, random_state):
    """Generate indices for a node"""
    np.random.seed(random_state)
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices[rank::world_size]


def test(n_samples, n_features, rank, world_size, split_by='samples', random_state=42):
    """Generate dataset for test purpose."""
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features // 2,
                               n_redundant=0, n_repeated=0, n_classes=2, random_state=random_state)
    y = y * 2 - 1
    if split_by == 'samples':
        indices = rank_indices(n_samples, rank, world_size, random_state)
        return X[indices, :], y[indices]
    else:
        indices = rank_indices(n_features, rank, world_size, random_state)
        return X[:, indices], y


from sklearn.utils.validation import check_array, FLOAT_DTYPES
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from sklearn.preprocessing import normalize as f_normalize
import numbers

def _preprocess_data(X, y, fit_intercept, normalize=False, copy=True,
                     sample_weight=None, return_mean=False, check_input=True):
    """Center and scale data.
    Centers data to have mean zero along axis 0. If fit_intercept=False or if
    the X is a sparse matrix, no centering is done, but normalization can still
    be applied. The function returns the statistics necessary to reconstruct
    the input data, which are X_offset, y_offset, X_scale, such that the output
        X = (X - X_offset) / X_scale
    X_scale is the L2 norm of X - X_offset. If sample_weight is not None,
    then the weighted mean of X and y is zero, and not the mean itself. If
    return_mean=True, the mean, eventually weighted, is returned, independently
    of whether X was centered (option used for optimization with sparse data in
    coordinate_descend).
    This is here because nearly all linear models will want their data to be
    centered. This function also systematically makes y consistent with X.dtype
    Copyright (c) 2007–2020 The scikit-learn developers.
    """
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=['csr', 'csc'],
                        dtype=FLOAT_DTYPES)
    elif copy:
        if issparse(X):
            X = X.copy()
        else:
            X = X.copy(order='K')

    y = np.asarray(y, dtype=X.dtype)

    if fit_intercept:
        if issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0)
            if not return_mean:
                X_offset[:] = X.dtype.type(0)

            if normalize:
                X_var *= X.shape[0]
                X_scale = np.sqrt(X_var, X_var)
                del X_var
                X_scale[X_scale == 0] = 1
                inplace_column_scale(X, 1. / X_scale)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)

        else:
            X_offset = np.average(X, axis=0, weights=sample_weight)
            X -= X_offset
            if normalize:
                X, X_scale = f_normalize(X, axis=0, copy=False,
                                         return_norm=True)
            else:
                X_scale = np.ones(X.shape[1], dtype=X.dtype)
        y_offset = np.average(y, axis=0, weights=sample_weight)
        y = y - y_offset
    else:
        if normalize:
            if issparse(X):
                _, X_var = mean_variance_axis(X, axis=0)
                # transform variance to norm in-place
                X_var *= X.shape[0]
                X_scale = np.sqrt(X_var, X_var)
                del X_var
                X_scale[X_scale == 0] = 1
                inplace_column_scale(X, 1. / X_scale)
            else:
                X, X_scale = f_normalize(X, axis=0, copy=False,
                                         return_norm=True)
        else:
            X_scale = np.ones(X.shape[1], dtype=X.dtype)
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


def load_dataset_by_rank(name, rank, world_size, dataset_size='all', datapoints=-1, split_by='features', dataset_path=None, random_state=42,
                         transpose=True, verbose=1, fit_intercept=False, normalize=False):
    r"""
    Assume the data_dir has following structure:

    split by samples
        - X/        
            - 0     # (n_samples_at_0, n_features)
            - 1
            - ...
        - y/        # (n_samples_at_0, )
            - 0
            - 1
            - ...
        - indices   # (n_samples, ) which is sequential order for all ranks.

    split by features
        - X/        
            - 0     # (n_samples, n_features_at_0)
            - 1
            - ...
        - y/        # (n_samples, )
            - 0
            - 1
            - ...
        - indices   # (n_features, ) which is sequential order for all ranks.

    Returns
    -------
    X : {array-like},
        Shape (n_samples, n_features_at_rank) if split by features;
        Shape (n_features, n_samples_at_rank) if split by samples;
    y : {array-like}
        labels to the data matrix
    """
    if dataset_path is None:
        dataset_path = os.path.join('data', name, split_by, f'{world_size}')
    X_file = os.path.join(dataset_path, 'X', str(rank))
    y_file = os.path.join(dataset_path, 'y', str(rank))

    # Check that the folder has expected strcuture
    if not (os.path.exists(X_file) and os.path.exists(y_file)):
        raise ValueError("dataset_path {} does not have expected file.".format(dataset_path))
    
    X, y = joblib.load(X_file), joblib.load(y_file)
    if verbose >= 3:
        print("RANK {:3}: X, y has been loaded: {} {}".format(rank, X.shape, y.shape))
    
    X_file_test = os.path.join(dataset_path, 'X_test', str(rank))
    y_file_test = os.path.join(dataset_path, 'y_test', str(rank))
    issplit = True
    if not (os.path.exists(X_file_test) and os.path.exists(y_file_test)):
        issplit = False
    X_test = y_test = None
    if issplit:
        X_test, y_test = joblib.load(X_file_test), joblib.load(y_file_test)
        if verbose >= 3:
            print("RANK {:3}: X_test, y_test has been loaded: {} {}".format(rank, X_test.shape, y_test.shape))

    if split_by == 'samples' and transpose:
        X = X.T
        if issplit:
            X_test = X_test.T

    
    if isinstance(X, np.ndarray) and not X.flags['F_CONTIGUOUS']:
        # The local coordinate solver (like scikit-learn's ElasticNet) requires X to be Fortran contiguous.
        # Since the time spent on converting the matrix can be very long, and CoCoA need to call solvers every round,
        # we perform such convertion before algorithm and disable check later.
        X = np.asfortranarray(X)
        if issplit:
            X_test = np.asfortranarray(X_test)
    
    if issparse(X):
        # For coordinate descent
        X = csc_matrix(X)
        if issplit:
            X_test = csc_matrix(X_test)

    # X, y, features_in_partition, n_samples, n_features
    return X, y, X_test, y_test#, intercept


def load_dataset(name, rank, world_size, dataset_size, datapoints, split_by, dataset_path=None, random_state=42, verbose=1):
    """Load dataset in the distributed environment.

    Load matrix `X` and label `y` for this node in the network. The matrix `X` can be split by rows or columns
    depending on the user's choice. For the number of rows and columns are the size of local data points and 
    features.

    Parameters
    ----------
    name : {str}
        name of the dataset.
    rank : {int}
        rank of the node in the network.
    world_size : {int}
        number of nodes in the network.
    dataset_size : {'small', 'all'}
        size of dataset.
    split_by : {'samples', 'features'}
        split the data matrix by `samples` or `features`.
    dataset_path : {str}, optional
        path to the dataset if non-synthesized (the default is None, which synthesized dataset)
    random_state : {int}, optional
        random seed.

    Returns
    -------
    X : {array-like}, shape (n_samples, n_features)
        local data matrix.
    y : {array-like}, shape (n_samples)
        labels to the data matrix
    """
    warnings.warn("deprecated: use load_dataset_by_rank instead", DeprecationWarning)
    percent = 1
    assert dataset_size in ['small', 'all']
    percent = 0.1 if dataset_size == 'small' else 1
    assert split_by in ['samples', 'features']
    n_features = 0
    features_in_partition = 0
    if name in ['test', 'test_sparse']:
        n_samples_test = 1000
        n_features_test = 100
        X, y = test(n_samples_test, n_features_test, rank, world_size, split_by=split_by, random_state=random_state)
        if name == 'test_sparse':
            X = csc_matrix(X)
    elif name == 'epsilon':
        X, y, features_in_partition, n_samples, n_features = Epsilon.dist_read(
            rank, world_size, dataset_path, percent=percent, split_by=split_by, random_state=random_state, datapoints=datapoints)
    elif name == 'url':
        X, y, features_in_partition, n_samples, n_features = URL.dist_read(
            rank, world_size, dataset_path, percent=percent, split_by=split_by, random_state=random_state, datapoints=datapoints)
    elif name == 'webspam':
        X, y, features_in_partition, n_samples, n_features = Webspam.dist_read(
            rank, world_size, dataset_path, percent=percent, split_by=split_by, random_state=random_state, datapoints=datapoints)
    elif name == 'rcv1':
        X, y, features_in_partition, n_samples, n_features = RCV1Test.dist_read(
            rank, world_size, dataset_path, percent=percent, split_by=split_by, random_state=random_state, datapoints=datapoints)
    else:
        raise NotImplementedError
    if verbose >= 2:
        print(f'Rank {rank:3} features: {features_in_partition}')

    # Transpose the matrix depending on the matrix split direction
    if split_by == 'samples':
        X = X.T

    if isinstance(X, np.ndarray) and not X.flags['F_CONTIGUOUS']:
        # The local coordinate solver (like scikit-learn's ElasticNet) requires X to be Fortran contiguous.
        # Since the time spent on converting the matrix can be very long, and CoCoA need to call solvers every round,
        # we perform such convertion before algorithm and disable check later.
        X = np.asfortranarray(X)

    if issparse(X):
        # For coordinate descent
        X = csc_matrix(X)

    return X, y
