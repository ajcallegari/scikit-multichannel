"""
Synthetic data generators and dummy objects for testing.
"""

import random
import numpy as np
import inspect
from sklearn.datasets import make_classification

from pipecaster.utils import Cloneable

__all__ = ['make_multi_input_classification',
           'make_multi_input_regression', 'DummyClassifier']


def make_multi_input_classification(n_informative_Xs=5,
                                    n_weak_Xs=0,
                                    n_random_Xs=0,
                                    weak_noise_sd=0.2,
                                    seed=None, **sklearn_params):
    """
    Get a synthetic classification dataset with multiple input matrices of
    three possible types: informative, weak, and random.

    Parameters
    ---------
    n_informative_Xs: int,
        The number of informative feature matrices generated using the
        scikit-learn make_classification function.
    n_weak_Xs: int, default=0
        The number of feature matrices generated by adding a scikit-learn
        make_classification matrix to a Gaussian random deviate with a mean
        of zero and standard deviation set by the weak_noise_sd param.
    n_random_Xs: int, default=0
        The number of matrices genereated by shuffling the rows of a feature
        matrix generated by the scikit-learn make_classification function.
    sklearn_params: paramter dict
        Paramters for the sklearn make_classification function.
    weak_noise_sd: float, default=0.2
        Standard deviation of the Gaussian noise term used to
        generate weak matrices.
    seed: int or None, default=None
        Seed for pseudorandom number generators.  Int values enable
        reproduciblility.
    sklearn_params: keyword arguments or dict
        Parameters for scikit-learn's make_classification function which
        generates the dataset.  Parameters are per-matrix, not for entire
        multi-matrix dataset.

    returns
    -------
    (Xs, y, X_types)
    Xs: list
        List of synthetic feature matrices in random order.  The rows in each
        matrix correspont to the samples in y.
    y: ndarray(dtype=int).shape(n_samples,)
        List of synthetic sample labels.
    X_types: list of strings,
        Description of the feature matrices: 'informative', 'weak', or 'random'
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        sklearn_params['random_state'] = seed

    def get_X(y):
        y = np.array(y)
        labels = np.unique(y)
        n_samples = {label: np.sum(y == label) for label in labels}
        chunk_size = 100

        X_aggregate, y_aggregate = None, None
        enough_labels = False
        while enough_labels is False:
            X_new, y_new = make_classification(**sklearn_params)
            if X_aggregate is None:
                X_aggregate = X_new
            else:
                X_aggregate = np.concatenate([X_aggregate, X_new], axis=0)

            if y_aggregate is None:
                y_aggregate = y_new
            else:
                y_aggregate = np.concatenate([y_aggregate, y_new])

            some_incompletes = False
            for label in labels:
                if np.sum(y_aggregate == label) < n_samples[label]:
                    some_incompletes = True

            if some_incompletes is False:
                enough_labels = True

            if seed is not None:
                sklearn_params['random_state'] += 1

        X = np.empty((len(y), X_new.shape[1]))
        for label in labels:
            label_mask = y == label
            label_X = X_aggregate[y_aggregate == label]
            X[label_mask] = label_X[:np.sum(label_mask)]
        return X

    X, y = make_classification(**sklearn_params)
    Xs = [get_X(y) for i in range(n_informative_Xs)]
    X_types = ['informative' for i in range(n_informative_Xs)]
    Xs += [get_X(y) + np.random.normal(loc=0, scale=weak_noise_sd,
                                       size=X.shape)
           for i in range(n_weak_Xs)]
    X_types += ['weak' for i in range(n_weak_Xs)]
    Xs += [np.random.permutation(get_X(y)) for i in range(n_random_Xs)]
    X_types += ['random' for i in range(n_random_Xs)]
    X_list = list(zip(Xs, X_types))
    random.shuffle(X_list)
    Xs, X_types = zip(*X_list)

    return list(Xs), y, list(X_types)

def make_regression(n_samples=500, n_features=100, n_informative=5,
                    offset=0, noise_sd=0, return_slopes=False,
                    random_state=None):
    """
    Generate a simple synthetic regression dataset where the target is
    related linearly to the informative features and the features are
    uncorrelated.

    Parameters
    ----------
    n_samples: int, default=500
        The number of sample to simulate.
    n_features: int, default=100
        The total number of features to simulate.
    n_informative: int, default=5
        The number of features that are linearly related to the target with
        slopes drawn uniformly from the interval [-1, 1) and feature values
        drawn randomly from the interval [0,1).  The remaining features are
        uncorrelated with the target and drawn from the interval [0,1).
    offset: float, default=0
        Constant in the linear equation:
            y = offset + slope_1 * feature_1 + slope_2 * feature_2 ...
    noise_sd: float, default=0
        The standard devation of a Gaussian noise term added to the features.
    random_state: int or None, default=None
        See for the pseudorandom number generator used to make features.

    Returns
    -------
    X: ndarray.shape(n_samples, n_features)
        Synthetic feature matrix (not normalized).
    y: ndarray.shape(n_samples,)
        Target values.
    slopes: ndarray.shape(n_features,)
        Slopes determining the relationship between the informative features
        and the target.  Zero value for uninformative features.  Slopes is only
        returned when the return_slopes paramter is True.
    """

    if random_state is not None:
        np.random.seed(random_state)
    n_random = n_features - n_informative

    slopes = np.array([2 * np.random.rand() - 1 for i in range(n_informative)])
    X_inf = np.stack([slopes * np.random.rand(n_informative)
                      for i in range(n_samples)])
    y = np.sum(slopes * X_inf, axis=1) + offset
    X_rand = np.stack([np.random.rand(n_random) for i in range(n_samples)])
    X = np.concatenate([X_inf, X_rand], axis=1)
    slopes = np.concatenate([slopes, np.zeros(n_random)])
    permutation_idx = np.random.permutation(n_features)
    X = X[: , permutation_idx].copy()
    X += np.random.normal(loc=0, scale=noise_sd, size=X.shape)
    slopes = slopes[permutation_idx].copy()

    return (X, y, slopes) if return_slopes else (X, y)


def make_multi_input_regression(n_informative_Xs=10,
                                n_weak_Xs=0,
                                n_random_Xs=0,
                                weak_noise_sd=0.2,
                                seed=None, **rgr_params):
    """
    Get a synthetic regression dataset with multiple input matrices of
    three possible types: informative, weak, and random.

    Parameters
    ---------
    n_informative_Xs: int,
        The number of informative feature matrices generated using the
        scikit-learn make_regression function.
    n_weak_Xs: int, default=0
        The number of feature matrices generated by adding a scikit-learn
        make_regression matrix to a Gaussian random deviate with a mean
        of zero and standard deviation set by the weak_noise_sd param.
    n_random_Xs: int, default=0
        The number of matrices genereated by shuffling the rows of a feature
        matrix generated by the scikit-learn make_regression function.
    sklearn_params: paramter dict
        Paramters for the sklearn make_regression function.
    weak_noise_sd: float, default=0.2
        Standard deviation of the Gaussian noise term used to
        generate weak matrices.
    seed: int or None, default=None
        Seed for pseudorandom number generators.  Int values enable
        reproduciblility.
    rgr_params: keyword arguments or dict
        Parameters for the make_regression function which
        generates the dataset.  Parameters are per-matrix, not for entire
        multi-matrix dataset.

    returns
    -------
    (Xs, y, X_types)
    Xs: list
        List of synthetic feature matrices in random order.  The rows in each
        matrix correspont to the samples in y.
    y: ndarray(dtype=int).shape(n_samples,)
        List of synthetic sample labels.
    X_types: list of strings,
        Description of the feature matrices: 'informative', 'weak', or 'random'
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        rgr_params['random_state'] = seed

    X_types = ['informative' for i in range(n_informative_Xs)]
    X_types += ['weak' for i in range(n_weak_Xs)]
    X_types += ['random' for i in range(n_random_Xs)]

    n_Xs = n_informative_Xs + n_weak_Xs + n_random_Xs

    sig = inspect.signature(make_regression)

    if 'n_informative' not in rgr_params:
        rgr_params['n_informative'] = \
            sig.parameters.get('n_informative').default

    if 'n_features' not in rgr_params:
        rgr_params['n_features'] = sig.parameters.get('n_features').default

    n_inf = rgr_params['n_informative']
    n_rand = rgr_params['n_features'] - n_inf
    rgr_params['n_features'] *= n_Xs
    rgr_params['n_informative'] *= n_Xs
    rgr_params['return_slopes'] = True

    X_pool, y, coef = make_regression(**rgr_params)
    informative_mask = coef != 0
    X_pool_inf = X_pool[:, informative_mask]
    X_pool_rand = X_pool[:, ~informative_mask]

    Xs = []
    for i, X_type in enumerate(X_types):
        X_inf = X_pool_inf[:, i*n_inf:(i+1)*n_inf]
        X_rand = X_pool_rand[:, i*n_rand:(i+1)*n_rand]
        X = np.concatenate([X_inf, X_rand], axis=1)
        X = X[:, np.random.permutation(X.shape[1])]
        if X_type == 'weak':
            X += np.random.normal(loc=0, scale=weak_noise_sd, size=X.shape)
        elif X_type == 'random':
            np.random.shuffle(X)
        Xs.append(X)

    tuples = list(zip(Xs, X_types))
    random.shuffle(tuples)
    Xs, X_types = zip(*tuples)

    return list(Xs), y, list(X_types)


class DummyClassifier(Cloneable):
    """
    Classifier that outputs random labels or probabilities and decouples
    argument passing overhead from fitting overhead or prediction overhead
    using futile cycles (e.g. it allows fast prediction with big input data or
    slow prediction with small input data, which is useful for distributed
    computing performance optimization).
    """
    def __init__(self, futile_cycles_fit=1000000, futile_cycles_pred=10):
        self.futile_cycles_fit = futile_cycles_fit
        self.futile_cycles_pred = futile_cycles_pred
        self._estimator_type = 'classifier'

    def fit(self, X, y=None, **fit_params):

        if y is not None:
            self.classes_, y_enc = np.unique(y, return_inverse=True)
        else:
            y_enc = y
        a = 0
        for i in range(self.futile_cycles_fit):
            a += 1
        return self

    def predict(self, X):
        a = 0
        for i in range(self.futile_cycles_pred):
            a += 1
        return np.random.choice(self.classes_, X.shape[0])

    def predict_proba(self, X):
        a = 0
        for i in range(self.futile_cycles_pred):
            a += 1
        return np.random.rand(X.shape[0], len(self.classes_))
