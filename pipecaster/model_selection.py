import numpy as np
import ray
import scipy.sparse as sp

from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

import pipecaster.utils as utils

__all__ = ['cross_val_score', 'cross_val_predict']

#### SINGLE MATRIX INPUTS ####

def fit_and_predict(predictor, Xs, y, train_indices, test_indices, predict_method, **fit_params):
        if utils.is_multi_input(predictor):
            X_trains = [X[train_indices] if X is not None else None for X in Xs]
            predictor.fit(X_trains, y[train_indices], **fit_params)
        else:
            predictor.fit(Xs[train_indices], y[train_indices], **fit_params)
        if hasattr(predictor, predict_method):
            predict_method = getattr(predictor, predict_method)
            if utils.is_multi_input(predictor):
                X_tests = [X[test_indices] if X is not None else None for X in Xs]
                predictions = predict_method(X_tests)
            else:
                predictions = predict_method(Xs[test_indices])
        else:
            raise AttributeError('invalid predict method')
            
        prediction_block = (predictions, test_indices)
        
        return prediction_block

@ray.remote
def ray_fit_and_predict(predictor, Xs, y, train_indices, test_indices, predict_method, **fit_params):
    return fit_and_predict(predictor, Xs, y, train_indices, test_indices, predict_method, **fit_params)

def cross_val_predict(predictor, Xs, y=None, groups=None, predict_method='predict', cv=None,
                      combine_splits=True, n_jobs=1, split_seed=None, **fit_params):
    """Sklearn's cross_val_predict function (v23.2) modified to enable stateful cloning with pipecaster.utility.get_clone() and faster multiprocessing/distributed computing with ray
    
    The bulk of this code is copied and pasted from scikit-learn, which request the following copyright notification:
    Copyright (c) 2007-2020 The scikit-learn developers.
    
    Generate cross-validated estimates for each input data point
    The data is split according to the cv parameter. Each sample belongs
    to exactly one test set, and its prediction is computed with an
    predictor fitted on the corresponding training set.
    Passing these predictions into an evaluation metric may not be a valid
    way to measure generalization performance. Results can differ from
    :func:`cross_validate` and :func:`cross_val_score` unless all tests sets
    have equal size and the metric decomposes over samples.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    predictor : predictor object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be, for example a list, or an array at least 2d.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the predictor is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    n_jobs : int, default=None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, defualt=None
        Parameters to pass to the fit method of the predictor.
    method : str, default='predict'
        Invokes the passed method name of the passed predictor. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.
    Returns
    -------
    predictions : ndarray
        This is the result of calling ``method``
    See also
    --------
    cross_val_score : calculate score for each CV split
    cross_validate : calculate one or more scores and timings for each CV split
    Notes
    -----
    In the case that one or more classes are absent in a training portion, a
    default score needs to be assigned to all instances for that class if
    ``method`` produces columns per class, as in {'decision_function',
    'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
    0.  In order to ensure finite output, we approximate negative infinity by
    the minimum finite float value for the dtype in other cases.
    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_predict
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> y_pred = cross_val_predict(lasso, X, y, cv=3)
    """
    
    # If classification methods produce multiple columns of output,
    # we need to manually encode classes to ensure consistent column ordering.
    encode = predict_method in ['decision_function', 'predict_proba',
                        'predict_log_proba'] and y is not None
    if encode:
        y = np.asarray(y)
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=np.int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc
            
    if type(cv) == int:
        if groups is not None:
            cv = GroupKFold(n_splits=cv, random_state=split_seed)
        else:
            if utils.is_classifier(predictor):
                cv = StratifiedKFold(n_splits=cv, random_state=split_seed)
            else:
                cv = KFold(n_splits=cv, random_state=split_seed)
                
    if utils.is_multi_input(predictor):
        live_Xs = [X for X in Xs if X is not None]
        splits = list(cv.split(live_Xs[0], y, groups))
    else:
        splits = list(cv.split(Xs, y, groups))
    
    if n_jobs < 2:
        prediction_blocks = [fit_and_predict(utils.get_clone(predictor), Xs, y, train_indices, test_indices, 
                                                    predict_method, **fit_params) 
                             for train_indices, test_indices in splits] 
    elif n_jobs > 1:
        try:
            ray.nodes()
        except RuntimeError:
            ray.init()
        Xs = ray.put(Xs)
        y = ray.put(y)
        jobs = [ray_fit_and_predict.remote(ray.put(utils.get_clone(predictor)), Xs, y, ray.put(train_indices), 
                                                  ray.put(test_indices), predict_method, **fit_params) 
                for train_indices, test_indices in splits] 
        prediction_blocks = ray.get(jobs)
    else:
        raise ValueError('Invalid n_jobs value: {}. Must be int greater than 0.'.format(n_jobs))
        
    if combine_splits == False:
        return prediction_blocks
    else:
        predictions, test_indices = zip(*prediction_blocks)
        if sp.issparse(predictions[0]):
            predictions = sp.vstack(predictions, format=predictions[0].format)
        elif encode and isinstance(predictions[0], list):
            assert NotImplementedError('Multi-output predictors not supported')
        else:
            predictions = np.concatenate(predictions)
        test_indices = np.concatenate(test_indices)

        return predictions[test_indices]
        
def cross_val_score(predictor, Xs, y=None, groups=None, scorer=None, predict_method='predict',
                    cv=3, n_jobs=1, split_seed=None, **fit_params):
    """Sklearn's cross_validate function (v23.2), modified to enable stateful cloning with pipecaster.utility.get_clone() and faster multiprocessing/distributed computing with ray
    
     The bulk of this code is copied and pasted from scikit-learn, which request the following copyright notification:
    Copyright (c) 2007-2020 The scikit-learn developers.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only
        a single value.
        Similar to :func:`cross_validate`
        but only a single metric is permitted.
        If None, the estimator's default scorer (if available) is used.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    n_jobs : int, default=None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
        .. versionadded:: 0.20
    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_score
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> print(cross_val_score(lasso, X, y, cv=3))
    [0.33150734 0.08022311 0.03531764]
    See Also
    ---------
    :func:`sklearn.model_selection.cross_validate`:
        To run cross-validation on multiple metrics and also to return
        train scores, fit times and score times.
    :func:`sklearn.model_selection.cross_val_predict`:
        Get predictions from each split of cross-validation for diagnostic
        purposes.
    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.
    """
    
    if scorer is None:
        if utils.is_classifier(predictor):
            scorer = accuracy_score
        elif utils.is_regressor(predictor):
            scorer = explained_variance_score
        
    split_blocks = cross_val_predict(predictor, Xs, y, groups, predict_method, cv, 
                                     combine_splits=False, n_jobs=n_jobs, split_seed=split_seed, **fit_params)
    scores = [scorer(y[test_indices], predictions) for predictions, test_indices in split_blocks]
    return scores
    