import ray
import scipy.sparse as sp
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_predict, _check_is_permutation, _enforce_prediction_order
import pipecaster as pc

def is_classifier(obj):
    return getattr(obj, "_estimator_type", None) == "classifier"

def fit_predict_deprecated(predictor, X, y, train_indices, test_indices, fit_params, method):
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    predictor.fit(X_train, y_train, **fit_params)
    prediction = getattr(predictor, method)(X_test)
    test_indices = test_indices
    
    return (prediction, test_indices)


@ray.remote
def ray_fit_predict(predictor, X, y, train_indices, test_indices, verbose, fit_params, method):
    return _fit_and_predict(predictor, X, y, train_indices, test_indices, fit_params, method)

def cross_val_predict(predictor, X, y, fit_params, cv=3, method = 'predict', n_jobs = 1, verbose = 0):
    """Replacement function for sklearn cross_val_predict that enables stateful cloning (pipecaster.get_clone()) and faster multiprocessing with ray
    
    argument
    --------
    predictor: object instance that implements the sklearn estimator & predictor interfaces
Parameters
    X : ndarray (n_samples, n_features), features
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), targets, 
    groups : array-like of shape (n_samples,), sample group labels
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
    n_jobs : int, The number of CPUs to use to do the computation.
    verbose : int, default=0
        The verbosity level.
    method : str, methdu used for prediction
    
    Returns
    -------
    predictions : ndarray (n_splits, n_samples, n_features), predictions
    """
    
    cv = check_cv(cv, y, classifier=is_classifier(predictor))
    
    if n_jobs == 1:
        prediction_blocks = [_fit_predict(pc.get_clone(predictor), X, y, train_indices, test_indices, verboset, fit_params, method)
                             for train_indices, test_indices in cv.split(X, y, groups)] 
    elif n_jobs > 1:
        X = ray.put(X)
        y = ray.put(y)
        fit_params = ray.put(fit_params)
        jobs = [ray_fit_predict.remote(pc.get_clone(predictor), X, y, train_indices, test_indices, fit_params, method)
                             for train_indices, test_indices in cv.split(X, y, groups)]
        prediction_blocks = ray.get(jobs)
        
    split_predictions, split_test_indices = zip(*prediction_blocks)
    shuffled_predictions = np.concatenate(split_predictions, axis = 0)
    test_indices = np.concatenate(split_test_indices)
    ordered_predictions = np.empty(shuffled_predictions.shape)
    ordered_predictions[test_indices] = shuffled_predictions
    
    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        # `predictions` is a list of method outputs from each fold.
        # If each of those is also a list, then treat this as a
        # multioutput-multiclass task. We need to separately concatenate
        # the method outputs for each label into an `n_labels` long list.
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]
    
    return ordered_predictions

def cross_val_predict(estimator, X, y=None, *, groups=None, cv=None,
                      n_jobs=None, verbose=0, fit_params=None, method='predict'):
    """Modified version of sklearn cross_val_predict that enables stateful cloning (pipecaster.get_clone()) and faster multiprocessing and distributed computing with ray
    
    The bulk of this code is copied and pasted from scikit-learn, which request the following copyright notification:
    Copyright (c) 2007-2020 The scikit-learn developers.
    
    Generate cross-validated estimates for each input data point
    The data is split according to the cv parameter. Each sample belongs
    to exactly one test set, and its prediction is computed with an
    estimator fitted on the corresponding training set.
    Passing these predictions into an evaluation metric may not be a valid
    way to measure generalization performance. Results can differ from
    :func:`cross_validate` and :func:`cross_val_score` unless all tests sets
    have equal size and the metric decomposes over samples.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
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
    fit_params : dict, defualt=None
        Parameters to pass to the fit method of the estimator.
    method : str, default='predict'
        Invokes the passed method name of the passed estimator. For
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
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # If classification methods produce multiple columns of output,
    # we need to manually encode classes to ensure consistent column ordering.
    encode = method in ['decision_function', 'predict_proba',
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

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    #parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        #pre_dispatch=pre_dispatch)
    #prediction_blocks = parallel(delayed(_fit_and_predict)(
        #clone(estimator), X, y, train, test, verbose, fit_params, method)
        #for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    
     if n_jobs == 1:
        prediction_blocks = [_fit_predict(pc.get_clone(predictor), X, y, train_indices, test_indices, verboset, 
                                          fit_params, method) for train_indices, test_indices in cv.split(X, y, groups)] 
    elif n_jobs > 1:
        X = ray.put(X)
        y = ray.put(y)
        fit_params = ray.put(fit_params)
        jobs = [ray_fit_predict.remote(pc.get_clone(predictor), X, y, train_indices, test_indices, fit_params, method)
                             for train_indices, test_indices in cv.split(X, y, groups)]
        prediction_blocks = ray.get(jobs)
        
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        # `predictions` is a list of method outputs from each fold.
        # If each of those is also a list, then treat this as a
        # multioutput-multiclass task. We need to separately concatenate
        # the method outputs for each label into an `n_labels` long list.
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]
    