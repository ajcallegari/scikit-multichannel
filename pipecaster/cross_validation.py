import numpy as np
import scipy.sparse as sp

from sklearn.metrics import balanced_accuracy_score, explained_variance_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

import pipecaster.utils as utils
import pipecaster.parallel as parallel

__all__ = ['cross_val_score', 'cross_val_predict']


def fit_and_predict(predictor, Xs, y, train_indices, test_indices,
                    predict_method_name, fit_params):
        """
        Clone, fit, and predict a singlechannel or multichannel model.
        """
        model = utils.get_clone(predictor)
        fit_params = {} if fit_params is None else fit_params

        if utils.is_multichannel(model):
            X_trains = [X[train_indices] if X is not None else None
                        for X in Xs]
            model.fit(X_trains, y[train_indices], **fit_params)
        else:
            model.fit(Xs[train_indices], y[train_indices], **fit_params)

        if hasattr(model, predict_method_name):
            predict_method = getattr(model, predict_method_name)
            if utils.is_multichannel(model):
                X_tests = [X[test_indices] if X is not None else None
                           for X in Xs]
                predictions = predict_method(X_tests)
            else:
                predictions = predict_method(Xs[test_indices])
        else:
            raise AttributeError('Predict method not found.')

        split_results = (predictions, test_indices)

        return split_results


def cross_val_predict(predictor, Xs, y=None, groups=None,
                      predict_method='predict', cv=None, combine_splits=True,
                      n_processes='max', fit_params=None):
    """
    Analog of the scikit-learn cross_val_predict function that supports both
    single and multichannel cross validation.

    Parameters
    ----------
    predictor : predictor instance implementing 'fit' and 'predict'
    Xs: ndarray.shape(n_samples, n_features) or list of ndarrays
        A single feature matrix, or a list of feature matrices, each with
        identical samples in the same order.
    y: nd.array(n_samples,) or list with length n_samples, default=None
        If list-like: Supervised learning target values.
        If None: Value used for unsupervised machine learning.
    groups: ndarray.shape(n_samples,) or list of n_samples, default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:'cv'
        instance (e.g., :class:'GroupKFold').
    predict_method: String or None, default='predict'
        Name of the method to use for predictions
    cv: None, int, or cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a '(Stratified)KFold',
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the predictor is a classifier and 'y' is
        either binary or multiclass, :class:'StratifiedKFold' is used. In all
        other cases, :class:'KFold' is used.
    combine_splits: bool, default=True
        If False: Return results for separate splits.
        If True: Concatenate results for splits into a single array.
    n_processes: int, default='max'
        The number of parallel fit/predict processes to run.
    fit_params: dict, defualt=None
        Auxiliary parameters to pass to the fit method of the predictor.

    Returns
    -------
    If combine_splits is False:
        Returns a list containing (predictions, sample indices) for each
        split.
    If combine_splits it True:
        Returns a single array with predictions for each sample in Xs
        in the order in which they were provided.
    """
    is_classifier = utils.is_classifier(predictor)
    if is_classifier and y is not None:
        classes_, y = np.unique(y, return_inverse=True)

    cv = int(5) if cv is None else cv

    if type(cv) == int:
        if groups is not None:
            cv = GroupKFold(n_splits=cv)
        else:
            if utils.is_classifier(predictor):
                cv = StratifiedKFold(n_splits=cv)
            else:
                cv = KFold(n_splits=cv)

    is_multichannel = utils.is_multichannel(predictor)
    if is_multichannel:
        live_Xs = [X for X in Xs if X is not None]
        splits = list(cv.split(live_Xs[0], y, groups))
    else:
        splits = list(cv.split(Xs, y, groups))

    args_list = [(predictor, Xs, y, train_indices, test_indices,
                  predict_method, fit_params)
                 for train_indices, test_indices in splits]

    n_jobs = len(args_list)
    n_processes = 1 if n_processes is None else n_processes
    if (type(n_processes) == int and n_jobs < n_processes):
        n_processes = n_jobs

    if n_processes == 'max' or n_processes > 1:
        try:
            shared_mem_objects = [Xs, y, fit_params]
            split_results = parallel.starmap_jobs(
                                fit_and_predict, args_list,
                                n_cpus=n_processes,
                                shared_mem_objects=shared_mem_objects)
        except Exception as e:
            print('parallel processing request failed with message {}'
                  .format(e))
            print('defaulting to single processor')
            n_processes = 1
    if n_processes == 1:
        # print('running a single process with {} jobs'.format(len(args_list)))
        split_results = [fit_and_predict(*args) for args in args_list]

    split_predictions, split_indices = zip(*split_results)

    if combine_splits is False:
        if is_classifier and predict_method == 'predict':
            split_predictions = [classes_[split_prediction]
                                 for split_prediction in split_predictions]
        split_results = zip(split_predictions, split_indices)
        return split_results
    else:
        sample_indices = np.concatenate(split_indices)
        predictions = np.concatenate(split_predictions)
        if is_classifier and predict_method == 'predict':
            predictions = classes_[predictions]
        return predictions[sample_indices]


def cross_val_score(predictor, Xs, y=None, groups=None, scorer='auto',
                    cv=3, n_processes=1, **fit_params):
    """
    Analog of the scikit-learn cross_val_score function that supports both
    single and multichannel cross validation.

    Parameters
    ----------
    predictor : predictor instance implementing 'fit' and 'predict'
    Xs: ndarray.shape(n_samples, n_features) or list of ndarrays
        A single feature matrix, or a list of feature matrices, each with
        identical samples in the same order.
    y: nd.array(n_samples,) or list with length n_samples, default=None
        If list-like: Supervised learning target values.
        If None: Value used for unsupervised machine learning.
    groups: ndarray.shape(n_samples,) or list of n_samples, default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:'cv'
        instance (e.g., :class:'GroupKFold').
    scorer : 'auto' or callable, default='auto'
        If 'auto': balanced_accuracy_score for classifiers or
            explained_variance_score for regressors
        If callable: A scorer object with signature
            'scorer(y_true, y_pred)' which returns a scalar figure of
            merit.
    cv: None, int, or cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a '(Stratified)KFold',
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the predictor is a classifier and 'y' is
        either binary or multiclass, :class:'StratifiedKFold' is used. In all
        other cases, :class:'KFold' is used.
    n_processes: int, default='max'
        The number of parallel fit/predict processes to run.
    fit_params: dict, defualt=None
        Auxiliary parameters to pass to the fit method of the predictor.

    Returns
    -------
    scores : List of scalar figure of merit scores, one for each split.

    """
    if scorer is None or scorer == 'auto':
        if utils.is_classifier(predictor):
            scorer = balanced_accuracy_score
        elif utils.is_regressor(predictor):
            scorer = explained_variance_score

    prediction_splits = cross_val_predict(predictor, Xs, y, groups, 'predict',
                                          cv, combine_splits=False,
                                          n_processes=n_processes,
                                          **fit_params)
    scores = [scorer(y[split_indices], split_prediction)
              for split_prediction, split_indices in prediction_splits]

    return scores
