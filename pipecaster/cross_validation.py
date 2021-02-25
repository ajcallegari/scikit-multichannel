"""
Cross validation functions supporting both MultichannelPipeline and
scikit-learn predictors.
"""

import numpy as np
import scipy.sparse as sp

from sklearn.metrics import balanced_accuracy_score, explained_variance_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

import pipecaster.utils as utils
import pipecaster.parallel as parallel

__all__ = ['cross_val_score', 'cross_val_predict']


def _fit_and_predict(predictor, Xs, y, train_indices, test_indices,
                    predict_method_name, fit_params):
        """
        Clone, fit, and predict a single channel or multichannel pipe.
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
    predictor : estimator/predictor instance
        Classifier or regressor that implements the scikit-learn estimator and
        predictor interfaces.
    Xs : list
        List of feature matrices and None spaceholders.
    y : list/array, default=None
        Optional targets for supervised ML.
    groups: list/array, default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used if cv parameter is set to GroupKFold.
    predict_method : str, default='predict'
        Name of the method to use for predictions
    cv : int, or callable, default=5
        - Set the cross validation method:
        - If int > 1: Use StratifiedKfold(n_splits=internal_cv) for
          classifiers or Kfold(n_splits=internal_cv) for regressors.
        - If None or 5: Use 5 splits with the default split generator.
        - If callable: Assumes interface like Kfold scikit-learn.
    combine_splits : bool, default=True
        - If True: Concatenate results for splits into a single array.
        - If False: Return results for separate splits.
    n_processes : int or 'max', default=1
        - If 1: Run all split computations in a single process.
        - If 'max': Run splits in multiple processes, using all
          available CPUs.
        - If int > 1: Run splits in multiple processes, using up to
          n_processes number of CPUs.
    fit_params : dict, default={}
        Auxiliary parameters sent to pipe fit_transform and fit methods.

    Returns
    -------
    Predictions
        - If combine_splits is False:
          Returns a list of tuples (predictions, sample indices), one for each
          split.
        - If combine_splits it True:
          Returns a single array with predictions for each sample in Xs
          in the order in which they were provided.

    Examples
    --------
    ::

        import pipecaster as pc
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(pc.ChannelEnsemble(GradientBoostingClassifier(), SVC()))

        predictions = pc.cross_val_predict(clf, Xs, y)
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
                                _fit_and_predict, args_list,
                                n_cpus=n_processes,
                                shared_mem_objects=shared_mem_objects)
        except Exception as e:
            print('parallel processing request failed with message {}'
                  .format(e))
            print('defaulting to single processor')
            n_processes = 1
    if n_processes == 1:
        # print('running a single process with {} jobs'.format(len(args_list)))
        split_results = [_fit_and_predict(*args) for args in args_list]

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
    predictor : estimator/predictor instance
        Classifier or regressor that implements the scikit-learn estimator and
        predictor interfaces.
    Xs : list
        List of feature matrices and None spaceholders.
    y : list/array, default=None
        Optional targets for supervised ML.
    groups: list/array, default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used if cv parameter is set to GroupKFold.
    scorer : 'auto' or callable, default='auto'
        - If 'auto': balanced_accuracy_score for classifiers or
          explained_variance_score for regressors
        - If callable: A scorer object that returns a scalar figure of
          merit score with signature:
          score = scorer(y_true, y_pred).
    cv : int, or callable, default=5
        - Set the cross validation method:
        - If int > 1: Use StratifiedKfold(n_splits=internal_cv) for
          classifiers or Kfold(n_splits=internal_cv) for regressors.
        - If None or 5: Use 5 splits with the default split generator.
        - If callable: Assumes interface like Kfold scikit-learn.
    n_processes : int or 'max', default=1
        - If 1: Run all split computations in a single process.
        - If 'max': Run splits in multiple processes, using all
          available CPUs.
        - If int > 1: Run splits in multiple processes, using up to
          n_processes number of CPUs.
    fit_params : dict, default={}
        Auxiliary parameters sent to pipe fit_transform and fit methods.

    Returns
    -------
    list
        List of scalar figure of merit scores, one for each split.

    Examples
    --------
    ::

        import pipecaster as pc
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(pc.ChannelEnsemble(GradientBoostingClassifier(), SVC()))

        pc.cross_val_score(clf, Xs, y)
        # ouput: [0.7647058823529411, 0.8455882352941176, 0.8180147058823529]
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
