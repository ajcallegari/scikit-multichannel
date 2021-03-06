"""
Cross validation functions supporting both MultichannelPipeline and
scikit-learn predictors.
"""

import numpy as np
import scipy.sparse as sp

from sklearn.metrics import balanced_accuracy_score, explained_variance_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

import pipecaster.utils as utils
import pipecaster.parallel as parallel

__all__ = ['cross_val_score', 'cross_val_predict']


def _fit_predict_split(predictor, Xs, y, train_indices, test_indices,
                    predict_method_names, fit_params):
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

        split_results = {}
        for predict_method_name in predict_method_names:
            predict_method = getattr(model, predict_method_name)
            if utils.is_multichannel(model):
                X_tests = [X[test_indices] if X is not None else None
                           for X in Xs]
                split_results[predict_method_name] = predict_method(X_tests)
            else:
                split_results[predict_method_name] = predict_method(
                                                        Xs[test_indices])

        split_results['indices'] = test_indices

        return split_results


def cross_val_predict(predictor, Xs, y=None, groups=None,
                      predict_methods=['predict'], cv=None,
                      combine_splits=True, n_processes=1, fit_params=None):
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
    predict_methods : str or list, default=['predict']
        Name of the method or methods to use for making predictions.
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
        - If combine_splits it True:
          Returns a single array for each prediction method containing
          predictions for each sample in the order that they appeared in the Xs
          parameter.  If only one prediction method is specified in the
          predict_methods parameter, a single array is returned, otherwise a
          dict indexed by the predict method name.
        - If combine_splits is False:
          Returns a dict indexed by prediction method where each value is a
          list of predictions, one for each split.  The dict also contains an
          'indices' entry containing a list of the sample indices for each
          split relative to the order in which they were provided in the Xs
          parameter.

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

    if isinstance(predict_methods, (tuple, list, np.ndarray)) is False:
        predict_methods = [predict_methods]

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
                  predict_methods, fit_params)
                 for train_indices, test_indices in splits]

    n_jobs = len(args_list)
    n_processes = 1 if n_processes is None else n_processes
    if (type(n_processes) == int and n_jobs < n_processes):
        n_processes = n_jobs

    if n_processes == 'max' or n_processes > 1:
        try:
            shared_mem_objects = [Xs, y, fit_params]
            split_results = parallel.starmap_jobs(
                                _fit_predict_split, args_list,
                                n_cpus=n_processes,
                                shared_mem_objects=shared_mem_objects)
        except Exception as e:
            print('parallel processing request failed with message {}'
                  .format(e))
            print('defaulting to single processor')
            n_processes = 1
    if n_processes == 1:
        # print('running a single process with {} jobs'.format(len(args_list)))
        split_results = [_fit_predict_split(*args) for args in args_list]

    # reorganize so splits are in lists
    split_results = {k:[res[k] for res in split_results]
                     for k in predict_methods + ['indices']}

    # decode classes where necessary
    if is_classifier and 'predict' in predict_methods:
        split_results['predict'] = [classes_[p]
                                    for p in split_results['predict']]

    if combine_splits is True:
        predictions = []
        sample_indices = np.concatenate(split_results['indices'])
        predictions = [np.concatenate(split_results[m])[sample_indices]
                       for m in  predict_methods]
        if len(predictions) == 1:
            return predictions[0]
        else:
            return {m:p for m, p in zip(predict_methods, predictions)}
    else:
        return split_results

def score_splits(split_results, y=None, predict_methods=['predict'],
                 scorers='auto'):
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
    predict_methods : str or list, default=['predict']
        Name of the method or methods to use for making predictions.
    scorers : {callable, list of callables, or 'auto'}, default='auto'
        - If 'auto':
            - balanced_accuracy_score for classifiers with predict()
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
        - If callable: A scorer that returns a scalar figure of merit score
          with signature: score = scorer(y_true, y_pred).
        - If list of callables: Ordered list of scorer objects that specify
          the scoring methods to use for each prediction method specified in
          the predict_methods parameter.
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
        - If 1 predict_method is specified:
          List of scalar figure of merit scores, one for each split.
        - If >1 predict_method is specified:
          Dict indexed by prediction method name where values are lists of
          scores the splits.

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

    is_classifier = utils.is_classifier(predictor)
    if is_classifier and y is not None:
        classes_, y = np.unique(y, return_inverse=True)

    if isinstance(predict_methods, (tuple, list, np.ndarray)) is False:
        predict_methods = [predict_methods]

    split_results = cross_val_predict(predictor, Xs, y, groups,
                                          predict_methods,
                                          cv, combine_splits=False,
                                          n_processes=n_processes,
                                          **fit_params)

    # set scorers if 'auto' is selected
    if type(scorers) == str and scorers == 'auto':
        scorers = []
        for method in predict_methods:
            if is_classifier and method == 'predict':
                scorers.append(balanced_accuracy_score)
            elif (is_classifier and method in
                  ['predict_proba', 'predict_log_proba', 'decision_function']):
                scorers.append(roc_auc_score)
            else:
                scorers.append(explained_variance_score)

    if (isinstance(scorers, (tuple, list, np.ndarray))) is False:
        scorers = [scorers]

    # drop redundant probs to make binary cls results compatible with sklearn
    if is_classifier and len(classes_) == 2:
        split_results = {m:[p[:, 1]
                             if m in ['predict_proba', 'predict_log_proba']
                             else p for p in split_results[m]]
                         for m in split_results}

    if len(scorers) != len(predict_methods):
        raise ValueError('Number of scorers must match number of '
                         'predict_methods or be set to auto.')

    # apply the scorers to the predictions
    scores = {m:[scorer(y[idx], p)
                 for p, idx in zip(split_results[m], split_results['indices'])]
              for m, scorer in zip(predict_methods, scorers)}

    if len(predict_methods) == 1:
        return scores[predict_methods[0]]
    else:
        return scores

def score_predictions(y_true, y_pred, predict_method, scorer,
                      is_classification):

    # set scorer if 'auto'
    if type(scorer) == str and scorer == 'auto':
        if is_classifier and predict_method == 'predict':
            scorer = balanced_accuracy_score
        elif (is_classifier and method in
              ['predict_proba', 'predict_log_proba', 'decision_function']):
            scorer = roc_auc_score
        else:
            scorer = explained_variance_score

    # drop redundant probs to make binary cls results compatible with sklearn
    if is_classifier and is_binary:
        y_pred = y_pred[:, 1]

    return scorer(y_pred, y_pred)

def score_splits(predictor, Xs, y=None, groups=None,
                    predict_methods = ['predict'], scorers='auto',
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
    predict_methods : str or list, default=['predict']
        Name of the method or methods to use for making predictions.
    scorers : {callable, list of callables, or 'auto'}, default='auto'
        - If 'auto':
            - balanced_accuracy_score for classifiers with predict()
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
        - If callable: A scorer that returns a scalar figure of merit score
          with signature: score = scorer(y_true, y_pred).
        - If list of callables: Ordered list of scorer objects that specify
          the scoring methods to use for each prediction method specified in
          the predict_methods parameter.
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
        - If 1 predict_method is specified:
          List of scalar figure of merit scores, one for each split.
        - If >1 predict_method is specified:
          Dict indexed by prediction method name where values are lists of
          scores the splits.

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

    is_classifier = utils.is_classifier(predictor)
    if is_classifier and y is not None:
        classes_, y = np.unique(y, return_inverse=True)

    if isinstance(predict_methods, (tuple, list, np.ndarray)) is False:
        predict_methods = [predict_methods]

    split_results = cross_val_predict(predictor, Xs, y, groups,
                                          predict_methods,
                                          cv, combine_splits=False,
                                          n_processes=n_processes,
                                          **fit_params)

    # set scorers if 'auto' is selected
    if type(scorers) == str and scorers == 'auto':
        scorers = []
        for method in predict_methods:
            if is_classifier and method == 'predict':
                scorers.append(balanced_accuracy_score)
            elif (is_classifier and method in
                  ['predict_proba', 'predict_log_proba', 'decision_function']):
                scorers.append(roc_auc_score)
            else:
                scorers.append(explained_variance_score)

    if (isinstance(scorers, (tuple, list, np.ndarray))) is False:
        scorers = [scorers]

    # drop redundant probs to make binary cls results compatible with sklearn
    if is_classifier and len(classes_) == 2:
        split_results = {m:[p[:, 1]
                             if m in ['predict_proba', 'predict_log_proba']
                             else p for p in split_results[m]]
                         for m in split_results}

    if len(scorers) != len(predict_methods):
        raise ValueError('Number of scorers must match number of '
                         'predict_methods or be set to auto.')

    # apply the scorers to the predictions
    scores = {m:[scorer(y[idx], p)
                 for p, idx in zip(split_results[m], split_results['indices'])]
              for m, scorer in zip(predict_methods, scorers)}

    if len(predict_methods) == 1:
        return scores[predict_methods[0]]
    else:
        return scores


def cross_val_score(predictor, Xs, y=None, groups=None,
                    predict_methods = ['predict'], scorers='auto',
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
    predict_methods : str or list, default=['predict']
        Name of the method or methods to use for making predictions.
    scorers : {callable, list of callables, or 'auto'}, default='auto'
        - If 'auto':
            - balanced_accuracy_score for classifiers with predict()
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
        - If callable: A scorer that returns a scalar figure of merit score
          with signature: score = scorer(y_true, y_pred).
        - If list of callables: Ordered list of scorer objects that specify
          the scoring methods to use for each prediction method specified in
          the predict_methods parameter.
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
        - If 1 predict_method is specified:
          List of scalar figure of merit scores, one for each split.
        - If >1 predict_method is specified:
          Dict indexed by prediction method name where values are lists of
          scores the splits.

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

    is_classifier = utils.is_classifier(predictor)
    if is_classifier and y is not None:
        classes_, y = np.unique(y, return_inverse=True)

    if isinstance(predict_methods, (tuple, list, np.ndarray)) is False:
        predict_methods = [predict_methods]

    split_results = cross_val_predict(predictor, Xs, y, groups,
                                          predict_methods,
                                          cv, combine_splits=False,
                                          n_processes=n_processes,
                                          **fit_params)

    # set scorers if 'auto' is selected
    if type(scorers) == str and scorers == 'auto':
        scorers = []
        for method in predict_methods:
            if is_classifier and method == 'predict':
                scorers.append(balanced_accuracy_score)
            elif (is_classifier and method in
                  ['predict_proba', 'predict_log_proba', 'decision_function']):
                scorers.append(roc_auc_score)
            else:
                scorers.append(explained_variance_score)

    if (isinstance(scorers, (tuple, list, np.ndarray))) is False:
        scorers = [scorers]

    # drop redundant probs to make binary cls results compatible with sklearn
    if is_classifier and len(classes_) == 2:
        split_results = {m:[p[:, 1]
                             if m in ['predict_proba', 'predict_log_proba']
                             else p for p in split_results[m]]
                         for m in split_results}

    if len(scorers) != len(predict_methods):
        raise ValueError('Number of scorers must match number of '
                         'predict_methods or be set to auto.')

    # apply the scorers to the predictions
    scores = {m:[scorer(y[idx], p)
                 for p, idx in zip(split_results[m], split_results['indices'])]
              for m, scorer in zip(predict_methods, scorers)}

    if len(predict_methods) == 1:
        return scores[predict_methods[0]]
    else:
        return scores
