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
import pipecaster.config as config
import pipecaster.parallel as parallel

__all__ = ['cross_val_score', 'cross_val_predict']

def _predict(model, Xs, test_indices, method_name):
    predict_method = getattr(model, method_name)
    if utils.is_multichannel(model):
        X_tests = [X[test_indices] if X is not None else None
                   for X in Xs]
        return predict_method(X_tests)
    else:
        return predict_method(Xs[test_indices])

def _fit_predict_split(predictor, Xs, y, train_indices, test_indices,
                    predict_method='predict', transform_method=None,
                    score_method=None, fit_params=None):
        """
        Clone, fit, and predict with a single channel or multichannel
        predictor.
        """
        is_classifier = utils.is_classifier(predictor)
        model = utils.get_clone(predictor)
        fit_params = {} if fit_params is None else fit_params

        if utils.is_multichannel(model):
            X_trains = [X[train_indices] if X is not None else None
                        for X in Xs]
            model.fit(X_trains, y[train_indices], **fit_params)
        else:
            model.fit(Xs[train_indices], y[train_indices], **fit_params)

        split_predictions = {}

        if predict_method is not None:
            split_predictions['predict'] = {}
            if predict_method == 'auto' and is_classifier:
                prediction_made = False
                for m in config.predict_method_precedence:
                    try:
                        y_pred = _predict(model, Xs, test_indices, m)
                        if y_pred is not None:
                            prediction_made = True
                    except:
                        pass
                    if prediction_made is True:
                        split_predictions['predict']['method'] = m
                        break
                if prediction_made == False:
                    raise AttributeError('failed to auto-detect prediction '
                                         'method')
            elif predict_method == 'auto' and is_classifier == False:
                y_pred = _predict(model, Xs, test_indices, 'predict')
                split_predictions['predict']['method'] = 'predict'
            else:
                y_pred = _predict(model, Xs, test_indices, predict_method)
                split_predictions['predict']['method'] = predict_method

            split_predictions['predict']['y_pred'] = y_pred

        if transform_method is not None:
            split_predictions['transform'] = {}
            if transform_method == 'auto':
                prediction_made = False
                for m in config.transform_method_precedence:
                    try:
                        y_pred = _predict(model, Xs, test_indices, m)
                        if y_pred is not None:
                            prediction_made = True
                    except:
                        pass
                    if prediction_made is True:
                        split_predictions['transform']['method'] = m
                        break
                if prediction_made == False:
                    raise AttributeError('failed to auto-detect transform '
                                         'method')
            elif transform_method == 'auto' and is_classifier == False:
                y_pred = _predict(model, Xs, test_indices, 'predict')
                split_predictions['transform']['method'] = 'predict'
            else:
                y_pred = _predict(model, Xs, test_indices, transform_method)
                split_predictions['transform']['method'] = transform_method

            split_predictions['transform']['y_pred'] = y_pred

        if score_method is not None:
            split_predictions['score'] = {}
            if score_method == 'auto':
                prediction_made = False
                for m in config.score_method_precedence:
                    try:
                        y_pred = _predict(model, Xs, test_indices, m)
                        if y_pred is not None:
                            prediction_made = True
                    except:
                        pass
                    if prediction_made is True:
                        split_predictions['score']['method'] = m
                        break
                if prediction_made == False:
                    raise AttributeError('failed to auto-detect score '
                                         'method')
            elif score_method == 'auto' and is_classifier == False:
                y_pred = _predict(model, Xs, test_indices, 'predict')
                split_predictions['score']['method'] = 'predict'
            else:
                y_pred = _predict(model, Xs, test_indices, score_method)
                split_predictions['score']['method'] = score_method

            split_predictions['score']['y_pred'] = y_pred

        return split_predictions, test_indices


def cross_val_predict(predictor, Xs, y=None, groups=None,
                      predict_method='predict', transform_method=None,
                      score_method=None, cv=None,
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
    predict_method : str, default='predict'
        - Name of the method used for predicting.
        - If 'auto' :
            - If classifier : method picked using
              config.predict_method_precedence order (default:
              predict->predict_proba->predict_log_proba->decision_function).
            - If regressor : 'predict'
    transform_method : str, default=None
        - Name of the prediction method to call when transforming (e.g. when
          outputting meta-features).
        - If 'auto' :
            - If classifier : method picked using
              config.transform_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'
    score_method : str, default=None
        - Name of prediction method used when scoring predictor performance.
        - If 'auto' :
            - If classifier : method picked using
              config.score_method_precedence order (default:
              ppredict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'
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
    dict
        - If combine_splits is True :
            {'predict':y_pred, 'transform':y_pred, 'score':y_pred)}
            Where y_pred = np.array(n_samples) or None if the type of
            prediction was not requested.  There will not be dict entries for
            prediction method parameters set to None (e.g. no 'transform' key
            when transform_method=None).
        - If combine_splits is False :
            {'predict':[], 'transform':[],
            'score':[], 'indices':[])}
            Where empty brackets indicate identically ordered lists with one
            list item per split.  List items are either prediction arrays or
            sample indices for the splits.  There will not be dict entries for
            prediction method parameters set to None (e.g. no 'transform' key
            when transform_method=None).

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
        predictions['predict']['y_pred']
        # output: [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, ...]
        y
        # output: [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, ...]
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
                  predict_method, transform_method, score_method, fit_params)
                 for train_indices, test_indices in splits]

    n_jobs = len(args_list)
    n_processes = 1 if n_processes is None else n_processes
    if (type(n_processes) == int and n_jobs < n_processes):
        n_processes = n_jobs

    if n_processes == 'max' or n_processes > 1:
        try:
            shared_mem_objects = [Xs, y, fit_params]
            job_results = parallel.starmap_jobs(
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
        job_results = [_fit_predict_split(*args) for args in args_list]

    split_predictions, split_indices = zip(*job_results)

    # reorganize so splits are in lists
    results = {k:{'y_pred':[sp[k]['y_pred'] for sp in split_predictions],
                  'method':split_predictions[0][k]['method']}
               for k in split_predictions[0]}

    # decode classes where necessary
    if is_classifier:
        for predict_method in results:
            if results[predict_method]['method'] == 'predict':
                results[predict_method]['y_pred'] = [classes_[p]
                    for p in results[predict_method]['y_pred']]

    if combine_splits is True:
        sample_indices = np.concatenate(split_indices)
        for predict_method in results:
            y_concat = np.concatenate(results[predict_method]['y_pred'])
            results[predict_method]['y_pred'] = y_concat[sample_indices]
    else:
        results['indices'] = split_indices

    return results


def score_predictions(y_true, y_pred, score_method, scorer,
                      is_classification, is_binary):
    """
    Score predictions with 'auto' method support.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # set scorer if 'auto'
    if type(scorer) == str and scorer == 'auto':
        if is_classification and score_method == 'predict':
            scorer = balanced_accuracy_score
        elif (is_classification and score_method in
              ['predict_proba', 'predict_log_proba', 'decision_function']):
            scorer = roc_auc_score
        else:
            scorer = explained_variance_score

    # drop redundant probs to make binary cls results compatible with sklearn
    if is_classification and is_binary and len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]

    return scorer(y_true, y_pred)


def cross_val_score(predictor, Xs, y=None, groups=None,
                    score_method = 'predict', scorer='auto',
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
    score_method : str, default='predict'
        Name of method called to make predictions for performance scoring. If
        'auto', methods are attempted in the order defined in
        config.score_method_precedence.
        Default: predict_proba->predict_log_proba->decision_function->predict.
    scorer : {callable, 'auto'}, default='auto'
        - Function calculating performance scores.
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer that returns a scalar figure of merit score
          with signature: score = scorer(y_true, y_pred).
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
        - If 1 predict_method is specified: List of scalar figure of merit
          scores, one for each split.
        - If >1 predict_method is specified:  Dict indexed by prediction method
          name where values are lists of scores the splits.

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
    is_binary = False
    if is_classifier and y is not None:
        classes_, y = np.unique(y, return_inverse=True)
        if len(classes_) == 2:
            is_binary = True

    split_results = cross_val_predict(predictor, Xs, y, groups,
                                      predict_method=None,
                                      transform_method=None,
                                      score_method=score_method,
                                      cv=cv, combine_splits=False,
                                      n_processes=n_processes,
                                      **fit_params)

    # score the predictions
    scores = [score_predictions(y[idx], yp, score_method, scorer,
                                is_classifier, is_binary)
              for yp, idx in zip(split_results['score']['y_pred'],
                                 split_results['indices'])]

    return scores
