import numpy as np
import scipy.sparse as sp

from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

import pipecaster.utils as utils
import pipecaster.parallel as parallel

__all__ = ['cross_val_score', 'cross_val_predict']

#### SINGLE MATRIX INPUTS ####

def fit_and_predict(predictor, Xs, y, train_indices, test_indices, predict_method_name, fit_params):
        predictor = utils.get_clone(predictor)
        fit_params = {} if fit_params is None else fit_params
        
        if utils.is_multichannel(predictor):
            X_trains = [X[train_indices] if X is not None else None for X in Xs]
            predictor.fit(X_trains, y[train_indices], **fit_params)
        else:
            predictor.fit(Xs[train_indices], y[train_indices], **fit_params)
            
        if hasattr(predictor, predict_method_name):
            predict_method = getattr(predictor, predict_method_name)
            if utils.is_multichannel(predictor):
                X_tests = [X[test_indices] if X is not None else None for X in Xs]
                predictions = predict_method(X_tests)
            else:
                predictions = predict_method(Xs[test_indices])
        else:
            raise AttributeError('invalid predict method')
            
        prediction_block = (predictions, test_indices)
        
        return prediction_block

def cross_val_predict(predictor, Xs, y=None, groups=None, predict_method='predict', cv=None,
                      combine_splits=True, n_processes='max', split_seed=None, fit_params=None):
    
    """Multichannel version of sklearn cross_val_predict.  Also supports single channel cross validation.

    Parameters
    ----------
    predictor : predictor instance implementing 'fit' and 'predict'
        
    Xs : array-like of shape (n_samples, n_features) or list of array-likes
        A single feature matrix, or a list of feature matrices with the same samples in the same order.  
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:'cv'
        instance (e.g., :class:'GroupKFold').
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a '(Stratified)KFold',
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the predictor is a classifier and 'y' is
        either binary or multiclass, :class:'StratifiedKFold' is used. In all
        other cases, :class:'KFold' is used.
    n_processes : int, default='max'
        The number of parallel processes to use to do the computation.
    fit_params : dict, defualt=None
        Auxiliary parameters to pass to the fit method of the predictor.
    method : str, default='predict'
        Invokes the passed method name of the passed predictor. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.
    Returns
    -------
    predictions : list(tuple) or ndarray 
        Returns the result of calling 'method'.
        When combine_splits == False:
            returns a list containing (predictions, sample indices) for each split
        When combine_splits == True:
            returns a single list with predictions for each sample in Xs
            in the order in which they were provided 
 
    """
    
    # If classification methods produce multiple columns of output,
    # we need to manually encode classes to ensure consistent column ordering.
    encode = predict_method in ['decision_function', 'predict_proba', 'predict_log_proba'] and y is not None
    
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
            
    cv = int(5) if cv is None else cv
            
    if type(cv) == int:
        if groups is not None:
            cv = GroupKFold(n_splits=cv, random_state=split_seed)
        else:
            if utils.is_classifier(predictor):
                cv = StratifiedKFold(n_splits=cv, random_state=split_seed)
            else:
                cv = KFold(n_splits=cv, random_state=split_seed)
    else:
        cv.random_state=split_seed
                
    if utils.is_multichannel(predictor):
        live_Xs = [X for X in Xs if X is not None]
        splits = list(cv.split(live_Xs[0], y, groups))
    else:
        splits = list(cv.split(Xs, y, groups))
        
    args_list = [(predictor, Xs, y, train_indices, test_indices, predict_method, fit_params) 
                for train_indices, test_indices in splits] 
    
    n_jobs = len(args_list)
    n_processes = 1 if n_processes is None else n_processes
    n_processes = n_jobs if (type(n_processes) == int and n_jobs < n_processes) else n_processes
            
    if n_processes == 'max' or n_processes > 1:
        try:
            shared_mem_objects = [Xs, y, fit_params]
            prediction_blocks = parallel.starmap_jobs(fit_and_predict, args_list, 
                                                      n_cpus=n_processes, shared_mem_objects=shared_mem_objects)
        except Exception as e:
            print('parallel processing request failed with message {}'.format(e))
            print('defaulting to single processor')
            n_processes = 1       
    if n_processes == 1:
        # print('running a single process with {} jobs'.format(len(args_list)))
        prediction_blocks = [fit_and_predict(*args) for args in args_list]
        
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
        
def cross_val_score(predictor, Xs, y=None, groups=None, scorer=explained_variance_score, predict_method='predict',
                    cv=3, n_processes=1, split_seed=None, **fit_params):
    """
    Multichannel channel version of scikit-learn's cross_val_score function.  
        
    Parameters
    ----------
    predictor : Scikit-learn conformant predictor instance
    Xs : array-like of shape (n_samples, n_features) or list of array-likes
        A single feature matrix, or a list of feature matrices with the same samples in the same order.  
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:'cv'
        instance (e.g., :class:'GroupKFold').
    scorer : callable, default=None
        a scorer callable object / function with signature
        'scorer(y_true, y_pred)' which should return only
        a single value.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a '(Stratified)KFold',
        - :term:'CV splitter',
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and 'y' is
        either binary or multiclass, :class:'StratifiedKFold' is used. In all
        other cases, :class:'KFold' is used.
    n_processes : int, default=None
        The number of CPUs to use to do the computation.
        'None' means 1 unless in a :obj:'joblib.parallel_backend' context.
        '-1' means using all processors. See :term:'Glossary <n_processes>'
        for more details.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    
    """
    if scorer is None:
        if utils.is_classifier(predictor):
            scorer = accuracy_score
        elif utils.is_regressor(predictor):
            scorer = explained_variance_score
        
    split_blocks = cross_val_predict(predictor, Xs, y, groups, predict_method, cv, 
                                     combine_splits=False, n_processes=n_processes, split_seed=split_seed, **fit_params)
    
    scores = [scorer(y[test_indices], predictions) for predictions, test_indices in split_blocks]
    return scores
    