import numpy as np
import ray
import scipy.sparse as sp
import scipy.stats

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_predict, _check_is_permutation, _enforce_prediction_order
from sklearn.utils.validation import indexable, _num_samples

import pipecaster.utils as utils

__all__ = ['TransformingPredictor', 'MetaClassifier']


@ray.remote
def ray_fit_and_predict(predictor, X, y, train_indices, test_indices, verbose, fit_params, method):
    return _fit_and_predict(predictor, X, y, train_indices, test_indices, verbose, fit_params, method)

def split_predict(predictor, X, y=None, *, groups=None, cv=None,
                      n_jobs=None, verbose=0, fit_params=None, method='predict'):
    """Slightly modified version of sklearn cross_val_predict for use in internal cv training. The modifications enable stateful cloning with pipecaster.utility.get_clone() and faster multiprocessing/distributed computing with ray
    
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
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=utils.is_classifier(predictor))

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
    
    if n_jobs == 1:
        prediction_blocks = [_fit_and_predict(utils.get_clone(predictor), X, y, train_indices, test_indices, verbose, 
                                          fit_params, method) for train_indices, test_indices in cv.split(X, y, groups)] 
        
    elif n_jobs > 1:
        try:
            ray.nodes()
        except RuntimeError:
            ray.init()
        splits = list(cv.split(X, y, groups))
        X = ray.put(X)
        y = ray.put(y)
        fit_params = ray.put(fit_params)
        jobs = [ray_fit_and_predict.remote(ray.put(utils.get_clone(predictor)), X, y, train_indices, test_indices, verbose, 
                                          fit_params, method) for train_indices, test_indices in splits] 
        prediction_blocks = ray.get(jobs)
        X = ray.get(X)
        y = ray.get(y)
        fit_params = ray.get(fit_params)
    else:
        raise ValueError('invalid n_jobs value: {}.  muste be int greater than 0'.format(n_jobs))
        
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

class TransformingPredictor:
    
    """Class that wraps scikit-learn predictors to provide transform() & fit_transform() methods, and training on internal splits.  Internal cross validation splits are used only during fit_transform() in order to reduce overfitting during downstream metaclassifier training.  Models trained on splits are temporary.  A persistent model trained on the full training set is also gererated by fit_transform() -- and also fit() -- and used for inference during calls to transform().
    
    arguments
    ---------
    predictor: instance of predictor object with sklearn interface
    method: string. indicates the name of the method to use for transformation or 'auto' for 'predict_proba' -> 'decision_function' -> 'predict'
    internal_cv: None, int or cross validation splitter (e.g. StratifiedKFold).  If None, 0, or 1, cross validation training is disabled.  For integers > 1, KFold is automatically used for regressors and StratifiedKFold used for classifiers.
    n_jobs: Number of parallel _fit_and_predict jobs to run during internal cv training.

    """
    
    def __init__(self, predictor, method='auto', internal_cv = 5, n_jobs = 1):
        self.predictor = predictor
        self.method = method
        self.internal_cv = internal_cv
        self.n_jobs = n_jobs
        
    def fit(self, X, y=None, **fit_params):
        if self.method == 'auto':
            if hasattr(self.predictor, 'predict_proba'):
                self.method_ = 'predict_proba' 
            elif hasattr(self.predictor, 'decision_function'):
                self.method_ = 'decision_function' 
            elif hasattr(self.predictor, 'predict'):
                self.method_ = 'predict' 
            else:
                raise TypeError('no method found for transform in ' + self.predictor.__class__.__name__)
        else:
            if hasattr(self.predictor, self.method):
                self.method_ = self.method 
            else:
                raise TypeError('{} method not found in {}'.format(self.method, pipe.__class__.__name__)) 
        self.predictor.fit(X, y, **fit_params)
        if hasattr(self.predictor, 'classes_'):
            self.classes_ = self.predictor.classes_
            self._estimator_type = 'classifier'
        else:
            self._estimator_type = 'regressor'
        
    def fit_transform(self, X, y, groups=None, **fit_params):
        self.fit(X, y, **fit_params)
        
        # internal cv training is disabled
        if self.internal_cv is None or (type(self.internal_cv) == int and self.internal_cv < 2):
            transform_method = getattr(self.predictor, self.method_)
            X = transform_method(X)
        # internal cv training is enabled
        else:
            X = split_predict(self.predictor, X, y, groups=groups, cv=self.internal_cv,
                      n_jobs=1, verbose=0, fit_params=fit_params, method=self.method_)
        return X.reshape(-1, 1) if len(X.shape) == 1 else X
    
    def transform(self, X):
        transform_method = getattr(self.predictor, self.method_)
        X = transform_method(X)
        return X.reshape(-1, 1) if len(X.shape) == 1 else X
                        
    def get_params(self, deep=False):
        return {'predictor':self.predictor,
                'method':self.method,
                'internal_cv':self.internal_cv,
                'n_jobs':self.n_jobs}
    
    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def _more_tags(self):
        return {'multiple_inputs': True}
    
    def get_clone(self):
        clone = TransformingPredictor(utils.get_clone(self.predictor), method=self.method, 
                                     internal_cv = self.internal_cv, n_jobs = self.n_jobs)
        for attr in ['classes_', '_estimator_type', 'method_']:
            if hasattr(self, attr):
                setattr(clone, attr, getattr(self, attr))
        return clone

class MetaClassifier:
    
    """Classifier and meta-classifier that takes multiple input matrices and generates a single prediction. 
    
    Parameters
    ----------
    classifier: str, or object instance with the sklearn interface, default='soft vote'
        Determines the metapredicition method: 'soft voting', 'hard voting', or stacked generalization using 
        the specified classifier 
    
    Notes
    -----
    Takes multiple inputs channels and ouputs to the channel with the lowest index number.
    Supports multiclass but not multilabel or multioutput classification
    Adds tranform functionality to both voting and stacking metaclassifiers to enable multilevel metaclassification.
        For soft voting, the transform is the mean probabilities of the multiple inputs
        For hard voting, the transform is the fraction of the input classifiers that picked the class - shape (n_samples, n_classes)
        For stacking, the transform is the predict_proba() method of the meta-clf algorithm if available otherwise the decision_function()
    
    """
    
    def __init__(self, classifier = 'soft vote'):
        self._estimator_type = 'classifier'
        
        if classifier not in ['soft vote', 'hard vote']:
            if hasattr(classifier, 'fit') == False:
                raise AttributeError('classifier object missing fit() method')
                
        self.classifier = classifier

        
    def fit(self, Xs, y=None, **fit_params):
        if y is not None:
            if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
                raise NotImplementedError('Multilabel and multi-output meta-classification not supported')
            self.classes_, y = np.unique(y, return_inverse=True)
            
        if self.classifier not in ['soft vote', 'hard vote']:
            self.transform_method_ = utils.get_transform_method(self.classifier)
            if hasattr(self.classifier, 'predict_proba'):
                self.pred_proba_method_ = getattr(self.classifier, 'predict_proba')
            elif hasattr(self.classifier, 'decision_function'):
                self.pred_proba_method_ = getattr(self.classifier, 'decision_function')
            if hasattr(self.classifier, 'transform') == False:
                self.transform_method_ = 'decision_function'
            else:
                self.pred_proba_method_ = None
                
            live_Xs = [X for X in Xs if X is not None]
            if len(Xs) < 1:
                raise ValueError('no inputs available to fit metaclassifier')
            meta_X = np.concatenate(live_Xs, axis=1)
            if y is None:
                self.classifier.fit(meta_X, **fit_params)
            else:
                self.classifier.fit(meta_X, y, **fit_params)
        
    def predict_proba(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        channel_predictions = [None for X in Xs]
            
        if self.classifier in ['soft vote', 'hard vote']:
            channel_predictions[0] = np.mean(live_Xs, axis=0)
            return channel_predictions
        else:
            meta_X = np.concatenate(live_Xs, axis=1)
            channel_predictions[0] =  self.pred_proba_method_(meta_X)
            return channel_predictions
                
    def decision_function(self, Xs):
        return self.predict_proba(Xs)
    
    def predict(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        if self.classifier == 'soft vote':
            mean_probs = np.mean(live_Xs, axis=0)
            decisions = np.argmax(mean_probs, axis=1)
            predictions = self.classes_[decisions]
        elif self.classifier == 'hard vote':
            input_decisions = np.stack([np.argmax(X, axis=1) for X in live_Xs])
            decisions = scipy.stats.mode(input_decisions, axis = 0)[0][0]
            predictions = self.classes_[decisions]
        else:
            meta_X = np.concatenate(live_Xs, axis=1)
            predictions = self.classifier.predict(meta_X)
            
        channel_predictions = [None for X in Xs]
        channel_predictions[0] = predictions
            
        return channel_predictions
            
    def transform(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        if self.classifier == 'soft vote':
            X_t = np.mean(live_Xs, axis=0)
        elif self.classifier == 'hard vote':
            input_predictions = [np.argmax(X, axis=1).reshape(-1,1) for X in live_Xs]
            input_predictions = np.concatenate(input_predictions, axis=1)
            n_samples = input_predictions.shape[0]
            n_classes = len(self.classes_)
            class_counts = [np.bincount(input_predictions[i,:], minlength=n_classes) for i in range(n_samples)]
            class_counts = np.stack(class_counts)
            class_counts /= len(live_Xs)
            X_t = class_counts
        else:
            if self.tranform_method_ is None:
                raise AttributeError('wrapped classifier lacks an accepted transform method')
            else:
                meta_X = np.concatenate(live_Xs, axis=1)
                X_t = self.tranform_method_(meta_X)
                
        if len(X_t.shape) == 1:
            X_t = X_t.reshape(-1, 1)
            
        Xs_t = [None for X in Xs]
        Xs_t[0] = X_t
        
        return Xs_t
                           
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)
                           
    def _more_tags(self):
        return {'metapredictor': True}
                                        
    def get_params(self, deep=False):
        return {'classifier':self.classifier, '_estimator_type':'classifier'}
    
    def set_params(self, **params):
        valid_params = self.get_params().keys()
        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                raise ValueError('invalid parameter name')
                           
    def get_clone(self):
        if self.classifier in ['soft vote', 'hard vote']:
            return MetaClassifier(self.classifier)
        else:
            return MetaClassifier(utils.get_clone(self.classifier))
        
        
class MetaRegressor:
    
    """Regressor and meta-regressor that takes multiple input matrices and generates a single prediction. 
    
    Parameters
    ----------
    regressor: str, or object instance with the sklearn regressor interface, default='mean voting'
        Determines the metapredicition method: 'mean voting', 'median voting', or stacked generalization using 
        the specified regressor 
    
    Notes
    -----
    Takes multiple inputs channels and ouputs to the channel with the lowest index number.
    Does not support multioutput regression (yet).
    Adds tranform functionality for multilevel metaclassification.
    
    """
    
    def __init__(self, regressor = 'mean voting'):
        self._estimator_type = 'regressor'
        
        if regressor not in ['mean voting', 'median voting']:
            if hasattr(regressor, 'fit') == False:
                raise AttributeError('regressor object missing fit() method')
                
        self.regressor = regressor
        
    def fit(self, Xs, y=None, **fit_params):
        if y is not None:
            if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
                raise NotImplementedError('Multi-output meta-regression not supported')
            
        if self.regressor not in ['mean voting', 'median voting']:
            self.transform_method_ = utils.get_transform_method(self.regressor)
            live_Xs = [X for X in Xs if X is not None]
            if len(Xs) < 1:
                raise ValueError('no inputs available to fit meta-regressor')
            meta_X = np.concatenate(live_Xs, axis=1)
            if y is None:
                self.regressor.fit(meta_X, **fit_params)
            else:
                self.regressor.fit(meta_X, y, **fit_params)
    
    def predict(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        if self.regressor == 'mean voting':
            predictions = np.mean(live_Xs, axis=0).reshape(-1)
        elif self.regressor == 'median voting':
            predictions = np.median(live_Xs, axis=0).reshape(-1)
        else:
            meta_X = np.concatenate(live_Xs, axis=1)
            predictions = self.regressor.predict(meta_X)
            
        channel_predictions = [None for X in Xs]
        channel_predictions[0] = predictions
            
        return channel_predictions
            
    def transform(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        if self.regressor == 'mean voting':
            predictions = np.mean(live_Xs, axis=0)
        elif self.regressor == 'median voting':
            predictions = np.median(live_Xs, axis=0)
        else:
            meta_X = np.concatenate(live_Xs, axis=1)
            predictions = self.regressor.predict(meta_X).reshape(-1,1)
            
        Xs_t = [None for X in Xs]
        Xs_t[0] = predictions
            
        return Xs_t
                           
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)
                           
    def _more_tags(self):
        return {'metapredictor': True}
                                        
    def get_params(self, deep=False):
        return {'regressor':self.regressor, '_estimator_type':'regressor'}
    
    def set_params(self, **params):
        valid_params = self.get_params().keys()
        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                raise ValueError('invalid parameter name')
                           
    def get_clone(self):
        if self.regressor in ['mean voting', 'median voting']:
            return MetaRegressor(self.regressor)
        else:
            return MetaRegressor(utils.get_clone(self.regressor))