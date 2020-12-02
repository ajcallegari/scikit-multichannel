import numpy as np
import ray
import scipy.stats

import pipecaster.utils as utils
from pipecaster.cross_validation import cross_val_predict

__all__ = ['TransformingPredictor', 'ChannelClassifier', 'ChannelRegressor']

class TransformingPredictor:
    
    """Wrapper class that provids scikit-learn predictors with transform()/fit_transform() methods and internal cross validation training so their predictions may be used for metaprediction. 
    
    arguments
    ---------
    predictor: object instance 
        The sklearn predictor to be wrapped
    transform_method: string. indicates the name of the method to use for transformation or 'auto' 
        for 'predict_proba' -> 'decision_function' -> 'predict'
    internal_cv: None, int or cross validation splitter (e.g. StratifiedKFold).  
        Set the traing set subsampling method used for fitting and tranforming when fit_tranform() is called.  If None, 0, or 1, 
        subsamplish is disabled.  For integers > 1, KFold is automatically used for regressors and StratifiedKFold used for 
        classifiers.  Cv subsampling ensures that models do not make predictions on their own training data samples, and is
        useful when predictions will subsequently be used to train a meta-classifier.
    n_jobs: Number of parallel _fit_and_predict jobs to run during internal cv training (ray multiprocessing).
    
    notes
    -----
    Models that are trained with internal cross validation subsamples when fit_transform() is called are used 
    transiently during pipeline fitting and are not persisted for subsequent inferences.  Inferences are made on 
    a differnt, persistent model that is trained on the full training set during calls to fit_transform() -- or fit().
    """
    
    def __init__(self, predictor, transform_method='auto', internal_cv=5, cv_scorer=None, cv_jobs=1):
        self.predictor = predictor
        self.transform_method = transform_method
        self.internal_cv = internal_cv
        self.cv_scorer = cv_scorer
        self.cv_jobs = cv_jobs
        
    def fit(self, X, y=None, **fit_params):
        if hasattr(self.predictor, 'fit') == False:
            raise AttributeError('missing fit method')
        self.predictor.fit(X, y, **fit_params)
        
        # detect the type of predictor
        if hasattr(self.predictor, 'classes_'):
            self.classes_ = self.predictor.classes_
        if utils.is_classifier(self.predictor):
            self._estimator_type = 'classifier'
        elif utils.is_regressor(self.predictor):
            self._estimator_type = 'regressor'
        else:
            self._estimator_type = 'unknown'
            
        if self.transform_method == 'auto':
            self.transform_method_ = utils.get_transform_method(self.predictor)
            if self.transform_method is None:
                raise AttributeError('missing transform method')
        else:
            if hasattr(self.predictor, self.transform_method):
                self.transform_method_ = getattr(self.predictor, self.transform_method)
            else:
                raise AttributeError('transform method {} not found'.format(self.transform_method))
        
    def fit_transform(self, X, y, groups=None, **fit_params):
        self.fit(X, y, **fit_params)
        
        # internal cv training is disabled
        if self.internal_cv is None or (type(self.internal_cv) == int and self.internal_cv < 2):
            X = self.transform_method_(X)
        # internal cv training is enabled
        else:
            X = cross_val_predict(self.predictor, X, y, groups=groups, cv=self.internal_cv,
                      n_jobs=self.cv_jobs, predict_method=self.transform_method_.__name__, **fit_params)
            if self.cv_scorer is not None:
                self.score_ = self.cv_scorer(y, X)
        return X.reshape(-1, 1) if len(X.shape) == 1 else X
    
    def transform(self, X):
        X = self.transform_method_(X)
        return X.reshape(-1, 1) if len(X.shape) == 1 else X
                        
    def get_params(self, deep=False):
        return {'predictor':self.predictor,
                'transform_method':self.transform_method,
                'internal_cv':self.internal_cv,
                'n_jobs':self.n_jobs}
    
    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def _more_tags(self):
        return {'multiple_inputs': True}
    
    def get_clone(self):
        clone = TransformingPredictor(utils.get_clone(self.predictor), transform_method=self.transform_method, 
                                      internal_cv=self.internal_cv, cv_scorer=self.cv_scorer, cv_jobs=self.cv_jobs)
        state_variables = ['transform_method_', '_estimator_type', 'classes_', 'score_']
        for var in state_variables:
            if hasattr(self, var):
                setattr(clone, var, getattr(self, var))
        return clone
            
class ChannelClassifier:
    
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
        self.classifier = classifier
        self._estimator_type = 'classifier'
        
    def fit(self, Xs, y=None, **fit_params):
        if y is not None:
            if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
                raise NotImplementedError('Multilabel and multi-output meta-classification not supported')
            self.classes_, y = np.unique(y, return_inverse=True)
            
        if self.classifier not in ['soft vote', 'hard vote']:
            if hasattr(self.classifier, 'fit') == False:
                raise AttributeError('missing fit() method')
            live_Xs = [X for X in Xs if X is not None]
            if len(Xs) > 0:
                meta_X = np.concatenate(live_Xs, axis=1)
                if y is None:
                    self.classifier.fit(meta_X, **fit_params)
                else:
                    self.classifier.fit(meta_X, y, **fit_params)
        
    def predict_proba(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        channel_predictions = [None for X in Xs]
        
        if len(live_Xs) > 0:
            if self.classifier in ['soft vote', 'hard vote']:
                channel_predictions[0] = np.mean(live_Xs, axis=0)
                return channel_predictions
            else:
                meta_X = np.concatenate(live_Xs, axis=1)
                if hasattr(self.classifier, 'predict_proba') == False:
                    raise AttributeError('missing predict_proba method')
                channel_predictions[0] = self.classifier.pred_proba(meta_X)
                return channel_predictions
                
    def decision_function(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        channel_predictions = [None for X in Xs]
        if len(live_Xs) > 0:
            if self.classifier in ['soft vote', 'hard vote']:
                channel_predictions[0] = np.mean(live_Xs, axis=0)
                return channel_predictions
            else:
                meta_X = np.concatenate(live_Xs, axis=1)
                if hasattr(self.classifier, 'decision_function') == False:
                    raise AttributeError('missing decision_function method')
                channel_predictions[0] = self.classifier.decision_function(meta_X)
                return channel_predictions
    
    def predict(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        channel_predictions = [None for X in Xs]
        if len(live_Xs) > 0:
            if self.classifier == 'soft vote':
                mean_probs = np.mean(live_Xs, axis=0)
                decisions = np.argmax(mean_probs, axis=1)
                channel_predictions[0] = self.classes_[decisions]
            elif self.classifier == 'hard vote':
                input_decisions = np.stack([np.argmax(X, axis=1) for X in live_Xs])
                decisions = scipy.stats.mode(input_decisions, axis = 0)[0][0]
                channel_predictions[0] = self.classes_[decisions]
            else:
                predict_method = utils.get_predict_method(self.classifier)
                if predict_method == None:
                    raise AttributeError('missing predict method')
                meta_X = np.concatenate(live_Xs, axis=1)
                channel_predictions[0] = predict_method(meta_X)
            
        return channel_predictions
            
    def transform(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        Xs_t = [None for X in Xs]
        
        if len(live_Xs) > 0:
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
                transform_method = utils.get_transform_method(self.classifier)
                if transform_method is None:
                    raise AttributeError('missing transform method')
                else:
                    meta_X = np.concatenate(live_Xs, axis=1)
                    X_t = transform_method(meta_X)

            if len(X_t.shape) == 1:
                X_t = X_t.reshape(-1, 1)
            Xs_t[0] = Xs_t
            
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
            return ChannelClassifier(self.classifier)
        else:
            return ChannelClassifier(utils.get_clone(self.classifier))
        
class ChannelRegressor:
    
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
        self.regressor = regressor
        self._estimator_type = 'regressor'
        
    def fit(self, Xs, y=None, **fit_params):
        if y is not None:
            if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
                raise NotImplementedError('Multi-output meta-regression not supported')
            
        if self.regressor not in ['mean voting', 'median voting']:
            if hasattr(self.regressor, 'fit') == False:
                raise AttributeError('missing fit() method')
            live_Xs = [X for X in Xs if X is not None]
            if len(Xs) > 0:
                meta_X = np.concatenate(live_Xs, axis=1)
                if y is None:
                    self.regressor.fit(meta_X, **fit_params)
                else:
                    self.regressor.fit(meta_X, y, **fit_params)
    
    def predict(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        channel_predictions = [None for X in Xs]
        if len(live_Xs) > 0:
            if self.regressor == 'mean voting':
                predictions = np.mean(live_Xs, axis=0).reshape(-1)
            elif self.regressor == 'median voting':
                predictions = np.median(live_Xs, axis=0).reshape(-1)
            else:
                predict_method = utils.get_predict_method(self.regressor)
                if predict_method is None:
                    raise AttributeError('missing predict method')
                meta_X = np.concatenate(live_Xs, axis=1)
                predictions = predict_method(meta_X)
            channel_predictions[0] = predictions
          
        return channel_predictions
            
    def transform(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        Xs_t = [None for X in Xs]

        if len(live_Xs) > 0:
            if self.regressor == 'mean voting':
                predictions = np.mean(live_Xs, axis=0)
            elif self.regressor == 'median voting':
                predictions = np.median(live_Xs, axis=0)
            else:
                transform_method = utils.get_transform_method(self.regressor)
                if transform_method is None:
                    raise AttributeError('missing transform method')
                meta_X = np.concatenate(live_Xs, axis=1)
                predictions = transform_method(meta_X).reshape(-1,1)
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
            return ChannelRegressor(self.regressor)
        else:
            return ChannelRegressor(utils.get_clone(self.regressor))