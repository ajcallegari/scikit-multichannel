import pipecaster.utils as utils
from pipecaster.utils import Cloneable, Saveable
from pipecaster.cross_validation import cross_val_predict

"""
Wrapper classes that provide single channel and multichannel predictors with transform/fit_transform methods, internal cv_training, and internal_cv performance scoring.  Used for meta-prediction and model selection.

Examples
--------
# give a single channel predictor transform and fit_transform methods
import pipecaster as pc
model = pc.transformer_wrappers.SingleChannel(model)

# give a single channel predictor transform & fit_transform methods, and internal_cv training
import pipecaster as pc
model = pc.transformer_wrappers.SingleChannelCV(model, internal_cv, cv_processes=1)

# give a multichannel predictor transform and fit_transform methods
import pipecaster as pc
model = pc.transformer_wrappers.Multichannel(model)

# give a multichannel predictor transform & fit_transform methods, and internal_cv training
import pipecaster as pc
model = pc.transformer_wrappers.MultichannelCV(model, internal_cv, cv_processes=1)

"""

# when converting a predictor to a transformer, use these methods for transforming
transform_method_precedence = ['predict_proba', 'decision_function', 'predict_log_proba', 'predict']
                        
def get_transform_method(pipe):
    for method in transform_method_precedence:
        if hasattr(pipe, method):
            return getattr(pipe, method)
    return None

def get_transform_method_name(pipe):
    for method in transform_method_precedence:
        if hasattr(pipe, method):
            return method
    return None

class SingleChannel(Cloneable, Saveable):
    """
    Wrapper class that provides scikit-learn conformant predictors with transform/fit_transform methods.
    
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
    The predictor is not cloned during construction or calls to fit(), but is cloned on get_clone() call. 
    """
    state_variables = ['classes_']
    
    def __init__(self, predictor=None):
        self._init_params(locals())
        utils.enforce_fit(predictor)
        self.transform_method_name = get_transform_method_name(predictor)
        if self.transform_method_name is None:
            raise NameError('predictor lacks a recognized method for conversion to transformer')
        self._expose_predictor_interface(predictor)
        self._esimtator_type = utils.detect_estimator_type(predictor)
        if self._esimtator_type is None:
            raise TypeError('could not detect predictor type')
        
    def _expose_predictor_interface(self, predictor):
        for method_name in utils.regognized_pred_methods:
            if hasattr(predictor, method_name):
                prediction_method = lambda self, X : self.predict_with_method(X, method_name)
                setattr(self, method_name, prediction_method)
        
    def set_transform_method(self, method_name):
        self.transform_method_name = method_name
                
    def fit(self, X, y=None, **fit_params):
        self.model = utils.get_clone(self.predictor)
        if y is None:
            self.model.fit(X, **fit_params)
        else:
            self.model.fit(X, y, **fit_params)
        if self._estimator_type == 'classifier':
            self.classes_ = self.model.classes_
            
        return self
    
    def predict_with_method(self, X, method_name):
        if hasattr(self, 'model'):
            if hasattr(self.model, method_name):
                predict_method = getattr(self.model, method_name)
                return predict_method(X)
            else:
                raise NameError('prediction method: {} not found in {}'.format(method_name, self.model))
        else:
            raise utils.FitError('prediction attempted before model fitting') 
            
    def transform(self, X):
        if hasattr(self, 'model'):
            transform_method = getattr(self.model, self.transform_method_name)
            X_t = transform_method(X)
            X_t = X_t.reshape(-1, 1) if self._estimator_type == 'regressor' and len(X_t.shape) == 1 else X_t
            return X_t
        else:
            raise utils.FitError('transform called before model fitting')
    
    def fit_transform(self, X, y=None, **fit_params):
        if hasattr(self.predictor, 'fit_transform'):
            self.model = utils.get_clone(self.predictor)
            if y is None:
                X_t = self.model.fit_transform(X, **fit_params)
            else:
                X_t = self.model.fit_transform(X, y, **fit_params)
        else:
            if y is None:
                self.fit(X, **fit_params)
            else:
                self.fit(X, y, **fit_params)
            transform_method = getattr(self.model, self.transform_method_name)
            X_t = transform_method(X)
            
        return X_t

    def _more_tags(self):
        return {'multichannel': False}
    
    def get_clone(self):
        clone = super().get_clone()
        if hasattr(self, 'model'):
            clone.model = utils.get_clone(self.model)
        return clone
    
class SingleChannelCV(SingleChannel):
    """
    Wrapper class that provides scikit-learn conformant predictors with transform/fit_transform methods and 
    internal cross validation functionality.
    
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
    fit().transform() is not the same as fit_tranform() because the latter uses internal cv training and inference.
    On calls to fit_transform() the model is fit on both the entire training set and cv splits of the training set.
    The model fit on the entire dataset is stored for inference on subsequent calls to predict(), predict_proba(), 
    decision_function(), or tranform().  The models fit on cv splits are used to make the predictions returned 
    by fit_transform but are not stored for future use. 
    """
    
    state_variables = ['transform_method_', '_estimator_type', 'classes_', 'score_']
    
    def __init__(self, predictor, internal_cv=5, split_seed=None, cv_processes=1, scorer=None):
        super()._init_params(locals())
        super().__init__(predictor)
                
    def fit_transform(self, X, y=None, groups=None, **fit_params):
        self.fit(X, y, **fit_params)
        
        # internal cv training is disabled
        if self.internal_cv is None or (type(self.internal_cv) == int and self.internal_cv < 2):
            X_t = self.transform(X)
        # internal cv training is enabled
        else:
            X_t = cross_val_predict(self.predictor, X, y, groups=groups, predict_method=self.transform_method_name, 
                                    cv=self.internal_cv, n_processes=self.cv_processes, 
                                    split_seed=self.split_seed, fit_params=fit_params)
            
            cross_val_predict(predictor, Xs, y=None, groups=None, predict_method='predict', cv=None,
                      combine_splits=True, n_processes='max', split_seed=None, fit_params=None)
            
            if self.scorer is not None:
                self.score_ = self.scorer(y, X)
            X_t = X_t.reshape(-1, 1) if self._estimator_type == 'regressor' and len(X_t.shape) == 1 else X_t
        
        return X_t

    def get_params(self, deep=False):
        return {p:getattr(self, p) for p in CvPredictor.params}
    
    def set_params(self, params):
        for key, value in params.items():
            if key in CvPredictor.params:
                setattr(self, key, value)
            else:
                raise AttributeError('invalid parameter name')

    def _more_tags(self):
        return {'multichannel': False}
    
    def get_clone(self):
        clone = CvPredictor(utils.get_clone(self.predictor), transform_method=self.transform_method,
                           internal_cv=self.internal_cv, scorer=self.scorer, cv_processes=self.cv_processes)
        for var in CvPredictor.state_variables:
            if hasattr(self, var):
                setattr(clone, var, getattr(self, var))
        return clone
    
    
class Multichannel(Cloneable, Saveable):
    
    def __init__(self, mutlichannel_predictor=None):
        super()._init_params(locals())
        utils.enforce_fit(predictor)
        utils.enforce_predictor(predictor)        
        self._estimator_type = utils.detect_estimator_type(mutlichannel_predictor)
        if self._estimator_type is None:
            raise AttributeError('could not detect predictor type')
        self.transform_method_name = get_transform_method_name(mutlichannel_predictor)
        self._expose_predictor_interface(mutlichannel_predictor)
        
    def _expose_predictor_interface(self, mutlichannel_predictor):
        for method_name in utils.recognized_pred_methods:
            if hasattr(mutlichannel_predictor, method_name):
                prediction_method = lambda self, Xs : self.predict_with_method(Xs, method_name)
                setattr(self, method_name, prediction_method)
                
    def fit(self, Xs, y=None, **fit_params):
        self.model = utils.get_clone(self.mutlichannel_predictor)
        if y is None:
            self.model.fit(Xs, **fit_params)
        else:
            self.model.fit(Xs, y, **fit_params)
        if hasattr(self.model, 'classes_'):
            self.classes_ = self.model.classes_
        return self
                
    def predict_with_method(self, Xs, method_name):
        if hasattr(self, 'model') == False:
            raise FitError('prediction attempted before call to fit()')
        X = np.concatenate(live_Xs, axis=1)
        prediction_method = getattr(self.model, method_name)
        return prediction_method(Xs)
    
    def transform(self, Xs):
        if hasattr(self, 'model') == False:
            raise FitError('transform attempted before call to fit()')
        transform_method = getattr(self.model, self.transform_method_name)
        return transform_method(Xs)
    
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y=None, **fit_params)
        return self.transform(Xs)
    
class MultichannelCV(Cloneable, Saveable):
    
    def __init__(self, mutlichannel_predictor=None, internal_cv=5, split_seed=None, cv_processes=1, scorer=None):
        super().__init__(mutlichannel_predictor)
        super()._init_params(locals())
        
    def fit_transform(self, Xs, y=None, groups=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        
        # internal cv training is disabled
        if self.internal_cv is None or (type(self.internal_cv) == int and self.internal_cv < 2):
            Xs_t = self.transform(Xs)
        # internal cv training is enabled
        else:
            Xs_t = cross_val_predict(self.predictor, Xs, y, groups=groups, predict_method=self.transform_method_name, 
                                     cv=self.internal_cv, combine_splits=True, n_processes=self.cv_processes, 
                                     split_seed=self.split_seed, fit_params=fit_params)
            
            if self.scorer is not None:
                self.score_ = self.scorer(y, Xs_t[0])
            X_t = X_t.reshape(-1, 1) if self._estimator_type == 'regressor' and len(X_t.shape) == 1 else X_t
        
        return X_t