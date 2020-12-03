import pipecaster.utils as utils
from pipecaster.cross_validation import cross_val_predict

__all__ = ['TransformingPredictor', 'CvPredictor']

class TransformingPredictor:
    
    """xxx. 
    
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
    params = ['predictor', 'transform_method']
    state_variables = ['transform_method_', '_estimator_type', 'classes_']
    
    def __init__(self, predictor, transform_method='auto'):
        self.predictor = predictor
        self.transform_method = transform_method
        
    def fit(self, X, y=None, **fit_params):
        if hasattr(self.predictor, 'fit') == False:
            raise AttributeError('missing fit method')
        self.predictor = utils.get_clone(self.predictor)
        if y is None:
            self.predictor.fit(X, **fit_params)
        else:
            self.predictor.fit(X, y, **fit_params)
        
        self._estimator_type = utils.detect_estimator_type(self.predictor)
            
        if self.transform_method == 'auto':
            self.transform_method_ = utils.get_transform_method(self.predictor)
            if self.transform_method_ is None:
                raise AttributeError('missing transform method')
        else:
            if hasattr(self.predictor, self.transform_method):
                self.transform_method_ = getattr(self.predictor, self.transform_method)
            else:
                raise AttributeError('transform method {} not found'.format(self.transform_method))
                
        return self
    
    def transform(self, X):
        X = self.transform_method_(X)
        X = X.reshape(-1, 1) if self._estimator_type == 'regressor' and len(X.shape) == 1 else X
        return X
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
                        
    def get_params(self, deep=False):
        return {p:getattr(self,p) for p in TransformingPredictor.params}
    
    def set_params(self, params):
        for key, value in params.items():
            if key in TransformingPredictor.params:
                setattr(self, key, value)
            else:
                raise AttributeError('invalid parameter name')

    def _more_tags(self):
        return {'multiple_inputs': True}
    
    def get_clone(self):
        clone = TransformingPredictor(utils.get_clone(self.predictor), transform_method=self.transform_method)
        for var in TransformingPredictor.state_variables:
            if hasattr(self, var):
                setattr(clone, var, getattr(self, var))
        return clone
    
class CvPredictor(TransformingPredictor):
    
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
    
    params = ['predictor', 'transform_method', 'internal_cv', 'cv_scorer', 'cv_jobs']
    state_variables = ['transform_method_', '_estimator_type', 'classes_', 'score_']
    
    def __init__(self, predictor, transform_method='auto', internal_cv=5, cv_scorer=None, cv_jobs=1):
        super().__init__(predictor, transform_method)
        self.internal_cv = internal_cv
        self.cv_scorer = cv_scorer
        self.cv_jobs = cv_jobs
                
    def fit_transform(self, X, y=None, groups=None, **fit_params):
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
        X = X.reshape(-1, 1) if self._estimator_type == 'regressor' and len(X.shape) == 1 else X
        return X

    def get_params(self, deep=False):
        return {p:getattr(self, p) for p in CvPredictor.params}
    
    def set_params(self, params):
        for key, value in params.items():
            if key in CvPredictor.params:
                setattr(self, key, value)
            else:
                raise AttributeError('invalid parameter name')

    def _more_tags(self):
        return {'multiple_inputs': True}
    
    def get_clone(self):
        clone = CvPredictor(utils.get_clone(self.predictor), transform_method=self.transform_method,
                           internal_cv=self.internal_cv, cv_scorer=self.cv_scorer, cv_jobs=self.cv_jobs)
        for var in CvPredictor.state_variables:
            if hasattr(self, var):
                setattr(clone, var, getattr(self, var))
        return clone