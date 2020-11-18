from sklearn.model_selection import cross_val_predict, StratifiedKFold

from pipecaster.utility import get_clone
from pipecaster.model_selection import cross_val_predict


class TransformingPredictor:
    
    """Class that wraps sklearn predictors to provide them with a transform() method and internal cross validation training.
    
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
        
    def fit(self, X, y, **fit_params):
        if self.method == 'auto':
            if hasattr(predictor, 'predict_proba'):
                self.method_ = 'predict_proba' 
            elif hasattr(predictor, 'decision_function'):
                self.method_ = 'decision_function' 
            elif hasattr(predictor, 'predict'):
                self.method_ = 'predict' 
            else:
                raise TypeError('no method found for transform in ' + self.predictor.__class__.__name__)
        else:
            if hasattr(predictor, self.method):
                self.method_ = self.method 
            else:
                raise TypeError('{} method not found in {}'.format(self.method, pipe.__class__.__name__)) 
        self.predictor.fit(X, y, **fit_params)
        if hasattr(self.predictor, 'classes_'):
            self.classes_ = self.predictor.classes_
            self._estimator_type = 'classifier'
        else:
            self._estimator_type = 'regressor'
        
    def fit_transform(self, X, y, groups=None, fit_params=None):
        self.fit(X, y, **fit_params)
        
        # internal cv training is disabled
        if self.internal_cv is None or (type(self.internal_cv) == int and self.internal_cv < 2):
            transform_method = getattr(self.predictor, self.method_)
            X = transform_method(X)
        # internal cv training is enabled
        else:
            X= pc.cross_val_predict(self.predictor, X, y, groups=groups, cv=self.internal_cv,
                      n_jobs=1, verbose=0, fit_params=fit_params, method=self.method_)
            
        return X.reshape(-1,1) if len(X.shape) == 1 else X
    
    def transform(self, X):
        transform_method = getattr(self.predictor, self.method_)
        X = transform_method(X)
        return X.reshape(-1,1) if len(X.shape) == 1 else X
                    
    def get_params(self, deep=False):
        return {'predictor':self.predictor,
                'method':self.method,
                'internal_cv':self.internal_cv,
                'n_jobs':self.n_jobs}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

    def _more_tags(self):
        return {'multiple_inputs': True}
    
    def get_clone(self):
        clone = TransformingPredictor(get_clone(self.predictor), method=self.method, 
                                     internal_cv = self.internal_cv, n_jobs = self.n_jobs)
        for attr in ['classes_', '_estimator_type', 'method_']:
            if hasattr(self, attr):
                setattr(clone, attr, getattr(self, attr))
        return clone

class MetaPredictor:
    
    """MetaPredictor concatenates predictions generated from multiple inputs and uses them to make metapredictions
    
    arguments
    ---------
    predictor: instance of a classifier or regressor with the sklearn interface
    
    Notes
    -----
    If you are only using one input, use sklearn's StackingClassifier and StackingRegressor instead.
    Stardard practice is to use this class on outputs from base classifiers trained using internal cross validation splitting
    to prevent overfitting.  This is done internally by pipecaster, which by wraps all predictors
    in the TransformingPredictor class.  When the number of training samples is limiting rather than overfitting, 
    you may get better performance by turning off internal cv splitting by using internal_cv = 1 in the constructor of 
    TransformingPredictor.
    
    """
    
    def __init__(self, predictor):
        self.predictor = predictor
        
    def fit(self, Xs, y, **fit_params):
        Xs = [X for X in Xs if X is not None]
        Xs = np.concatenate(Xs, axis=1)
        self.predictor.fit(Xs, y, **fit_params)
                           
    def transform(self, Xs):
        Xs = [X for X in Xs if X is not None]
        X = np.concatenate(Xs, axis=1)
        if hasattr(self.predictor, 'transform'):
            X = self.predictor.transform(X)                
        elif hasattr(self.predictor, 'predict_proba'):
            X = self.predictor.predict_proba(X)
        elif hasattr(self.predictor, 'decision_function'):
            X = self.predictor.decision_function(X)         
        elif hasattr(self.predictor, 'predict'):
            X = self.predictor.predict(X)
        else:
            raise TypeError('predictor wrapped by metapredictor lacks interface for transform \
                            (transform, predict_proba, or predict')
        return X
                           
    def fit_transform(self, Xs, y, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)
                           
    def predict(self, Xs):
        Xs = [X for X in Xs if X is not None]
        X = np.concatenate(Xs, axis=1)
        return self.predict(X)
                           
    def predict_proba(self, Xs):
        Xs = [X for X in Xs if X is not None]
        X = np.concatenate(Xs, axis=1)
        return self.predict_proba(X)   
                           
    def _more_tags(self):
        return {'metapredictor': True}
                           
    def clone(self):
        return MetaPredictor(get_clone(self.predictor))