from sklearn.model_selection import cross_val_predict, StratifiedKFold

from pipecaster import pipeline

class TransformingPredictor:
    
    def __init__(self, predictor, method='auto', internal_cv = 5):
        self.predictor = predictor
        self.method = method
        self.internal_cv = internal_cv
        
    def fit(self, X, y, **fit_params):
         if self.method == 'auto'
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
        
    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        transform_method = getattr(self.predictor, self.method_)
        if self.is_classifier:
            if type(self.internal_cv) == int:
                if self.internal_cv < 2:
                    
                else:
                for train_index, test_index in skf.split(X, y):
                    print("TRAIN:", train_index, "TEST:", test_index)
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]            
                else:
                
                split_generator = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
            
        elif self.is_regressor:
            
            
        predictions = cross_val_predict(pipeline.get_clone(self.predictor), X, y, cv=deepcopy(cv),
                                   method=meth, n_jobs=self.n_jobs,
                                   fit_params=fit_params,
                                   verbose=self.verbose)

        
        X = transform(X)
        return X.reshape(-1,1) if len(X.shape) == 1 else X


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
        self.predictor.fit(Xs, y, **fit_params
                           
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
        return MetaPredictor(pipeline.get_clone(self.predictor))