import ray

import pipecaster.utils as utils
from pipecaster.channel_metaprediction import TransformingPredictor


class PredictorStack:
    
    def __init__(self, base_predictors, meta_predictor, internal_cv=5, predictor_jobs=1, cv_jobs=1):
        self.base_predictors = base_predictors
        self.meta_predictor = meta_predictor
        self.internal_cv = internal_cv
        self.predictor_jobs = predictor_jobs
        self.cv_jobs = cv_jobs
        
    def fit(self, X, y=None, **fit_params):
        estimator_types = [p._estimator_type for p in self.base_predictors]
        if len(estimator_types) != len(set(estimator_types)):
            raise ValueError('base_predictors must be of uniform type (e.g. all classifiers or all regressors)')
        self._estimator_type = estimator_types[0]
        self.base_predictors = [utils.get_clone(p) if type(p) == TransformingPredictor else 
                                TransformingPredictor(p, internal_cv=self.internal_cv, cv_jobs=self.cv_jobs)]
        
        predictions_list = []
        for predictor in self.base_predictors:
            if y is None:
                predictions_list.append(predictor.fit_transform(X, **fit_params))
                predictor.fit(X, **fit_params)
            else:
                predictions_list.append(predictor.fit_transform(X, y, **fit_params))
                predictor.fit(X, y, **fit_params)

        meta_X = np.concatenate(predictions_list, axis=1)
        self.meta_predictor = utils.get_clone(meta_predictor)
        self.meta_predictor.fit(meta_X, y, **fit_params)
        
    def predict(self, X):
        predictions_list = [p.transform(X) for p in self.base_predictors]
        meta_X = np.concatenate(predictions_list, axis=1)
        
        return self.meta_predictor.predict(meta_X)
        
    def transform(self, X):
        predictions_list = [p.transform(X) for p in self.base_predictors]
        meta_X = np.concatenate(predictions_list, axis=1)
        
        return self.meta_predictor.transform(meta_X)
        
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        self.tranform(X
        
    def get_params(self, ):
        
    def set_params(self, ):
        
    def get_clone(self, ):
        

class ClassifierStack:
    
    def __init__(self, base_classifiers, meta_classifier='soft voting', internal_cv=5, cv_scorer='auto', 
                 score_selector=None, classifier_jobs=1, cv_jobs=1):
        
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier
        self.internal_cv = internal_cv
        self.cv_scorer = cv_scorer
        self.score_selector = score_selector
        self.classifier_jobs = classifier_jobs
        self.cv_jobs = cv_jobs
        self._estimator_type = 'classifier'
    
    @staticmethod
    def _get_score(model_scorer, predictor, X, y, **fit_params):
        if X is None:
            return None
        else:
            return model_scorer(model_scorer, X, y, **fit_params)
    
    @ray.remote
    def _ray_get_score(model_scorer, predictor, X, y, **fit_params):
        return _get_score(model_scorer, predictor, X, y, **fit_params)
    
    def fit(self, X, y, **fit_params):
        estimator_type = self._estimator_type
        scores = []
        for i, predictor in enumerate(predictors):
            if type(predictor) != TransformingPredictor:
                predictor = TransformingPredictor(predictor, transform_method=transform_method, 
                                                  internal_cv=self.internal_cv, cv_scorer=self.cv_scorer, 
                                                  cv_jobs=self.cv_jobs)
            predictors[i] = utils.get_clone(predictor)
            predictors[i].fit_transform(X, y, **fit_params)
            scores.append(predictors[i].score_)
                
        self.predictors = [utils.get_clone(p) for p in self.predictors]
        self.predictors = [TransformingPredictor(p) if X is not None else None for X, p in zip(Xs, self.predictors)]
            
        if self.channel_jobs < 2:
            performance_scores = [ChannelModelSelector._get_score(self.model_scorer, predictor, X, y, **fit_params) 
                                  for predictor, X in zip(self.predictors, Xs)]
        else:
            y = ray.put(y)
            model_scorer = ray.put(self.model_scorer)
            jobs = [ChannelModelSelector._ray_get_score.remote(model_scorer, ray.put(predictor), 
                                                               ray.put(X), y, **fit_params) 
                    for predictor, X in zip(self.predictors, Xs)]
            performance_scores = ray.get(jobs)
        
        selected_indices = set(self.score_selector(performance_scores))
        self.predictors = [self.predictors[i] if i in selected_indices else None for i in range(len(Xs))]
            
    def transform(self, Xs):
        return [p.transform(Xs) if p is not None else None for p in self.predictors]
            
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return [p.transform(Xs) if p is not None else None for p in self.predictors]
    
    def get_params(self, deep=False):
        return {'predictors':self.predictors,
                'model_scorer':self.model_scorer,
                'score_selector':self.score_selector,
                'channel_jobs':self.channel_jobs,
                'cv_jobs':self.cv_jobs}
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ['predictors', 'model_scorer','score_selector', 'channel_jobs', 'cv_jobs']:
                setattr(self, key, value)
            else:
                raise AttributeError('{} not a valid ChannelModelSelector parameter'.format(key))
                
    def get_selection_indices(self):
        return [i for i, p in enumerate(self.predictors) if p is not None]
                
    def get_clone(self):
        clone = ChannelModelSelector(utils.get_list_clone(self.predictors), self.model_scorer, 
                                     self.score_selector, self.channel_jobs, self.cv_jobs)
        return clone