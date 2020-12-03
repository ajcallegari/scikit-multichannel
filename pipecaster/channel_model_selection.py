import numpy as np
import ray

import pipecaster.utils as utils
from pipecaster.transforming_predictors import CvPredictor
from pipecaster.score_selection import RankScoreSelector
from pipecaster.channel_scoring import CvPerformanceScorer
from pipecaster.cross_validation import cross_val_score 
from sklearn.metrics import accuracy_score

__all__ = ['SelectKBestModels']

class CvModelScorer:
    
    def __init__(self, cv, scorer, channel_jobs=1, cv_jobs=1):
        self.cv = cv
        self.scorer = scorer
        self.channel_jobs = channel_jobs
        self.cv_jobs = cv_jobs
    
    def __call__(self, predictor, X, y, **fit_params):
        if X is None:
            return None
        else:
            scores = cross_val_score(predictor, X, y, scorer=self.scorer, cv=self.cv, n_jobs=self.cv_jobs, 
                                     verbose=0, fit_params=fit_params)
            return np.mean(scores)
    
    def get_clone(self):
        return PerformanceScorer(self.cv, self.scorer, self.channel_jobs, self.cv_jobs)
    
class ChannelModelSelector:
    
    def __init__(self, predictors, model_scorer, score_selector, channel_jobs=1, cv_jobs=1):
        self.predictors = predictors
        self.model_scorer = model_scorer
        self.score_selector = score_selector
        self.channel_jobs = channel_jobs
        self.cv_jobs = cv_jobs
    
    @staticmethod
    def _get_score(model_scorer, predictor, X, y, **fit_params):
        if X is None:
            return None
        else:
            return model_scorer(model_scorer, X, y, **fit_params)
    
    @ray.remote
    def _ray_get_score(model_scorer, predictor, X, y, **fit_params):
        return _get_score(model_scorer, predictor, X, y, **fit_params)
    
    def fit(self, Xs, y, **fit_params):
        
        is_listlike = isinstance(self.predictors, (list, tuple, np.ndarray))
        if is_listlike:
            if len(Xs) != len(self.predictors):
                raise ValueError('predictor list length does not match input list length')
            else:
                self.predictors = [utils.get_clone(p) if X is not None else None for X, p in zip(Xs, self.predictors)]
        else:
            self.predictors = [utils.get_clone(self.predictors) if X is not None else None for X in Xs]
        self.predictors = [CvPredictor(p) if X is not None else None for X, p in zip(Xs, self.predictors)]
            
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
            
    def fit_transform(self, Xs, y, **fit_params):
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
    
class SelectKBestModels(ChannelModelSelector):
    
    def __init__(self, predictors, cv=3, scorer=accuracy_score, k=1, channel_jobs=1, cv_jobs=1):
        self.predictors = predictors
        self.cv = cv
        self.scorer = scorer
        self.k = k
        self.channel_jobs = channel_jobs
        self.cv_jobs = cv_jobs
        super().__init__(predictors, CvModelScorer(cv, scorer, channel_jobs, cv_jobs), RankScoreSelector(k))

    def __str__(self, verbose = True):
        return utils.get_descriptor(self.__class__.__name__, self.get_params(), verbose)
    
    def __repr__(self, verbose = True):
        return self.__str__(verbose)  
    
    def get_params(self):
        return {'predictors':self.predictors,
                'cv':self.cv,
                'scorer':self.scorer,
                'k':self.k,
                'channel_jobs':self.channel_jobs,
                'cv_jobs':self.cv_jobs}
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ['predictors', 'cv', 'scorer', 'k', 'channel_jobs', 'cv_jobs']:
                setattr(self, key, value)
            else:
                raise AttributeError('{} not a valid SelectKBestModels parameter'.format(key))   
                
    def get_clone(self):
        return SelectKBestModels(utils.get_clone(self.predictors), self.cv, self.scorer, 
                                     self.k, self.channel_jobs, self.cv_jobs)