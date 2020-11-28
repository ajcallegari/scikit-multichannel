import numpy as np
import ray

import pipecaster.utils as utils
from pipecaster.metaprediction import TransformingPredictor
from pipecaster.score_selection import RankScoreSelector
from pipecaster.channel_scoring import CvPerformanceScorer


__all__ = ['SelectKBestModels']


class CvModelScorer:
    
    def __init__(self, cv, scoring, channel_jobs=1, cv_jobs=1):
        self.cv = cv
        self.scoring = scoring
        self.channel_jobs = channel_jobs
        self.cv_jobs = cv_jobs
    
    def __call__(self, predictor, X, y, **fit_params):
        if X is None:
            return None
        else:
            scores = cross_val_score(predictor, X, y, scoring=self.scoring, cv=self.cv, n_jobs=self.cv_jobs, 
                                     verbose=0, fit_params=fit_params)
            return np.mean(scores)
    
    def get_clone(self):
        return PerformanceScorer(self.cv, self.scoring, self.channel_jobs, self.cv_jobs)
    
class ChannelModelSelector:
    
    def __init__(self, predictors, model_scorer, score_selector, channel_jobs=1, cv_jobs=1):
        self.predictors = predictors
        self.model_scorer = model_scorer
        self.score_selector = score_selector
        self.channel_jobs = channel_jobs
        self.cv_jobs = cv_jobs
    
    @staticmethod
    def _get_score(predictor, X, y, **fit_params):
        if X is None:
            return None
        else:
            return self.performance_scorer(predictor, X, y, **fit_params)
    
    @ray.remote
    def _ray_get_score(predictor, X, y, **fit_params):
        return _get_score(predictor, X, y, **fit_params)
    
    def fit(self, Xs, y=None, **fit_params):
        is_listlike = isinstance(self.predictors, (list, tuple, np.ndarray))
        if is_listlike:
            if len(Xs) != len(self.predictors):
                raise ValueError('predictor list length does not match input list length')
            else:
                self.predictors = [utils.get_clone(p) if X is not None else None for X, p in zip(Xs, self.predictors)]
        else:
            self.predictors = [utils.get_clone(self.predictors) if X is not None else None for X in Xs]
        self.predictors = [TransformingPredictor(p) if X is not None else None for X, p in zip(Xs, self.predictors)]
            
        if self.channel_jobs < 2:
            performance_scores = [ChannelModelSelector._get_score(predictor, X, y, **fit_params) 
                                  for predictor, X in zip(self.predictors, Xs)]
        else:
            y = ray.put(y)
            jobs = [ChannelModelSelector._ray_get_score.remote(ray.put(predictor), ray.put(X), y, **fit_params) 
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
                'performance_scorer':self.performance_scorer,
                'channel_scorer':self.channel_scorer,
                'score_selector':self.score_selector,
                'channel_jobs':self.channel_jobs,
                'cv_jobs':self.cv_jobs}
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ['predictors', 'performance_scorer', 'channel_scorer', 'score_selector', 'channel_jobs', 'cv_jobs']:
                setattr(self, key, value)
            else:
                raise AttributeError('{} not a valid ChannelModelSelector parameter'.format(key))
                
    def get_clone(self):
        clone = ChannelSelector(self.channel_scorer, self.score_selector)
        if hasattr(self, 'selected_indices_'):
            clone.selected_indices_ = self.selected_indices_
        return clone
    
class SelectKBestModels(ChannelModelSelector):
    
    def __init__(self, predictors, cv=3, scoring='accuracy', k=1, channel_jobs=1, cv_jobs=1):
        self.predictors = predictors
        self.cv = cv
        self.scoring = scoring
        self.k = k
        self.channel_jobs = channel_jobs
        self.cv_jobs = cv_jobs
        super().__init__(predictors, CvPerformanceScorer(probe, cv, scoring, channel_jobs, cv_jobs), RankScoreSelector(k))

    def __str__(self, verbose = True):
        return utils.get_descriptor(self.__class__.__name__, self.get_params(), verbose)
    
    def __repr__(self, verbose = True):
        return self.__str__(verbose)  
    
    def get_params(self):
        return {'predictors':self.predictors,
                'cv':self.cv,
                'scoring':self.scoring,
                'k':self.k,
                'channel_jobs':self.channel_jobs,
                'cv_jobs':self.cv_jobs}
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ['predictors', 'cv', 'scoring', 'k', 'channel_jobs', 'cv_jobs']:
                setattr(self, key, value)
            else:
                raise AttributeError('{} not a valid SelectKBestModels parameter'.format(key))    
    def get_clone(self):
        return SelectKBestPerformers(utils.get_clone(self.probe), self.cv, self.scoring, 
                                     self.k, self.channel_jobs, self.cv_jobs)