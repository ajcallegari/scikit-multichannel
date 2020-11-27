import numpy as np
import ray

from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score

import pipecaster.utils as utils
from pipecaster.metaprediction import TransformingPredictor

__all__ = ['SelectKBestChannels', 'SelectKBestPerformers']

class AggregateFeatureScorer:
    
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum):
        self.feature_scorer = feature_scorer
        self.aggregator = aggregator
        
    def __call__(self, X, y):
        if X is None:
            return None
        else:
            score_func_ret = self.feature_scorer(X, y)
            if isinstance(score_func_ret, (list, tuple)):
                scores = np.array(score_func_ret[0]).astype(float)
            else:
                scores = np.array(score_func_ret).astype(float)
            return self.aggregator(scores)
    
    def get_clone(self):
        return AggregateFeatureScorer(self.feature_scorer, self.aggregator)
    
class CvPerformanceScorer:
    
    def __init__(self, predictor, cv, scoring, channel_jobs=1, cv_jobs=1):
        self.predictor = predictor
        self.cv = cv
        self.scoring = scoring
        self.channel_jobs = channel_jobs
        self.cv_jobs = cv_jobs
    
    def __call__(self, X, y, **fit_params):
        if X is None:
            return None
        else:
            scores = cross_val_score(self.predictor, X, y, scoring=self.scoring, cv=self.cv, n_jobs=self.cv_jobs, 
                                     verbose=0, fit_params=fit_params)
            return np.mean(scores)
    
    def get_clone(self):
        return PerformanceScorer(self.cv, self.scoring, self.channel_jobs, self.cv_jobs)
        
class RankScoreSelector:
    
    def __init__(self, k=3):
        self.k=k
        
    def __call__(self, channel_scores):
        k = self.k if self.k < len(channel_scores) else len(channel_scores) - 1
        return set(np.argsort(channel_scores)[-k:])
    
    def get_clone(self):
        return RankScoreSelector(self.k)

class ChannelSelector:
    
    def __init__(self, channel_scorer, score_selector, channel_jobs=1):
        self.channel_scorer = channel_scorer
        self.score_selector = score_selector
        self.channel_jobs = channel_jobs
        
    @ray.remote
    def _get_channel_score(channel_scorer, X, y, **fit_params):
        if X is None:
            return None
        return channel_scorer(X, y, **fit_params)
    
    def fit(self, Xs, y=None, **fit_params):
        
        if self.channel_jobs < 2:
            channel_scores = [self.channel_scorer(X, y, **fit_params) if X is not None else None for X in Xs]
        else:
            channel_scorer = ray.put(self.channel_scorer)
            y = ray.put(y)
            jobs = [ChannelSelector._get_channel_score(channel_scorer, ray.put(X), y, **fit_params)
                    for X in Xs]
            channel_score = ray.get(jobs)
            
        self.selected_indices_ = self.score_selector(channel_scores)
            
    def transform(self, Xs):
        return [Xs[i] if (i in self.selected_indices_) else None for i in range(len(Xs))]
            
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)
    
    def get_params(self, deep=False):
        return {'channel_scorer':self.channel_scorer,
                'score_selector':self.score_selector}
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ['channel_scorer', 'score_selector']:
                setattr(self, key, value)
            else:
                raise AttributeError('{} not a valid ChannelSelector parameter'.format(key))
                
    def get_clone(self):
        clone = ChannelSelector(self.channel_scorer.clone(), self.score_selector.clone())
        if hasattr(self, 'selected_indices_'):
            clone.selected_indices_ = self.selected_indices_
        return clone
                
class SelectKBestChannels(ChannelSelector):
    
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, k=3):
        self.feature_scorer = feature_scorer
        self.aggregator = aggregator
        self.k = k
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator), RankScoreSelector(k))
        
    def __str__(self, verbose = True):
        return utils.get_descriptor(self.__class__.__name__, self.get_params(), verbose)
    
    def __repr__(self, verbose = True):
        return self.__str__(verbose)
    
    def get_clone(self):
        return SelectKBestChannels(self.feature_scorer, self.aggregator, self.k)
    
class SelectKBestPerformers(ChannelSelector):
    
    def __init__(self, probe, cv, scoring, k, channel_jobs=1, cv_jobs=1):
        self.probe = probe
        self.cv = cv
        self.scoring = scoring
        self.k = k
        self.channel_jobs = channel_jobs
        self.cv_jobs = cv_jobs
        super().__init__(CvPerformanceScorer(probe, cv, scoring, channel_jobs, cv_jobs), RankScoreSelector(k))

    def __str__(self, verbose = True):
        return utils.get_descriptor(self.__class__.__name__, self.get_params(), verbose)
    
    def __repr__(self, verbose = True):
        return self.__str__(verbose)  
    
    def get_clone(self):
        return SelectKBestPerformers(utils.get_clone(self.probe), self.cv, self.scoring, 
                                     self.k, self.channel_jobs, self.cv_jobs)
    
class ChannelModelSelector:
    
    def __init__(self, predictors, performance_scorer, score_selector, channel_jobs=1, cv_jobs=1):
        self.predictors = predictors
        self.performance_scorer = performance_scorer
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
        return {'channel_scorer':self.channel_scorer,
                'score_selector':self.score_selector}
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ['channel_scorer', 'score_selector']:
                setattr(self, key, value)
            else:
                raise AttributeError('{} not a valid ChannelSelector parameter'.format(key))
                
    def get_clone(self):
        clone = ChannelSelector(self.channel_scorer, self.score_selector)
        if hasattr(self, 'selected_indices_'):
            clone.selected_indices_ = self.selected_indices_
        return clone