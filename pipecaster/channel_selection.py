import numpy as np
import ray

from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_score

import pipecaster.utils as utils


__all__ = ['SelectKBestChannels', 'SelectKBestPerformers']

class AggregateFeatureScorer:
    
    def __init__(self, feature_scorer, aggregator=np.sum):
        self.feature_scorer = feature_scorer
        self.aggregator = aggregator
        
    def __call__(self, Xs, y):
        X_scores = []
        for X in Xs:
            if X is None:
                X_scores.append(np.nan)
            else:
                score_func_ret = self.feature_scorer(X, y)
                if isinstance(score_func_ret, (list, tuple)):
                    scores = np.array(score_func_ret[0]).astype(float)
                else:
                    scores = np.array(score_func_ret).astype(float)
                X_scores.append(self.aggregator(scores))
                
        return X_scores
        
class PerformanceScorer:
    
    def __init__(self, probe, cv, scoring, channel_jobs=1, cv_jobs=1):
        self.probe = probe
        self.cv = cv
        self.scoring = scoring
        self.channel_jobs = channel_jobs
        self.cv_jobs = cv_jobs
    
    @staticmethod
    def _get_performance_score(probe, X, y, scoring, cv, cv_jobs, **fit_params):
        if X is None:
            return None
        else:
            scores = cross_val_score(probe, X, y, scoring=scoring, cv=cv, n_jobs=cv_jobs, 
                                     verbose=0, fit_params=fit_params)
            return np.mean(scores)
    
    @ray.remote
    def _ray_get_performance_score(probe, X, y, scoring, cv, cv_jobs, **fit_params):
        return PerformanceScorer._get_performance_score(probe, X, y, scoring, cv, cv_jobs, **fit_params)
        
    def __call__(self, Xs, y, **fit_params):
        
        if self.channel_jobs < 2:
            X_scores = [PerformanceScorer._get_performance_score(self.probe, X, y, self.scoring, self.cv, 
                                                                 self.cv_jobs, **fit_params)
                        for X in Xs]
        elif self.channel_jobs > 1:
            jobs = [PerformanceScorer._ray_get_performance_score.remote(ray.put(self.probe), ray.put(X), ray.put(y), 
                                                                        self.scoring, self.cv, self.cv_jobs, 
                                                                        **fit_params) 
                    for X in Xs]
            X_scores = ray.get(jobs)
                
        return X_scores
        
class RankScoreSelector:
    
    def __init__(self, k=3):
        self.k=k
        
    def __call__(self, channel_scores):
        k = self.k if self.k < len(channel_scores) else len(channel_scores) - 1
        return set(np.argsort(channel_scores)[-k:])

class ChannelSelector:
    
    def __init__(self, channel_scorer, score_selector):
        self.channel_scorer = channel_scorer
        self.score_selector = score_selector
    
    def fit(self, Xs, y=None, groups=None, **fit_params):
        channel_scores = self.channel_scorer(Xs, y, **fit_params)
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
        clone = ChannelSelector(self.channel_scorer, self.score_selector)
        if hasattr(self, 'selected_indices_'):
            clone.selected_indices_ = self.selected_indices_
        return clone
                
class SelectKBestChannels(ChannelSelector):
    
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, k=3):
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator), RankScoreSelector(k))
        
    def __str__(self, verbose = True):
        return utils.get_descriptor(self.__class__.__name__, self.get_params(), verbose)
    
    def __repr__(self, verbose = True):
        return self.__str__(verbose)
    
class SelectKBestPerformers(ChannelSelector):
    
    def __init__(self, probe, cv, scoring, k, channel_jobs=1, cv_jobs=1):
        super().__init__(PerformanceScorer(probe, cv, scoring, channel_jobs, cv_jobs), RankScoreSelector(k))

    def __str__(self, verbose = True):
        return utils.get_descriptor(self.__class__.__name__, self.get_params(), verbose)
    
    def __repr__(self, verbose = True):
        return self.__str__(verbose)    