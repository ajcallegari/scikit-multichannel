import numpy as np
import ray

from sklearn.feature_selection import f_classif

import pipecaster.utils as utils
from pipecaster.score_selection import RankScoreSelector
from pipecaster.channel_scoring import AggregateFeatureScorer, CvPerformanceScorer

__all__ = ['SelectKBestChannels', 'SelectKBestPerformers']

        
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
    
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, k=1):
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
    
    def __init__(self, probe, cv=3, scoring='accuracy', k=1, channel_jobs=1, cv_jobs=1):
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
    