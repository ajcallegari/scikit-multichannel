import numpy as np
import ray

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif

import pipecaster.utils as utils
from pipecaster.utils import Clonab, Saveable
from pipecaster.score_selection import RankScoreSelector
from pipecaster.channel_scoring import AggregateFeatureScorer, CvPerformanceScorer

__all__ = ['ChannelSelector', 'SelectKBestChannels', 'SelectKBestPerformers']
        
class ChannelSelector(Clonable, Saveable):
    
    state_variable = ['selected_indices_', 'channel_scores_']
    
    def __init__(self, channel_scorer, score_selector, channel_jobs=1):
        super()._init_params(locals())
        
    @ray.remote
    def _get_channel_score(channel_scorer, X, y, **fit_params):
        if X is None:
            return None
        return channel_scorer(X, y, **fit_params)
    
    def fit(self, Xs, y=None, **fit_params):
        
        if self.channel_jobs < 2:
            self.channel_scores_ = [self.channel_scorer(X, y, **fit_params) if X is not None else None for X in Xs]
        else:
            channel_scorer = ray.put(self.channel_scorer)
            y = ray.put(y)
            jobs = [ChannelSelector._get_channel_score(channel_scorer, ray.put(X), y, **fit_params)
                    for X in Xs]
            self.channel_scores_ = ray.get(jobs)
            
        self.selected_indices_ = self.score_selector(self.channel_scores_)
        
    def get_channel_scores(self):
        if hasattr(self, 'channel_scores_'):
            return self.channel_scores_
        else:
            raise utils.FitError('channel scores requested before they were computed by a call to fit()')
            
    def transform(self, Xs):
        return [Xs[i] if (i in self.selected_indices_) else None for i in range(len(Xs))]
            
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)
                
    def get_selection_indices(self):
        return self.selected_indices_

class SelectKBestChannels(ChannelSelector):
    
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, k=1):
        super()._init_params(locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator), RankScoreSelector(k))
    
class SelectKBestPerformers(ChannelSelector):
    
    def __init__(self, probe, cv=3, scorer=accuracy_score, k=1, channel_jobs=1, cv_jobs=1):
        super()._init_params(locals())
        super().__init__(CvPerformanceScorer(probe, cv, scorer, channel_jobs, cv_jobs), RankScoreSelector(k))
    