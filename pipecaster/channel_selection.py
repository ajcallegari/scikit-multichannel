import numpy as np
from sklearn.feature_selection import f_classif

__all__ = ['SelectKBestChannels']

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
    
    def fit(self, Xs, y=None, **fit_params):
        channel_scores = self.channel_scorer(Xs, y)
        self.selected_indices_ = self.score_selector(channel_scores)
            
    def transform(self, Xs):
        return [Xs[i] if (i in self.selected_indices_) else None for i in range(len(Xs))]
            
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y)
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
        string_ = self.__class__.__name__ + '('
        
        if verbose:
            argstrings = []
            for k, v in self.get_params().items():
                argstring = k + '=' 
                if hasattr(v, '__name__'):
                    argstring += v.__name__
                elif hasattr(v, '__str__'):
                    argstring += v.__str__()
                elif type(v) in [str, int, float]:
                    argstring += v
                else:
                    argstring += 'NA'
                argstrings.append(argstring)
            string_ += ', '.join(argstrings)
                    
        return  string_ + ')'
    
    def __repr__(self, verbose = True):
        return self.__str__(verbose)

    