import numpy as np

from sklearn.feature_selection import f_classif

from pipecaster.cross_validation import cross_val_score

__all__ = ['AggregateFeatureScorer', 'CvPerformanceScorer']

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
    
    def __init__(self, predictor, cv, scorer, channel_jobs=1, cv_jobs=1):
        self.predictor = predictor
        self.cv = cv
        self.scorer = scorer
        self.channel_jobs = channel_jobs
        self.cv_jobs = cv_jobs
    
    def __call__(self, X, y, **fit_params):
        if X is None:
            return None
        else:
            scores = cross_val_score(self.predictor, X, y, scorer=self.scorer, cv=self.cv, n_jobs=self.cv_jobs, **fit_params)
            return np.mean(scores)
    
    def get_clone(self):
        return PerformanceScorer(self.cv, self.scorer, self.channel_jobs, self.cv_jobs)