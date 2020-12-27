import numpy as np

from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import f_classif

from pipecaster.cross_validation import cross_val_score
from pipecaster.utils import Cloneable, Saveable

__all__ = ['AggregateFeatureScorer', 'CvPerformanceScorer']

class AggregateFeatureScorer(Cloneable, Saveable):
    
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum):
        self._params_to_attributes(AggregateFeatureScorer.__init__, locals())
        
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
    
class CvPerformanceScorer(Cloneable, Saveable):
    
    def __init__(self, predictor_probe=None, cv=5, scorer=explained_variance_score, cv_processes=1):
        self._params_to_attributes(CvPerformanceScorer.__init__, locals())
    
    def __call__(self, X, y, **fit_params):
        if X is None:
            return None
        else:
            scores = cross_val_score(self.predictor_probe, X, y, scorer=self.scorer, cv=self.cv, 
                                     n_processes=self.cv_processes, **fit_params)
            return np.mean(scores)
    