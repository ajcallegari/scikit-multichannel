import numpy as np

from pipecaster.utils import Cloneable, Saveable

"""
Callables for choosing figure of merit scores (e.g. for feature selection or channel selection).
Each returns an array of indices of the selected scores (indices into the list of all scores).

"""

class RankScoreSelector(Cloneable, Saveable):
        
    def __init__(self, k):
        self.k = k if k >=1 else k
        
    def __call__(self, scores):
        k = len(scores) if self.k > len(scores) else self.k
        return set(np.argsort(scores)[-k:])
    
class PctRankScoreSelector(Cloneable, Saveable):
        
    def __init__(self, pct, n_min=1):
        self._params_to_attributes(PctRankScoreSelector.__init__, locals())
        
    def __call__(self, scores):
        k = int(len(scores) * self.pct/100.0)
        k = 1 if k < 1 else k
        k = len(scores) if k > len(scores) else k
        selected_indices = np.argsort(scores)[-k:]
        if len(selected_indices) < self.n_min:
            selected_indices = RankScoreSelector(self.n_min)(scores)
        return selected_indices
    
class CutoffScoreSelector(Cloneable, Saveable):
    
    def __init__(self, cutoff=0.0, n_min=1):
        self._params_to_attributes(CutoffScoreSelector.__init__, locals())

        
    def __call__(self, scores):
        selected_indices = np.flatnonzero(np.array(scores) > self.cutoff)
        if len(selected_indices) < self.n_min:
            selected_indices = RankScoreSelector(self.n_min)(scores)
        return selected_indices
    
class VarianceCutoffScoreSelector(Cloneable, Saveable):
        
    def __init__(self, variance_cutoff=2.0, get_variance=np.nanstd, get_baseline=np.nanmean, n_min=1):
        self._params_to_attributes(VarianceCutoffScoreSelector.__init__, locals())
        
    def __call__(self, scores):
        scores = np.array(scores)
        variance = self.get_variance(scores)
        baseline = self.get_baseline(scores)
        selected_indices = np.flatnonzero(scores >= baseline + variance_cutoff * variance)
        if len(selected_indices) < self.n_min:
            selected_indices = RankScoreSelector(self.n_min)(scores)
        return selected_indices
    
