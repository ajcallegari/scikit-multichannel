import numpy as np

from pipecaster.utils import Cloneable, Saveable

class RankScoreSelector(Cloneable, Saveable):
        
    def __init__(self, k):
        self.k = k if k >=1 else k
        
    def __call__(self, channel_scores):
        k = self.k if self.k <= len(channel_scores) else len(channel_scores)
        return set(np.argsort(channel_scores)[-k:])