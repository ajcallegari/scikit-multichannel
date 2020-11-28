import numpy as np

class RankScoreSelector:
    
    def __init__(self, k=3):
        self.k=k
        
    def __call__(self, channel_scores):
        k = self.k if self.k < len(channel_scores) else len(channel_scores) - 1
        return set(np.argsort(channel_scores)[-k:])
    
    def get_clone(self):
        return RankScoreSelector(self.k)