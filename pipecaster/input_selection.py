import numpy as np
from sklearn.feature_selection import f_classif

class SelectKBestInputs:
            
    def __init__(self, score_func=f_classif, aggregator=np.sum, k=3):
        self.score_func = score_func
        self.aggregator = aggregator
        self.k = k
        
    def fit(self, Xs, y, **fit_params):
        k = self.k if self.k < len(Xs) else len(Xs) - 1
        X_scores = []
        
        for X in Xs:
            if X is None:
                X_scores.append(np.nan)
            else:
                score_func_ret = self.score_func(X, y)
                if isinstance(score_func_ret, (list, tuple)):
                    scores = np.array(score_func_ret[0]).astype(float)
                else:
                    scores = np.array(score_func_ret).astype(float)
                X_scores.append(aggregator(scores))
            
        self.selected_indices_ = np.argsort(X_scores)[-k:]
            
    def transform(self, Xs, y=None):
        Xs_t = np.full(len(Xs), None)
        Xs_t[self.selected_indices_] = Xs[self.selected_indices_]
        
        return Xs_t, y
    
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs, y)
    
    def get_params(self, deep=False):
        return {'score_func':self.score_func,
                'aggregator':self.aggregator,
                'k':self.k}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

    def _more_tags(self):
        return {'multiple_inputs': True}