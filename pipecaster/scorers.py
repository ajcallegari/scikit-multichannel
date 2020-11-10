import numpy as np
import pandas as pd
from scipy import stats

import pipelines as pl

#### FEATURE SCORERS ####

# mean squared error with negative sign to make higher score better
def nmse_score_func(X,y):
    y = y.reshape(-1,1)
    return -np.mean(np.square(y - X), axis = 0)

# inverse mean squared error 
def imse_score_func(X,y):
    y = y.reshape(-1,1)
    return 1/np.mean(np.square(y - X), axis = 0)

# mean absolute error with negative sign to make higher score better
def nmae_score_func(X,y):
    y = y.reshape(-1,1)
    return -np.mean(np.abs(y - X), axis = 0)

# inverse absolute error 
def imae_score_func(X,y):
    y = y.reshape(-1,1)
    return 1/np.mean(np.abs(y - X), axis = 0)

def pearson_score_func(X, y):
    scores = [abs(pearsonr(X[:,i],y)[0]) for i in range(X.shape[1])]
    return np.array(scores).astype(float)

def spearman_score_func(X, y):
    scores = [abs(spearmanr(X[:,i],y)[0]) for i in range(X.shape[1])]
    return np.array(scores).astype(float)

def r2_score_func(X,y):
    y = y.reshape(-1,1)
    y_mean = np.mean(y)
    denominator = np.sum(np.square(y - y_mean))
    numerator = np.sum(np.square(y - X), axis = 0)
    valid_mask = denominator != 0
    scores = np.empty(X.shape[1])
    scores[valid_mask] = 1 - numerator[valid_mask] / denominator
    scores[~valid_mask] = 0
    return scores

#### MATRIX SCORERS  ####

class XSumScorer:
    
    def __init__(self, score_func):
        self.score_func = score_func
        
    def __call__(self, X, y):
        return np.sum(self.score_func(X, y))
    
class XMeanScorer:
    
    def __init__(self, score_func):
        self.score_func = score_func
        
    def __call__(self, X, y):
        return np.mean(self.score_func(X, y))

#### SELECTORS ####

class RankSelector:
    
    def __init__(top_n):
        self.top_n = top_n
        
    def __call__(scores):
        indices = np.argsort(-scores)
        indices = indices[np.isfinite(indices)]
        n = top_n if top_n <= len(indices) else len(indices)
        return indices[:n]
    
class RelativeValueSelector:
    
    def __init__(cutoff, min_selected, var = 'stdev', baseline = 'mean'):
        self.cutoff = cutoff
        self.min_selected = min_selected
        self.var = var
        self.baseline = baseline
        
    def __call__(scores):
        
        scores = np.array(scores)
        
        if baseline == 'mean':
            baseline = np.nanmean(scores)
        elif baseline == 'median':
            baseline = np.nanmedian(scores)
            
        if var == 'stdev':
            var =  np.nanstd(scores)
        elif var == 'iqr':
            var = stats.iqr(scores)
            
        indices = np.flatnonzero(scores >= baseline + var * self.cutoff)
        if len(indices) < self.min_selected:
            indices = np.argsort(-scores)
            indices = indices[np.isfinite(indices)]
            n = self.min_selected if self.min_selected <= len(indices) else len(indices)
            indices = indices[:n]
            
        return indices

#### X TRANSFORMERS ####

class ScalingFSelector():
    """ Sklearn conformant transformer that finds the n most important features and then reduces the dimensionality
        of feature matrices to include only these features.  Allow features to be scale by importance score. 
    """
    
    def __init__(self, score_func, n_features, scale = False, protected_cols = None):
        """
            args:
                score_func:  feature scoring function, larger scores = larger importance
                n_features: number of top features to select (in addition to protected features)
                scale: multiply feature values by their importance score
                protected_cols: set or list-like of columns indices to be selected regardless of importance. 
                    these columns are not scaled if scale = True 
        """
        
        self.score_func = score_func
        self.n_features = n_features
        self.scale = scale
        self.protected_cols = None if protected_cols is None else np.array(protected_cols).astype(int)
        
    def fit(self, X, y):
        
        if self.protected_cols is None:
            score_func_ret = self.score_func(X, y)
            if isinstance(score_func_ret, (list, tuple)):
                scores = np.array(score_func_ret[0]).astype(float)
            else:
                scores = np.array(score_func_ret).astype(float)
            self.selected_cols = np.argsort(-scores)[:self.n_features].copy()
            if self.scale == True:
                self.weights = scores[self.selected_cols].copy()
                self.weights -= np.min(self.weights)
        else:
            all_cols = np.arange(X.shape[1])
            protected_mask = np.array([c in set(self.protected_cols) for c in all_cols]).astype(bool)
            unprotected_mask = np.invert(protected_mask)
            unprotected_cols = np.flatnonzero(unprotected_mask)

            score_func_ret = self.score_func(X[:, unprotected_cols], y)
            if isinstance(score_func_ret, (list, tuple)):
                scores = np.array(score_func_ret[0]).astype(float)
            else:
                scores = np.array(score_func_ret).astype(float)

            sorted_subset_cols = np.argsort(-scores)
            # convert subset indices into superset indices
            sorted_cols = unprotected_cols[sorted_subset_cols]
            self.selected_cols = sorted_cols[:self.n_features].copy()

            if self.scale == True:
                self.weights = scores[sorted_subset_cols[:self.n_features]].copy()   
                self.weights -= np.min(self.weights)
                                
    def transform(self, X):
        if self.protected_cols is None:
            if self.scale == True:
                return X[:, self.selected_cols] * self.weights
            else:
                return X[:, self.selected_cols]
        else:
            X_protected = X[:, self.protected_cols]
            X_selected = X[:, self.selected_cols]
            if self.scale == True:
                X_selected *= self.weights
            return np.concatenate((X_protected, X_selected), axis = 1)
        
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)
    
class AutoFSelector:
    
    def __init__(self, score_func, sigma, min_features = 5, baseline = 'mean', scale = False, protected_cols = None):
        self.score_func = score_func
        self.sigma = sigma
        self.min_features = min_features
        self.baseline = baseline
        self.scale = scale
        self.protected_cols = None if protected_cols is None else np.array(protected_cols).astype(int)
        
    def fit(self, X, y):
        
        if self.protected_cols is None:
            score_func_ret = self.score_func(X, y)
            if isinstance(score_func_ret, (list, tuple)):
                scores = np.array(score_func_ret[0]).astype(float)
            else:
                scores = np.array(score_func_ret).astype(float)

            if self.baseline == 'mean':
                mean, sd = np.nanmean(scores), np.nanstd(scores)
                cutoff = mean + self.sigma * sd
            elif self.baseline == 'median':
                median, sd = np.nanmedian(scores), np.nanstd(scores)
                cutoff = median + self.sigma * sd
            self.selected_cols = np.flatnonzero(scores > cutoff)
            if len(self.selected_cols) < self.min_features:
                self.selected_cols = np.argsort(-scores)[0:self.min_features].copy()
            if self.scale == True:
                self.weights = scores[self.keep_indices]
                self.weights -= np.min(self.weights)
                
        else:
            all_cols = np.arange(X.shape[1])
            protected_mask = np.array([c in set(self.protected_cols) for c in all_cols]).astype(bool)
            unprotected_mask = np.invert(protected_mask)
            unprotected_cols = np.flatnonzero(unprotected_mask)

            score_func_ret = self.score_func(X[:, unprotected_cols], y)
            if isinstance(score_func_ret, (list, tuple)):
                scores = np.array(score_func_ret[0]).astype(float)
            else:
                scores = np.array(score_func_ret).astype(float)
                
            if self.baseline == 'mean':
                mean, sd = np.nanmean(scores), np.nanstd(scores)
                cutoff = mean + self.sigma * sd
            elif self.baseline == 'median':
                median, sd = np.nanmedian(scores), np.nanstd(scores)
                cutoff = median + self.sigma * sd
                
            subset_keep_cols = np.flatnonzero(scores > cutoff)
            if len(subset_keep_cols) < self.min_features:
                subset_keep_cols = np.argsort(-scores)[0:self.min_features]

            # convert subset indices into superset indices
            self.selected_cols = unprotected_cols[subset_keep_cols]

            if self.scale == True:
                self.weights = scores[subset_keep_cols] 
                self.weights -= np.min(self.weights)
        
    def transform(self, X):
        if self.protected_cols is None:
            if self.scale == True:
                return X[:, self.selected_cols] * self.weights
            else:
                return X[:, self.selected_cols]
        else:
            X_protected = X[:, self.protected_cols]
            X_selected = X[:, self.selected_cols]
            if self.scale == True:
                X_selected *= self.weights
            return np.concatenate((X_protected, X_selected), axis = 1)
        
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)

#### Xs TRANSFORMERS ####

class MXTransformer:
    
    def __init__(self, transformer_types, params_list):
        assert len(tranformer_types) == len(params_list), 'parameters must be supplied for each transformer type'
        self.transformer_types = transformer_types
        self.params_list = params_list
        
    def fit(self, Xs, y):
        # if only 1 transformer type was supplied broadast that type across all feature matrices
        if len(self.transformer_types) == 1:
            self.transformers = [self.transformer_types[0](self.params_list[0]) for X in Xs]
        # if a list of transformer types was supplied instantiated each
        else:
            assert len(self.transformer_types) == len(Xs), \
                'The number of transformer types must exactly match the number of feature matrices'
            self.transformers = [t(p).fit(X,y) for t, p, X in zip(self.transformer_types, self.params_list, Xs)]
                    
    def transform(self, Xs, y):
        assert len(self.transformers) == len(Xs), 'Number of number of transformers must match number of matrices'
        return [t.transform(X, y) if hasattr(t,'transform') else t.predict(X).reshape(-1,1)
                for t, X in zip(self.transformers, Xs)], y
    
    def fit_transform(self, Xs, y):
        self.fit(Xs, y)
        return self.transform(Xs, y)
    
class MXSelector:
    
    def __init__(self, X_scorer, X_selector, protected_indices = None):
        self.X_scorer = X_scorer
        self.X_selector = X_selector
        if protected_indices is None:
            self.protected_indices = set([])
        else:
            self.protected_indices = set(protected_indices)
        
    def fit(self, Xs, y):
        scores = [self.X_scorer(X, y) if i not in self.protected_indices else np.nan for i, X in enumerate(Xs)]
        self.selected_indices = set(self.X_selector(scores))
        
    def transform(self, Xs, y):
        return [X for i, X in enumerate(Xs) if (i in self.keep_indices or i in self.protected_indices)], y
    
    def fit_transform(self, Xs, y):
        self.fit(Xs, y)
        return self.tranform(Xs, y)
    
class FXSelector:
    
    def __init__(self, F_scorer, X_scorer, X_selector, protected_X_indices = None):
        self.F_scorer = F_scorer
        self.X_scorer = X_scorer
        self.X_selector = X_selector
        if protected_X_indices is None:
            self.protected_X_indices = set([])
        else:
            self.protected_X_indices = set(protected_X_indices)
            
    def fit(self, Xs, y):
        scores = [self.X_scorer(X, y) if i not in self.protected_indices else np.nan for i, X in enumerate(Xs)]
        self.selected_indices = set(self.X_selector(scores))
        
    def transform(self, Xs, y):
        return [X for i, X in enumerate(Xs) if (i in self.keep_indices or i in self.protected_indices)], y
    
    def fit_transform(self, Xs, y):
        self.fit(Xs, y)
        return self.tranform(Xs, y)

class XConcatenator:
    
    def fit(self, Xs, y):
        return Xs, y
    
    def transform(self, Xs, y):
        return [np.concatenate(Xs, axis = 1)], y
    
    def fit_transform(self, Xs, y):
        return self.transform(Xs, y)
    
class XVoter:
    
    def __init__(self, method = 'mean'):
        self.method = method
    
    def fit(self, Xs, y):
        return Xs, y
    
    def transform(self, Xs, y):
        assert len(set([X.shape for X in Xs])) == 1, 'shapes must be identical for each X to use XVoter'
        if method == 'mean':
            return [np.mean(np.stack(Xs), axis = 0)], y
        if method == 'median':
            return [np.median(np.stack(Xs), axis = 0)], y
    
    def fit_transform(self, Xs, y):
        return self.transform(Xs, y)    

class MultiPipeline:
    
    def __init__(self, pipeline_blueprints):
        self.pipeline_blueprints = pipeline_blueprints
        
    def fit(Xs, y):
        
        self.transformers = []
        
        for X in Xs:
            self.transformers = [t[0](t[1]) for t in pipeline_blueprints]