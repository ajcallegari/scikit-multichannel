import numpy as np
import ray

from sklearn.metrics import accuracy_score

import pipecaster.utils as utils
from pipecaster.utils import Cloneable, Saveable
from pipecaster.predicting_transformers import CvTranformer
from pipecaster.score_selection import RankScoreSelector
from pipecaster.channel_scoring import CvPerformanceScorer
from pipecaster.cross_validation import cross_val_score 

__all__ = ['CvModelScorer', 'ChannelModelSelector', 'SelectKBestModels']

class CvModelScorer(Cloneable):
    
    def __init__(self, cv, scorer, channel_processes=1, cv_processes=1):
        self._params_to_attributes(locals())
    
    def __call__(self, predictor, X, y, **fit_params):
        if X is None:
            return None
        else:
            scores = cross_val_score(predictor, X, y, scorer=self.scorer, cv=self.cv, n_jobs=self.cv_processes, 
                                     verbose=0, fit_params=fit_params)
            return np.mean(scores)
    
class ChannelModelSelector(Cloneable, Saveable):
    """
    
    notes
    -----
    predictors can be list of predictors or single to be broadcast
    
    """
    
    def __init__(self, predictors=None, internal_cv=5, scorer=explained_variance_score, 
                 score_selector=RankScoreSelector(3), channel_processes=1, cv_processes=1):
        self._params_to_attributes(locals())
    
    @staticmethod
    def _fit_job(predictor, X, y, fit_params):
        model = utils.get_clone(predictor)
        if y is None:
            predictions = model.fit_transform(X, **fit_params)
        else:
            predictions = model.fit_transform(X, y, **fit_params)
            
        return model, predictions
        
    def fit(self, Xs, y=None, **fit_params):
        
        is_listlike = isinstance(self.predictors, (list, tuple, np.ndarray))
        if is_listlike:
            if len(Xs) != len(self.predictors):
                raise ValueError('predictor list length does not match input list length')
            else:
                predictors = [p if X is not None else None for X, p in zip(Xs, self.predictors)]
        else:
            predictors = [self.predictors if X is not None else None for X in Xs]
        
        for i, p in enumerate(predictors):
            if Xs[i] is None:
                predictors[i] = None
                continue
            if type(p) == CvTransformer:
                raise TypeError('CvTransformer found in predictors. Disallowed to enable uniform CvTransformer scoring')
            predictors[i] = CvTransformer(p, internal_cv=self.internal_cv, scorer=self.scorer, cv_processes=1)
        
        active_X_indices = [i for i, X in enumerate(Xs) if X is not None]
        args_list = [(p, X, y, fit_params) for p, X in zip(predictors, Xs) if X is not None]
        
        n_processes = self.channel_processes
        if n_processes is not None and n_processes > 1:
            try:
                shared_mem_objects = [X, y, fit_params]
                fit_results = parallel.starmap_jobs(ChannelModelSelector._fit_job, args_list, 
                                                    n_cpus=self.n_processes, 
                                                    shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'.format(e))
                print('defaulting to single processor')
                n_processes = 1       
        if n_processes is None or n_processes <= 1:
            # print('running a single process with {} jobs'.format(len(args_list)))
            fit_results = [ChannelModelSelector._fit_job(**args) for args in args_list]
                
        models, predictions_list = zip(*fit_results)
        if self.scorer is not None and self.score_selector is not None: 
            model_scores = [p.score_ for p in models]
            selected_indices = self.score_selector(model_scores)
            
        channel_models = [None for X in Xs]
        channel_predictions = [None for X in Xs]
        for i in selected_indices:
            self.models[active_X_indices[i]] = models[i]
            channel_predictions[active_X_indices[i]] = predictions_list[i]
           
        selected_indices = set(self.score_selector(performance_scores))
        self.predictors = [self.predictors[i] if i in selected_indices else None for i in range(len(Xs))]
            
    def transform(self, Xs):
        return [p.transform(Xs) if p is not None else None for p in self.predictors]
            
    def fit_transform(self, Xs, y, **fit_params):
        self.fit(Xs, y, **fit_params)
        return [p.transform(Xs) if p is not None else None for p in self.predictors]
    
    def get_params(self, deep=False):
        return {'predictors':self.predictors,
                'model_scorer':self.model_scorer,
                'score_selector':self.score_selector,
                'channel_processes':self.channel_processes,
                'cv_processes':self.cv_processes}
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ['predictors', 'model_scorer','score_selector', 'channel_processes', 'cv_processes']:
                setattr(self, key, value)
            else:
                raise AttributeError('{} not a valid ChannelModelSelector parameter'.format(key))
                
    def get_selection_indices(self):
        return [i for i, p in enumerate(self.predictors) if p is not None]
                
    def get_clone(self):
        clone = ChannelModelSelector(utils.get_list_clone(self.predictors), self.model_scorer, 
                                     self.score_selector, self.channel_processes, self.cv_processes)
        return clone
    
class SelectKBestModels(ChannelModelSelector):
    
    def __init__(self, predictors, cv=3, scorer=accuracy_score, k=1, channel_processes=1, cv_processes=1):
        self._params_to_attributes(locals())
        super().__init__(predictors, CvModelScorer(cv, scorer, channel_processes, cv_processes), RankScoreSelector(k))