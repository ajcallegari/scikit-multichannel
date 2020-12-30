import numpy as np
import ray
import functools

from sklearn.metrics import explained_variance_score, balanced_accuracy_score
from sklearn.feature_selection import f_classif

import pipecaster.utils as utils
from pipecaster.utils import Cloneable, Saveable
from pipecaster.score_selection import RankScoreSelector
from pipecaster.channel_scoring import AggregateFeatureScorer, CvPerformanceScorer
import pipecaster.parallel as parallel
from pipecaster.cross_validation import cross_val_score 
import pipecaster.transform_wrappers as transform_wrappers
from pipecaster.score_selection import RankScoreSelector

__all__ = ['ChannelSelector', 'ModelSelector', 'SelectKBestScores', 'SelectKBestProbes', 'SelectKBestModels']
        
class ChannelSelector(Cloneable, Saveable):
    
    state_variables = ['selected_indices_', 'channel_scores_']
    
    def __init__(self, channel_scorer=None, score_selector=None, channel_processes=1):
        self._params_to_attributes(ChannelSelector.__init__, locals())
        
    def _get_channel_score(channel_scorer, X, y, fit_params):
        if X is None:
            return None
        return channel_scorer(X, y, **fit_params)
    
    def fit(self, Xs, y=None, **fit_params):
        args_list = [(self.channel_scorer, X, y, fit_params) for X in Xs]
        n_processes = self.channel_processes
        if n_processes is not None and n_processes > 1:
            try:
                shared_mem_objects = [y, fit_params]
                self.channel_scores_ = parallel.starmap_jobs(ChannelSelector._get_channel_score, args_list, 
                                                             n_cpus=n_processes, shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'.format(e))
                print('defaulting to single processor')
                n_processes = 1       
        if n_processes is None or n_processes <= 1:
            self.channel_scores_ = [self.channel_scorer(X, y, **fit_params) if X is not None else None for X in Xs]
            
        self.selected_indices_ = self.score_selector(self.channel_scores_)
        
    def get_channel_scores(self):
        if hasattr(self, 'channel_scores_'):
            return self.channel_scores_
        else:
            raise utils.FitError('Channel scores not found. They are only available after call to fit().')
            
    def get_support(self):
        if hasattr(self, 'selected_indices_'):
            return self.selected_indices_
        else:
            raise utils.FitError('Must call fit before getting selection information')
            
    def transform(self, Xs):
        return [Xs[i] if (i in self.selected_indices_) else None for i in range(len(Xs))]
            
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)
                
    def get_selection_indices(self):
        return self.selected_indices_

class SelectKBestScores(ChannelSelector):
    
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, k=1, channel_processes=1):
        self._params_to_attributes(SelectKBestScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator), RankScoreSelector(k), channel_processes)
    
class SelectKBestProbes(ChannelSelector):
        
    def __init__(self, predictor_probe=None, cv=3, scorer='auto', k=1, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectKBestProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv, scorer, cv_processes), 
                         RankScoreSelector(k), channel_processes)
    
class ModelSelector(Cloneable, Saveable):
    """
    Multichannel predictor that tests a single-channel predictor on each channel, measures the performance of each
    using cross validation, and outputs the predictions of the best models. Note: does not concatenate outputs of multiple
    models, so if more than one channel predictor is selected then an additional voting or meta-prediction step is
    required to generate a single prediction.
    
    Parameters
    ----------
    predictors: list of sklearn conformant predictors (clasifiers or regressors) of len(Xs) or single predictor, default=None
        If list, then one predictor will be applied to each channel in the listed order.
        If single predictor, predictor is cloned and broadcast across all input channels.
    cv: int or cross validation splitter instance (e.g. StratifiedKFold()), default=5
        Set the cross validation method.  If int, defaults to KFold() for regressors orr
        StratifiedKFold() for classifiers.  
    scorer : callable or 'auto', default='auto'
        Figure of merit score used for selecting models via internal cross validation.
        If a callable, the object should have the signature 'scorer(y_true, y_pred)' and return
        a single value.  
        If 'auto' regressors will be scored with explained_variance_score and classifiers
        with balanced_accuracy_score.
    score_selector: callable, default=RankScoreSelector(3)
        A scorer callable object / function with signature 'score_selector(scores)' which should 
        return a list of the indices of the predictors/channels to be selected
    channel_processes: int
        Number of parallel processes to run for each predictor during model fitting
    cv_processes: int
        Number of parallel processes to run for each cross validation split during model fitting
        
    notes
    -----
    predictors can be list of predictors (one per channel) or single predictor to be broadcast over all channels
    currently only supports single channel predictors (e.g. scikit-learn conformant predictor)
    predict_proba and decision_function are treated as synonymous
    """
    state_variables = ['classes_', 'selected_indices_']
    
    def __init__(self, predictors=None, cv=5, scorer='auto', 
                 score_selector=RankScoreSelector(3), channel_processes=1, cv_processes=1):
        self._params_to_attributes(ModelSelector.__init__, locals())
        
        if predictors is None:
            raise ValueError('No predictors found.')
            
        if isinstance(predictors, (list, tuple, np.ndarray)):
            estimator_types = [p._estimator_type for p in predictors]
        else:
            estimator_types = [predictors._estimator_type]
        if len(set(estimator_types)) != 1:
            raise TypeError('Predictors must be of uniform type (e.g. all classifiers or all regressors).')
        self._estimator_type = estimator_types[0] 
        if scorer == 'auto':
            if utils.is_classifier(self):
                self.scorer = balanced_accuracy_score
            elif utils.is_regressor(self):
                self.scorer = explained_variance_score
            else:
                raise AttributeError('predictor type required for automatic assignment of scoring metric')
        self._expose_predictor_interface(predictors)
        
    def _expose_predictor_interface(self, predictors):
        if isinstance(predictors, (tuple, list, np.ndarray)):
            if len(predictors) == 1:
                method_set = utils.get_prediction_method_names(predictors[0])
            elif len(predictors) > 1:
                prediction_methods = [utils.get_prediction_method_names(p) for p in predictors]
                # make predict_proba and decision_funcion synonymous to enable meta-prediction with transform_wrapper
                for ms in prediction_methods:
                    if 'predict_proba' in ms:
                        ms.append('decision_function')
                    elif 'decision_function' in ms:
                        ms.append('predict_proba')
                method_set = set(prediction_methods[0]).intersection(*prediction_methods[1:])
        else:
            method_set = utils.get_prediction_method_names(predictors)
            
        if len(method_set) == 0:
            raise TypeError('No uniform predicion interface detected.')
        if 'predict' not in method_set:
            raise TypeError('Missing predict method in predictors argument. Predict() method required for all predictors.')
        for method_name in method_set:
            prediction_method = functools.partial(self.predict_with_method, method_name=method_name)
            setattr(self, method_name, prediction_method)
    
    @staticmethod
    def _fit_job(predictor, X, y, fit_params):
        model = utils.get_clone(predictor)
        if y is None:
            predictions = model.fit_transform(X, **fit_params)
        else:
            predictions = model.fit_transform(X, y, **fit_params)
            
        return model, predictions
        
    def fit(self, Xs, y=None, **fit_params):
        
        # broadcast predictors if necessary
        is_listlike = isinstance(self.predictors, (list, tuple, np.ndarray))
        if is_listlike:
            if len(Xs) != len(self.predictors):
                raise ValueError('predictor list length does not match input list length')
            else:
                predictors = self.predictors
        else:
            predictors = [self.predictors if X is not None else None for X in Xs]
            
        # wrap predictors for transforming, remove predictors if channel is dead
        for i, predictor in enumerate(predictors):
            if Xs[i] is None:
                predictors[i] = None
                continue
            if type(predictor) in [transform_wrappers.SingleChannelCV, transform_wrappers.MultichannelCV]:
                raise TypeError('CV transform_wrapper found in predictors (disallowed to promote uniform wrapping)')
                
            predictors[i] = transform_wrappers.SingleChannelCV(predictor, internal_cv=self.cv, 
                                                               scorer=self.scorer, cv_processes=self.cv_processes)
        
        # build list of active channels for parallel processing
        active_X_indices = [i for i, X in enumerate(Xs) if X is not None]
        args_list = [(p, X, y, fit_params) for p, X in zip(predictors, Xs) if X is not None]
        
        n_processes = self.channel_processes
        if n_processes is not None and n_processes > 1:
            try:
                shared_mem_objects = [y, fit_params]
                fit_results = parallel.starmap_jobs(ModelSelector._fit_job, args_list, 
                                                    n_cpus=self.channel_processes, 
                                                    shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'.format(e))
                print('defaulting to single processor')
                n_processes = 1    
                
        if n_processes is None or n_processes <= 1:
            # print('running a single process with {} jobs'.format(len(args_list)))
            fit_results = [ModelSelector._fit_job(*args) for args in args_list]
                
        models, predictions_list = zip(*fit_results) 
        model_scores = [p.score_ for p in models]
            
        # expand active lists back into channel lists
        channel_models = [None for X in Xs]
        channel_scores = [None for X in Xs]
        for i, j in enumerate(active_X_indices):
            channel_models[j] = models[i]
            channel_scores[j] = model_scores[i]
        self.selected_indices_ = self.score_selector(channel_scores)
        
        # store only the selected models for future use
        self.models = [m if i in self.selected_indices_ else None for i, m in enumerate(channel_models)]
        
        # make predict_proba and decision_function synonymous to enable metaprediction with mixed nomenclature
        for model in self.models:
            if model is not None:
                if hasattr(model, 'predict_proba'):
                    setattr(model, 'decision_function', model.predict_proba)
                elif hasattr(model, 'decision_function'):
                    setattr(model, 'predict_proba', model.decision_function)
            
    def predict_with_method(self, Xs, method_name):
        predictions = [None for X in Xs]
        for i in self.selected_indices_:
            prediction_method = getattr(self.models[i], method_name)
            predictions[i] = prediction_method(Xs[i])

        return predictions

    def get_selected_indices(self):
        return self.selected_indices_
                
    def get_clone(self):
        clone = super().get_clone()
        if hasattr(self, 'models'):
            clone.models = [utils.get_clone(m) if m is not None else None for m in self.models]
        return clone
    
class SelectKBestModels(ModelSelector):
    
    def __init__(self, predictors, cv=5, scorer='auto', k=1, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectKBestModels.__init__, locals())
        super().__init__(predictors, cv, scorer, RankScoreSelector(k), channel_processes, cv_processes)
            