import numpy as np
import ray
import scipy.stats
import functools

from sklearn.metrics import explained_variance_score

import pipecaster.utils as utils
from pipecaster.utils import Cloneable, Saveable
import pipecaster.transform_wrappers as transform_wrappers
from pipecaster.score_selection import RankScoreSelector

__all__ = ['SoftVotingClassifier', 'HardVotingClassifier', 'AggregatingRegressor', 
           'SelectivePredictorStack', 'ChannelConcatenator', 'MultichannelPredictor']

class SoftVotingClassifier(Cloneable, Saveable):
    """
    Meta-classifier that combines inferences from multiple base classifiers by averaging their predictions.
    
    Notes
    -----
    This class operates on a single feature matrix produced by the concatenation of mutliple predictions,
    i.e. a meta-feature matrix.  The predicted classes are inferred from the order of the meta-feature 
    matrix columns.  
    """
    
    state_variables = ['classes_']
    
    def __init__(self):
        self._estimator_type = 'classifier'
    
    def fit(self, X, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        return self
        
    def _decatenate(self, X):
        n_classes = len(self.classes_)
        if X.shape[1] % n_classes != 0:
            raise ValueError('''Number of meta-features not divisible by number of classes. This can happen if 
                                base classifiers were trained on different subsamples with different number of 
                                classes.  Pipecaster uses StratifiedKFold to prevent this, but GroupKFold can lead
                                to violations.  Someone need to make StratifiedGroupKFold''')
        Xs = [X[:, i:i+n_classes] for i in range(n_classes)]
        return Xs
    
    def predict_proba(self, X):
        Xs = self._decatenate(X)
        return np.mean(Xs, axis=0)
                             
    def predict(self, X):
        mean_probs = self.predict_proba(X)
        decisions = np.argmax(mean_probs, axis=1)
        predictions = self.classes_[decisions]
        return predictions    
    
class HardVotingClassifier(Cloneable, Saveable):
    """
    Meta-classifier that combines inferences from multiple base classifiers by 
       outputting the most frequently predicted class (i.e. the modal class).
    
    Notes
    -----
    This class operates on a single feature matrix produced by the concatenation of mutliple predictions,
    i.e. a meta-feature matrix.  The predicted classes are inferred from the order of the meta-feature 
    matrix columns.
    
    This implementation of hard voting also adds a predict_proba function to be used in the event that hard outputs are 
    needed for additional stacking.  Predict_proba() outputs the fraction of the input classifiers that picked the class 
    - shape (n_samples, n_classes).
    
    """    
    state_variables = ['classes_']
    
    def __init__(self):
        self._estimator_type = 'classifier'
    
    def fit(self, X, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        return self
        
    def _decatenate(self, meta_X):
        n_classes = len(self.classes_)
        if X.shape[1] % n_classes != 0:
            raise ValueError('''Number of meta-features not divisible by number of classes. This can happen if 
                                base classifiers were trained on different subsamples with different number of 
                                classes.  Pipecaster uses StratifiedKFold to prevent this, but GroupKFold can lead
                                to violations.  Someone need to make StratifiedGroupKFold''')
        Xs = [meta_X[:, i : i + n_classes] for i in range(n_classes)]
        return Xs
    
    def predict(self, X):
        Xs = self._decatenate(X)
        input_decisions = np.stack([np.argmax(X, axis=1) for X in Xs])
        decisions = scipy.stats.mode(input_decisions, axis = 0)[0][0]
        predictions = self.classes_[decisions]
        return predictions
    
    def predict_proba(self, X):
        Xs = self._decatenate(X)
        input_predictions = [np.argmax(X, axis=1).reshape(-1,1) for X in Xs]
        input_predictions = np.concatenate(input_predictions, axis=1)
        n_samples = input_predictions.shape[0]
        n_classes = len(self.classes_)
        class_counts = [np.bincount(input_predictions[i,:], minlength=n_classes) for i in range(n_samples)]
        class_counts = np.stack(class_counts)
        class_counts /= len(live_Xs)
        return class_counts

class AggregatingRegressor(Cloneable, Saveable):
    """
    A meta-regressor that uses an aggregator function to combine inferences from multiple base regressors.
    
    notes
    -----
    Currently supports only supports single output regressors.
    """
    
    state_variables = []
    
    def __init__(self, aggregator=np.mean):
        self._init_params(locals())
        self._estimator_type = 'regressor'
    
    def fit(self, X=None, y=None, **fit_params):
        return self
    
    def predict(self, X):
        return self.aggregator(X, axis=1)
    
    def transform(self, X):
        return self.aggregator(X, axis=1).reshape(-1,1)
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

class SelectivePredictorStack(Cloneable, Saveable):
    
    """
    A single-input channel ensemble predictor that selects base predictors based on internal cross validation performance. 
    
    Parameters
    ---------
    base_predictors: iterable, default=None
        Ensemble of sklearn conformant classifiers or regressors that generate inferences treated as meta-features.
        These predictors are automatically wrapped with the transform_wrappers.SingleChannelCV class to provide 
        cross validation training and transformer functionality during calls to fit_transform().
    meta_predictor: predictor, default=None
        Sklearn conformant classifier or regresor that makes predictions from meta-features.
    internal_cv: None, int, sklearn cross-validation generator, default=5
        Method for ensuring that meta-predictions are not generated from training data.
        Note - internal_cv is generally a good idea to prevent overfitting, but in some instances the advantage
            conferred by reducing overfitting can be smaller than the disadvantage of starving the
            base classifiers for limited data.  In these instances, internal cv can be inactivated by setting 
            this argument to None or 1.
    scorer : callable, default=explained_variance_score
        a scorer callable object / function with signature
        'scorer(y_true, y_pred)' which should return only
        a single value.    
    base_processes: int or 'max', default=1
        The number of parallel processes to run for base predictor fitting. 
    cv_processes: int or 'max', default=1
        The number of parallel processes to run for internal cross validation.
    
    notes:
    Compatible with both sklearn and pipecaster
    Predictor cloning occurs at fit time.
    
    """
    state_variables = ['classes_']
    
    def __init__(self, base_predictors=None, meta_predictor=None, internal_cv=5, 
                 scorer=explained_variance_score, score_selector=RankScoreSelector(k=3), 
                 base_processes=1, cv_processses=1):
        self._init_params(locals())
                             
        estimator_types = [p._estimator_type for p in base_predictors]
        if len(set(estimator_types)) != 1:
            raise TypeError('base_predictors must be of uniform type (e.g. all classifiers or all regressors)')
        if meta_predictor._estimator_type != estimator_types[0]:
            raise TypeError('meta_predictor must be same type as base_predictors (e.g. classifier or regressor)')
        self._estimator_type = estimator_types[0]
        self._expose_predictor_interface(meta_predictor)
            
    def _expose_predictor_interface(self, meta_predictor):
        for method_name in utils.recognized_pred_methods:
            if hasattr(meta_predictor, method_name):
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
        
    def fit(self, X, y=None, **fit_params):
        
        if self._estimator_type == 'classifier' and y is not None:
            self.classes_, y = np.unique(y, return_inverse=True)
        
        self.base_predictors = [transform_wrappers.SingleChannelCV(p, internal_cv=self.internal_cv, scorer=self.scorer, 
                                                                   cv_processes=self.cv_processes)
                                for p in self.base_predictors]
            
        args_list = [(p, X, y, fit_params) for p in self.base_predictors]
        
        n_processes = self.base_processes
        if n_processes is not None and n_processes > 1:
            try:
                shared_mem_objects = [X, y, fit_params]
                fit_results = parallel.starmap_jobs(PredictorStack._fit_job, args_list, n_cpus=n_processes, 
                                                    shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'.format(e))
                print('defaulting to single processor')
                n_processes = 1       
        if n_processes is None or n_processes <= 1:
            # print('running a single process with {} jobs'.format(len(args_list)))
            fit_results = [PredictorStack._fit_job(**args) for args in args_list]
                
        self.base_models, predictions_list = zip(*fit_results)
        if self.scorer is not None and self.score_selector is not None: 
            base_model_scores = [p.score_ for p in self.base_models]
            selected_indices = self.score_selector(base_model_scores)
            self.base_models = [m for i, m in enumerate(self.base_models) if i in selected_indices]
            predictions_list = [p for i, p in enumerate(predictions_list) if i in selected_indices]
        meta_X = np.concatenate(predictions_list, axis=1)
        self.meta_model = utils.get_clone(self.meta_predictor)
        self.meta_model.fit(meta_X, y, **fit_params)
        if hasattr(self.meta_model, 'classes_'):
            self.classes = self.meta_model.classes_
            
        return self
    
    def predict_with_method(self, X, method_name):
        """
        Make inferences by calling the indicated method on the metaclassifier after creating meta-features
        
        Parameters
        ----------
        X: ndarray.shape(n_samples, n_features)
        method_name: str
        
        Returns
        -------
        ndarray(n_samples,) for regressions or classifiers called with "predict"
        ndarray(n_sample, n_classes) for classifiers called with predict_proba, decision_function, or predict_log_proba
        """
        if hasattr(self, 'base_models') == False or hasattr(self, 'meta_model') == False: 
            raise utils.FitError('prediction attempted before model fitting')
        predictions_list = [p.transform(X) for p in self.base_models]
        meta_X = np.concatenate(predictions_list, axis=1)
        prediction_method = getattr(self.meta_model, method_name)
        predictions = self.prediction_method(meta_X)
        if self._estimator_type == 'classifier' and method_name == 'predict':
            predictions = self.classes_[predictions]
        return predictions
                
    def _more_tags(self):
        return {'multichannel': False}
    
    def get_clone(self):
        clone = super().get_clone()
        if hasattr(self, 'base_models'):
            clone.base_models = [utils.get_clone(m) for m in self.base_models]
        if hasattr(self, 'meta_model'):
            clone.meta_model = utils.get_clone(self.meta_model)
    
class ChannelConcatenator(Cloneable, Saveable):
    
    def fit(self, Xs, y=None, **fit_params):
        pass
    
    def transform(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        Xs_t = [None for X in Xs]
        Xs_t[0] = np.concatenate(live_Xs, axis=1) if len(live_Xs) > 0 else None
        return Xs_t
    
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)

class MultichannelPredictor(Cloneable, Saveable):
    
    """Predictor (meta-predictor) that takes matrices from multiple input channels, concatenates them to 
       create a single feature matrix, and outputs a single prediction into the first channel. 
    
    Parameters
    ----------
    classifier: str, or object instance with the sklearn interface, default='soft vote'
        Determines the metapredicition method: 'soft voting', 'hard voting', or stacked generalization using 
        the specified classifier 
    
    Notes
    -----
    Supports multiclass but not multilabel or multioutput classification.
    With single channel predictors, you need to wrap them to make them into transformers due to sklearn
    interface limitations.  Since multichannel functionality is native to pipecaster, I opted to
    skip the wrapper step and make all predictors also transformers by default.
    
    """
    state_variables = ['classes_']
    
    def __init__(self, predictor=None):
        self._init_params(locals())
        utils.enforce_fit(predictor)
        utils.enforce_predict(predictor)
        self._esimtator_type = utils.detect_estimator_type(predictor)
        if self._esimtator_type is None:
            raise TypeError('could not detect predictor type')
        self._expose_predictor_interface(predictor)
            
    def _expose_predictor_interface(self, predictor):
        for method_name in utils.recognized_pred_methods:
            if hasattr(predictor, method_name):
                prediction_method = functools.partial(self.predict_with_method, method_name=method_name)
                setattr(self, method_name, prediction_method)
                
    def fit(self, Xs, y=None, **fit_params):
        self.model = utils.get_clone(self.predictor)
        live_Xs = [X for X in Xs if X is not None]
        
        if len(Xs) > 0:
            X = np.concatenate(live_Xs, axis=1)
            if y is None:
                self.model.fit(X, **fit_params)
            else:
                self.model.fit(X, y, **fit_params)
            if hasattr(self.model, 'classes_'):
                self.classes_ = self.model.classes_
        return self
                
    def predict_with_method(self, Xs, method_name):
        if hasattr(self, 'model') == False:
            raise FitError('prediction attempted before call to fit()')
        live_Xs = [X for X in Xs if X is not None]
        channel_predictions = [None for X in Xs]
        
        if len(live_Xs) > 0:
            X = np.concatenate(live_Xs, axis=1)
            prediction_method = getattr(self.model, method_name)
            channel_predictions[0] = prediction_method(X)
        
        return channel_predictions
                           
    def _more_tags(self):
        return {'multichannel':True}
    
    def get_clone(self):
        clone = super().get_clone()
        if hasattr(self, 'model'):
            clone.model = utils.get_clone(self.model)
        return clone