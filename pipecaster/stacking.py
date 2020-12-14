import ray
import scipy.stats

from sklearn.metrics import explained_variance_score

import pipecaster.utils as utils
from pipecaster.utils import Parameterized, Saveable
from pipecaster.channel_metaprediction import TransformingPredictor
from pipecaster.score_selection import RankScoreSelector

__all__ = ['SoftVotingClassifier']

class SoftVotingClassifier(Cloneable, Saveable):
    
    state_variables = ['classes_', '_estimator_type']
    
    def __init__(self):
        self._estimator_type = 'classifier'
    
    def fit(self, Xs, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        
    def _decatenate(self.meta_X):
        n_classes = len(self.classes_)
        Xs = [meta_X[:, i:i+n_classes] for i in range(n_classes)]
        return Xs
    
    def predict(self, X):
        Xs = self._decatenate(X)
        mean_probs = np.mean(Xs, axis=0)
        decisions = np.argmax(mean_probs, axis=1)
        predictions = self.classes_[decisions]
        return predictions
    
    def transform(self, X):
        Xs = self._decatenate(X)
        return np.mean(Xs, axis=0)
    
    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
class HardVotingClassifier(Cloneable, Saveable):
    
    state_variables = ['classes_', '_estimator_type']
    
    def __init__(self):
        self._estimator_type = 'classifier'
    
    def fit(self, X, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        
    def _decatenate(self, meta_X):
        n_classes = len(self.classes_)
        Xs = [meta_X[:, i:i+n_classes] for i in range(n_classes)]
        return Xs
    
    def predict(self, X):
        Xs = self._decatenate(X)
        input_decisions = np.stack([np.argmax(X, axis=1) for X in Xs])
        decisions = scipy.stats.mode(input_decisions, axis = 0)[0][0]
        predictions = self.classes_[decisions]
        return predictions
    
    def transform(self, X):
        Xs = self._decatenate(X)
        input_predictions = [np.argmax(X, axis=1).reshape(-1,1) for X in Xs]
        input_predictions = np.concatenate(input_predictions, axis=1)
        n_samples = input_predictions.shape[0]
        n_classes = len(self.classes_)
        class_counts = [np.bincount(input_predictions[i,:], minlength=n_classes) for i in range(n_samples)]
        class_counts = np.stack(class_counts)
        class_counts /= len(live_Xs)
        
        return class_counts
    
    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
class AggregatingRegressor(Cloneable, Saveable):
    
    state_variables = ['classes_', '_estimator_type']
    
    def __init__(self, aggregator=np.mean):
        self._init_params(locals())
        self._estimator_type = 'regressor'
    
    def fit(self, X=None, y=None, **fit_params):
        pass
    
    def predict(self, X):
        return self.aggregator(X, axis=1)
    
    def transform(self, X):
        return self.aggregator(X, axis=1).reshape(-1,1)
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

class PredictorStack(Cloneable, Saveable):
    
    """
    
    notes:
    cloning occurs at fit time
    
    """
    
    def __init__(self, base_predictors=None, meta_predictor=None, internal_cv=5, base_fit_jobs=1, cv_jobs=1):
        self._init_params(locals())
        
    @staticmethod
    def _fit_job(predictor, X, y, **fit_params):
        if y is None:
            predictions = predictor.fit_transform(X, **fit_params))
            predictor.fit(X, **fit_params)
        else:
            predictions = predictor.fit_transform(X, y, **fit_params))
            predictor.fit(X, y, **fit_params)
            
        return predictor, predictions
    
    @ray.remote
    def _ray_fit_job(predictor, X, y, **fit_params):
        return _fit_job(predictor, X, y, **fit_params)
        
    def fit(self, X, y=None, **fit_params):
        estimator_types = [p._estimator_type for p in self.base_predictors]
        if len(estimator_types) != len(set(estimator_types)):
            raise TypeError('base_predictors must be of uniform type (e.g. all classifiers or all regressors)')
        if estimator_types[0] != meta_predictor._estimator_type:
            raise TypeError('meta_predictor must be same type as base_predictors (e.g. classifier or regressor)')
        self._estimator_type = estimator_types[0]
        self.base_predictors = [utils.get_clone(p) if type(p) == TransformingPredictor else 
                                TransformingPredictor(p, internal_cv=self.internal_cv, cv_jobs=self.cv_jobs)]
        
        if self.base_fit_jobs == 1:
            base_fit_results = [_fit_job(bp, X, y, **fit_params) for bp in self.base_predictors]
            predictor, predictions = zip(*base_fit_results)
        else:
            
   

        meta_X = np.concatenate(predictions_list, axis=1)
        self.meta_predictor = utils.get_clone(meta_predictor)
        self.meta_predictor.fit(meta_X, y, **fit_params)
        
    def predict(self, X):
        predictions_list = [p.transform(X) for p in self.base_predictors]
        meta_X = np.concatenate(predictions_list, axis=1)
        
        return self.meta_predictor.predict(meta_X)
        
    def transform(self, X):
        predictions_list = [p.transform(X) for p in self.base_predictors]
        meta_X = np.concatenate(predictions_list, axis=1)
        
        return self.meta_predictor.transform(meta_X)
        
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.tranform(X)
        
    def _more_tags(self):
        return {'multichannel': False}
    
class SelectiveStack(PredictorStack):
    
    def __init__(base_predictors=None, meta_predictor=None, internal_cv=5, scorer=explained_variance_score,
                 score_selector=RankScoreSelector(k=2), base_jobs=1, cv_jobs=1):
        
        self._init_params(locals())
        super().__init__(base_predictors=None, meta_predictor=None, internal_cv=5, base_jobs=1, cv_jobs=1)

class MultiChannelPredictor(Cloneable, Saveable):
    
    """Predictor and meta-predictor that takes multiple input matrices and generates a single prediction. 
    
    Parameters
    ----------
    classifier: str, or object instance with the sklearn interface, default='soft vote'
        Determines the metapredicition method: 'soft voting', 'hard voting', or stacked generalization using 
        the specified classifier 
    
    Notes
    -----
    Takes multiple inputs channels and ouputs to the channel with the lowest index number.
    Supports multiclass but not multilabel or multioutput classification
    Adds tranform functionality to both voting and stacking metaclassifiers to enable multilevel metaclassification.
        For soft voting, the transform is the mean probabilities of the multiple inputs
        For hard voting, the transform is the fraction of the input classifiers that picked the class - shape (n_samples, n_classes)
        For stacking, the transform is the predict_proba() method of the meta-clf algorithm if available otherwise the decision_function()
    
    """
    
    state_variables = ['classes_', '_estimator_type']
    
    def __init__(self, predictor=None):
        self._init_params(locals())
        self._esimtator_type = utils.detect_estimator_type(predictor)
        if self._esimtator_type == None:
            raise TypeError('could not detect estimator type')
        if hasattr(predictor, 'fit') == False:
            raise AttributeError('predictor missing fit() method')
            
        if hasattr(predictor, 'predict_proba'):
            setattr(self, 'predict_proba', self._predict_proba)
        if hasattr(predictor, 'decision_function'):
            setattr(self, 'decision_function', self._decision_function)
        if hasattr(predictor, 'transform'):
            setattr(self, 'transform', self._transform)
        if hasattr(predictor, 'fit_transform'):
            setattr(self, 'fit_transform', self._decision_function) 
                
    def fit(self, Xs, y=None, **fit_params):
        
        live_Xs = [X for X in Xs if X is not None]
        if len(Xs) > 0:
            meta_X = np.concatenate(live_Xs, axis=1)
            if y is None:
                self.predictor.fit(meta_X, **fit_params)
            else:
                self.predictor.fit(meta_X, y, **fit_params)
                
    def predict(self, Xs):
        predict_method = utils.get_predict_method(self.predictor)
        if predict_method == None:
            raise AttributeError('missing predict method')   
            
        live_Xs = [X for X in Xs if X is not None]
        channel_predictions = [None for X in Xs]
        if len(live_Xs) > 0:
            meta_X = np.concatenate(live_Xs, axis=1)
            channel_predictions[0] = predict_method(meta_X)
        
        return channel_predictions

    def _predict_proba(self, Xs):
        
        live_Xs = [X for X in Xs if X is not None]
        channel_predictions = [None for X in Xs]
        if len(live_Xs) > 0:
            meta_X = np.concatenate(live_Xs, axis=1)
            channel_predictions[0] = self.predictor.pred_proba(meta_X)
            
        return channel_predictions
                
    def _decision_function(self, Xs):
        
        live_Xs = [X for X in Xs if X is not None]
        channel_predictions = [None for X in Xs]
        if len(live_Xs) > 0:
            meta_X = np.concatenate(live_Xs, axis=1)
            channel_predictions[0] = self.predictor.decision_function(meta_X)
            
        return channel_predictions
    
    def _transform(self, Xs):
        
        live_Xs = [X for X in Xs if X is not None]
        Xs_t = [None for X in Xs]
        if len(live_Xs) > 0:
            meta_X = np.concatenate(live_Xs, axis=1)
            Xs_t[0] = self.predictor.trasform(meta_X)
            return Xs_t
        
        return Xs_t
    
    def _fit_transform(self, Xs, y=None, **fit_params):
        
        live_Xs = [X for X in Xs if X is not None]
        Xs_t = [None for X in Xs]
        if len(live_Xs) > 0:
            meta_X = np.concatenate(live_Xs, axis=1)
            Xs_t[0] = self.predictor.fit_transform(meta_X)
            return Xs_t
    
        return Xs_t
                           
    def _more_tags(self):
        return {'multichannel':True}