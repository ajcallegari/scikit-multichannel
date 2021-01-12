import numpy as np
import ray
import scipy.stats
import functools

from sklearn.metrics import explained_variance_score, balanced_accuracy_score

import pipecaster.utils as utils
from pipecaster.utils import Cloneable, Saveable
import pipecaster.transform_wrappers as transform_wrappers
from pipecaster.score_selection import RankScoreSelector

__all__ = ['SoftVotingClassifier', 'HardVotingClassifier',
           'AggregatingRegressor', 'SelectivePredictorStack',
           'MultichannelPredictor']


class SoftVotingClassifier(Cloneable, Saveable):
    """
    Meta-classifier that combines inferences from multiple base classifiers by
    averaging their predictions.

    Notes
    -----
    This class operates on a single feature matrix produced by the
    concatenation of mutliple predictions, i.e. a meta-feature matrix.
    The predicted classes are inferred from the order of the meta-feature
    matrix columns.
    """
    state_variables = ['classes_']

    def __init__(self):
        self._estimator_type = 'classifier'

    def fit(self, X, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output \
                                      meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def _decatenate(self, meta_X):
        n_classes = len(self.classes_)
        if meta_X.shape[1] % n_classes != 0:
            raise ValueError(
                '''Number of meta-features not divisible by number of classes.
                This can happen if base classifiers were trained on different
                subsamples with different number of classes.  Pipecaster uses
                StratifiedKFold to prevent this, but GroupKFold can lead to
                violations.  Someone need to make StratifiedGroupKFold''')
        Xs = [meta_X[:, i:i+n_classes]
              for i in range(0, meta_X.shape[1], n_classes)]
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
    This class operates on a single feature matrix produced by the
    concatenation of mutliple predictions, i.e. a meta-feature matrix.  The
    predicted classes are inferred from the order of the meta-feature
    matrix columns.

    This implementation of hard voting also adds a predict_proba function to
    be used in the event that hard outputs are needed for additional model
    stacking.  Predict_proba() outputs the fraction of the input classifiers
    that picked the class - shape (n_samples, n_classes).
    """
    state_variables = ['classes_']

    def __init__(self):
        self._estimator_type = 'classifier'

    def fit(self, X, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output \
                                      meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def _decatenate(self, meta_X):
        n_classes = len(self.classes_)
        if meta_X.shape[1] % n_classes != 0:
            raise ValueError(
                '''Number of meta-features not divisible by number of classes.
                This can happen if base classifiers were trained on different
                subsamples with different number of classes.  Pipecaster uses
                StratifiedKFold to prevent this, but GroupKFold can lead to
                violations.  Someone need to make StratifiedGroupKFold''')
        Xs = [meta_X[:, i:i+n_classes]
              for i in range(0, meta_X.shape[1], n_classes)]
        return Xs

    def predict(self, X):
        """
        Return the modal class predicted by the base classifiers.
        """
        Xs = self._decatenate(X)
        input_decisions = np.stack([np.argmax(X, axis=1) for X in Xs])
        decisions = scipy.stats.mode(input_decisions, axis=0)[0][0]
        predictions = self.classes_[decisions]
        return predictions

    def predict_proba(self, X):
        """
        Return the fraction of the base classifiers that picked the class -
            shape (n_samples, n_classes)
        """
        Xs = self._decatenate(X)
        input_predictions = [np.argmax(X, axis=1).reshape(-1, 1) for X in Xs]
        input_predictions = np.concatenate(input_predictions, axis=1)
        n_samples, n_votes = input_predictions.shape
        n_classes = len(self.classes_)
        class_counts = [np.bincount(input_predictions[i, :],
                                    minlength=n_classes)
                        for i in range(n_samples)]
        class_counts = np.stack(class_counts).astype(float)
        class_counts /= n_votes

        return class_counts


class AggregatingRegressor(Cloneable, Saveable):
    """
    A meta-regressor that uses an aggregator function to combine inferences
    from multiple base regressors.

    Notes
    -----
    Currently supports only supports single output regressors.
    """
    state_variables = []

    def __init__(self, aggregator=np.mean):
        self._params_to_attributes(AggregatingRegressor.__init__, locals())
        self._estimator_type = 'regressor'

    def fit(self, X=None, y=None, **fit_params):
        return self

    def predict(self, X):
        return self.aggregator(X, axis=1)

    def transform(self, X):
        return self.aggregator(X, axis=1).reshape(-1, 1)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class SelectivePredictorStack(Cloneable, Saveable):
    """
    A scikit-learn corformant single channel ensemble predictor stack that
    can select base predictors based on performance in an internal cross
    validation test.

    Parameters
    ---------
    base_predictors: list of predictors, default=None
        Ensemble of scikit-learn conformant base predictors (either all
        classifiers or all regressors).
    meta_predictor: predictor, default=None
        Sklearn conformant classifier or regresor that makes predictions from
        the inference of the base predictors.
    internal_cv: int, None, sklearn cross-validation generator, default=5
        Method for estimating performance of base classifiers and ensuring that
        they not generate predictions from their training samples during
        training of the meta-predictor.
        If int i: If classifier, StratifiedKfold(n_splits=i), if regressor
            KFold(n_splits=i) for
        If None: default value of 5
        If not int or None: Will be treated as a scikit-learn
            conformant split generator like KFold
    scorer : callable or 'auto', default='auto'
        Figure of merit score used for selecting models during internal cross
        validation.
        If a callable: the object should have the signature
            'scorer(y_true, y_pred)' and return a scalar figure of merit.
        If 'auto': balanced_accuracy_score for classifiers or
            explained_variance_score for regressors
    score_selector: callable, default=RankScoreSelector(k=3)
        A callable with the pattern: selected_indices = callable(scores)
    base_transform_method: string or None, default=None
        Set the prediction method name used to generate meta-features from
        base predictors.  If None, the precedence order specified in
        transform_wrappers will be used to automatically pick a method.
    base_processes: int or 'max', default=1
        The number of parallel processes to run for base predictor fitting.
    cv_processes: int or 'max', default=1
        The number of parallel processes to run for internal cross validation.

    notes:
    Compatible with both sklearn and pipecaster
    """
    state_variables = ['classes_', 'scores_', 'selected_indices_']

    def __init__(self, base_predictors=None, meta_predictor=None,
                 internal_cv=5, scorer='auto',
                 score_selector=RankScoreSelector(k=3),
                 base_transform_method=None,
                 base_processes=1, cv_processes=1):
        self._params_to_attributes(SelectivePredictorStack.__init__, locals())
        estimator_types = [p._estimator_type for p in base_predictors]
        if len(set(estimator_types)) != 1:
            raise TypeError('base_predictors must be of uniform type \
                            (e.g. all classifiers or all regressors)')
        self._estimator_type = estimator_types[0]
        if scorer == 'auto':
            if utils.is_classifier(self):
                self.scorer = balanced_accuracy_score
            elif utils.is_regressor(self):
                self.scorer = explained_variance_score
            else:
                raise AttributeError('predictor type required for automatic \
                                     assignment of scoring metric')
        self._expose_predictor_interface(meta_predictor)

    def _expose_predictor_interface(self, meta_predictor):
        for method_name in utils.recognized_pred_methods:
            if hasattr(meta_predictor, method_name):
                prediction_method = functools.partial(self.predict_with_method,
                                                      method_name=method_name)
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

        self.base_predictors = [transform_wrappers.SingleChannelCV(
                                    p, self.base_transform_method,
                                    self.internal_cv, self.cv_processes,
                                    self.scorer)
                                for p in self.base_predictors]

        args_list = [(p, X, y, fit_params) for p in self.base_predictors]
        n_jobs = len(args_list)
        n_processes = 1 if self.base_processes is None else self.base_processes
        if (type(n_processes) == int and n_jobs < n_processes):
            n_processes = n_jobs
        if n_processes == 'max' or n_processes > 1:
            try:
                shared_mem_objects = [X, y, fit_params]
                fit_results = parallel.starmap_jobs(
                                SelectivePredictorStack._fit_job, args_list,
                                n_cpus=n_processes,
                                shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'
                      .format(e))
                print('defaulting to single processor')
                n_processes = 1
        if n_processes is None or n_processes <= 1:
            fit_results = [SelectivePredictorStack._fit_job(*args)
                           for args in args_list]

        self.base_models, predictions_list = zip(*fit_results)
        if self.scorer is not None and self.score_selector is not None:
            base_model_scores = [p.score_ for p in self.base_models]
            selected_indices = self.score_selector(base_model_scores)
            self.base_models = [m for i, m in enumerate(self.base_models)
                                if i in selected_indices]
            predictions_list = [p for i, p in enumerate(predictions_list)
                                if i in selected_indices]
        self.scores_ = base_model_scores
        self.selected_indices_ = selected_indices
        meta_X = np.concatenate(predictions_list, axis=1)
        self.meta_model = utils.get_clone(self.meta_predictor)
        self.meta_model.fit(meta_X, y, **fit_params)
        if hasattr(self.meta_model, 'classes_'):
            self.classes = self.meta_model.classes_

        return self

    def get_model_scores(self):
        if hasattr(self, 'scores_'):
            return self.scores_
        else:
            raise utils.FitError('Base model scores not found. They are only \
                                 available after call to fit().')

    def get_support(self):
        if hasattr(self, 'selected_indices_'):
            return self.selected_indices_
        else:
            raise utils.FitError('Must call fit before getting selection \
                                 information')

    def get_base_models(self, unwrap=True):
        if hasattr(self, 'base_models') is False:
            raise FitError('No base models found. Call fit() or \
                           fit_transform() to create base models')
        else:
            return [transform_wrappers.unwrap_model(m)
                    for m in self.base_models]

    def predict_with_method(self, X, method_name):
        """
        Make inferences by calling the indicated method on the meta-predictor
        after creating meta-features.

        Parameters
        ----------
        X: ndarray.shape(n_samples, n_features)
        method_name: String

        Returns
        -------
        Meta-predictions.
        if method_name is 'predict': ndarray(n_samples,)
        if method_name is 'predict_proba', 'decision_function', or
            'predict_log_proba':
            ndarray(n_sample, n_classes)
        """
        if (hasattr(self, 'base_models') is False or
                hasattr(self, 'meta_model') is False):
            raise utils.FitError('prediction attempted before model fitting')
        predictions_list = [p.transform(X) for p in self.base_models]
        meta_X = np.concatenate(predictions_list, axis=1)
        prediction_method = getattr(self.meta_model, method_name)
        predictions = prediction_method(meta_X)
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
        return clone


class MultichannelPredictor(Cloneable, Saveable):
    """
    Predictor or meta-predictor that takes matrices from multiple input
    channels, concatenates them to create a single input feature matrix, and
    outputs a single prediction list into the first channel.

    Parameters
    ----------
    predictor: scikit-learn conformant classifier or regressor

    Notes
    -----
    Class uses reflection to expose its prediction interface durining
    initializtion, so class methods will typically not be identical to
    class instance methods.
    """
    state_variables = ['classes_']

    def __init__(self, predictor=None):
        self._params_to_attributes(MultichannelPredictor.__init__, locals())
        utils.enforce_fit(predictor)
        utils.enforce_predict(predictor)
        self._estimator_type = utils.detect_predictor_type(predictor)
        if self._estimator_type is None:
            raise TypeError('could not detect predictor type')
        self._expose_predictor_interface(predictor)

    def _expose_predictor_interface(self, predictor):
        for method_name in utils.recognized_pred_methods:
            if hasattr(predictor, method_name):
                prediction_method = functools.partial(
                                        self.predict_with_method,
                                        method_name=method_name)
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
        if hasattr(self, 'model') is False:
            raise FitError('prediction attempted before call to fit()')
        live_Xs = [X for X in Xs if X is not None]

        if len(live_Xs) > 0:
            X = np.concatenate(live_Xs, axis=1)
            prediction_method = getattr(self.model, method_name)
            predictions = prediction_method(X)

        return predictions

    def _more_tags(self):
        return {'multichannel': True}

    def get_clone(self):
        clone = super().get_clone()
        if hasattr(self, 'model'):
            clone.model = utils.get_clone(self.model)
        return clone
