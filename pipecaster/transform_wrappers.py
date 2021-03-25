"""
Wrapper classes for internal ML models.

MultichannelPipelines treat all internal component as transfomers (i.e.
invoking fit/transform/fit_transform).  As a consequence, when predictors are
used internally (e.g. for voting or stacking) a transformer interface must be
added to the internal predictors.  In practice, this means choosing a
prediction method to use when transforming, converting 1D outputs to 2D
outputs, and applying internal cross validation training when required.

:class:`SingleChannel` and :class:`Multichannel` classes add a transformer
interface to single channel and multichannel predictors respectively.

:class:`SingleChannelCV` and :class:`MultichannelCV` classes add a transformer
interface and internal cross validaiton training to single channel and
multichannel predictors respectively.  Internal cross validation (internal cv)
training is typically used when outputs of a base predictor will be used to
train a meta-predictor.  It guarantees that base predictors do not make
inferences on their own training samples (1).  Internal cv training can improve
meta-predictor accuracy if overfitting is a limiting problem, or it can reduce
metapredictor accuracy if the number of training samples is limiting.

(1) Wolpert, David H. "Stacked generalization." Neural networks 5.2
    (1992): 241-259.
"""

import functools
import numpy as np
from sklearn.metrics import log_loss

import pipecaster.utils as utils
import pipecaster.config as config
from pipecaster.utils import Cloneable, Saveable
from pipecaster.cross_validation import cross_val_predict, score_predictions

__all__ = ['make_transformer', 'make_cv_transformer', 'unwrap_predictor',
           'unwrap_model']

def make_transformer(predictor, transform_method='auto'):
    """
    Add transform methods to a predictor.

    Parameters
    ----------
    predictor : scikit-learn predictor or multichannel predictor
        Predictor to wrap.
    transform_method : str, default='auto'
        - Name of the prediction method to call when transforming (e.g. when
          outputting meta-features).
        - If 'auto' :
            - If classifier : method picked using
              config.transform_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'

    Returns
    -------
    Predictor/transformer
        A wrapped predictor with both predictor and transformer interfaces.

    Examples
    --------
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=2)
        clf = pc.MultichannelPipeline(n_channels=5)
        base_clf = GradientBoostingRegressor()
        base_clf = pc.make_transformer(base_clf)
        clf.add_layer(base_clf)
        clf.add_layer(pc.SoftVotingClassifier())
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.8529411764705882, 0.9411764705882353, 0.96875]
    """
    if utils.is_multichannel(predictor):
        return Multichannel(predictor, transform_method)
    else:
        return SingleChannel(predictor, transform_method)


def make_cv_transformer(predictor, transform_method='auto', internal_cv=5,
             score_method='auto', scorer='auto', cv_processes=1):
    """
    Add internal cross validation training and transform methods to a
    predictor.

    Parameters
    ----------
    predictor : scikit-learn predictor or multichannel predictor
        Predictor to wrap.
    transform_method : str, default='auto'
        - Name of the prediction method to call when transforming (e.g. when
          outputting meta-features).
        - If 'auto' :
            - If classifier : method picked using
              config.transform_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'
    internal_cv : int, None, or callable, default=5
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int > 1: StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If {None, 1}: disable internal cv.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    score_method : str, default='auto'
        - Name of prediction method used when scoring predictor performance.
        - If 'auto' :
            - If classifier : method picked using
              config.score_method_precedence order (default:
              ppredict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'
    scorer : callable, default='auto'
        Callable that computes a figure of merit score for the internal_cv run.
        The score is exposed as score_ attribute during fit_transform().
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer with signature: score = scorer(y_true, y_pred).
    cv_processes : int or 'max', default=1
        - The number of parallel processes to run for internal cross
          validation.
        - If int : Use up to cv_processes number of processes.
        - If 'max' : Use all available CPUs.

    Returns
    -------
    Predictor/transformer
        A wrapped predictor with both predictor and transformer interfaces.
        Internal cross_validation training occurs during calls to
        fit_transform().

    Examples
    --------
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=2)
        clf = pc.MultichannelPipeline(n_channels=5)
        base_clf = GradientBoostingRegressor()
        base_clf = pc.make_cv_transformer(base_clf)
        clf.add_layer(base_clf)
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.8529411764705882, 0.9080882352941176, 1.0]
    """
    if utils.is_multichannel(predictor):
        return MultichannelCV(predictor, transform_method, internal_cv,
                              score_method, scorer, cv_processes)
    else:
        return SingleChannelCV(predictor, transform_method, internal_cv,
                              score_method, scorer, cv_processes)


class SingleChannel(Cloneable, Saveable):
    """
    Add transformer interface to a scikit-learn predictor.

    Wrapper class that provides scikit-learn conformant predictors with
    transform() and fit_transform methods().

    Parameters
    ----------
    predictor : predictor instance
        The scikit-learn conformant estimator/predictor to wrap.
    transform_method : str, default='auto'
        - Name of the prediction method to call when transforming (e.g. when
          outputting meta-features).
        - If 'auto' :
            - If classifier : method picked using
              config.transform_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'

    Examples
    --------
    Model stacking, classification:
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = GradientBoostingClassifier()
        base_clf = pc.transform_wrappers.SingleChannel(base_clf)
        clf.add_layer(base_clf, pipe_processes='max')
        clf.add_layer(pc.MultichannelPredictor(SVC()))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.8529411764705882, 0.8216911764705883, 0.9099264705882353]

    Model stacking, regression:
    ::
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.svm import SVR
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_regression(n_informative_Xs=7,
                                                  n_random_Xs=3)
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = GradientBoostingRegressor()
        base_clf = pc.transform_wrappers.SingleChannel(base_clf)
        clf.add_layer(base_clf, pipe_processes=1)
        clf.add_layer(pc.MultichannelPredictor(SVR()))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.077183453, 0.067682880449, 0.07849665]

    Notes
    -----
    This class uses reflection to expose the predictor methods found in the
    object that it wraps, so the method attributes in a SingleChannel instance
    are not usually identical to the method attributes of the SingleChannel
    class.
    """

    def __init__(self, predictor, transform_method='auto'):
        self._params_to_attributes(SingleChannel.__init__, locals())
        utils.enforce_fit(predictor)
        utils.enforce_predict(predictor)
        self._add_predictor_interface(predictor)
        self._set_estimator_type(predictor)

    def _set_estimator_type(self, predictor):
        if hasattr(predictor, '_estimator_type') is True:
            self._estimator_type = predictor._estimator_type

    def _add_predictor_interface(self, predictor):
        for method_name in config.recognized_pred_methods:
            if hasattr(predictor, method_name):
                prediction_method = functools.partial(self.predict_with_method,
                                                      method_name=method_name)
                setattr(self, method_name, prediction_method)

    def _add_model_interface(self, model, X):
        detected_methods = utils.detect_predict_methods(model, X)
        for method_name in detected_methods:
            prediction_method = functools.partial(self.predict_with_method,
                                                  method_name=method_name)
            setattr(self, method_name, prediction_method)

    def _remove_predictor_interface(self):
        for method_name in config.recognized_pred_methods:
            if hasattr(self, method_name):
                delattr(self, method_name)

    def set_transform_method(self, method_name):
        self.transform_method = method_name
        return self

    def get_transform_method(self):
        if self.transform_method == 'auto':
            method_name = utils.get_transform_method(self)
            if method_name is None:
                raise NameError('model lacks a recognized method for '
                                'conversion to transformer')
        else:
            method_name = self.transform_method

        return method_name

    def fit(self, X, y=None, **fit_params):
        self.model = utils.get_clone(self.predictor)
        is_classifier = utils.is_classifier(self.predictor)
        if y is None:
            self.model.fit(X, **fit_params)
        else:
            if is_classifier:
                self.classes_, y = np.unique(y, return_inverse=True)
            self.model.fit(X, y, **fit_params)

        self._set_estimator_type(self.model)
        self._remove_predictor_interface()
        self._add_model_interface(self.model, X)

        return self

    def predict_with_method(self, X, method_name):
        if hasattr(self, 'model') is False:
            raise utils.FitError('prediction attempted before model fitting')
        if hasattr(self.model, method_name):
            predict_method = getattr(self.model, method_name)
            predictions = predict_method(X)
        else:
            raise NameError('prediction method {} not found in {} attributes'
                            .format(method_name, self.model))
        if utils.is_classifier(self) and method_name == 'predict':
            predictions = self.classes_[predictions]

        return predictions

    def transform(self, X):
        if hasattr(self, 'model'):
            transformer = getattr(self.model, self.get_transform_method())
            X_t = transformer(X)
            # convert output array to output matrix:
            if len(X_t.shape) == 1:
                X_t = X_t.reshape(-1, 1)
            # drop redundant prob output from binary classifiers:
            elif (len(X_t.shape) == 2 and X_t.shape[1] == 2 and
                  utils.is_classifier(self.model)):
                X_t = X_t[:, 1].reshape(-1, 1)
            return X_t
        else:
            raise utils.FitError('transform called before model fitting')

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def _more_tags(self):
        return {'multichannel': False}

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        if hasattr(self, 'model'):
            clone.model = utils.get_clone(self.model)
            clone._set_estimator_type(self.model)
            clone._remove_predictor_interface()
            clone._add_predictor_interface(self)
        return clone

    def get_descriptor(self, verbose=1):
        return '{' + utils.get_descriptor(self.predictor, verbose,
                                          self.get_params()) + '}tr'


class SingleChannelCV(SingleChannel):
    """
    Add transformer interface and internal cross validation training to
    scikit-learn predictor.

    Wrapper class that provides predictors with transform() and fit_transform()
    methods, and internal cross validation training with performance scoring.

    Parameters
    ----------
    predictor : predictor instance
        The scikit-learn conformant predictor to wrap.
    transform_method : str, default='auto'
        - Name of the prediction method to call when transforming (e.g. when
          outputting meta-features).
        - If 'auto' :
            - If classifier : method picked using
              config.transform_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'
    internal_cv : int, None, or callable, default=5
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int > 1: StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If {None, 1}: disable internal cv.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    score_method : str, default='auto'
        - Name of prediction method used when scoring predictor performance.
        - If 'auto' :
            - If classifier : method picked using
              config.score_method_precedence order (default:
              ppredict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'
    scorer : callable, default='auto'
        Callable that computes a figure of merit score for the internal_cv run.
        The score is exposed as score_ attribute during fit_transform().
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer with signature: score = scorer(y_true, y_pred).
    cv_processes : int or 'max', default=1
        - The number of parallel processes to run for internal cross
          validation.
        - If int : Use up to cv_processes number of processes.
        - If 'max' : Use all available CPUs.

    Examples
    --------
    Model stacking:
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = GradientBoostingClassifier()
        base_clf = pc.transform_wrappers.SingleChannelCV(base_clf)
        clf.add_layer(base_clf, pipe_processes='max')
        clf.add_layer(pc.MultichannelPredictor(SVC()))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.9411764705882353, 0.8897058823529411, 0.9963235294117647]

    Notes
    -----
    fit().transform() is not the same as fit_tranform() because only the latter
    uses internal cv training and inference.
    On calls to fit_transform() the model is fit on both the entire training
    set and cv splits of the training set. The model fit on the entire dataset
    is stored for futer inference.  The models fit on cv splits are used
    to make the outputs of fit_transform() but are not stored for future use.

    This class uses reflection to expose the predictor methods found in the
    object that it wraps, so the method attributes in a SingleChannelCV
    instance are usually not identical to the method attributes of the
    SingleChannelCV class.
    """

    def __init__(self, predictor, transform_method='auto', internal_cv=5,
                 score_method='auto', scorer='auto', cv_processes=1):
        self._params_to_attributes(SingleChannelCV.__init__, locals())
        super().__init__(predictor, transform_method)

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        is_classifier = utils.is_classifier(self.predictor)
        if y is not None and is_classifier:
            self.classes_, y = np.unique(y, return_inverse=True)

        self.fit(X, y, **fit_params)
        transform_method = self.get_transform_method()

        # if internal cv training is disabled
        if (self.internal_cv is None or
                (type(self.internal_cv) == int and self.internal_cv < 2)):
            y_transform = self.transform(X)
        # internal cv training is enabled
        else:
            split_results = cross_val_predict(self.predictor, X, y,
                                    groups=groups,
                                    predict_method=None,
                                    transform_method=transform_method,
                                    score_method=self.score_method,
                                    cv=self.internal_cv, combine_splits=True,
                                    n_processes=self.cv_processes,
                                    fit_params=fit_params)

            y_transform = split_results['transform']['y_pred']
            y_score = split_results['score']['y_pred']
            is_binary = (True if is_classifier and len(self.classes_) == 2
                         else False)

            score_method = split_results['score']['method']
            self.score_ = score_predictions(y, y_score, score_method,
                                            self.scorer, is_classifier,
                                            is_binary)

        # convert output array to output matrix:
        X_t = y_transform
        if len(X_t.shape) == 1:
            X_t = X_t.reshape(-1, 1)

        # drop the redundant prob output from binary classifiers:
        elif (len(X_t.shape) == 2 and X_t.shape[1] == 2 and
              utils.is_classifier(self.model)):
            X_t = X_t[:, 1].reshape(-1, 1)

        return X_t

    def _more_tags(self):
        return {'multichannel': False}

    def get_descriptor(self, verbose=1):
        return '{' + utils.get_descriptor(self.predictor, verbose,
                                          self.get_params()) + '}cvtr'

    def get_clone(self):
        clone = super().get_clone()
        if hasattr(self, 'scores_'):
            clone.scores_ = self.scores_
        return clone


class Multichannel(Cloneable, Saveable):
    """
    Add transformer interface to a multichannel predictor.

    Wrapper class that provides pipecaster's multichannel predictors with
    transform() and fit_transform methods().

    Parameters
    ----------
    multichannel_predictor : multichannel predictor instance
        The predictor to wrap.
    transform_method : str, default='auto'
        - Name of the prediction method to call when transforming (e.g. when
          outputting meta-features).
        - If 'auto' :
            - If classifier : method picked using
              config.transform_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'

    Examples
    --------
    model stacking:
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.MultichannelPredictor(GradientBoostingClassifier())
        base_clf = pc.transform_wrappers.Multichannel(base_clf)
        clf.add_layer(5, base_clf, 5, base_clf, pipe_processes=1)
        clf.add_layer(pc.MultichannelPredictor(SVC()))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.9411764705882353, 0.9411764705882353, 0.8768382352941176]

    Notes
    -----
    This class uses reflection to expose the predictor methods found in the
    object that it wraps, so the method attributes in a Multichannel instance
    are not usually identical to the method attributes of the Multichannel
    class.
    """

    def __init__(self, multichannel_predictor, transform_method='auto'):
        self._params_to_attributes(Multichannel.__init__, locals())
        utils.enforce_fit(multichannel_predictor)
        utils.enforce_predict(multichannel_predictor)
        utils.enforce_predict(multichannel_predictor)
        self._add_predictor_interface(multichannel_predictor)
        self._set_estimator_type(multichannel_predictor)

    def _set_estimator_type(self, predictor):
        if hasattr(predictor, '_estimator_type') is True:
            self._estimator_type = predictor._estimator_type

    def _add_predictor_interface(self, predictor):
        for method_name in config.recognized_pred_methods:
            if hasattr(predictor, method_name):
                prediction_method = functools.partial(self.predict_with_method,
                                                      method_name=method_name)
                setattr(self, method_name, prediction_method)

    def _add_model_interface(self, model, Xs):
        detected_methods = utils.detect_predict_methods(model, Xs)
        for method_name in detected_methods:
            prediction_method = functools.partial(self.predict_with_method,
                                                  method_name=method_name)
            setattr(self, method_name, prediction_method)

    def _remove_predictor_interface(self):
        for method_name in config.recognized_pred_methods:
            if hasattr(self, method_name):
                delattr(self, method_name)

    def get_transform_method(self):
        if self.transform_method == 'auto':
            if hasattr(self, 'model'):
                method_name = utils.get_transform_method(self.model)
            else:
                method_name = utils.get_transform_method(
                                                self.multichannel_predictor)
            if method_name is None:
                raise NameError('model lacks a recognized method for '
                                'conversion to transformer')
        else:
            method_name = self.transform_method

        return method_name

    def fit(self, Xs, y=None, **fit_params):
        self.model = utils.get_clone(self.multichannel_predictor)
        if y is None:
            self.model.fit(Xs, **fit_params)
        else:
            if utils.is_classifier(self.model):
                self.classes_, y = np.unique(y, return_inverse=True)
            self.model.fit(Xs, y, **fit_params)

        self._set_estimator_type(self.model)
        self._remove_predictor_interface()
        self._add_model_interface(self.model, Xs)

        return self

    def predict_with_method(self, Xs, method_name):
        if hasattr(self, 'model') is False:
            raise FitError('prediction attempted before call to fit()')
        prediction_method = getattr(self.model, method_name)
        predictions = prediction_method(Xs)
        if utils.is_classifier(self) and method_name == 'predict':
            predictions = self.classes_[predictions]
        return predictions

    def transform(self, Xs):
        if hasattr(self, 'model') is False:
            raise FitError('transform attempted before call to fit()')
        tansformer = getattr(self.model, self.get_transform_method())
        predictions = np.array(tansformer(Xs))

        # convert output array to output matrix:
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        # drop the redundant prob output from binary classifiers:
        elif (len(predictions.shape) == 2 and predictions.shape[1] == 2
              and utils.is_classifier(self.model)):
            predictions = predictions[:, 1].reshape(-1, 1)

        Xs_t = [predictions if i == 0 else None for i, X in enumerate(Xs)]
        return Xs_t

    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)

    def get_descriptor(self, verbose=1):
        return '{' + utils.get_descriptor(self.multichannel_predictor, verbose,
                                          self.get_params()) + '}tr'

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        if hasattr(self, 'model'):
            clone.model = utils.get_clone(self.model)
            clone._set_estimator_type(self.model)
            clone._remove_predictor_interface()
            clone._add_predictor_interface(self)
        return clone


class MultichannelCV(Multichannel):
    """
    Add transformer interface and internal cross validation training to
    multichannel predictor.

    Wrapper class that provides pipecaster's multichannel predictors with
    transform() and fit_transform() methods, and internal cross validation
    training with performance scoring.

    Parameters
    ----------
    multichannel_predictor : multichannel_predictor instance
        The pipecaster predictor to wrap.
    transform_method : str, default='auto'
        - Name of the prediction method to call when transforming (e.g. when
          outputting meta-features).
        - If 'auto' :
            - If classifier : method picked using
              config.transform_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'
    internal_cv : int, None, or callable, default=5
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int > 1: StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If {None, 1}: disable internal cv.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    score_method : str, default='auto'
        - Name of prediction method used when scoring predictor performance.
        - If 'auto' :
            - If classifier : method picked using
              config.score_method_precedence order (default:
              ppredict_proba->predict_log_proba->decision_function->predict).
            - If regressor : 'predict'
    scorer : callable, default='auto'
        Callable that computes a figure of merit score for the internal_cv run.
        The score is exposed as score_ attribute during fit_transform().
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer with signature: score = scorer(y_true, y_pred).
    cv_processes : int or 'max', default=1
        - The number of parallel processes to run for internal cross
          validation.
        - If int : Use up to cv_processes number of processes.
        - If 'max' : Use all available CPUs.

    Examples
    --------
    model stacking:
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.MultichannelPredictor(GradientBoostingClassifier())
        base_clf = pc.transform_wrappers.MultichannelCV(base_clf)
        clf.add_layer(5, base_clf, 5, base_clf, pipe_processes=1)
        clf.add_layer(pc.MultichannelPredictor(SVC()))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.8823529411764706, 0.9393382352941176, 0.9080882352941176]

    Notes
    -----
    fit().transform() is not the same as fit_tranform() because only the latter
    uses internal cv training and inference.
    On calls to fit_transform() the model is fit on both the entire training
    set and cv splits of the training set. The model fit on the entire dataset
    is stored for future inferences.  The models fit on cv splits are used
    to make the outputs of fit_transform() but are not stored for future use.

    This class uses reflection to expose the predictor methods found in the
    object that it wraps, so the method attributes in a MultichannelCV
    instance are usually not identical to the method attributes of the
    MultichannelCV class.
    """

    def __init__(self, multichannel_predictor, transform_method='auto',
                 internal_cv=5, score_method='auto',
                 scorer='auto', cv_processes=1):
        self._params_to_attributes(MultichannelCV.__init__, locals())
        super().__init__(multichannel_predictor, transform_method)

    def fit_transform(self, Xs, y=None, groups=None, **fit_params):
        is_classifier = utils.is_classifier(self.multichannel_predictor)
        if y is not None and is_classifier:
            self.classes_, y = np.unique(y, return_inverse=True)

        self.fit(Xs, y, **fit_params)
        transform_method = self.get_transform_method()

        # if internal cv training is disabled
        if (self.internal_cv is None or
                (type(self.internal_cv) == int and self.internal_cv < 2)):
            y_transform = self.transform(X)
        # internal cv training is enabled
        else:
            split_results = cross_val_predict(self.multichannel_predictor,
                                    Xs, y,
                                    groups=groups,
                                    predict_method=None,
                                    transform_method=transform_method,
                                    score_method=self.score_method,
                                    cv=self.internal_cv, combine_splits=True,
                                    n_processes=self.cv_processes,
                                    fit_params=fit_params)

            y_transform = split_results['transform']['y_pred']
            y_score = split_results['score']['y_pred']
            is_binary = (True if is_classifier and len(self.classes_) == 2
                         else False)
            score_method = split_results['score']['method']
            self.score_ = score_predictions(y, y_score, score_method,
                                            self.scorer, is_classifier,
                                            is_binary)

        # convert predictions to transformed matrix:
        X_t = y_transform
        if len(X_t.shape) == 1:
            X_t = X_t.reshape(-1, 1)

        # drop the redundant prob output from binary classifiers:
        elif (len(X_t.shape) == 2 and X_t.shape[1] == 2 and
              utils.is_classifier(self.model)):
            X_t = X_t[:, 1].reshape(-1, 1)

        Xs_t = [None for X in Xs]
        Xs_t[0] = X_t

        return Xs_t

    def get_descriptor(self, verbose=1):
        return '{' + utils.get_descriptor(self.multichannel_predictor, verbose,
                                          self.get_params()) + '}cvtr'

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'score_'):
            clone.score_ = self.score_
        return clone


def unwrap_predictor(pipe):
    """
    Return a predictor that is wrapped in a transform wrapper.
    """
    if type(pipe) not in [SingleChannel, SingleChannelCV, Multichannel,
                          MultichannelCV]:
        return pipe
    if type(pipe) in [Multichannel, MultichannelCV]:
        return pipe.multichannel_predictor
    else:
        return pipe.predictor


def unwrap_model(pipe):
    """
    Return a model that is wrapped in a transform wrapper.
    """
    if type(pipe) not in [SingleChannel, SingleChannelCV, Multichannel,
                          MultichannelCV]:
        return pipe
    if hasattr(pipe, 'model') is True:
        return pipe.model
    else:
        raise utils.FitError('no model found')
