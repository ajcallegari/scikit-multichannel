"""
Wrapper classes for model stacking.

MultichannelPipelines treat internal component as transfomers (i.e. invoking
fit/transform/fit_transform).  As a consequence, when predictors are used
internally in stacked architectures, a transformer interface must be added to
the internal predictors.  In practice, this just means choosing a prediction
method to use when transforming and converting 1D outputs to 2D outputs.

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

import pipecaster.utils as utils
from pipecaster.utils import Cloneable, Saveable
from pipecaster.cross_validation import cross_val_predict

# choose methods in this order when generating transform outputs from predictor
transform_method_precedence = ['predict_proba', 'decision_function',
                               'predict_log_proba', 'predict']


def get_transform_method(pipe):
    """
    Get reference to a transform method.

    Parameters
    ----------
    pipe: pipe instance

    Returns
    -------
    Reference to a pipe method that can be used for transforming.
    """
    for method_name in transform_method_precedence:
        if hasattr(pipe, method_name):
            return getattr(pipe, method_name)
    return None


def get_transform_method_name(pipe):
    """
    Get transform method name for a pipe using transform_method_precedence.

    Parameters
    ----------
    pipe: pipe instance

    Returns
    -------
    Name (str) of a pipe method that can be used for transforming.
    """
    for method_name in transform_method_precedence:
        if hasattr(pipe, method_name):
            return method_name
    return None


class SingleChannel(Cloneable, Saveable):
    """
    Add transformer interface to a scikit-learn predictor.

    Wrapper class that provides scikit-learn conformant predictors with
    transform() and fit_transform methods().

    Parameters
    ----------
    predictor : predictor instance
        The scikit-learn conformant predictor to wrap.
    transform_method_name : string, default='auto'
        Name of the prediction method to used for generating outputs on call to
        transform or fit_transform. If 'auto', the method is automatically
        chosen using the order specified in transform_method_precedence.

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
        base_clf = pc.transform_wrappers.SingleChannel(
                                                GradientBoostingClassifier())
        clf.add_layer(base_clf, pipe_processes='max')
        clf.add_layer(pc.MultichannelPredictor(SVC()))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.8529411764705882, 0.8216911764705883, 0.9099264705882353]

    Notes
    -----
    This class uses reflection to expose the predictor methods found in the
    object that it wraps, so the method attributes in a SingleChannel instance
    are not usually identical to the method attributes of the SingleChannel
    class.
    """
    state_variables = ['classes_']

    def __init__(self, predictor, transform_method_name='auto'):
        self._params_to_attributes(SingleChannel.__init__, locals())
        utils.enforce_fit(predictor)
        if transform_method_name == 'auto':
            self.transform_method_name = get_transform_method_name(predictor)
            if self.transform_method_name is None:
                raise NameError('predictor lacks a recognized method for \
                                conversion to transformer')
        predict_methods = utils.get_predict_methods(predictor)
        self._set_predictor_interface(predict_methods)
        self._estimator_type = utils.detect_predictor_type(predictor)
        if self._estimator_type is None:
            raise TypeError('could not detect predictor type for {}'
                            .format(predictor))

    def _set_predictor_interface(self, predict_method_names):
        for method_name in utils.recognized_pred_methods:
            is_available = method_name in predict_method_names
            is_exposed = hasattr(self, method_name)

            if is_available and is_exposed is False:
                prediction_method = functools.partial(self.predict_with_method,
                                                      method_name=method_name)
                setattr(self, method_name, prediction_method)
            elif is_exposed and is_available is False:
                delattr(self, method_name)

    def set_transform_method(self, method_name):
        self.transform_method_name = method_name
        return self

    def fit(self, X, y=None, **fit_params):
        self.model = utils.get_clone(self.predictor)
        if y is None:
            self.model.fit(X, **fit_params)
        else:
            self.model.fit(X, y, **fit_params)
        if self._estimator_type == 'classifier':
            self.classes_ = self.model.classes_
        predict_methods = utils.get_predict_methods(self.model)
        self._set_predictor_interface(predict_methods)
        return self

    def predict_with_method(self, X, method_name):
        if hasattr(self, 'model') is False:
            raise utils.FitError('prediction attempted before model fitting')
        if hasattr(self.model, method_name):
            predict_method = getattr(self.model, method_name)
            return predict_method(X)
        else:
            raise NameError('prediction method {} not found in {} attributes'
                            .format(method_name, self.model))

    def transform(self, X):
        if hasattr(self, 'model'):
            transform_method = getattr(self.model, self.transform_method_name)
            X_t = transform_method(X)
            # convert output array to output matrix:
            if len(X_t.shape) == 1:
                X_t = X_t.reshape(-1, 1)
            # drop the redundant prob output from binary classifiers:
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
        clone = super().get_clone()
        if hasattr(self, 'model'):
            clone.model = utils.get_clone(self.model)
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
    transform_method_name: string, default='auto'
        Name of the prediction method to used for generating outputs on call to
        transform or fit_transform. If 'auto', the method is automatically
        chosen using the order specified in transform_method_precedence.
    internal_cv : int, None, or callable, default=5
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int : StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If None : default value of 5.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    cv_processes : int or 'max', default=1
        - The number of parallel processes to run for internal cross
          validation.
        - If int : Use up to cv_processes number of processes.
        - If 'max' : Use all available CPUs.
    scorer : callable, default=None
        Callable that computes a figure of merit score for the internal_cv run.
        Expected pattern: score = scorer(y_true, y_pred). The cross validation
        score is exposed through creation of a score_ attribute during calls to
        fit_transform().

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
        base_clf = pc.transform_wrappers.SingleChannelCV(
                                                GradientBoostingClassifier())
        clf.add_layer(base_clf, pipe_processes='max')
        clf.add_layer(pc.MultichannelPredictor(SVC()))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.9411764705882353, 0.9411764705882353, 0.9375]

    Notes
    -----
    fit().transform() is not the same as fit_tranform() because only the latter
    uses internal cv training and inference.
    On calls to fit_transform() the model is fit on both the entire training
    set and cv splits of the training set. The model fit on the entire dataset
    is stored for inference on subsequent calls to predict(), predict_proba(),
    decision_function(), or tranform().  The models fit on cv splits are used
    to make the outputs of fit_transform() but are not stored for future use.

    This class uses reflection to expose the predictor methods found in the
    object that it wraps, so the method attributes in a SingleChannelCV
    instance are usually not identical to the method attributes of the
    SingleChannelCV class.
    """
    state_variables = ['score_']

    def __init__(self, predictor, transform_method_name='auto', internal_cv=5,
                 cv_processes=1, scorer=None):
        self._inherit_state_variables(super())
        self._params_to_attributes(SingleChannelCV.__init__, locals())
        super().__init__(predictor, transform_method_name)
        self.internal_cv = 5 if internal_cv is None else internal_cv

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        self.fit(X, y, **fit_params)

        # internal cv training is disabled
        if (self.internal_cv is None or
                (type(self.internal_cv) == int and self.internal_cv < 2)):
            X_t = self.transform(X)
        # internal cv training is enabled
        else:
            X_t = cross_val_predict(self.predictor, X, y, groups=groups,
                                    predict_method=self.transform_method_name,
                                    cv=self.internal_cv, combine_splits=True,
                                    n_processes=self.cv_processes,
                                    fit_params=fit_params)

            if self.scorer is not None:
                if self.transform_method_name in ['predict_proba',
                                                  'decision_function',
                                                  'predict_log_proba']:
                    self.score_ = self.scorer(y, utils.classify_samples(X_t))
                else:
                    self.score_ = self.scorer(y, X_t)
        # convert output array to output matrix:
        if len(X_t.shape) == 1:
            X_t = X_t.reshape(-1, 1)
        # drop the redundant prob output from binary classifiers:
        elif (len(X_t.shape) == 2 and X_t.shape[1] == 2 and
              utils.is_classifier(self.model)):
            X_t = X_t[:, 1].reshape(-1, 1)

        return X_t

    def _more_tags(self):
        return {'multichannel': False}

    def get_clone(self):
        return super().get_clone()

    def get_descriptor(self, verbose=1):
        return '{' + utils.get_descriptor(self.predictor, verbose,
                                          self.get_params()) + '}cvtr'


class Multichannel(Cloneable, Saveable):
    """
    Add transformer interface to a multichannel predictor.

    Wrapper class that provides pipecaster's multichannel predictors with
    transform() and fit_transform methods().

    Parameters
    ----------
    multichannel_predictor : multichannel_predictor instance
        The predictor to wrap.
    transform_method_name : string, default='auto'
        Name of the prediction method to used for generating outputs on call to
        transform or fit_transform. If 'auto', the method is automatically
        chosen using the order specified in transform_method_precedence.

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
        base_clf = pc.transform_wrappers.Multichannel(
            pc.MultichannelPredictor(GradientBoostingClassifier()))
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
    state_variables = ['classes_']

    def __init__(self, multichannel_predictor, transform_method_name='auto'):
        self._params_to_attributes(Multichannel.__init__, locals())
        utils.enforce_fit(multichannel_predictor)
        utils.enforce_predict(multichannel_predictor)
        self._estimator_type = utils.detect_predictor_type(
                                     multichannel_predictor)
        if self._estimator_type is None:
            raise AttributeError('could not detect predictor type')
        if transform_method_name == 'auto':
            self.transform_method_name = get_transform_method_name(
                                            multichannel_predictor)
            if self.transform_method_name is None:
                raise TypeError('missing recognized method for transforming \
                                with a predictor')
        predict_methods = utils.get_predict_methods(multichannel_predictor)
        self._set_predictor_interface(predict_methods)

    def _set_predictor_interface(self, predict_method_names):
        for method_name in utils.recognized_pred_methods:
            is_available = method_name in predict_method_names
            is_exposed = hasattr(self, method_name)

            if is_available and is_exposed is False:
                prediction_method = functools.partial(self.predict_with_method,
                                                      method_name=method_name)
                setattr(self, method_name, prediction_method)
            elif is_exposed and is_available is False:
                delattr(self, method_name)

    def fit(self, Xs, y=None, **fit_params):
        self.model = utils.get_clone(self.multichannel_predictor)
        if y is None:
            self.model.fit(Xs, **fit_params)
        else:
            if utils.is_classifier(self.model):
                self.classes_, y = np.unique(y, return_inverse=True)
            self.model.fit(Xs, y, **fit_params)
        predict_methods = utils.get_predict_methods(self.model)
        self._set_predictor_interface(predict_methods)
        return self

    def predict_with_method(self, Xs, method_name):
        if hasattr(self, 'model') is False:
            raise FitError('prediction attempted before call to fit()')
        prediction_method = getattr(self.model, method_name)
        return prediction_method(Xs)

    def transform(self, Xs):
        if hasattr(self, 'model') is False:
            raise FitError('transform attempted before call to fit()')
        transform_method = getattr(self.model, self.transform_method_name)
        predictions = np.array(transform_method(Xs))

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
    transform_method_name: string, default='auto'
        Name of the prediction method to used for generating outputs on call to
        transform or fit_transform. If 'auto', the method is automatically
        chosen using the order specified in transform_method_precedence.
    internal_cv : int, None, or callable, default=5
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int : StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If None : default value of 5.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    cv_processes : int or 'max', default=1
        - The number of parallel processes to run for internal cross
          validation.
        - If int : Use up to cv_processes number of processes.
        - If 'max' : Use all available CPUs.
    scorer : callable, default=None
        Callable that computes a figure of merit score for the internal_cv run.
        Expected pattern: score = scorer(y_true, y_pred). The cross validation
        score is exposed through creation of a score_ attribute during calls to
        fit_transform().

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
        base_clf = pc.transform_wrappers.MultichannelCV(
            pc.MultichannelPredictor(GradientBoostingClassifier()))
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
    is stored for inference on subsequent calls to predict(), predict_proba(),
    decision_function(), or tranform().  The models fit on cv splits are used
    to make the outputs of fit_transform() but are not stored for future use.

    This class uses reflection to expose the predictor methods found in the
    object that it wraps, so the method attributes in a MultichannelCV
    instance are usually not identical to the method attributes of the
    MultichannelCV class.
    """
    state_variables = ['score_']

    def __init__(self, multichannel_predictor, transform_method_name='auto',
                 internal_cv=5, cv_processes=1, scorer=None):
        internal_cv = 5 if internal_cv is None else internal_cv
        self._params_to_attributes(MultichannelCV.__init__, locals())
        self._inherit_state_variables(super())
        super().__init__(multichannel_predictor, transform_method_name)

    def fit_transform(self, Xs, y=None, groups=None, **fit_params):

        if y is not None and utils.is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)

        self.fit(Xs, y, **fit_params)

        # internal cv training is disabled
        if (self.internal_cv is None or
                (type(self.internal_cv) == int and self.internal_cv < 2)):
            Xs_t = self.transform(Xs)
        # internal cv training is enabled
        else:
            predictions = cross_val_predict(
                                  self.multichannel_predictor, Xs, y,
                                  groups=groups,
                                  predict_method=self.transform_method_name,
                                  cv=self.internal_cv, combine_splits=True,
                                  n_processes=self.cv_processes,
                                  fit_params=fit_params)

            Xs_t = [None for X in Xs]

            # convert output array to output matrix:
            if len(predictions.shape) == 1:
                Xs_t[0] = predictions.reshape(-1, 1)
            # drop the redundant prob output from binary classifiers:
            elif (len(predictions.shape) == 2 and predictions.shape[1] == 2
                  and utils.is_classifier(self.model)):
                Xs_t[0] = predictions[:, 1].reshape(-1, 1)
            else:
                Xs_t[0] = predictions

            if self.scorer is not None:
                if utils.is_classifier(self) and len(predictions.shape) > 1:
                    predictions = util.classify_samples(predictions,
                                                        self.classes_)
                self.score_ = self.scorer(y, predictions)

        return Xs_t

    def get_descriptor(self, verbose=1):
        return '{' + utils.get_descriptor(self.multichannel_predictor, verbose,
                                          self.get_params()) + '}cvtr'


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
