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
from sklearn.metrics import log_loss

import pipecaster.utils as utils
from pipecaster.utils import Cloneable, Saveable
from pipecaster.cross_validation import cross_val_predict

# choose methods in this order when generating transform outputs from predictor
transform_method_precedence = ['decision_function', 'predict_proba',
                               'predict_log_proba', 'predict']

score_method_precedence = ['decision_function', 'predict_proba',
                               'predict_log_proba', 'predict']

def get_transform_method(pipe):
    """
    Get prediction method name using transform_method_precedence.

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


def get_score_method(pipe):
    """
    Get prediction method name using score_method_precedence.

    Parameters
    ----------
    pipe: pipe instance

    Returns
    -------
    Name (str) of a pipe method that can be used for making predictons for
    performance estimation.
    """
    for method_name in score_method_precedence:
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
    transform_method : str, default='auto'
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

    def __init__(self, predictor, transform_method='auto'):
        self._params_to_attributes(SingleChannel.__init__, locals())
        utils.enforce_fit(predictor)

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
        self.transform_method = method_name
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
            if self.transform_method == 'auto':
                transform_method = get_transform_method(self.model)
                if transform_methodis None:
                    raise NameError('model lacks a recognized method for \
                                    conversion to transformer')
            else:
                transform_method = self.transform_method
            transformer = getattr(self.model, transform_method)
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
    transform_method: string, default='auto'
        Name of the prediction method to used for transforming. If 'auto',
        method chosen using the transform_method_precedence order.
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
    score_method : str, default='auto'
        - Name of prediction method used when scoring predictor performance.
        - if 'auto' :
            - If classifier : method picked using score_method_precedence order
            - If regressor : 'predict'
    scorer : callable, default='auto'
        Callable that computes a figure of merit score for the internal_cv run.
        The score is exposed as score_ attribute during fit_transform().
        - If 'auto':
            - balanced_accuracy_score for classifiers with predict()
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
        - If callable: A scorer with signature: score = scorer(y_true, y_pred).

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
    is stored for futer inference.  The models fit on cv splits are used
    to make the outputs of fit_transform() but are not stored for future use.

    This class uses reflection to expose the predictor methods found in the
    object that it wraps, so the method attributes in a SingleChannelCV
    instance are usually not identical to the method attributes of the
    SingleChannelCV class.
    """

    def __init__(self, predictor, transform_method='auto', internal_cv=5,
                 cv_processes=1, score_method='auto', scorer='auto'):
        self._params_to_attributes(SingleChannelCV.__init__, locals())
        super().__init__(predictor, transform_method)

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        self.fit(X, y, **fit_params)

        # internal cv training is disabled
        if (self.internal_cv is None or
                (type(self.internal_cv) == int and self.internal_cv < 2)):
            X_t = self.transform(X)
        # internal cv training is enabled
        else:
            if self.score_method == 'auto':
                score_method = get_score_method(self.model)
                if score_method is None:
                    raise NameError('model lacks a recognized method for '
                                    'making scorable predictions.')
            else:
                score_method = self.score_method

            if self.scorer == 'auto':
                if utils.is_regressor(self.model):
                    scorer = explained_variance_score
                else if score_method is 'predict':
                    scorer = balanced_accuracy_score
                elif score_method in ['predict_proba', 'decision_function'
                                           'predict_log_proba']:
                    scorer = roc_auc_score
            else:
                scorer = self.scorer

            if self.transform_method == 'auto':
                transform_method = get_transform_method(self.model)
                if transform_methodis None:
                    raise NameError('model lacks a recognized method for \
                                    conversion to transformer')
            else:
                transform_method = self.transform_method

            predict_methods = [transform_method]
            if score_method != transform_method:
                predict_methods.append(score_method)

            preds = cross_val_predict(self.predictor, X, y, groups=groups,
                                    predict_methods=predict_methods,
                                    cv=self.internal_cv, combine_splits=True,
                                    n_processes=self.cv_processes,
                                    fit_params=fit_params)

            if self.transform_method in ['predict_proba',
                                              'predict_log_proba']:
                y_pred = [utils.classify_sample(p) for p in X_t]
                if self.scorer is None:
                    self.score_ = -log_loss(y, X_t)
                else:
                    self.score_ = self.scorer(y, y_pred)
            if self.transform_method == 'decision_function':
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
    transform_method : string, default='auto'
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

    def __init__(self, multichannel_predictor, transform_method='auto'):
        self._params_to_attributes(Multichannel.__init__, locals())
        utils.enforce_fit(multichannel_predictor)
        utils.enforce_predict(multichannel_predictor)
        self._estimator_type = utils.detect_predictor_type(
                                     multichannel_predictor)
        if self._estimator_type is None:
            raise AttributeError('could not detect predictor type')
        if transform_method == 'auto':
            self.transform_method = get_transform_method(
                                            multichannel_predictor)
            if self.transform_method is None:
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
        tansformer = getattr(self.model, self.transform_method)
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
    transform_method: string, default='auto'
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
    is stored for future inferences.  The models fit on cv splits are used
    to make the outputs of fit_transform() but are not stored for future use.

    This class uses reflection to expose the predictor methods found in the
    object that it wraps, so the method attributes in a MultichannelCV
    instance are usually not identical to the method attributes of the
    MultichannelCV class.
    """

    def __init__(self, multichannel_predictor, transform_method='auto',
                 internal_cv=5, cv_processes=1, scorer=None):
        internal_cv = 5 if internal_cv is None else internal_cv
        self._params_to_attributes(MultichannelCV.__init__, locals())
        super().__init__(multichannel_predictor, transform_method)

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
                                  predict_method=self.transform_method,
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
