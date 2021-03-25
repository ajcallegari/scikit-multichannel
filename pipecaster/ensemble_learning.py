"""
Pipeline components for ensemble prediction.
"""

import numpy as np
import pandas as pd
import ray
import scipy.stats
import functools

from sklearn.metrics import explained_variance_score, balanced_accuracy_score
from sklearn.model_selection import ParameterGrid

import pipecaster.utils as utils
import pipecaster.config as config
from pipecaster.utils import Cloneable, Saveable
import pipecaster.transform_wrappers as transform_wrappers
from pipecaster.score_selection import RankScoreSelector
import pipecaster.parallel as parallel

__all__ = ['SoftVotingClassifier', 'SoftVotingDecision',
           'HardVotingClassifier', 'AggregatingRegressor',
           'SoftVotingMetaClassifier', 'SoftVotingMetaDecision',
           'HardVotingMetaClassifier', 'AggregatingMetaRegressor', 'Ensemble',
           'GridSearchEnsemble', 'MultichannelPredictor', 'ChannelEnsemble']


class SoftVotingClassifier(Cloneable, Saveable):
    """
    Make ensemble predictions from averaged predict_proba outputs.

    This multichannel pipeline component takes a list of predict_proba outputs
    from an ensemble of of base classifiers and predicts with the averaged
    probs.

    Notes
    -----
    - To vote with decision_function outputs use the SoftVotingDecision or
      HardVotingClassifier classes.
    - To mix classifiers with different scalar output methods, use the
      HardVotingClassifier class.

    Examples
    --------
    SoftVotingMetaClassifier as a meta-predictor for ChannelEnsemble.
    ::

        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=5,
                                                      n_random_Xs=5)

        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        base_clf = KNeighborsClassifier()
        base_clf = pc.transform_wrappers.SingleChannel(base_clf)
        clf.add_layer(base_clf)
        clf.add_layer(pc.SoftVotingClassifier())
        pc.cross_val_score(clf, Xs, y)
        # output: [0.9411764705882353, 0.8768382352941176, 0.9080882352941176]

    """

    def __init__(self):
        self._estimator_type = 'classifier'

    def fit(self, Xs, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output '
                                      'meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def predict_proba(self, Xs):
        live_Xs = [X for X in Xs if X is not None]

        # expand binary classification probs to 2 columns
        if len(self.classes_) == 2:
            for i, X in enumerate(live_Xs):
                X_expanded = np.empty((X.shape[0], 2))
                X = X.reshape(-1)
                X_expanded[:, 0] =  1.0 - X
                X_expanded[:, 1] =  X
                live_Xs[i] = X_expanded

        return np.mean(live_Xs, axis=0)

    def predict(self, Xs):
        mean_probs = self.predict_proba(Xs)
        decisions = np.argmax(mean_probs, axis=1)
        predictions = self.classes_[decisions]
        return predictions

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        return clone


class SoftVotingDecision(Cloneable, Saveable):
    """
    Take average of an ensemble of decision_function outputs.

    This pipeline component takes scalar decision_function outputs from a prior
    pipeline classifier stage and averages them.  Averaged decision_function
    outputs can be used as meta-features for model stacking (see example
    below).

    Notes
    -----
        - The ensemble of inputs must be concatenated into a single
          meta-feature matrix in a prior stage.

        - This class implements estimator and tranformer interfaces but
          lacks a complete predictor interface (i.e. 'predict' method) because
          there is not a standard method for generating a categorical
          predictions from an ensemble of decision_function outputs.

    Examples
    --------
    ::

        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc
        from sklearn.metrics import balanced_accuracy_score

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=6,
                                                      n_random_Xs=3)

        clf = pc.MultichannelPipeline(n_channels=9)
        clf.add_layer(StandardScaler())
        base_clf = pc.make_cv_transformer(SVC(),
                                          transform_method='decision_function')
        clf.add_layer(base_clf)
        meta_clf1 = pc.SoftVotingDecision()
        clf.add_layer(3, meta_clf1, 3, meta_clf1, 3, meta_clf1)
        meta_clf2 = pc.MultichannelPredictor(GradientBoostingClassifier())
        clf.add_layer(meta_clf2)
        pc.cross_val_score(clf, Xs, y, score_method='predict',
                                        scorer=balanced_accuracy_score)
        # [0.8823529411764706, 0.8823529411764706, 0.7040441176470589]
    """

    def __init__(self):
        self._estimator_type = 'classifier'

    def fit(self, Xs, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output '
                                      'meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def decision_function(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        return np.mean(live_Xs, axis=0)

    def transform(self, Xs):
        Xs_t = [None for X in Xs]
        Xs_t[0] = self.decision_function(Xs)
        return Xs_t

    def fit_transform(self, Xs, y, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        return clone


class HardVotingClassifier(Cloneable, Saveable):
    """
    Predict using the most frequent class in a prediction ensemble.

    This pipeline component takes categorical predictions from a prior pipeline
    stage and uses them to make an ensemble prediction.  The prediction
    is made by taking the modal prediction (i.e. most frequently predicted
    class) of the classifiers in the ensemble.

    Examples
    --------
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=5,
                                                      n_random_Xs=5)

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = GradientBoostingClassifier()
        base_clf = pc.make_transformer(base_clf, transform_method='predict')
        clf.add_layer(base_clf)
        clf.add_layer(pc.HardVotingClassifier())
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.8823529411764706, 0.9117647058823529, 0.8216911764705883]
    """

    def __init__(self):
        self._estimator_type = 'classifier'

    def fit(self, Xs, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output '
                                      'meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def predict(self, Xs):
        """
        Return the modal class predicted by the base classifiers.
        """
        live_Xs = [X for X in Xs if X is not None]
        y_preds = np.concatenate(live_Xs, axis=1)
        y_pred = scipy.stats.mode(y_preds, axis=1)[0].reshape(-1)
        try:
            return self.classes_[y_pred]
        except IndexError:
            raise IndexError('Could not decoding category labels.  Try setting'
                             ' the base predictor transform_method to predict')

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        return clone


class AggregatingRegressor(Cloneable, Saveable):
    """
    Predict with aggregated outputs of a regressor ensemble.

    This multichannel component takes a list of predictions from a prior
    pipeline stage and converts them into a single prediction using an
    aggregator function (e.g. np.mean).

    Notes
    -----
    Only supports single output regressors.

    Parameters
    ----------
    aggregator : callable, default=np.mean
        Function for converting multiple regression predictions into a single
        prediction.  Signature: prediction = aggregator(predictions).

    Examples
    --------
    ::

        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_regression(n_informative_Xs=3)

        clf = pc.MultichannelPipeline(n_channels=3)
        base_clf = GradientBoostingRegressor()
        clf.add_layer(pc.make_transformer(base_clf))
        clf.add_layer(pc.AggregatingRegressor(np.mean))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.4126896021655341, 0.3661579493175574, 0.2878593262097652]
    """

    def __init__(self, aggregator=np.mean):
        self.aggregator = aggregator
        self._estimator_type = 'regressor'

    def fit(self, Xs=None, y=None, **fit_params):
        return self

    def predict(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        y_preds = np.concatenate(live_Xs, axis=1)
        return self.aggregator(y_preds, axis=1)


class SoftVotingMetaClassifier(Cloneable, Saveable):
    """
    Soft voting meta-classifier.

    This component can be used in contexts where you would ordinarily use an
    ML algorithm for meta-classification.  Like ML meta-classifiers,
    SoftVotingMetaClassifier takes an input vector formed by concatenating the
    predictions of the base predictors.  The prediction behavior is identical
    to :class:`SoftVotingClassifier`.

    SoftVotingMetaClassifier can be used as a standalone pipeline component or
    be used as the the meta_predictor paramter to MultichannelPredictor,
    Ensemble, and ChannelEnsemble components.

    The inputs, which must be concatenated into a single meta-feature matrix in
    a prior stage, are decatenated and predicted classes inferred from the
    order of the meta-feature matrix columns.

    Notes
    -----
    - To vote with decision_function outputs use the SoftVotingMetaDecision or
      HardVotingMetaClassifier classes.
    - To mix classifiers with different scalar output methods, use the
      HardVotingMetaClassifier class.

    Examples
    --------
    SoftVotingMetaClassifier as a meta-predictor for ChannelEnsemble.
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = GradientBoostingClassifier()
        meta_clf = pc.SoftVotingMetaClassifier()
        clf.add_layer(pc.ChannelEnsemble(base_clf, meta_clf))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.9117647058823529, 0.8180147058823529, 0.9117647058823529]

    SoftVotingMetaClassifier as a predictor for MultichannelPredictor.
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.make_transformer(GradientBoostingClassifier())
        meta_clf = pc.SoftVotingMetaClassifier()
        clf.add_layer(base_clf)
        clf.add_layer(pc.MultichannelPredictor(meta_clf))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.82352941176, 0.8474264705882353, 0.9080882352]

    SoftVotingMetaClassifier as a standalone pipeline component.
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.make_transformer(GradientBoostingClassifier())
        clf.add_layer(base_clf)
        clf.add_layer(pc.ChannelConcatenator())
        clf.add_layer(1, pc.SoftVotingMetaClassifier())
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.8823529411764706, 0.8492647058823529, 0.8455882352941176]
    """

    def __init__(self):
        self._estimator_type = 'classifier'

    def fit(self, X, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output '
                                      'meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def _decatenate(self, meta_X):
        n_classes = len(self.classes_)
        if n_classes == 2:
            # expand binary classification probs to 2 columns
            n_rows, n_cols = meta_X.shape[0], 2 * meta_X.shape[1]
            Xs_expanded = np.empty((n_rows, n_cols))
            Xs_expanded[:, range(0, n_cols, 2)] =  1.0 - meta_X
            Xs_expanded[:, range(1, n_cols + 1, 2)] =  meta_X
            meta_X = Xs_expanded

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

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        return clone


class SoftVotingMetaDecision(Cloneable, Saveable):
    """
    Take average of an ensemble of decision_function outputs.

    This component can be used in limited contexts where you would ordinarily
    use an ML algorithm to generate meta-features from an ensemble (see example
    below).  Like ML meta-classifiers, SoftVotingMetaDecision takes an input
    vector formed by concatenating the predictions of the base predictors.
    Unlike ML models, SoftVotingMetaDecision can't make predictions, it can
    only output meta-features to be used for additional meta-classification
    (there is not a standard method for generating a categorical predictions
    from an ensemble of decision_function outputs).

    Examples
    --------
    ::

        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        base_clf = pc.make_transformer(SVC(),
                                       transform_method='decision_function')
        clf.add_layer(base_clf)
        meta_clf1 = pc.SoftVotingMetaDecision()
        clf.add_layer(2, meta_clf1, 2, meta_clf1, 2, meta_clf1, 2,
                      meta_clf1, 2, meta_clf1)
        meta_clf2 = pc.MultichannelPredictor(GradientBoostingClassifier())
        clf.add_layer(meta_clf2)
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.35294117647058826, 0.84375, 0.5808823529411764]
    """

    def __init__(self):
        self._estimator_type = 'classifier'

    def fit(self, X, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output '
                                      'meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def _decatenate(self, meta_X):
        n_classes = len(self.classes_)
        if n_classes > 2:
            raise NotImplementedError('Only binary classification is '
                                      'supported but {} classes found.'
                                      .format(n_classes))
        Xs = [meta_X[:, i:i+1] for i in range(0, meta_X.shape[1])]
        return Xs

    def decision_function(self, X):
        Xs = self._decatenate(X)
        return np.mean(Xs, axis=0)

    def transform(self, X):
        return self.decision_function(X).reshape(-1, 1)

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        return clone


class HardVotingMetaClassifier(Cloneable, Saveable):
    """
    Predict with the most frquent class in ensemble of predictions.

    This component can be used in contexts where you would ordinarily use an
    ML algorithm for meta-classification.  Like ML meta-classifiers,
    HardVotingMetaClassifier takes an input vector formed by concatenating the
    predictions of the base predictors.  The prediction behavior is identical
    to :class:`HardVotingClassifier`.

    The ensemble of inputs for HardVotingMetaClassifier must be concatenated
    into a single matrix in a prior stage.

    Examples
    --------
    HardVotingMetaClassifier as a meta-predictor for ChannelEnsemble.
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = GradientBoostingClassifier()
        meta_clf = pc.HardVotingMetaClassifier()
        clf.add_layer(pc.ChannelEnsemble(base_clf, meta_clf,
                                         base_transform_methods='predict'))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.7647058823529411, 0.6985294117647058, 0.9080882352941176]

    HardVotingMetaClassifier as a predictor for MultichannelPredictor.
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.transform_wrappers.SingleChannel(
            GradientBoostingClassifier(), transform_method='predict')
        meta_clf = pc.HardVotingMetaClassifier()
        clf.add_layer(base_clf)
        clf.add_layer(pc.MultichannelPredictor(meta_clf))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.7352941176470589, 0.7849264705882353, 0.7536764705882353]

    HardVotingMetaClassifier as a standalone pipeline component.
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.transform_wrappers.SingleChannel(
            GradientBoostingClassifier(), transform_method='predict')
        clf.add_layer(base_clf)
        clf.add_layer(pc.ChannelConcatenator())
        clf.add_layer(1, pc.HardVotingMetaClassifier())
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.7352941176470589, 0.6654411764705883, 0.78125]
    """

    def __init__(self):
        self._estimator_type = 'classifier'

    def fit(self, X, y, **fit_params):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output '
                                      'meta-classification not supported')
        self.classes_, y = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        """
        Return the modal class predicted by the base classifiers.
        """
        predictions = scipy.stats.mode(X, axis=1)[0].reshape(-1)
        try:
            return self.classes_[predictions]
        except IndexError:
            raise IndexError('Could not decoding category labels.  Try setting'
                             ' the base predictor transform_method to predict')

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        return clone


class AggregatingMetaRegressor(Cloneable, Saveable):
    """
    Predict with aggregated outputs of a regressor ensemble.

    This pipeline component can take the place of ML algorithms that function
    as meta-regressors.  It cannot predict, but can output meta-features
    created by applying an aggeragator function to the predictions of the base
    regressors.  Can be used alone or as a meta-predictor within
    MultichannelPredictor, Ensemble, and ChannelEnsemble pipeline components.

    Notes
    -----
    Only supports single output regressors.

    Parameters
    ----------
    aggregator : callable, default=np.mean
        Function for converting multiple regression predictions into a single
        prediction.  Signature: prediction = aggregator(predictions).

    Examples
    --------
    ::

        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_regression(n_informative_Xs=3)

        # recommended use style
        clf = pc.MultichannelPipeline(n_channels=3)
        base_clf = GradientBoostingRegressor()
        meta_clf = pc.AggregatingMetaRegressor(np.mean)
        clf.add_layer(pc.ChannelEnsemble(base_clf, meta_clf))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.38792868647764933, 0.42338725182412085, 0.41323951948]

        # alternative use style 1:
        clf = pc.MultichannelPipeline(n_channels=3)
        base_clf = pc.make_transformer(GradientBoostingRegressor())
        clf.add_layer(base_clf)
        clf.add_layer(pc.ChannelConcatenator())
        clf.add_layer(1, pc.AggregatingMetaRegressor(np.mean))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.3895750977165564, 0.42519719457611027, 0.4154415]

        # alternative use style 2:
        clf = pc.MultichannelPipeline(n_channels=3)
        base_clf = pc.make_transformer(GradientBoostingRegressor())
        meta_clf = pc.AggregatingMetaRegressor(np.mean)
        clf.add_layer(base_clf)
        clf.add_layer(pc.MultichannelPredictor(meta_clf))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.39292990230137037, 0.42187542250236043, 0.413277619]
    """

    def __init__(self, aggregator=np.mean):
        self._params_to_attributes(AggregatingMetaRegressor.__init__, locals())
        self._estimator_type = 'regressor'

    def fit(self, X=None, y=None, **fit_params):
        return self

    def predict(self, X):
        return self.aggregator(X, axis=1)


class Ensemble(Cloneable, Saveable):
    """
    Model ensemble with optional in-pipeline model selection.

    This pipeline component makes inferences with a set of base predictors by
    either selecting the single best base predictor during fitting or by
    pooling base predictor inferences with a meta-predictor.

    The meta-predictor may be a voting or aggregating algorithm (e.g.
    SoftVotingMetaClassifier, AggregatingMetaRegressor) or a scikit-learn conformant ML
    algorithm.  In the latter case, it is standard practice to use internal
    cross validation training of the base classifiers to prevent them from
    making inferences on training samples (1).  To activate internal cross
    validation training, set the internal_cv constructor argument of Ensemble
    (cross validation is only used to generate outputs for meta-predictor
    training; the whole training set is always used to train the final base
    predictor models).

    Ensemble also takes advantage of internal cross validation
    to enable in-pipeline screening of base predictors during model
    fitting. To enable model selection, provide a score_selector (e.g.
    those found in the :mod:`pipecaster.score_selection` module) to the
    contructor.

    (1) Wolpert, David H. "Stacked generalization."
    Neural networks 5.2 (1992): 241-259.

    Parameters
    ----------
    base_predictors : predictor instance or list of predictor instances
        Ensemble of scikit-learn conformant base predictors (either all
        classifiers or all regressors).
    meta_predictor : predictor instance, default=None
        Scikit-learn conformant classifier or regressor that makes predictions
        from the base predictor inferences.  This parameter is optional when
        the internal_cv and score_selector parameters are set, in which case
        predictions from the top performing model will be used in the absence
        of a meta-predictor.
    base_transform_methods : str or list, default='auto'
        - Set the name of the base predictor methods to use for generating
          meta-features.
        - if 'auto' :
            - If classifier : Method picked using
              config.transform_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict)
            - If regressor : 'predict'
        - If str other than 'auto' : Name is broadcast over all base
          predictors.
        - If list : List of names or 'auto', one per base predictor.
    internal_cv : int, None, or callable, default=None
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int > 1: StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If {None, 1}: disable internal cv.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    base_score_methods : {str, list}, default='auto'
        - Name or names of prediction method(s) used when scoring predictor
          performance.
        - if 'auto' :
            - If classifier : Method picked using
              config.score_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict)
            - If regressor : 'predict'
        - If str other than 'auto' : Name is broadcast over all base
          predictors.
        - If list : List of names or 'auto', one per base predictor.
    scorer : {callable, 'auto'}, default='auto'
        - Method for calculating performance scores.
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer that returns a scalar figure of merit score
          with signature: score = scorer(y_true, y_pred).
      score_selector : callable or None, default=None
        - Method for selecting models from the ensemble.
        - If callable : Selector with signature:
          selected_indices = callable(scores).
        - If None :  All models will be retained in the ensemble if there is
          a meta-predictor, otherwise RankScoreSelector(k=1) is used.
    disable_cv_train : bool, default=False
        - If False : cv predictions will be used to train the meta-predictor.
        - If True : cv predictions not used to train the meta-predictor.
    base_processes : int or 'max', default=1
        - The number of parallel processes to run for base predictor fitting.
        - If int : Use up to base_processes number of processes.
        - If 'max' : Use all available CPUs.
    cv_processes : int or 'max', default=1
        - The number of parallel processes to run for internal cross
          validation.
        - If int : Use up to cv_processes number of processes.
        - If 'max' : Use all available CPUs.

    Examples
    --------
    Ensemble voting:
    ::

        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import Pipeline
        import pipecaster as pc

        X, y = make_classification(n_classes=2, n_samples=500, n_features=100,
                                   n_informative=5, class_sep=0.6)

        predictors = [MLPClassifier(), LogisticRegression(),
                      KNeighborsClassifier(),  GradientBoostingClassifier(),
                      RandomForestClassifier(), GaussianNB()]

        ensemble_clf = pc.Ensemble(
                         base_predictors=predictors,
                         meta_predictor=pc.SoftVotingMetaClassifier(),
                         base_processes='max')

        clf = Pipeline([('scaler', StandardScaler()),
                        ('ensemble_clf', ensemble_clf)])

        pc.cross_val_score(clf, X, y)
        # output: [0.8142570281124498, 0.8262335054503729, 0.8429152148664344]

    Ensemble voting with model selection:
    ::

        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import Pipeline
        import pipecaster as pc

        X, y = make_classification(n_classes=2, n_samples=500, n_features=100,
                                   n_informative=5, class_sep=0.6)

        predictors = [MLPClassifier(), LogisticRegression(),
                      KNeighborsClassifier(),  GradientBoostingClassifier(),
                      RandomForestClassifier(), GaussianNB()]

        ensemble_clf = pc.Ensemble(
                         base_predictors=predictors,
                         meta_predictor=pc.SoftVotingMetaClassifier(),
                         internal_cv=5, scorer='auto',
                         score_selector=pc.RankScoreSelector(k=2),
                         disable_cv_train=True, base_processes='max')

        clf = Pipeline([('scaler', StandardScaler()),
                        ('ensemble_clf', ensemble_clf)])

        pc.cross_val_score(clf, X, y)
        # output: [0.8625215146299483, 0.8440189328743546, 0.8373493975903614]

        clf.fit(X, y)
        # Models selected by the Ensemble:
        ensemble_clf = clf.named_steps['ensemble_clf']
        [p for i, p in enumerate(predictors)
         if i in ensemble_clf.get_support()]
        # output: [GradientBoostingClassifier(), RandomForestClassifier()]

    Stacked generalization:
    ::

        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        import pipecaster as pc

        X, y = make_classification(n_classes=2, n_samples=500, n_features=100,
                                   n_informative=5, class_sep=0.6)

        predictors = [MLPClassifier(), LogisticRegression(),
                      KNeighborsClassifier(),  GradientBoostingClassifier(),
                      RandomForestClassifier(), GaussianNB()]

        ensemble_clf = pc.Ensemble(
                         base_predictors=predictors,
                         meta_predictor=SVC(),
                         internal_cv=5,
                         disable_cv_train=False,
                         base_processes='max')

        clf = Pipeline([('scaler', StandardScaler()),
                        ('ensemble_clf', ensemble_clf)])

        pc.cross_val_score(clf, X, y, score_method='predict')
        # output: [0.7541594951233506, 0.7360154905335627, 0.7289156626506024]

    Model selection (no meta-prediction):
    ::

        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import Pipeline
        import pipecaster as pc

        X, y = make_classification(n_classes=2, n_samples=500, n_features=100,
                                   n_informative=5, class_sep=0.6)

        predictors = [MLPClassifier(), LogisticRegression(),
                      KNeighborsClassifier(),  GradientBoostingClassifier(),
                      RandomForestClassifier(), GaussianNB()]

        ensemble_clf = pc.Ensemble(
                         base_predictors=predictors,
                         meta_predictor=None,
                         internal_cv=5,
                         scorer='auto',
                         base_processes='max')

        clf = Pipeline([('scaler', StandardScaler()),
                        ('ensemble_clf', ensemble_clf)])

        pc.cross_val_score(clf, X, y)
        # output: [0.8443775100401607, 0.7972022955523672, 0.8617886178861789]

        # inspect models selected by Ensemble
        clf.fit(X, y)
        ensemble_clf = clf.named_steps['ensemble_clf']
        selected_models = [p for i, p in enumerate(predictors)
            if i in ensemble_clf.get_support()]
        selected_models
        # output: [GradientBoostingClassifier()]
    """

    def __init__(self, base_predictors, meta_predictor=None,
                 base_transform_methods='auto',
                 internal_cv=None,
                 base_score_methods='auto',
                 scorer='auto',
                 score_selector=None,
                 disable_cv_train=False,
                 base_processes=1, cv_processes=1):
        self._params_to_attributes(Ensemble.__init__, locals())

        if isinstance(base_predictors, (tuple, list, np.ndarray)):
            for predictor in base_predictors:
                utils.enforce_fit(predictor)
                utils.enforce_predict(predictor)
        else:
            utils.enforce_fit(base_predictors)
            utils.enforce_predict(base_predictors)

        if internal_cv is None and score_selector is not None:
            raise ValueError('Must choose an internal cv method when channel '
                             'selection is activated')

        if meta_predictor is None and score_selector is None:
            self.score_selector = RankScoreSelector(k=1)

        # expose availbable predictor interface (may change after fit)
        if meta_predictor is not None:
            utils.enforce_fit(meta_predictor)
            utils.enforce_predict(meta_predictor)
            self._set_estimator_type(meta_predictor)
            self._add_predictor_interface(meta_predictor)
        else:
            if isinstance(base_predictors, (tuple, list, np.ndarray)):
                for predictor in base_predictors:
                    self._set_estimator_type(predictor)
                    self._add_predictor_interface(predictor)
            else:
                self._set_estimator_type(base_predictors)
                self._add_predictor_interface(base_predictors)

    def _set_estimator_type(self, predictor):
        if hasattr(predictor, '_estimator_type') is True:
            self._estimator_type = predictor._estimator_type
        else:
            if predictor._estimator_type != self._estimator_type:
                print('All estimators in ensumble must of same type.')

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

    @staticmethod
    def _fit_job(predictor, X, y, internal_cv, transform_method,
                 cv_processes, score_method, scorer, fit_params):
        model = utils.get_clone(predictor)
        model = transform_wrappers.SingleChannel(model,
                                                 transform_method)
        predictions = model.fit_transform(X, y, **fit_params)

        cv_predictions, score = None, None
        if internal_cv is not None:
            predict_methods = list(set([transform_method, score_method]))
            model_cv = utils.get_clone(predictor)
            model_cv = transform_wrappers.SingleChannelCV(
                                                model_cv, transform_method,
                                                internal_cv, score_method,
                                                scorer, cv_processes)
            cv_predictions = model_cv.fit_transform(X, y, **fit_params)
            score = model_cv.score_

        return model, predictions, cv_predictions, score

    def fit(self, X, y=None, **fit_params):

        if self._estimator_type == 'classifier' and y is not None:
            self.classes_, y = np.unique(y, return_inverse=True)

        if isinstance(self.base_transform_methods, (tuple, list, np.ndarray)):
            if len(self.base_transform_methods) != len(self.base_predictors):
                raise utils.FitError('Number of base predict methods did not '
                                     'match number of base predictors.')
            transform_methods = self.base_transform_methods
        else:
            transform_methods = [self.base_transform_methods
                                 for p in self.base_predictors]

        if isinstance(self.base_score_methods, (tuple, list, np.ndarray)):
            if len(self.base_score_methods) != len(self.base_predictors):
                raise utils.FitError('Number of base predict methods did not '
                                     'match number of base predictors.')
            score_methods = self.base_score_methods
        else:
            score_methods = [self.base_score_methods
                             for p in self.base_predictors]

        args_list = [(p, X, y, self.internal_cv, tm, self.cv_processes,
                      sm, self.scorer, fit_params)
                     for p, tm, sm in zip(self.base_predictors,
                                          transform_methods,
                                          score_methods)]
        n_jobs = len(args_list)
        n_processes = 1 if self.base_processes is None else self.base_processes
        n_processes = (n_jobs
                       if (type(n_processes) == int and n_jobs < n_processes)
                       else n_processes)
        if n_processes == 'max' or n_processes > 1:
            try:
                shared_mem_objects = [X, y, self.cv_processes, self.scorer,
                                      fit_params]
                fit_results = parallel.starmap_jobs(
                                Ensemble._fit_job, args_list,
                                n_cpus=n_processes,
                                shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'
                      .format(e))
                print('defaulting to single processor')
                n_processes = 1

        if type(n_processes) == int and n_processes <= 1:
            fit_results = [Ensemble._fit_job(*args)
                           for args in args_list]

        models, predictions, cv_predictions, scores = zip(*fit_results)

        if scores is not None:
            self.scores_ = list(scores)

        if self.score_selector is not None:
            self.selected_indices_ = self.score_selector(scores)
        else:
            self.selected_indices_ = [
                i for i, p in enumerate(self.base_predictors)]

        if self.internal_cv is not None and self.disable_cv_train is False:
            predictions = cv_predictions

        self.base_models = [m for i, m in enumerate(models)
                            if i in self.selected_indices_]
        predictions = [p for i, p in enumerate(predictions)
                       if i in self.selected_indices_]

        if self.meta_predictor is None and len(self.selected_indices_) > 1:
           raise utils.FitError('A meta_predictor is required when more than '
                                'one base predictors is selected.')
        elif self.meta_predictor is None and len(self.selected_indices_) == 1:
            if hasattr(self.base_models[0], 'classes_'):
               self.classes_ = self.base_models[0].classes_
        elif self.meta_predictor is not None:
            meta_X = np.concatenate(predictions, axis=1)
            self.meta_model = utils.get_clone(self.meta_predictor)
            self.meta_model.fit(meta_X, y, **fit_params)
            if hasattr(self.meta_model, 'classes_'):
                self.classes_ = self.meta_model.classes_

        self._remove_predictor_interface()
        if hasattr(self, 'meta_model'):
            self._set_estimator_type(self.meta_model)
            self._add_model_interface(self.meta_model, meta_X)
        else:
            self._set_estimator_type(self.base_models[0])
            self._add_model_interface(self.base_models[0], X)

        return self

    def get_model_scores(self):
        if hasattr(self, 'scores_'):
            return self.scores_
        else:
            raise utils.FitError('Base model scores not found. They are only '
                                 'available after call to fit().')

    def get_support(self):
        if hasattr(self, 'selected_indices_'):
            return self.selected_indices_
        else:
            raise utils.FitError('Must call fit before getting selection '
                                 'information')

    def get_base_models(self):
        if hasattr(self, 'base_models') is False:
            raise FitError('No base models found. Call fit() or '
                           'fit_transform() to create base models')
        else:
            return self.base_models

    def predict_with_method(self, X, method_name):
        """
        Make ensemble predictions.

        Users will not generally call this method directly because available
        preditors are exposed through reflection for scikit-learn compliant
        prediction methods:
            - ensemble.predict()
            - ensemble.predict_proba()
            - ensemble.predict_log_proba()
            - ensemble.decision_function()

        Parameters
        ----------
        X: ndarray.shape(n_samples, n_features)
            Feature matrix.
        method_name: str
            Name of the meta-predictor prediction method to invoke.

        Returns
        -------
        Ensemble predictions.
            - If method_name is 'predict' or 'decision_function':
              ndarray(n_samples,)
            - If method_name is 'predict_proba', or
              'predict_log_proba': ndarray(n_samples, n_classes)
        """
        if hasattr(self, 'base_models') is False:
            raise utils.FitError('prediction attempted before model fitting')

        if self.meta_predictor is None:
            prediction_method = getattr(self.base_models[0], method_name)
            predictions = prediction_method(X)
        else:
            predictions_list = [m.transform(X) for m in self.base_models]
            meta_X = np.concatenate(predictions_list, axis=1)
            prediction_method = getattr(self.meta_model, method_name)
            predictions = prediction_method(meta_X)

        if self._estimator_type == 'classifier' and method_name == 'predict':
            predictions = self.classes_[predictions]
        return predictions

    def get_screen_results(self):
        df = pd.DataFrame({'model':self.base_predictors,
                           'performance':self.scores_,
                           'selections':['+++' if i in self.get_support()
                                         else '-' for i, p
                                         in enumerate(self.scores_)]})
        return df.set_index('model')

    def _more_tags(self):
        return {'multichannel': False}

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        if hasattr(self, 'scores_'):
            clone.scores_ = self.scores_.copy()
        if hasattr(self, 'selected_indices_'):
            clone.selected_indices_ = self.selected_indices_.copy()
        if hasattr(self, 'base_models'):
            clone.base_models = [utils.get_clone(m) if m is not None else None
                                 for m in self.base_models]
        if hasattr(self, 'meta_model'):
            clone.meta_model = utils.get_clone(self.meta_model)
        clone._add_predictor_interface(self)
        return clone


class GridSearchEnsemble(Ensemble):
    """
    Model ensemble with in-pipeline hyperparameter screening.

    This pipeline component makes inferences with a set of base predictors by
    either selecting the single best base predictor during fitting or by
    pooling base predictor inferences with a meta-predictor.  The base
    predictor ensemble consists of a single algorithm and an ensemble of
    hyperparameters.  GridSearchEnsemble uses internal cross validation to
    enable in-pipeline screening of hyperparameters during model fitting. To
    enable hyperparameter screening, provide a score_selector to the contructor
    (e.g. those found in the :mod:`pipecaster.score_selection` module).

    The optional meta-predictor may be a voting or aggregating
    algorithm (e.g. SoftVotingMetaClassifier, AggregatingMetaRegressor) or a
    scikit-learn conformant ML algorithm.  In the latter case, it is standard
    practice to use internal cross validation training of the base classifiers
    to prevent them from making inferences on training samples (1).  To
    activate internal cross validation training, set the internal_cv
    constructor argument of GridSearchEnsemble (cross validation is only used
    to generate outputs for meta-predictor training; the whole training set is
    always used to train the final base predictor models).

    (1) Wolpert, David H. "Stacked generalization."
    Neural networks 5.2 (1992): 241-259.

    Parameters
    ----------
    param_dict: dict, default=None
        Dict of parameters and values to be screened in a grid search
        (Cartesian product of the value lists).  E.g.:

        {'param1':[value1, value2, value3],
        'param2':[value1, value2, value3]}
    base_predictor_cls : class, default=None
        Predictor class to be used for base prediction.  Must implement the
        scikit-learn estimator and predictor interfaces.
    meta_predictor : predictor instance, default=None
        Scikit-learn conformant classifier or regressor that makes predictions
        from the base predictor inferences.  This parameter is optional when
        the internal_cv and score_selector parameters are set, in which case
        predictions from the top performing model will be used in the absence
        of a meta-predictor.
    base_transform_methods : str or list, default='auto'
        - Set the name of the base predictor methods to use for generating
          meta-features.
        - if 'auto' :
            - If classifier : Method picked using
              config.transform_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict)
            - If regressor : 'predict'
        - If str other than 'auto' : Name is broadcast over all base
          predictors.
        - If list : List of names or 'auto', one per base predictor.
    internal_cv : int, None, or callable, default=None
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int > 1: StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If {None, 1}: disable internal cv.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    base_score_methods : {str, list}, default='auto'
        - Name or names of prediction method(s) used when scoring predictor
          performance.
        - if 'auto' :
            - If classifier : Method picked using
              config.score_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict)
            - If regressor : 'predict'
        - If str other than 'auto' : Name is broadcast over all base
          predictors.
        - If list : List of names or 'auto', one per base predictor.
    scorer : {callable, 'auto'}, default='auto'
        - Method for calculating performance scores.
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer that returns a scalar figure of merit score
          with signature: score = scorer(y_true, y_pred).
      score_selector : callable or None, default=None
        - Method for selecting models from the ensemble.
        - If callable : Selector with signature:
          selected_indices = callable(scores).
        - If None :  All models will be retained in the ensemble if there is
          a meta-predictor, otherwise RankScoreSelector(k=1) is used.
    disable_cv_train : bool, default=False
        - If False : cv predictions will be used to train the meta-predictor.
        - If True : cv predictions not used to train the meta-predictor.
    base_processes : int or 'max', default=1
        - The number of parallel processes to run for base predictor fitting.
        - If int : Use up to base_processes number of processes.
        - If 'max' : Use all available CPUs.
    cv_processes : int or 'max', default=1
        - The number of parallel processes to run for internal cross
          validation.
        - If int : Use up to cv_processes number of processes.
        - If 'max' : Use all available CPUs.

    Examples
    --------
    ::

            from sklearn.datasets import make_classification
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score

            import pipecaster as pc

            screen = {
                 'learning_rate':[0.1, 10],
                 'n_estimators':[5, 25],
            }

            X, y = make_classification()
            clf = pc.GridSearchEnsemble(
                             param_dict=screen,
                             base_predictor_cls=GradientBoostingClassifier,
                             meta_predictor=pc.SoftVotingMetaClassifier(),
                             internal_cv=5, scorer='auto',
                             score_selector=pc.RankScoreSelector(k=2),
                             base_processes=pc.count_cpus())
            clf.fit(X, y)
            clf.get_screen_results()
            # output: (outputs a dataframe with the screen results)

            cross_val_score(clf, X, y, scoring='balanced_accuracy', cv=5)
            # output: array([0.9 , 0.85, 0.85, 0.85, 0.65])
    """

    def __init__(self, param_dict=None, base_predictor_cls=None,
                 meta_predictor=None,
                 base_transform_methods='auto',
                 internal_cv=5,
                 base_score_methods='auto',
                 scorer='auto',
                 score_selector=RankScoreSelector(k=3),
                 disable_cv_train=False,
                 base_processes=1, cv_processes=1):
        self._params_to_attributes(GridSearchEnsemble.__init__, locals())
        self.params_list_ = list(ParameterGrid(self.param_dict))
        base_predictors = [self.base_predictor_cls(**ps)
                                   for ps in self.params_list_]
        super().__init__(base_predictors, meta_predictor,
                         base_transform_methods, internal_cv,
                         base_score_methods, scorer, score_selector,
                         disable_cv_train, base_processes, cv_processes)

    def fit(self, X, y=None, **fit_params):

        super().fit(X, y, **fit_params)

    def get_results(self):
        """
        Get the results of the screen.

        Returns
        -------
        selections : list
            List containing '+' for selected paramters and '-' for unselected
            paramters in the order of params_list.
        params_list : list
            List of grid search parameter sets.
        scores : list
            List of performance scores for each set of parameters set in the
            order of params_list.
        """
        if hasattr(self, 'base_models') is True:
            selections = [True if i in self.selected_indices_ else False
                          for i, p in enumerate(self.params_list_)]
            return selections, self.params_list_, self.scores_

    def get_screen_results(self):
        """
        Get pandas DataFrame with screen results.
        """
        selections, params_list, scores = self.get_results()
        selections = ['+++' if s is True else '-' for s in selections]
        df = pd.DataFrame({'selections':selections,
                           'parameters':params_list, 'performance':scores})
        df.sort_values('performance', ascending=False, inplace=True)
        return df.set_index('parameters')

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'params_list'):
            clone.params_list = self.params_list.copy()

        return clone


class MultichannelPredictor(Cloneable, Saveable):
    """
    Predict with mutliple intput channels.

    This pipeline component concatenates mutliple inputs and into a single
    vector and uses it as input for a scikit-learn compatible predictor.
    Can be used for prediction or meta-prediction.

    Parameters
    ----------
    predictor : predictor instance
        Classifier or regressor that implements the scikit-learn estimator and
        predictor interfaces.

    Notes
    -----
    Class uses reflection to expose its prediction interface during
    initialization, so instances of this class will generally have more
    methods than the class itself.

    Examples
    --------
    Prediction with multiple input channels:
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.9117647058823529, 0.8768382352941176, 0.9099264705882353]

    Meta-prediction (stacked generalization):
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.make_cv_transformer(GradientBoostingClassifier())
        clf.add_layer(base_clf, pipe_processes='max')
        clf.add_layer(pc.MultichannelPredictor(SVC()))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.9117647058823529, 0.90625, 0.9099264705882353]
    """

    def __init__(self, predictor):
        self._params_to_attributes(MultichannelPredictor.__init__, locals())
        utils.enforce_fit(predictor)
        utils.enforce_predict(predictor)
        self._set_estimator_type(predictor)
        self._add_predictor_interface(predictor)

    def _set_estimator_type(self, predictor):
        if hasattr(predictor, '_estimator_type') is True:
            self._estimator_type = predictor._estimator_type
        else:
            if predictor._estimator_type != self._estimator_type:
                print('All estimators in ensumble must of same type.')

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

        self._set_estimator_type(self.model)
        self._remove_predictor_interface()
        self._add_model_interface(self.model, X)
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
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        if hasattr(self, 'model'):
            clone.model = utils.get_clone(self.model)
        clone._add_predictor_interface(self)
        return clone

    def get_descriptor(self, verbose=0, params=None):
        return utils.get_descriptor(self.predictor, verbose, params) + '_MC'


class ChannelEnsemble(Cloneable, Saveable):
    """
    Model ensemble that takes multiple input channels (one model per channel).

    This pipeline component makes inferences with a set of base predictors by
    either selecting the single best base predictor during fitting or by
    pooling base predictor inferences with a meta-predictor.

    The meta-predictor may be a voting or aggregating algorithm (e.g.
    SoftVotingMetaClassifier, AggregatingMetaRegressor) or a scikit-learn conformant ML
    algorithm.  In the latter case, it is standard practice to use internal
    cross validation training of the base classifiers to prevent them from
    making inferences on training samples (1).  To activate internal cross
    validation training, set the internal_cv constructor argument of
    ChannelEnsemble (cross validation is only used to generate outputs for
    meta-predictor training, the whole training set is always used to train the
    final base predictor models).

    ChannelEnsemble also takes advantage of internal cross validation
    to enable in-pipeline screening of base predictors during model
    fitting. To enable model selection, provide a score_selector (e.g.
    those found in the :mod:`pipecaster.score_selection` module) to the
    contructor.

    (1) Wolpert, David H. "Stacked generalization."
    Neural networks 5.2 (1992): 241-259.

    Parameters
    ----------
    base_predictors : {predictor instance, list}
        - Ensemble of scikit-learn conformant base predictors (either all
          classifiers or all regressors).
        - If predictor instance: Instance to be cloned and broadcast across
          all inputs.
        - If list: List with one predictor instance per channel.
    meta_predictor : predictor instance, default=None
        Scikit-learn conformant classifier or regressor that makes predictions
        from base predictor inferences.  This parameter is optional when
        internal_cv and score_selector parameters are set, in which case
        predictions from the top performing model will be used in the absence
        of a meta-predictor.
    base_transform_methods : str or list, default='auto'
        - Set the name of the base predictor methods to use for generating
          meta-features.
        - if 'auto' :
            - If classifier : Method picked using
              config.transform_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict)
            - If regressor : 'predict'
        - If str other than 'auto' : Name is broadcast over all base
          predictors.
        - If list : List of names or 'auto', one per base predictor.
    internal_cv : int, None, or callable, default=None
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int > 1: StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If {None, 1}: disable internal cv.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    base_score_methods : {str, list}, default='auto'
        - Name or names of prediction method(s) used when scoring predictor
          performance.
        - if 'auto' :
            - If classifier : Method picked using
              config.score_method_precedence order (default:
              predict_proba->predict_log_proba->decision_function->predict)
            - If regressor : 'predict'
        - If str other than 'auto' : Name is broadcast over all base
          predictors.
        - If list : List of names or 'auto', one per base predictor.
    scorer : {callable, 'auto'}, default='auto'
        - Method for calculating performance scores.
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer that returns a scalar figure of merit score
          with signature: score = scorer(y_true, y_pred).
      score_selector : callable or None, default=None
        - Method for selecting models from the ensemble.
        - If callable : Selector with signature:
          selected_indices = callable(scores).
        - If None :  All models will be retained in the ensemble if there is
          a meta-predictor, otherwise RankScoreSelector(k=1) is used.
    disable_cv_train : bool, default=False
        - If False : cv predictions will be used to train the meta-predictor.
        - If True : cv predictions not used to train the meta-predictor.
    base_processes : int or 'max', default=1
        - The number of parallel processes to run for base predictor fitting.
        - If int : Use up to base_processes number of processes.
        - If 'max' : Use all available CPUs.
    cv_processes : int or 'max', default=1
        - The number of parallel processes to run for internal cross
          validation.
        - If int : Use up to cv_processes number of processes.
        - If 'max' : Use all available CPUs.

    Examples
    --------
    Channel Voting:
    ::

        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.make_transformer(GradientBoostingClassifier())
        meta_clf = pc.SoftVotingMetaClassifier()
        clf.add_layer(pc.ChannelEnsemble(base_clf, meta_clf),
                      pipe_processes='max')

        pc.cross_val_score(clf, Xs, y, score_method='predict')
        # output: [0.85294117647, 0.821691176470, 0.909926470]

    Channel voting with model selection:
    ::

        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.make_transformer(GradientBoostingClassifier())
        meta_clf = pc.SoftVotingMetaClassifier()
        clf.add_layer(pc.ChannelEnsemble(base_clf, meta_clf, internal_cv=3,
                                     score_selector=pc.RankScoreSelector(k=3)),
                      pipe_processes='max')

        pc.cross_val_score(clf, Xs, y, score_method='predict')
        # output: [0.8235294117647058, 0.8455882352941176, 0.8768382352941176]

        clf.fit(Xs, y)
        df = clf.get_model(0, 0).get_screen_results()
        df['input'] = X_types
        df
        # output: DataFrame with screen results compared to informative inputs

    Stacked Generalization:
    ::

        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.make_transformer(GradientBoostingClassifier())
        meta_clf = SVC()
        clf.add_layer(pc.ChannelEnsemble(base_clf, meta_clf, internal_cv=3),
                      pipe_processes='max')

        pc.cross_val_score(clf, Xs, y, score_method='predict')
        # output: [0.9117647058823529, 0.8823529411764706, 0.96875]

    Model selection without ensemble prediction:
    ::

        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.make_transformer(GradientBoostingClassifier())
        clf.add_layer(pc.ChannelEnsemble(base_clf, internal_cv=3,
                                         score_selector=pc.RankScoreSelector(k=1)),
                      pipe_processes='max')

        pc.cross_val_score(clf, Xs, y, score_method='predict')
        # output: [0.823529411, 0.8768382352, 0.75919117647]

        clf.fit(Xs, y)
        df = clf.get_model(0, 0).get_screen_results()
        df['input'] = X_types
        df
        # output: DataFrame with screen results compared to informative inputs
    """

    def __init__(self, base_predictors, meta_predictor=None,
                 base_transform_methods='auto',
                 internal_cv=None,
                 base_score_methods='auto',
                 scorer='auto',
                 score_selector=None,
                 disable_cv_train=False,
                 base_processes=1, cv_processes=1):
        self._params_to_attributes(ChannelEnsemble.__init__, locals())

        if isinstance(base_predictors, (tuple, list, np.ndarray)):
            for predictor in base_predictors:
                utils.enforce_fit(predictor)
                utils.enforce_predict(predictor)
        else:
            utils.enforce_fit(base_predictors)
            utils.enforce_predict(base_predictors)

        if internal_cv is None and score_selector is not None:
            raise ValueError('Must choose an internal cv method when channel '
                             'selection is activated')

        if meta_predictor is None and score_selector is None:
            self.score_selector = RankScoreSelector(k=1)

        # expose available predictor interface (may change after fit)
        if meta_predictor is not None:
            utils.enforce_fit(meta_predictor)
            utils.enforce_predict(meta_predictor)
            self._set_estimator_type(meta_predictor)
            self._add_predictor_interface(meta_predictor)
        else:
            if isinstance(base_predictors, (tuple, list, np.ndarray)):
                for predictor in base_predictors:
                    self._set_estimator_type(predictor)
                    self._add_predictor_interface(predictor)
            else:
                self._set_estimator_type(base_predictors)
                self._add_predictor_interface(base_predictors)

    def _set_estimator_type(self, predictor):
        if hasattr(predictor, '_estimator_type') is True:
            self._estimator_type = predictor._estimator_type
        else:
            if predictor._estimator_type != self._estimator_type:
                print('All estimators in ensumble must of same type.')

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

    @staticmethod
    def _fit_job(predictor, X, y, transform_method, internal_cv,
                 score_method, scorer, cv_processes, fit_params):
        if X is None:
            return None, None, None, None
        model = utils.get_clone(predictor)
        model = transform_wrappers.SingleChannel(model,
                                                 transform_method)
        predictions = model.fit_transform(X, y, **fit_params)

        cv_predictions, score = None, None
        if internal_cv is not None:
            model_cv = utils.get_clone(predictor)
            model_cv = transform_wrappers.SingleChannelCV(
                                                model_cv, transform_method,
                                                internal_cv, score_method,
                                                scorer, cv_processes)
            cv_predictions = model_cv.fit_transform(X, y, **fit_params)
            score = model_cv.score_

        return model, predictions, cv_predictions, score

    def fit(self, Xs, y=None, **fit_params):

        if self._estimator_type == 'classifier' and y is not None:
            self.classes_, y = np.unique(y, return_inverse=True)

        if isinstance(self.base_predictors, (tuple, list, np.ndarray)):
            if len(self.base_predictors) != len(Xs):
                raise utils.FitError('Number of base predictors did not match '
                                     'number of input channels')
            predictors = self.base_predictors
        else:
            predictors = [self.base_predictors for X in Xs]

        if isinstance(self.base_transform_methods, (tuple, list, np.ndarray)):
            if len(self.base_transform_methods) != len(Xs):
                raise utils.FitError('Number of base predict methods did not '
                                     'match number of input channels')
            transform_methods = self.base_transform_methods
        else:
            transform_methods = [self.base_transform_methods for X in Xs]

        if isinstance(self.base_score_methods, (tuple, list, np.ndarray)):
            if len(self.base_score_methods) != len(Xs):
                raise utils.FitError('Number of base score methods did not '
                                     'match number of input channels')
            score_methods = self.base_score_methods
        else:
            score_methods = [self.base_score_methods for X in Xs]

        args_list = [(p, X, y, tm, self.internal_cv, sm, self.scorer,
                      self.cv_processes, fit_params)
                     for p, X, tm, sm in zip(predictors, Xs,
                                             transform_methods,
                                             score_methods)]

        n_jobs = len(args_list)
        n_processes = 1 if self.base_processes is None else self.base_processes
        n_processes = (n_jobs
                       if (type(n_processes) == int and n_jobs < n_processes)
                       else n_processes)
        if n_processes == 'max' or n_processes > 1:
            try:
                shared_mem_objects = [y, self.internal_cv, self.cv_processes,
                                      self.scorer, fit_params]
                fit_results = parallel.starmap_jobs(
                                ChannelEnsemble._fit_job, args_list,
                                n_cpus=n_processes,
                                shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'
                      .format(e))
                print('defaulting to single processor')
                n_processes = 1

        if type(n_processes) == int and n_processes <= 1:
            fit_results = [ChannelEnsemble._fit_job(*args)
                           for args in args_list]

        models, predictions, cv_predictions, scores = zip(*fit_results)
        self.base_models = models
        if scores is not None:
            self.scores_ = list(scores)

        if self.score_selector is not None:
            self.selected_indices_ = self.score_selector(scores)
        else:
            self.selected_indices_ = [i for i, X in enumerate(Xs)
                                      if X is not None]

        if self.internal_cv is not None and self.disable_cv_train is False:
            predictions = cv_predictions

        self.base_models = [m if i in self.selected_indices_ else None
                            for i, m in enumerate(self.base_models)]
        predictions = [p for i, p in enumerate(predictions)
                       if i in self.selected_indices_]

        if self.meta_predictor is None and len(self.selected_indices_) > 1:
            raise utils.FitError('A meta_predictor is required when more than '
                                 'one base predictors is selected.')
        elif self.meta_predictor is None and len(self.selected_indices_) == 1:
            model = self.base_models[self.selected_indices_[0]]
            if hasattr(model, 'classes_'):
               self.classes_ = model.classes_
        elif self.meta_predictor is not None:
            meta_X = np.concatenate(predictions, axis=1)
            self.meta_model = utils.get_clone(self.meta_predictor)
            self.meta_model.fit(meta_X, y, **fit_params)
            if hasattr(self.meta_model, 'classes_'):
                self.classes_ = self.meta_model.classes_

        self._remove_predictor_interface()
        if hasattr(self, 'meta_model'):
            self._set_estimator_type(self.meta_model)
            self._add_model_interface(self.meta_model, meta_X)
        else:
            model = self.base_models[self.selected_indices_[0]]
            X = Xs[self.selected_indices_[0]]
            self._set_estimator_type(model)
            self._add_model_interface(model, X)

        return self

    def get_model_scores(self):
        if hasattr(self, 'scores_'):
            return self.scores_
        else:
            raise utils.FitError('Base model scores not found. They are only '
                                 'available after call to fit().')

    def get_support(self):
        if hasattr(self, 'selected_indices_'):
            return self.selected_indices_
        else:
            raise utils.FitError('Must call fit before getting selection '
                                 'information')

    def get_base_models(self):
        if hasattr(self, 'base_models') is False:
            raise FitError('No base models found. Call fit() or '
                           'fit_transform() to create base models')
        else:
            return self.base_models

    def predict_with_method(self, Xs, method_name):
        """
        Make channel ensemble predictions with specified method.

        Users will not generally call this method directly because available
        preditors are exposed through reflection for scikit-learn compliant
        prediction methods:
            - ensemble.predict()
            - ensemble.predict_proba()
            - ensemble.predict_log_proba()
            - ensemble.decision_function()

        Parameters
        ----------
        Xs: list of (ndarray.shape(n_samples, n_features) or None)
            List of input feature matrices (or None spaceholders).
        method_name: str
            Name of the meta-predictor prediction method to invoke.

        Returns
        -------
        Ensemble predictions.
            - If method_name is 'predict': ndarray(n_samples,)
            - If method_name is 'predict_proba', 'decision_function', or
              'predict_log_proba': ndarray(n_samples, n_classes)
        """
        if hasattr(self, 'base_models') is False:
            raise utils.FitError('prediction attempted before model fitting')

        if self.meta_predictor is None:
            sel_idx = self.selected_indices_[0]
            prediction_method = getattr(self.base_models[sel_idx], method_name)
            predictions = prediction_method(Xs[sel_idx])
        else:
            predictions_list = [m.transform(X) for i, (m, X) in
                                enumerate(zip(self.base_models, Xs))
                                if i in self.selected_indices_]
            meta_X = np.concatenate(predictions_list, axis=1)
            prediction_method = getattr(self.meta_model, method_name)
            predictions = prediction_method(meta_X)

        if self._estimator_type == 'classifier' and method_name == 'predict':
            predictions = self.classes_[predictions]
        return predictions

    def get_screen_results(self):
        df = pd.DataFrame({'performance':self.scores_,
                             'selections':['+++' if i in self.get_support()
                                           else '-' for i, p
                                           in enumerate(self.scores_)]})
        df.index.name='channel'
        return df

    def _more_tags(self):
        return {'multichannel': False}

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        if hasattr(self, 'scores_'):
            clone.scores_ = self.scores_.copy()
        if hasattr(self, 'selected_indices_'):
            clone.selected_indices_ = self.selected_indices_.copy()
        if hasattr(self, 'base_models'):
            clone.base_models = [utils.get_clone(m) if m is not None else None
                                 for m in self.base_models]
        if hasattr(self, 'meta_model'):
            clone.meta_model = utils.get_clone(self.meta_model)
        clone._add_predictor_interface(self)
        return clone
