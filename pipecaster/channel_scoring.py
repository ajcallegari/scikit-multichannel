"""
Scorers that estimate the predictive value of a feature matrix.

Signature:
    score = channel_scorer(X, y)
"""

import numpy as np

from sklearn.metrics import explained_variance_score, balanced_accuracy_score
from sklearn.feature_selection import f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from pipecaster.cross_validation import cross_val_score, score_predictions
from pipecaster.utils import Cloneable, Saveable
import pipecaster.utils as utils

__all__ = ['AggregateFeatureScorer', 'CvPerformanceScorer']


class AggregateFeatureScorer(Cloneable, Saveable):
    """
    Channel scorer that computes an aggregate feature score.

    Callable class that computes features scores using a feature_scorer object
    then computes an aggregate matrix score from the set of feature scores
    using an aggregator object (e.g. np.mean, np.median, np.sum).

    Parameters
    ----------
    feature_scorer : callable
        Scorer that returns figure of merit scores for individual features with
        the signature: scores = scorer(y_true, y_pred)
    aggregator : callable
        Callable that generates a scalar aggregate figure of merit score
        from individual features scores (e.g. np.mean) with the signature:
        score = aggregator(scores)

    Examples
    --------
    ::

        import numpy as np
        import pipecaster as pc
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import f_classif

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=10)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(pc.ChannelSelector(
                        channel_scorer=pc.AggregateFeatureScorer(f_classif,
                                                                 np.mean),
                        score_selector=pc.RankScoreSelector(3)))
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
        pc.cross_val_score(clf, Xs, y)
        # output: [0.9705882352941176, 0.9117647058823529, 0.9411764705882353]
    """
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum):
        self._params_to_attributes(AggregateFeatureScorer.__init__, locals())

    def __call__(self, X, y):
        """
        Get an aggregate feature score.

        Parameters
        ----------
        X: ndarray.shape(n_samples, n_features)
            Feature matrix.
        y: list or ndarray.shape(n_samples,)
            Supervised machine learning targets.
        """
        if X is None:
            return None
        else:
            score_func_ret = self.feature_scorer(X, y)
            if isinstance(score_func_ret, (list, tuple)):
                scores = np.array(score_func_ret[0]).astype(float)
            else:
                scores = np.array(score_func_ret).astype(float)
            return self.aggregator(scores)


class CvPerformanceScorer(Cloneable, Saveable):
    """
    Channel scorer that computes performance of a predictor probe using cross
    validation.

    Callable class that estimates the predictive value of a feature matrix
    using a machine learning probe and cross validation.

    Parameters
    ----------
    predictor_probe : predictor
        Scikit-learn estimator/predictor, usually with low complexity and
        high speed, used to estimate the predictive value of a feature matrix.
    cv : None, int, or callable, default=5
        - Set the cross validation method:
        - If 1 : Internal cv training is inactivated.
        - If int > 1: StratifiedKFold(n_splits=internal_cv) for classifiers and
          KFold(n_splits=internal_cv) for regressors.
        - If None : The default value of 5 is used.
        - If callable : Assumes interface like scikit-learn KFold.
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
        - Set the number of processes used during cross validation:
        - If 1 : Run all split computations in a single process.
        - If 'max' : Run each split in a different process, using all available
          CPUs.
        - If int > 1 : Run each split in a different process, using up to
          cv_processes number of CPUs.

    Examples
    --------
    ::

        import numpy as np
        import pipecaster as pc
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import f_classif

        probe = GradientBoostingClassifier(n_estimators=5)

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=10)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(pc.ChannelSelector(
                        channel_scorer=pc.CvPerformanceScorer(probe),
                        score_selector=pc.RankScoreSelector(3)))
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
        pc.cross_val_score(clf, Xs, y)
        # output: [0.9117647058823529, 0.9393382352941176, 0.9393382352941176]

    """

    def __init__(self, predictor_probe, cv=5,
                 score_method='auto', scorer='auto',
                 cv_processes=1):
        self._params_to_attributes(CvPerformanceScorer.__init__, locals())

    def __call__(self, X, y, **fit_params):
        """
        Get figure of merit score.

        Parameters
        ----------
        X: ndarray.shape(n_samples, n_features)
            Feature matrix.
        y: list/array of length n_samples, default=None
            Targets for supervised ML.
        fit_params: dict, defualt=None
            Auxiliary parameters to pass to the fit method of the probe.
        """
        if X is None:
            return None
        else:
            scores = cross_val_score(self.predictor_probe, X, y,
                                     score_method=self.score_method,
                                     scorer=self.scorer,
                                     cv=self.cv, n_processes=self.cv_processes,
                                     **fit_params)
            return np.mean(scores)
