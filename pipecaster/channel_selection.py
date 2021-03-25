"""
Components that select channels to send to next pipeline stage.

Selections are made using channel scorers and score selectors like those found
in: :mod:`pipecaster.channel_scoring` and :mod:`pipecaster.score_selection`.
"""

import numpy as np
import ray
import functools

from sklearn.metrics import explained_variance_score, balanced_accuracy_score
from sklearn.feature_selection import f_classif

import pipecaster.utils as utils
from pipecaster.utils import Cloneable, Saveable
from pipecaster.score_selection import RankScoreSelector
from pipecaster.channel_scoring import AggregateFeatureScorer
from pipecaster.channel_scoring import CvPerformanceScorer
import pipecaster.parallel as parallel
from pipecaster.cross_validation import cross_val_score
import pipecaster.transform_wrappers as transform_wrappers
from pipecaster.score_selection import RankScoreSelector, PctRankScoreSelector
from pipecaster.score_selection import HighPassScoreSelector
from pipecaster.score_selection import VarianceHighPassScoreSelector

__all__ = ['ChannelSelector', 'SelectKBestScores', 'SelectPercentBestScores',
           'SelectHighPassScores', 'SelectVarianceHighPassScores',
           'SelectKBestProbes', 'SelectPercentBestProbes',
           'SelectHighPassProbes', 'SelectVarianceHighPassProbes']


class ChannelSelector(Cloneable, Saveable):
    """
    Select channels using channel_scorer and score_selector objects.

    Parameters
    ----------
    channel_scorer : callable, default=None
        Callable that provide a figure of merit score for a channel with the
        signature:  score = channel_scorer(X, y).
    score_selector : callable, default=None
        Callable that returns a list of indices of selected channels with the
        signature: selected_indices = score_selector(scores).
    channel_processes : int or 'max', default=1
        - Set the number of processes used during training:
        - If 1: Run all channel computations in a single process.
        - If 'max': Run each channel in a different process, using all
          available CPUs.
        - If int > 1: Run each channel in a different process, using up to
          channel_processes number of CPUs.

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
        # output: [0.9117647058823529, 0.8786764705882353, 0.9375]
    """

    def __init__(self, channel_scorer=None, score_selector=None,
                 channel_processes=1):
        self._params_to_attributes(ChannelSelector.__init__, locals())

    def _get_channel_score(channel_scorer, X, y, fit_params):
        if X is None:
            return None
        return channel_scorer(X, y, **fit_params)

    def fit(self, Xs, y=None, **fit_params):
        """
        Fit pipes and select channels.

        Parameters
        ----------
        Xs : list
            List of input feature matrices (or None for dead channels).
        y : list/array, default=None
            Optional targets for supervised ML.
        fit_params : dict, defualt=None
            Auxiliary parameters to pass to the fit method of the predictors.

        Returns
        -------
        self
        """
        args_list = [(self.channel_scorer, X, y, fit_params) for X in Xs]
        n_processes = self.channel_processes
        if n_processes is not None and n_processes > 1:
            try:
                shared_mem_objects = [y, fit_params]
                self.channel_scores_ = parallel.starmap_jobs(
                                 ChannelSelector._get_channel_score, args_list,
                                 n_cpus=n_processes,
                                 shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'
                      .format(e))
                print('defaulting to single processor')
                n_processes = 1
        if n_processes is None or n_processes <= 1:
            self.channel_scores_ = [self.channel_scorer(X, y, **fit_params)
                                    if X is not None else None for X in Xs]

        self.selected_indices_ = self.score_selector(self.channel_scores_)
        return self

    def get_channel_scores(self):
        """
        Get list of figure of merit scores (one per channel).
        """
        if hasattr(self, 'channel_scores_'):
            return self.channel_scores_
        else:
            raise utils.FitError('Channel scores not found. They are only \
                                 available after call to fit().')

    def get_support(self):
        """
        Get indices of channels selected during fitting.
        """
        if hasattr(self, 'selected_indices_'):
            return self.selected_indices_
        else:
            raise utils.FitError('Must call fit before getting selection \
                                 information')

    def transform(self, Xs):
        """
        Pass through selected matrices only.

        Parameters
        ----------
        Xs : list
            List of input feature matrices (or None value placeholders).

        Returns
        -------
        Xs_t : list
            List containing selected feature matrices and None value
            placeholders for matrices that were not selected.
        """
        return [Xs[i] if (i in self.selected_indices_) else None
                for i in range(len(Xs))]

    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)

    def get_selection_indices(self):
        return self.selected_indices_

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'selected_indices_'):
            clone.selected_indices_ = self.selected_indices_.copy()
        if hasattr(self, 'channel_scores_'):
            clone.channel_scores_ = self.channel_scores_.copy()
        return clone


class SelectKBestScores(ChannelSelector):
    """
    Select fixed number of channels based on aggregate feature score.

    Parameters
    ----------
    feature_scorer : callable, default=None
        Callable that returns a scalar figure of merit score for each feature
        with signature: scores = feature_scorer(X, y).
    aggregator : callable, default=np.sum
        Function that converts feature scores into a single aggregate matrix
        score with signature: matrix_score = aggregator(features_scores).
    k: int, default=1
        The number of channels to select.  Selected channels are those with
        the highest matrix_scores.
    channel_processes : int or 'max', default=1
        - Set the number of processes used during training:
        - If 1: Run all channel computations in a single process.
        - If 'max': Run each channel in a different process, using all
          available CPUs.
        - If int > 1: Run each channel in a different process, using up to
          channel_processes number of CPUs.

    Examples
    --------
    ::

        import numpy as np
        import pipecaster as pc
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import f_classif

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        clf.add_layer(pc.SelectKBestScores(f_classif, np.sum, 3))
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))

        clf.fit(Xs, y)
        selections = clf.get_model(1,0).get_support()
        # show selected input types (random or informative)
        [t for i, t in enumerate(X_types) if i in selections]
        # output: ['informative', 'informative', 'informative']

        pc.cross_val_score(clf, Xs, y)
        # output: [1.0, 0.96875, 1.0]
    """
    def __init__(self, feature_scorer=None, aggregator=np.sum, k=1,
                 channel_processes=1):
        self._params_to_attributes(SelectKBestScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator),
                         RankScoreSelector(k), channel_processes)


class SelectPercentBestScores(ChannelSelector):
    """
    Select percentage of channels based on aggregate feature score.

    Parameters
    ----------
    feature_scorer : callable, default=None
        Callable that returns a scalar figure of merit score for each feature
        with signature: scores = feature_scorer(X, y).
    aggregator: callable, default=np.sum
        Function that converts feature scores into a single aggregate matrix
        score with signature: matrix_score = aggregator(features_scores).
    pct : float
        The percentage of channels to select.
    channel_processes : int or 'max', default=1
        - Set the number of processes used during training:
        - If 1: Run all channel computations in a single process.
        - If 'max': Run each channel in a different process, using all
          available CPUs.
        - If int > 1: Run each channel in a different process, using up to
          channel_processes number of CPUs.

    Examples
    --------
    ::

        import numpy as np
        import pipecaster as pc
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import f_classif

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        clf.add_layer(pc.SelectPercentBestScores(f_classif, np.sum, 30))
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
        clf.fit(Xs, y)

        # show selected input types (random or informative)
        selections = clf.get_model(1,0).get_support()
        [t for i, t in enumerate(X_types) if i in selections]
        # output: ['informative', 'informative', 'informative']

        pc.cross_val_score(clf, Xs, y)
        # output: [0.9117647058823529, 0.9080882352941176, 0.8492647058823529]
    """
    def __init__(self, feature_scorer=None, aggregator=np.sum, percent=33,
                 channel_processes=1):
        self._params_to_attributes(SelectPercentBestScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator),
                         PctRankScoreSelector(percent), channel_processes)


class SelectHighPassScores(ChannelSelector):
    """
    Select channels with aggregate features score above an absolute cutoff.

    Parameters
    ----------
    feature_scorer : callable, default=None
        Callable that returns a scalar figure of merit score for each feature
        with signature: scores = feature_scorer(X, y).
    aggregator : callable, default=np.sum
        Function that converts feature scores into a single aggregate matrix
        score with signature: matrix_score = aggregator(features_scores).
    cutoff : float, default=0.0
        Items with scores above this value are selected.
    n_min : int, default=1
        The minimum number of items to select.  Takes precedence over cutoff.
    channel_processes : int or 'max', default=1
        - Set the number of processes used during training:
        - If 1: Run all channel computations in a single process.
        - If 'max': Run each channel in a different process, using all
          available CPUs.
        - If int > 1: Run each channel in a different process, using up to
          channel_processes number of CPUs.

    Examples
    --------
    ::

        import numpy as np
        import pipecaster as pc
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import f_classif

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        clf.add_layer(pc.SelectHighPassScores(f_classif, cutoff=100, n_min=1))
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))

        clf.fit(Xs, y)
        selections = clf.get_model(1,0).get_support()
        # show selected input types (random or informative)
        [t for i, t in enumerate(X_types) if i in selections]
        # output: ['informative', 'informative', 'informative']

        pc.cross_val_score(clf, Xs, y)
        # output: [0.8823529411, 0.9099264705, 0.941176470]
    """
    def __init__(self, feature_scorer=None, aggregator=np.sum, cutoff=0,
                 n_min=1, channel_processes=1):
        self._params_to_attributes(SelectHighPassScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator),
                         HighPassScoreSelector(cutoff, n_min),
                         channel_processes)


class SelectVarianceHighPassScores(ChannelSelector):
    """
    Select channels with aggregate feature scores above a relative cutoff.

    Computes an aggregate feature score for each channel and selects channels
    with scores above a cutoff value definied relative to a statistic
    describing score variance and a statistic describing the baseline:

    *cutoff = baseline + variance_cutoff * variance*

    Parameters
    ----------
    feature_scorer : callable, default=None
        Callable that returns a scalar figure of merit score for each feature
        with signature: scores = feature_scorer(X, y).
    aggregator : callable, default=np.sum
        Function that converts feature scores into a single aggregate matrix
        score with signature: matrix_score = aggregator(features_scores).
    variance_cutoff : float, default=2.0
        The number of units of variance used to define the cutoff.
    get_variance : callable, default=np.nanstd
        Callable that provides a scalar measure of the variability of the
        scores (e.g. np.nanstd, scipy.stats.iqr) with the signature:
        variance = get_variance(aggregate_scores)
    get_baseline : callable, default=np.nanmean
        Callable that provides a scalar baseline score (e.g. np.nanmean or
        np.nanmedian) with the signature:
        baseline = get_baseline(aggregate_scores)
    n_min : int, default=1
        The minimum number of items to select.  Takes precedence over other
        parameters.
    channel_processes : int or 'max', default=1
        - Set the number of processes used during training:
        - If 1: Run all channel computations in a single process.
        - If 'max': Run each channel in a different process, using all
          available CPUs.
        - If int > 1: Run each channel in a different process, using up to
          channel_processes number of CPUs.

    Examples
    --------
    ::

        import numpy as np
        import pipecaster as pc
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import f_classif

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        clf.add_layer(pc.SelectVarianceHighPassScores(
                            f_classif, np.sum, variance_cutoff=1,
                            get_variance=np.nanstd, get_baseline=np.nanmean,
                            n_min=1))
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
        clf.fit(Xs, y)

        # show selected input types (random or informative)
        selections = clf.get_model(1,0).get_support()
        [t for i, t in enumerate(X_types) if i in selections]
        # output: ['informative', 'informative', 'informative']

        pc.cross_val_score(clf, Xs, y)
        # output: [0.9411764705882353, 0.9375, 0.9393382352941176]
    """


    def __init__(self, feature_scorer=None, aggregator=np.sum,
                 variance_cutoff=2.0, get_variance=np.nanstd,
                 get_baseline=np.nanmean, n_min=1,
                 channel_processes=1):
        self._params_to_attributes(SelectVarianceHighPassScores.__init__,
                                   locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator),
                         VarianceHighPassScoreSelector(
                                             variance_cutoff, get_variance,
                                             get_baseline, n_min),
                         channel_processes)


class SelectKBestProbes(ChannelSelector):
    """
    Select fixed number of channels based on ML performance.

    This pipe component estimates the predictive value of each feature matrix
    using cross validation of an ML probe.  Probe performance is calculated
    using a metric specified by the scorer argument of __init__() and by taking
    the mean score of the cross validation splits.  A fixed number of matrices
    with the highest probe performances are passed through as outputs.

    Parameters
    ----------
    predictor_probe : predictor, default=None
        Predictor instance with the scikit-learn estimator & predictor
        interfaces.  Used to estimate channel information content.
    cv : int, or callable, default=5
        - Set the cross validation method:
        - If int > 1: Use StratifiedKfold(n_splits=internal_cv) for
          classifiers or Kfold(n_splits=internal_cv) for regressors.
        - If None or 5: Use 5 splits with the default split generator.
        - If callable: Assumes interface like Kfold scikit-learn.
    score_method : str, default='auto'
        - Name of prediction method used when scoring probe performance.
        - if 'auto' :
            - If classifier : Method picked using
              config.score_method_precedence order.
            - If regressor : 'predict'
    scorer : callable, default='auto'
        Callable that computes a figure of merit score for the probe.
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer with signature: score = scorer(y_true, y_pred).
    k : int, default=1
        Number of channels to select.  Selected channels are those with the
        highest probe performance scores.
    channel_processes : int or 'max', default=1
        - Set the number of processes used during training:
        - If 1: Run all channel computations in a single process.
        - If 'max': Run each channel in a different process, using all
          available CPUs.
        - If int > 1: Run each channel in a different process, using up to
          channel_processes number of CPUs.

    Examples
    --------
    ::

        import numpy as np
        import pipecaster as pc
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        probe = GradientBoostingClassifier(n_estimators=5, max_depth=3)
        clf.add_layer(pc.SelectKBestProbes(predictor_probe=probe, cv=5,
                                           scorer='auto', k=3))
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
        clf.fit(Xs, y)

        # show selected input types (random or informative)
        selections = clf.get_model(1, 0).get_support()
        [t for i, t in enumerate(X_types) if i in selections]
        # output: ['informative', 'informative', 'informative']

        pc.cross_val_score(clf, Xs, y)
        # output: [0.8235294117647058, 0.9080882352941176, 0.9411764705882353]
    """
    def __init__(self, predictor_probe=None, cv=5,
                 score_method='auto', scorer='auto', k=1,
                 channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectKBestProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv,
                                             score_method, scorer,
                                             cv_processes),
                         RankScoreSelector(k), channel_processes)


class SelectPercentBestProbes(ChannelSelector):
    """
    Select a fixed percentage of channels based on ML performance.

    This pipe component estimates the predictive value of each feature matrix
    using cross validation of an ML probe.  Probe performance is calculated
    using a metric specified by the scorer argument of __init__() and by taking
    the mean score of the cross validation splits.  A specified percentage of
    the matrices with the highest probe performances are passed through as
    outputs.

    Parameters
    ----------
    predictor_probe : predictor, default=None
        Predictor instance with the scikit-learn estimator & predictor
        interfaces.  Used to estimate channel information content.
    cv : int, or callable, default=5
        - Set the cross validation method.
        - If int > 1: Use StratifiedKfold(n_splits=internal_cv) for
          classifiers or Kfold(n_splits=internal_cv) for regressors.
        - If None or 5: Use 5 splits with the default split generator.
        - If callable: Assumes interface like Kfold scikit-learn.
    score_method : str, default='auto'
        - Name of prediction method used when scoring probe performance.
        - if 'auto' :
            - If classifier : Method picked using
              config.score_method_precedence order.
            - If regressor : 'predict'
    scorer : callable, default='auto'
        Callable that computes a figure of merit score for the probe.
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer with signature: score = scorer(y_true, y_pred).
    pct: float
        The percentage of channels to select.
    channel_processes : int or 'max', default=1
        - Set the number of processes used during training:
        - If 1: Run all channel computations in a single process.
        - If 'max': Run each channel in a different process, using all
          available CPUs.
        - If int > 1: Run each channel in a different process, using up to
          channel_processes number of CPUs.

    Examples
    --------
    ::

        import numpy as np
        import pipecaster as pc
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        clf.add_layer(pc.SelectPercentBestProbes(
                    predictor_probe=GradientBoostingClassifier(n_estimators=5),
                    cv=5, scorer='auto', pct=30))
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
        clf.fit(Xs, y)

        # show selected input types (random or informative)
        selections = clf.get_model(1, 0).get_support()
        [t for i, t in enumerate(X_types) if i in selections]
        # output: ['informative', 'informative', 'informative']

        pc.cross_val_score(clf, Xs, y)
        # output: [0.8823529411764706, 1.0, 0.96875]
    """
    def __init__(self, predictor_probe=None, cv=3,
                 score_method='auto', scorer='auto', pct=33,
                 channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectPercentBestProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv,
                                             score_method, scorer,
                                             cv_processes),
                         PctRankScoreSelector(pct),
                         channel_processes)


class SelectHighPassProbes(ChannelSelector):
    """
    Select channels based on ML performance and an absolute cutoff.

    This pipe component estimates the predictive value of each feature matrix
    using cross validation of an ML probe.  Probe performance is calculated
    using a metric specified by the scorer argument of __init__() and by taking
    the mean score of the cross validation splits.  Matrices with probe
    performance scores that exceed a specified cutoff are passed through as
    outputs.

    Parameters
    ----------
    predictor_probe : predictor, default=None
        Predictor instance with the scikit-learn estimator & predictor
        interfaces.  Used to estimate channel information content.
    cv : int, or callable, default=5
        - Set the cross validation method.
        - If int > 1: Use StratifiedKfold(n_splits=internal_cv) for
          classifiers or Kfold(n_splits=internal_cv) for regressors.
        - If None or 5: Use 5 splits with the default split generator.
        - If callable: Assumes interface like Kfold scikit-learn.
    score_method : str, default='auto'
        - Name of prediction method used when scoring probe performance.
        - if 'auto' :
            - If classifier : Method picked using
              config.score_method_precedence order.
            - If regressor : 'predict'
    scorer : callable, default='auto'
        Callable that computes a figure of merit score for the probe.
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer with signature: score = scorer(y_true, y_pred).
    cutoff: float, default=0.0
        Channels with probe performance scores above this value are selected.
    n_min: int, default=1
        Minimum number of channels to select.  Takes precedence over cutoff.
    channel_processes : int or 'max', default=1
        - Set the number of processes used during training.
        - If 1: Run all channel computations in a single process.
        - If 'max': Run each channel in a different process, using all
          available CPUs.
        - If int > 1: Run each channel in a different process, using up to
          channel_processes number of CPUs.

    Examples
    --------
    ::

        import numpy as np
        import pipecaster as pc
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import f_classif

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        clf.add_layer(pc.SelectHighPassProbes(
                    predictor_probe=GradientBoostingClassifier(n_estimators=5),
                    cv=5, scorer='auto', cutoff=0.75, n_min=1))
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
        clf.fit(Xs, y)

        # show selected input types (random or informative)
        selections = clf.get_model(1, 0).get_support()
        [t for i, t in enumerate(X_types) if i in selections]
        # output: ['informative', 'informative', 'informative']

        pc.cross_val_score(clf, Xs, y)
        # output: [0.8823529, 0.970588235, 0.94117647]
    """
    def __init__(self, predictor_probe=None, cv=3, score_method='auto',
                 scorer='auto', cutoff=0.0, n_min=1, channel_processes=1,
                 cv_processes=1):
        self._params_to_attributes(SelectHighPassProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv,
                                             score_method, scorer,
                                             cv_processes),
                         HighPassScoreSelector(cutoff, n_min),
                         channel_processes)


class SelectVarianceHighPassProbes(ChannelSelector):
    """
    Select channels based on ML performance and a relative cutoff.

    This pipe component estimates the predictive value of each feature matrix
    using cross validation of an ML probe.  Probe performance is calculated
    using a metric specified by the scorer argument of __init__() and by taking
    the mean score of the cross validation splits.  Matrices are passed through
    as outputs if their probe performances exceed a cutoff value definied
    relative to a statistic describing score variance and a statistic
    describing the baseline:

    *cutoff = baseline + variance_cutoff * variance*

    Parameters
    ----------
    predictor_probe : predictor, default=None
        Predictor instance with the scikit-learn estimator & predictor
        interfaces.  Used to estimate channel information content.
    cv : int, or callable, default=5
        - Set the cross validation method.
        - If int > 1 : Use StratifiedKfold(n_splits=internal_cv) for
          classifiers or Kfold(n_splits=internal_cv) for regressors.
        - If None or 5 : Use 5 splits with the default split generator.
        - If callable : Assumes interface like Kfold scikit-learn.
    score_method : str, default='auto'
        - Name of prediction method used when scoring probe performance.
        - if 'auto' :
            - If classifier : Method picked using
              config.score_method_precedence order.
            - If regressor : 'predict'
    scorer : callable, default='auto'
        Callable that computes a figure of merit score for the probe.
        - If 'auto':
            - explained_variance_score for regressors with predict()
            - roc_auc_score for classifiers with {predict_proba,
              predict_log_proba, decision_function}
            - balanced_accuracy_score for classifiers with only predict()
        - If callable: A scorer with signature: score = scorer(y_true, y_pred).
    variance_cutoff : float, default=2.0
        The number of units of performance variance used to define the cutoff.
    get_variance : callable, default=np.nanstd
        Callable that provides a scalar measure of the variability of the
        scores (e.g. np.nanstd, scipy.stats.iqr) with the signature:
        variance = get_variance(performance_scores)
    get_baseline : callable, default=np.nanmean
        Callable that provides a scalar baseline score (e.g. np.nanmean or
        np.nanmedian) with the signature:
        baseline = get_baseline(performance_scores)
    n_min : int, default=1
        The minimum number of items to select.  Takes precedence over other
        parameters.
    channel_processes : int or 'max', default=1
        - Set the number of processes used during training.
        - If 1 : Run all channel computations in a single process.
        - If 'max': Run each channel in a different process, using all
          available CPUs.
        - If int > 1: Run each channel in a different process, using up to
          channel_processes number of CPUs.

    Examples
    --------
    ::

        import numpy as np
        import pipecaster as pc
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import f_classif

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        clf.add_layer(pc.SelectHighPassProbes(
                    predictor_probe=GradientBoostingClassifier(n_estimators=5),
                    cv=5, scorer='auto', cutoff=0.1, n_min=1))
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
        clf.fit(Xs, y)

        # show selected input types (random or informative)
        selections = clf.get_model(1, 0).get_support()
        [t for i, t in enumerate(X_types) if i in selections]
        # output: ['informative', 'informative', 'informative']

        pc.cross_val_score(clf, Xs, y)
        # output: [0.85294, 0.9393, 0.90808]
    """
    def __init__(self, predictor_probe=None, cv=3,
                 score_method='auto', scorer='auto',
                 variance_cutoff=2.0,
                 get_variance=np.nanstd, get_baseline=np.nanmean, n_min=1,
                 channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectVarianceHighPassProbes.__init__,
                                   locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv,
                                             score_method, scorer,
                                             cv_processes),
                         VarianceHighPassScoreSelector(variance_cutoff,
                                                       get_variance,
                                                       get_baseline, n_min),
                         channel_processes)
