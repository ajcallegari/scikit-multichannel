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

"""
Module with classes for in-pipeline selection of input channels in a
MultichannelPipeline based on various criteria.
"""

__all__ = ['ChannelSelector', 'ModelSelector',
           'SelectKBestScores', 'SelectPercentBestScores',
           'SelectHighPassScores', 'SelectVarianceHighPassScores',
           'SelectKBestProbes', 'SelectPercentBestProbes',
           'SelectHighPassProbes', 'SelectVarianceHighPassProbes',
           'SelectKBestModels', 'SelectPercentBestModels',
           'SelectHighPassModels', 'SelectVarianceHighPassModels']


class ChannelSelector(Cloneable, Saveable):
    """
    Multichannel pipe that scores and selects channels during calls to pipeline
    fit().

    Parameters
    ----------
    channel_scorer: callable, default=None
        Callable object that provide a figure of merit score for a channel.
        Signature:  score = channel_scorer(X, y)
    score_selector: callable, default=None
        Callable object that returns a list of indices of selected
        channels. Signature: selected_indices = score_selector(scores)
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.

    Example
    -------
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
    >>> [0.9705882352941176, 0.9117647058823529, 0.9411764705882353]
    """
    state_variables = ['selected_indices_', 'channel_scores_']

    def __init__(self, channel_scorer=None, score_selector=None,
                 channel_processes=1):
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

    def get_channel_scores(self):
        if hasattr(self, 'channel_scores_'):
            return self.channel_scores_
        else:
            raise utils.FitError('Channel scores not found. They are only \
                                 available after call to fit().')

    def get_support(self):
        if hasattr(self, 'selected_indices_'):
            return self.selected_indices_
        else:
            raise utils.FitError('Must call fit before getting selection \
                                 information')

    def transform(self, Xs):
        return [Xs[i] if (i in self.selected_indices_) else None
                for i in range(len(Xs))]

    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)

    def get_selection_indices(self):
        return self.selected_indices_


class SelectKBestScores(ChannelSelector):
    """
    ChannelSelector that computes an aggregate feature score for each channel
    and selects a fixed number of the top-scoring channels.

    Parameters
    ----------
    feature_scorer: callable, default=f_classif
        Callable that returns a figure of merit score for each featurs.
        Pattern: scores = feature_scorer(X, y)
    aggregator: callable, default=np.sum
        Callable that aggregates individual features scores to create a single
        scalar matrix score.  Pattern:
        aggregate_score = aggregator(feature_scores)
    k: int, default=1
        The number of channels to select.  The selected channels are those
        with this highest aggregate feature scores.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.

    Example
    -------
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
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.8235294117647058, 0.9375, 0.8823529411764706]
    """
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, k=1,
                 channel_processes=1):
        self._params_to_attributes(SelectKBestScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator),
                         RankScoreSelector(k), channel_processes)


class SelectPercentBestScores(ChannelSelector):
    """
    ChannelSelector that computes an aggregate feature score for each channel
    and selects a specified percentage of the top-scoring channels.

    Parameters
    ----------
    feature_scorer: callable, default=f_classif
        Callable that returns a figure of merit score for each featurs.
        Pattern: scores = feature_scorer(X, y)
    aggregator: callable, default=np.sum
        Callable that aggregates individual features scores to create a single
        scalar matrix score.  Pattern:
        aggregate_score = aggregator(feature_scores)
    pct: float
        The percentage of channels to select.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.

    Example
    -------
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
    selections = clf.get_model(1,0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.9411764705882353, 0.9411764705882353, 0.9705882352941176]
    """
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, percent=33,
                 channel_processes=1):
        self._params_to_attributes(SelectPercentBestScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator),
                         PctRankScoreSelector(percent), channel_processes)


class SelectHighPassScores(ChannelSelector):
    """
    ChannelSelector that computes an aggregate feature score for each channel
    and selects channels with scores that exceed a specified cutoff value.

    Parameters
    ----------
    feature_scorer: callable, default=f_classif
        Callable that returns a figure of merit score for each featurs.
        Pattern: scores = feature_scorer(X, y)
    aggregator: callable, default=np.sum
        Callable that aggregates individual features scores to create a single
        scalar matrix score.  Pattern:
        aggregate_score = aggregator(feature_scores)
    cutoff: float, default=0.0
        Score that defines the selection.  Items with scores above this value
        are selected.
    n_min: int, default=1
        The minimum number of items to select.  Takes precedence over cutoff.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.

    Example
    -------
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
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.8823529411764706, 0.9099264705882353, 0.9411764705882353]
    """
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, cutoff=0,
                 n_min=1, channel_processes=1):
        self._params_to_attributes(SelectHighPassScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator),
                         HighPassScoreSelector(cutoff, n_min),
                         channel_processes)


class SelectVarianceHighPassScores(ChannelSelector):
    """
    ChannelSelector that computes an aggregate feature score for each channel
    and selects channels with scores above a cutoff value definied relative
    to a statistic describing score variance and a statistic describing the
    baseline:  cutoff = baseline + variance_cutoff * variance.

    Parameters
    ----------
    feature_scorer: callable, default=f_classif
        Callable that returns a figure of merit score for each featurs.
        Pattern: scores = feature_scorer(X, y)
    aggregator: callable, default=np.sum
        Callable that aggregates individual features scores to create a single
        scalar matrix score.  Pattern:
        aggregate_score = aggregator(feature_scores)
    variance_cutoff: float, default=2.0
        The number of units of variance used to define the cutoff.  Items with
        scores above variance_cutoff * variance will will be selected.
    get_variance: callable, default=np.nanstd
        Callable that provides a scalar measure of the variability of the
        scores (e.g. np.nanstd, scipy.stats.iqr).
        Pattern: variance = get_variance(scores)
    get_baseline: callable, default=np.nanmean
        Callable that provides a scalar baseline score (e.g. np.nanmean or
        np.nanmedian).
        Pattern: baseline = get_baseline(scores)
    n_min: int, default=1
        The minimum number of items to select.  Takes precedence over other
        parameters.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.

    Example
    -------
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
    selections = clf.get_model(1,0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.9411764705882353, 0.9375, 0.9393382352941176]
    """
    def __init__(self, feature_scorer=f_classif, aggregator=np.sum,
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
    ChannelSelector that computes a predictor probe cross validation score for
    each channel and selects a fixed number of the top-scoring channels.

    Parameters
    ----------
    predictor_probe: predictor instance, default=None
        Predictor instance with the scikit-learn estimator & predictor
        interfaces.  Used to estimate channel inforation content.
    cv: int, or callable, default=5
        Set the cross validation method.
        if int > 1: Use StratifiedKfold(n_splits=internal_cv) for
            classifiers or Kfold(n_splits=internal_cv) for regressors.
        If None or 5: Use 5 splits with the default split generator
        If callable: Assumes scikit-learn interface like Kfold
    scorer: callable or 'auto', default='auto'
        Callable that provides a figure of merit used for scoring probes
        during cross validation. Signature:
        score = scorer(y_true, y_pred)
        'auto': Score regressors with explained_variance_score and classifiers
            with balanced_accuracy_score.
    k: int, default=1
        The number of channels to select.  The selected channels are those
        with this highest aggregate feature scores.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.

    Example
    -------
    import numpy as np
    import pipecaster as pc
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier

    Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                        n_random_Xs=7)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(StandardScaler())
    clf.add_layer(pc.SelectKBestProbes(
                    predictor_probe=GradientBoostingClassifier(n_estimators=5),
                    cv=5, scorer='auto', k=3))
    clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))

    clf.fit(Xs, y)
    selections = clf.get_model(1, 0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.8235294117647058, 0.9080882352941176, 0.9411764705882353]
    """
    def __init__(self, predictor_probe=None, cv=5, scorer='auto', k=1,
                 channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectKBestProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv,
                                             scorer, cv_processes),
                         RankScoreSelector(k), channel_processes)


class SelectPercentBestProbes(ChannelSelector):
    """
    ChannelSelector that computes a predictor probe cross validation score for
    each channel and selects a specified percentage of the top-scoring
    channels.

    Parameters
    ----------
    predictor_probe: predictor instance, default=None
        Predictor instance with the scikit-learn estimator & predictor
        interfaces.  Used to estimate channel inforation content.
    cv: int, or callable, default=5
        Set the cross validation method.
        if int > 1: Use StratifiedKfold(n_splits=internal_cv) for
            classifiers or Kfold(n_splits=internal_cv) for regressors.
        If None or 5: Use 5 splits with the default split generator
        If callable: Assumes scikit-learn interface like Kfold
    scorer: callable or 'auto', default='auto'
        Callable that provides a figure of merit used for scoring probes
        during cross validation. Signature:
        score = scorer(y_true, y_pred)
        'auto': Score regressors with explained_variance_score and classifiers
            with balanced_accuracy_score.
    pct: float
        The percentage of channels to select.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.

    Example
    -------
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
    selections = clf.get_model(1, 0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.8823529411764706, 1.0, 0.96875]
    """
    def __init__(self, predictor_probe=None, cv=3, scorer='auto', pct=33,
                 channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectPercentBestProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv, scorer,
                                             cv_processes),
                         PctRankScoreSelector(pct),
                         channel_processes)


class SelectHighPassProbes(ChannelSelector):
    """
    ChannelSelector that computes a predictor probe cross validation score for
    each channel and selects channels with scores that exceed a specified
    cutoff value.

    Parameters
    ----------
    predictor_probe: predictor instance, default=None
        Predictor instance with the scikit-learn estimator & predictor
        interfaces.  Used to estimate channel inforation content.
    cv: int, or callable, default=5
        Set the cross validation method.
        if int > 1: Use StratifiedKfold(n_splits=internal_cv) for
            classifiers or Kfold(n_splits=internal_cv) for regressors.
        If None or 5: Use 5 splits with the default split generator
        If callable: Assumes scikit-learn interface like Kfold
    scorer: callable or 'auto', default='auto'
        Callable that provides a figure of merit used for scoring probes
        during cross validation. Signature:
        score = scorer(y_true, y_pred)
        'auto': Score regressors with explained_variance_score and classifiers
            with balanced_accuracy_score.
    cutoff: float, default=0.0
        Score that defines the selection.  Items with scores above this value
        are selected.
    n_min: int, default=1
        The minimum number of items to select.  Takes precedence over cutoff.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.

    Example
    -------
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
    selections = clf.get_model(1, 0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.8823529411764706, 0.9705882352941176, 0.9411764705882353]
    """
    def __init__(self, predictor_probe=None, cv=3, scorer='auto', cutoff=0.0,
                 n_min=1, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectHighPassProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv,
                                             scorer, cv_processes),
                         HighPassScoreSelector(cutoff, n_min),
                         channel_processes)


class SelectVarianceHighPassProbes(ChannelSelector):
    """
    ChannelSelector that computes a predictor probe cross validation score for
    each channel and selects channels with scores above a cutoff value definied
    relative to a statistic describing score variance and a statistic
    describing the baseline:  cutoff = baseline + variance_cutoff * variance.

    Parameters
    ----------
    predictor_probe: predictor instance, default=None
        Predictor instance with the scikit-learn estimator & predictor
        interfaces.  Used to estimate channel inforation content.
    cv: int, or callable, default=5
        Set the cross validation method.
        if int > 1: Use StratifiedKfold(n_splits=internal_cv) for
            classifiers or Kfold(n_splits=internal_cv) for regressors.
        If None or 5: Use 5 splits with the default split generator
        If callable: Assumes scikit-learn interface like Kfold
    scorer: callable or 'auto', default='auto'
        Callable that provides a figure of merit used for scoring probes
        during cross validation. Signature:
        score = scorer(y_true, y_pred)
        'auto': Score regressors with explained_variance_score and classifiers
            with balanced_accuracy_score.
    variance_cutoff: float, default=2.0
        The number of units of variance used to define the cutoff.  Items with
        scores above variance_cutoff * variance will will be selected.
    get_variance: callable, default=np.nanstd
        Callable that provides a scalar measure of the variability of the
        scores (e.g. np.nanstd, scipy.stats.iqr).
        Pattern: variance = get_variance(scores)
    get_baseline: callable, default=np.nanmean
        Callable that provides a scalar baseline score (e.g. np.nanmean or
        np.nanmedian).
        Pattern: baseline = get_baseline(scores)
    n_min: int, default=1
        The minimum number of items to select.  Takes precedence over other
        parameters.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.

    Example
    -------
    import numpy as np
    import pipecaster as pc
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.feature_selection import f_classif

    Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                        n_random_Xs=7)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(StandardScaler())
    clf.add_layer(pc.SelectVarianceHighPassProbes(
                    predictor_probe=GradientBoostingClassifier(n_estimators=5),
                    cv=5, scorer='auto', variance_cutoff=1,
                    get_variance=np.nanstd, get_baseline=np.nanmean,
                    n_min=1))
    clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
    clf.fit(Xs, y)
    selections = clf.get_model(1, 0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.8529411764705882, 0.9393382352941176, 0.9080882352941176]
    """
    def __init__(self, predictor_probe=None, cv=3, scorer='auto',
                 variance_cutoff=2.0,
                 get_variance=np.nanstd, get_baseline=np.nanmean, n_min=1,
                 channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectVarianceHighPassProbes.__init__,
                                   locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv,
                                             scorer, cv_processes),
                         VarianceHighPassScoreSelector(variance_cutoff,
                                                       get_variance,
                                                       get_baseline, n_min),
                         channel_processes)


class ModelSelector(Cloneable, Saveable):
    """
    Model and channel selector that computes a model cross validation score
    for each channel model, selects a subset of the channels, and outputs the
    predictions of the selected models.

    Note: Does not concatenate outputs from multiple models, so if more than
    one channel predictor is selected then an additional voting or
    meta-prediction layer is required to generate a single prediction.

    Parameters
    ----------
    predictors: predictor instance or list of instances, default=None
        Predictor(s) instances that implement the scikit-learn estimator &
            predictor interfaces.
        List: One predictor will be applied to each channel in the listed
            order.
        Single predictor: Predictor is cloned and broadcast across all input
            channels.
    cv: int, or callable, default=5
        Set the cross validation method.
        if int > 1: Use StratifiedKfold(n_splits=internal_cv) for
            classifiers or Kfold(n_splits=internal_cv) for regressors.
        If None or 5: Use 5 splits with the default split generator
        If callable: Assumes scikit-learn interface like Kfold
    scorer: callable or 'auto', default='auto'
        Callable that provides a figure of merit used for scoring models
        during cross validation. Signature:
            score = scorer(y_true, y_pred)
        'auto': Score regressors with explained_variance_score and classifiers
            with balanced_accuracy_score.
    score_selector: callable, default=None
        Callable object that returns a list of indices of selected
        channels. Signature:
            selected_indices = score_selector(scores)
    cv_transform: bool, default=True
        True: Output cv predictions when fit_transform is called (use for
            model stacking).
        False: Inactivate internal cv training and output whole training set
            predictions when fit_transform is called.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.
    cv_processes: int or 'max', default=1
        Number of parallel processes to run for each cross validation split
        during model fitting.   If 'max', all available CPUs will be used.

    Notes:
        Predict_proba and decision_function are treated as synonymous.

    Example
    -------
    import numpy as np
    import pipecaster as pc
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC

    Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                        n_random_Xs=7)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(pc.ModelSelector(predictors=GradientBoostingClassifier(),
                                   cv=5, scorer='auto',
                                   score_selector=pc.RankScoreSelector(3)))
    clf.add_layer(pc.MultichannelPredictor(SVC()))
    clf.fit(Xs, y)
    selections = clf.get_model(0, 0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.9411764705882353, 0.9375, 0.9705882352941176]
    """
    state_variables = ['classes_', 'selected_indices_']

    def __init__(self, predictors=None, cv=5, scorer='auto',
                 score_selector=RankScoreSelector(3), cv_transform=True,
                 channel_processes=1, cv_processes=1):
        self._params_to_attributes(ModelSelector.__init__, locals())

        if predictors is None:
            raise ValueError('No predictors found.')

        if isinstance(predictors, (list, tuple, np.ndarray)):
            estimator_types = [p._estimator_type for p in predictors]
        else:
            estimator_types = [predictors._estimator_type]
        if len(set(estimator_types)) != 1:
            raise TypeError('Predictors must be of uniform type (e.g. all \
                            classifiers or all regressors).')
        self._estimator_type = estimator_types[0]
        if scorer == 'auto':
            if utils.is_classifier(self):
                self.scorer = balanced_accuracy_score
            elif utils.is_regressor(self):
                self.scorer = explained_variance_score
            else:
                raise AttributeError('predictor type required for automatic \
                                     assignment of scoring metric')

    @staticmethod
    def _fit_predict_score(predictor, X, y, fit_params, cv,
                           scorer, cv_processes):

        if X is None:
            return None, None, None

        if type(predictor) in [transform_wrappers.SingleChannelCV,
                               transform_wrappers.MultichannelCV]:
            raise TypeError('CV transform_wrapper found in predictors \
                            (disallowed to promote uniform wrapping)')

        predictor = transform_wrappers.unwrap_predictor(predictor)
        model = utils.get_clone(predictor)
        model = transform_wrappers.SingleChannelCV(model, internal_cv=cv,
                                                   scorer=scorer,
                                                   cv_processes=cv_processes)
        if y is None:
            cv_predictions = model.fit_transform(X, **fit_params)
        else:
            cv_predictions = model.fit_transform(X, y, **fit_params)

        return model, cv_predictions, model.score_

    def _expose_predictor_interface(self, model):
        method_set = utils.get_prediction_method_names(model)
        for method_name in method_set:
            prediction_method = functools.partial(self.predict_with_method,
                                                  method_name=method_name)
            setattr(self, method_name, prediction_method)

    def fit_transform(self, Xs, y=None, **fit_params):

        # broadcast predictors if necessary
        is_listlike = isinstance(self.predictors, (list, tuple, np.ndarray))
        if is_listlike:
            if len(Xs) != len(self.predictors):
                raise ValueError('predictor list length does not match input \
                                 list length')
            else:
                predictors = self.predictors
        else:
            predictors = [self.predictors if X is not None else None
                          for X in Xs]

        cv, scorer, cv_processes = self.cv, self.scorer, self.cv_processes
        args_list = [(p, X, y, fit_params, cv, scorer, cv_processes)
                     for p, X in zip(predictors, Xs)]

        n_jobs = len(args_list)
        n_processes = (1 if self.channel_processes is None
                       else self.channel_processes)
        if (type(n_processes) == int and n_jobs < n_processes):
            n_processes = n_jobs
        if n_processes == 'max' or n_processes > 1:
            try:
                shared_mem_objects = [y, fit_params, cv, scorer, cv_processes]
                job_results = parallel.starmap_jobs(
                                ModelSelector._fit_predict_score, args_list,
                                n_cpus=self.channel_processes,
                                shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'
                      .format(e))
                print('defaulting to single processor')
                n_processes = 1
        if n_processes is None or n_processes <= 1:
            job_results = [ModelSelector._fit_predict_score(*args)
                           for args in args_list]

        models, cv_predictions, model_scores = zip(*job_results)
        self.selected_indices_ = self.score_selector(model_scores)
        self.models = [m if i in set(self.selected_indices_) else None
                       for i, m in enumerate(models)]

        if len(self.selected_indices_ == 1):
            self._expose_predictor_interface(
                                    self.models[self.selected_indices_[0]])

        if self.cv_transform is True:
            Xs_t = [p if i in set(self.selected_indices_) else None
                    for i, p in enumerate(cv_predictions)]
        else:
            Xs_t = [model.transform(X) if i in set(self.selected_indices_)
                    else None for i, (model, X) in enumerate(zip(models, Xs))]
        return Xs_t

    def fit(self, Xs, y=None, **fit_params):
        self.fit_transform(Xs, y, **fit_params)

    def transform(self, Xs):
        if hasattr(self, 'models') is False:
            raise utils.FitError('Tranform called before model fitting.')
        return [m.transform(X) if m is not None else None
                for m, X in zip(self.models, Xs)]

    def predict_with_method(self, Xs, method_name):
        if hasattr(self, 'models') is False:
            raise utils.FitError('prediction attempted before model fitting')
        if len(self.selected_indices_) != 1:
            raise ValueErrror('To predict with a ModelSelector, exactly 1 \
                              model must be selected.')
        selected_index = self.selected_indices[0]
        prediction_method = getattr(self.models[selected_index], method_name)
        predictions = prediction_method(Xs[selected_index])
        if utils.is_classifier(self) and method_name == 'predict':
            predictions = self.classes_[predictions]

        return predictions

    def get_support(self):
        return self.selected_indices_

    def get_clone(self):
        clone = super().get_clone()
        if hasattr(self, 'models'):
            clone.models = [utils.get_clone(m) if m is not None else None
                            for m in self.models]
        return clone


class SelectKBestModels(ModelSelector):
    """
    Model and channel selector that computes a model cross validation score
    for each channel model and selects a fixed number of the
    top-scoring models.

    Parameters
    ----------
    predictors: predictor instance or list of instances, default=None
        Predictor(s) instances that implement the scikit-learn estimator &
            predictor interfaces.
        List: One predictor will be applied to each channel in the listed
            order.
        Single predictor: Predictor is cloned and broadcast across all input
            channels.
    cv: int, or callable, default=5
        Set the cross validation method.
        if int > 1: Use StratifiedKfold(n_splits=internal_cv) for
            classifiers or Kfold(n_splits=internal_cv) for regressors.
        If None or 5: Use 5 splits with the default split generator
        If callable: Assumes scikit-learn interface like Kfold
    scorer: callable or 'auto', default='auto'
        Callable that provides a figure of merit used for scoring models
        during cross validation. Signature:
            score = scorer(y_true, y_pred)
        'auto': Score regressors with explained_variance_score and classifiers
            with balanced_accuracy_score.
    k: int, default=1
        The number of channels to select.  The selected channels are those
        with this highest aggregate feature scores.
    cv_transform: bool, default=True
        True: Output cv predictions when fit_transform is called (use for
            model stacking).
        False: Inactivate internal cv training and output whole training set
            predictions when fit_transform is called.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.
    cv_processes: int or 'max', default=1
        Number of parallel processes to run for each cross validation split
        during model fitting.   If 'max', all available CPUs will be used.

    Example
    -------
    import numpy as np
    import pipecaster as pc
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC

    Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                        n_random_Xs=7)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(pc.SelectKBestModels(predictors=GradientBoostingClassifier(),
                                       cv=5, scorer='auto', k=3))
    clf.add_layer(pc.MultichannelPredictor(SVC()))
    clf.fit(Xs, y)
    selections = clf.get_model(0, 0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[1.0, 1.0, 0.9411764705882353]
    """
    def __init__(self, predictors, cv=5, scorer='auto', k=1,
                 cv_transform=True, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectKBestModels.__init__, locals())
        super().__init__(predictors, cv, scorer, RankScoreSelector(k),
                         cv_transform, channel_processes, cv_processes)


class SelectPercentBestModels(ModelSelector):
    """
    Model and channel selector that computes a model cross validation score
    for each channel model and selects a specified percentage of the
    top-scoring models.

    Parameters
    ----------
    predictors: predictor instance or list of instances, default=None
        Predictor(s) instances that implement the scikit-learn estimator &
            predictor interfaces.
        List: One predictor will be applied to each channel in the listed
            order.
        Single predictor: Predictor is cloned and broadcast across all input
            channels.
    cv: int, or callable, default=5
        Set the cross validation method.
        if int > 1: Use StratifiedKfold(n_splits=internal_cv) for
            classifiers or Kfold(n_splits=internal_cv) for regressors.
        If None or 5: Use 5 splits with the default split generator
        If callable: Assumes scikit-learn interface like Kfold
    scorer: callable or 'auto', default='auto'
        Callable that provides a figure of merit used for scoring models
        during cross validation. Signature:
            score = scorer(y_true, y_pred)
        'auto': Score regressors with explained_variance_score and classifiers
            with balanced_accuracy_score.
    pct: float
        The percentage of channels to select.
    cv_transform: bool, default=True
        True: Output cv predictions when fit_transform is called (use for
            model stacking).
        False: Inactivate internal cv training and output whole training set
            predictions when fit_transform is called.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.
    cv_processes: int or 'max', default=1
        Number of parallel processes to run for each cross validation split
        during model fitting.   If 'max', all available CPUs will be used.

    Example
    -------
    import numpy as np
    import pipecaster as pc
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC

    Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                        n_random_Xs=7)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(pc.SelectPercentBestModels(
                                    predictors=GradientBoostingClassifier(),
                                    cv=5, scorer='auto', pct=30))
    clf.add_layer(pc.MultichannelPredictor(SVC()))
    clf.fit(Xs, y)
    selections = clf.get_model(0, 0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.8823529411764706, 0.96875, 1.0]
    """
    def __init__(self, predictors, cv=5, scorer='auto', pct=33,
                 cv_transform=True, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectPercentBestModels.__init__, locals())
        super().__init__(predictors, cv, scorer,
                         PctRankScoreSelector(pct),
                         cv_transform, channel_processes, cv_processes)


class SelectHighPassModels(ModelSelector):
    """
    Model and channel selector that computes a model cross validation score
    for each channel model and selects models with scores that exceed a
    specified cutoff value.

    Parameters
    ----------
    predictors: predictor instance or list of instances, default=None
        Predictor(s) instances that implement the scikit-learn estimator &
            predictor interfaces.
        List: One predictor will be applied to each channel in the listed
            order.
        Single predictor: Predictor is cloned and broadcast across all input
            channels.
    cv: int, or callable, default=5
        Set the cross validation method.
        if int > 1: Use StratifiedKfold(n_splits=internal_cv) for
            classifiers or Kfold(n_splits=internal_cv) for regressors.
        If None or 5: Use 5 splits with the default split generator
        If callable: Assumes scikit-learn interface like Kfold
    scorer: callable or 'auto', default='auto'
        Callable that provides a figure of merit used for scoring models
        during cross validation. Signature:
            score = scorer(y_true, y_pred)
        'auto': Score regressors with explained_variance_score and classifiers
            with balanced_accuracy_score.
    cutoff: float, default=0.0
        Score that defines the selection.  Items with scores above this value
        are selected.
    n_min: int, default=1
        The minimum number of items to select.  Takes precedence over cutoff.
    cv_transform: bool, default=True
        True: Output cv predictions when fit_transform is called (use for
            model stacking).
        False: Inactivate internal cv training and output whole training set
            predictions when fit_transform is called.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.
    cv_processes: int or 'max', default=1
        Number of parallel processes to run for each cross validation split
        during model fitting.   If 'max', all available CPUs will be used.

    Example
    -------
    import numpy as np
    import pipecaster as pc
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC

    Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                        n_random_Xs=7)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(pc.SelectHighPassModels(
                                    predictors=GradientBoostingClassifier(),
                                    cv=5, scorer='auto', cutoff=0.75, n_min=1))
    clf.add_layer(pc.MultichannelPredictor(SVC()))
    clf.fit(Xs, y)
    selections = clf.get_model(0, 0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.9705882352941176, 0.9705882352941176, 0.9117647058823529]
    """
    def __init__(self, predictors, cv=5, scorer='auto', cutoff=0.0, n_min=1,
                 cv_transform=True, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectHighPassModels.__init__, locals())
        super().__init__(predictors, cv, scorer,
                         HighPassScoreSelector(cutoff, n_min),
                         cv_transform, channel_processes, cv_processes)


class SelectVarianceHighPassModels(ModelSelector):
    """
    Model and channel selector that computes a predictor cross validation score
    for each channel model and selects models with scores above a cutoff value
    definied relative to a statistic describing score variance and a statistic
    describing the baseline: cutoff = baseline + variance_cutoff * variance.

    Parameters
    ----------
    predictors: predictor instance or list of instances, default=None
        Predictor(s) instances that implement the scikit-learn estimator &
            predictor interfaces.
        List: One predictor will be applied to each channel in the listed
            order.
        Single predictor: Predictor is cloned and broadcast across all input
            channels.
    cv: int, or callable, default=5
        Set the cross validation method.
        if int > 1: Use StratifiedKfold(n_splits=internal_cv) for
            classifiers or Kfold(n_splits=internal_cv) for regressors.
        If None or 5: Use 5 splits with the default split generator
        If callable: Assumes scikit-learn interface like Kfold
    scorer: callable or 'auto', default='auto'
        Callable that provides a figure of merit used for scoring models
        during cross validation. Signature:
            score = scorer(y_true, y_pred)
        'auto': Score regressors with explained_variance_score and classifiers
            with balanced_accuracy_score.
    variance_cutoff: float, default=2.0
        The number of units of variance used to define the cutoff.  Items with
        scores above variance_cutoff * variance will will be selected.
    get_variance: callable, default=np.nanstd
        Callable that provides a scalar measure of the variability of the
        scores (e.g. np.nanstd, scipy.stats.iqr).
        Pattern: variance = get_variance(scores)
    get_baseline: callable, default=np.nanmean
        Callable that provides a scalar baseline score (e.g. np.nanmean or
        np.nanmedian).
        Pattern: baseline = get_baseline(scores)
    n_min: int, default=1
        The minimum number of items to select.  Takes precedence over other
        parameters.
    cv_transform: bool, default=True
        True: Output cv predictions when fit_transform is called (use for
            model stacking).
        False: Inactivate internal cv training and output whole training set
            predictions when fit_transform is called.
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.
    cv_processes: int or 'max', default=1
        Number of parallel processes to run for each cross validation split
        during model fitting.   If 'max', all available CPUs will be used.

    Example
    -------
    import numpy as np
    import pipecaster as pc
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC

    Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                        n_random_Xs=7)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(pc.SelectVarianceHighPassModels(
                        predictors=GradientBoostingClassifier(),
                        cv=5, scorer='auto', variance_cutoff=1.0,
                        get_variance=np.nanstd, get_baseline=np.nanmean,
                        n_min=1))
    clf.add_layer(pc.MultichannelPredictor(SVC()))
    clf.fit(Xs, y)
    selections = clf.get_model(0, 0).get_support()
    # show selected input types (random or informative)
    [t for i, t in enumerate(X_types) if i in selections]
    >>>['informative', 'informative', 'informative']

    pc.cross_val_score(clf, Xs, y)
    >>>[0.9117647058823529, 0.8823529411764706, 0.96875]
    """
    def __init__(self, predictors, cv=5, scorer='auto',
                 variance_cutoff=2.0, get_variance=np.nanstd,
                 get_baseline=np.nanmean,  n_min=1, cv_transform=True,
                 channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectVarianceHighPassModels.__init__,
                                   locals())
        super().__init__(predictors, cv, scorer,
                         VarianceHighPassScoreSelector(variance_cutoff,
                                                       get_variance,
                                                       get_baseline, n_min),
                         cv_transform, channel_processes, cv_processes)
