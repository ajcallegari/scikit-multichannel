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
MultichannelPipeline.
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
    channel_scorer : callable, default=None
        Callable object that provide a figure of merit score for a channel.
        Signature:  score = channel_scorer(X, y)
    score_selector: callable, default=None
        Callable object that returns a list of the indices of the selected
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
                    channel_scorer=pc.AggregateFeatureScorer(f_classif, np.mean),
                    score_selector=pc.RankScoreSelector(3)))
    clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))
    pc.cross_val_score(clf, Xs, y)

    output: [0.9705882352941176, 0.9117647058823529, 0.9411764705882353]

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

    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, k=1, channel_processes=1):
        self._params_to_attributes(SelectKBestScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator), RankScoreSelector(k), channel_processes)


class SelectPercentBestScores(ChannelSelector):

    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, percent=33, channel_processes=1):
        self._params_to_attributes(SelectPercentBestScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator),
                         PctRankScoreSelector(percent), channel_processes)


class SelectHighPassScores(ChannelSelector):

    def __init__(self, feature_scorer=f_classif, aggregator=np.sum, cutoff=0, n_min=1, channel_processes=1):
        self._params_to_attributes(SelectHighPassScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator),
                         HighPassScoreSelector(cutoff, n_min), channel_processes)


class SelectVarianceHighPassScores(ChannelSelector):

    def __init__(self, feature_scorer=f_classif, aggregator=np.sum,
                 variance_cutoff=2.0, get_variance=np.nanstd, get_baseline=np.nanmean, n_min=1,
                 channel_processes=1):
        self._params_to_attributes(SelectVarianceHighPassScores.__init__, locals())
        super().__init__(AggregateFeatureScorer(feature_scorer, aggregator),
                         VarianceHighPassScoreSelector(variance_cutoff, get_variance, get_baseline, n_min),
                         channel_processes)


class SelectKBestProbes(ChannelSelector):

    def __init__(self, predictor_probe=None, cv=3, scorer='auto', k=1, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectKBestProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv, scorer, cv_processes),
                         RankScoreSelector(k), channel_processes)


class SelectPercentBestProbes(ChannelSelector):

    def __init__(self, predictor_probe=None, cv=3, scorer='auto', percent=33, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectPercentBestProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv, scorer, cv_processes),
                         PctRankScoreSelector(percent), channel_processes)


class SelectHighPassProbes(ChannelSelector):

    def __init__(self, predictor_probe=None, cv=3, scorer='auto', cutoff=0.0, n_min=1, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectHighPassProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv, scorer, cv_processes),
                         HighPassScoreSelector(cutoff, n_min), channel_processes)


class SelectVarianceHighPassProbes(ChannelSelector):

    def __init__(self, predictor_probe=None, cv=3, scorer='auto', variance_cutoff=2.0,
                 get_variance=np.nanstd, get_baseline=np.nanmean, n_min=1,
                 channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectVarianceHighPassProbes.__init__, locals())
        super().__init__(CvPerformanceScorer(predictor_probe, cv, scorer, cv_processes),
                         VarianceHighPassScoreSelector(variance_cutoff, get_variance, get_baseline, n_min),
                         channel_processes)


class ModelSelector(Cloneable, Saveable):
    """
    Multichannel predictor that tests a single-channel predictor on each channel, measures the performance of each
    using cross validation, and outputs the predictions of the best models. Note: does not concatenate outputs of multiple
    models, so if more than one channel predictor is selected then an additional voting or meta-prediction step is
    required to generate a single prediction.

    Parameters
    ----------
    predictors: list of sklearn conformant predictors (clasifiers or regressors) of len(Xs) or single predictor, default=None
        If list, then one predictor will be applied to each channel in the listed order.
        If single predictor, predictor is cloned and broadcast across all input channels.
    cv: int or cross validation splitter instance (e.g. StratifiedKFold()), default=5
        Set the cross validation method.  If int, defaults to KFold() for regressors orr
        StratifiedKFold() for classifiers.
    scorer : callable or 'auto', default='auto'
        Figure of merit score used for selecting models via internal cross validation.
        If a callable, the object should have the signature 'scorer(y_true, y_pred)' and return
        a single value.
        If 'auto' regressors will be scored with explained_variance_score and classifiers
        with balanced_accuracy_score.
    score_selector: callable, default=RankScoreSelector(3)
        A scorer callable object / function with signature 'score_selector(scores)' which should
        return a list of the indices of the predictors/channels to be selected
    cv_transform: bool, default=True
        True: output cv predictions when fit_transform is called (use for making meta-features)
        False: output whole training set predictions when fit_transform is called
    channel_processes: int or 'max', default=1
        Number of parallel processes to run for each channel during model
        fitting.  If 'max', all available CPUs will be used.
    cv_processes: int or 'max', default=1
        Number of parallel processes to run for each cross validation split
        during model fitting.   If 'max', all available CPUs will be used.

    notes
    -----
    predictors can be list of predictors (one per channel) or single predictor to be broadcast over all channels
    currently only supports single channel predictors (e.g. scikit-learn conformant predictor)
    predict_proba and decision_function are treated as synonymous
    """
    state_variables = ['classes_', 'selected_indices_']

    def __init__(self, predictors=None, cv=5, scorer='auto',
                 score_selector=RankScoreSelector(3), cv_transform=True, channel_processes=1, cv_processes=1):
        self._params_to_attributes(ModelSelector.__init__, locals())

        if predictors is None:
            raise ValueError('No predictors found.')

        if isinstance(predictors, (list, tuple, np.ndarray)):
            estimator_types = [p._estimator_type for p in predictors]
        else:
            estimator_types = [predictors._estimator_type]
        if len(set(estimator_types)) != 1:
            raise TypeError('Predictors must be of uniform type (e.g. all classifiers or all regressors).')
        self._estimator_type = estimator_types[0]
        if scorer == 'auto':
            if utils.is_classifier(self):
                self.scorer = balanced_accuracy_score
            elif utils.is_regressor(self):
                self.scorer = explained_variance_score
            else:
                raise AttributeError('predictor type required for automatic assignment of scoring metric')

    @staticmethod
    def _fit_predict_score(predictor, X, y, fit_params, cv, scorer, cv_processes):

        if X is None:
            return None, None, None

        if type(predictor) in [transform_wrappers.SingleChannelCV, transform_wrappers.MultichannelCV]:
            raise TypeError('CV transform_wrapper found in predictors (disallowed to promote uniform wrapping)')

        predictor = transform_wrappers.unwrap_predictor(predictor)
        model = utils.get_clone(predictor)
        model = transform_wrappers.SingleChannelCV(model, internal_cv=cv, scorer=scorer,
                                                   cv_processes=cv_processes)
        if y is None:
            cv_predictions = model.fit_transform(X, **fit_params)
        else:
            cv_predictions = model.fit_transform(X, y, **fit_params)

        return model, cv_predictions, model.score_

    def _expose_predictor_interface(self, model):
        method_set = utils.get_prediction_method_names(model)
        for method_name in method_set:
            prediction_method = functools.partial(self.predict_with_method, method_name=method_name)
            setattr(self, method_name, prediction_method)

    def fit_transform(self, Xs, y=None, **fit_params):

        # broadcast predictors if necessary
        is_listlike = isinstance(self.predictors, (list, tuple, np.ndarray))
        if is_listlike:
            if len(Xs) != len(self.predictors):
                raise ValueError('predictor list length does not match input list length')
            else:
                predictors = self.predictors
        else:
            predictors = [self.predictors if X is not None else None for X in Xs]

        cv, scorer, cv_processes = self.cv, self.scorer, self.cv_processes
        args_list = [(p, X, y, fit_params, cv, scorer, cv_processes)
                     for p, X in zip(predictors, Xs)]

        n_jobs = len(args_list)
        n_processes = 1 if self.channel_processes is None else self.channel_processes
        n_processes = n_jobs if (type(n_processes) == int and n_jobs < n_processes) else n_processes
        if n_processes == 'max' or n_processes > 1:
            try:
                shared_mem_objects = [y, fit_params, cv, scorer, cv_processes]
                job_results = parallel.starmap_jobs(ModelSelector._fit_predict_score, args_list,
                                                    n_cpus=self.channel_processes,
                                                    shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'.format(e))
                print('defaulting to single processor')
                n_processes = 1
        if n_processes is None or n_processes <= 1:
            # print('running a single process with {} jobs'.format(len(args_list)))
            job_results = [ModelSelector._fit_predict_score(*args) for args in args_list]

        models, cv_predictions, model_scores = zip(*job_results)
        self.selected_indices_ = self.score_selector(model_scores)
        # store only the selected models for future use
        self.models = [m if i in set(self.selected_indices_) else None for i, m in enumerate(models)]

        if len(self.selected_indices_ == 1):
            self._expose_predictor_interface(self.models[self.selected_indices_[0]])

        if self.cv_transform == True:
            Xs_t = [p if i in set(self.selected_indices_) else None
                                for i, p in enumerate(cv_predictions)]
        else:
            Xs_t = [model.transform(X) if i in set(self.selected_indices_) else None
                                for i, (model, X) in enumerate(zip(models, Xs))]
        return Xs_t

    def fit(self, Xs, y=None, **fit_params):
        self.fit_transform(Xs, y, **fit_params)

    def transform(self, Xs):
        if hasattr(self, 'models') == False:
            raise utils.FitError('Tranform called before model fitting.')
        return [m.transform(X) if m is not None else None for m, X in zip(self.models, Xs)]

    def predict_with_method(self, Xs, method_name):
        if hasattr(self, 'models') == False:
            raise utils.FitError('prediction attempted before model fitting')
        if len(self.selected_indices_) != 1:
            raise ValueErrror('To predict with a ModelSelector, exactly 1 model must be selected.')
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
            clone.models = [utils.get_clone(m) if m is not None else None for m in self.models]
        return clone


class SelectKBestModels(ModelSelector):

    def __init__(self, predictors, cv=5, scorer='auto', k=1, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectKBestModels.__init__, locals())
        super().__init__(predictors, cv, scorer, RankScoreSelector(k), channel_processes, cv_processes)


class SelectPercentBestModels(ModelSelector):

    def __init__(self, predictors, cv=5, scorer='auto', percent=33, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectPercentBestModels.__init__, locals())
        super().__init__(predictors, cv, scorer, PctRankScoreSelector(percent), channel_processes, cv_processes)


class SelectHighPassModels(ModelSelector):

    def __init__(self, predictors, cv=5, scorer='auto', cutoff=0.0, n_min=1, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectHighPassModels.__init__, locals())
        super().__init__(predictors, cv, scorer, HighPassScoreSelector(cutoff, n_min),
                         channel_processes, cv_processes)


class SelectVarianceHighPassModels(ModelSelector):

    def __init__(self, predictors, cv=5, scorer='auto',
                 variance_cutoff=2.0, get_variance=np.nanstd, get_baseline=np.nanmean,
                 n_min=1, channel_processes=1, cv_processes=1):
        self._params_to_attributes(SelectVarianceHighPassModels.__init__, locals())
        super().__init__(predictors, cv, scorer,
                         VarianceHighPassScoreSelector(variance_cutoff, get_variance, get_baseline, n_min),
                         channel_processes, cv_processes)
