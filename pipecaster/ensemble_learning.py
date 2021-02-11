"""
Pipeline components for voting and model stacking.
"""

import numpy as np
import pandas as pd
import ray
import scipy.stats
import functools

from sklearn.metrics import explained_variance_score, balanced_accuracy_score
from sklearn.model_selection import ParameterGrid

import pipecaster.utils as utils
from pipecaster.utils import Cloneable, Saveable
import pipecaster.transform_wrappers as transform_wrappers
from pipecaster.score_selection import RankScoreSelector
import pipecaster.parallel as parallel

__all__ = ['SoftVotingClassifier', 'HardVotingClassifier',
           'AggregatingRegressor', 'EnsemblePredictor', 'GridSearchEnsemble',
           'MultichannelPredictor', 'ChannelEnsemblePredictor']


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

    Example
    -------
    from sklearn.ensemble import GradientBoostingClassifier
    import pipecaster as pc

    Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                  n_random_Xs=7)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(GradientBoostingClassifier())
    clf.add_layer(pc.ChannelConcatenator())
    clf.add_layer(pc.SoftVotingClassifier())
    pc.cross_val_score(clf, Xs, y, cv=3)
    >>>[0.8235294117647058, 0.7849264705882353, 0.7886029411764706]

    # alternative use style:
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(GradientBoostingClassifier())
    clf.add_layer(pc.MultichannelPredictor(pc.SoftVotingClassifier()))
    pc.cross_val_score(clf, Xs, y, cv=3)
    >>>[0.8823529411764706, 0.875, 0.8474264705882353]
    """
    state_variables = ['classes_']

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
        if meta_X.shape[1] % n_classes != 0:
            raise ValueError(
                '''Number of meta-features not divisible by number of classes.
                This can happen if base classifiers were trained on different
                subsamples with different number of classes.  Pipecaster uses
                StratifiedKFold to prevent this, but GroupKFold can lead to
                violations.  Someone needs to make StratifiedGroupKFold''')
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

    Example
    -------
    from sklearn.ensemble import GradientBoostingClassifier
    import pipecaster as pc

    Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                  n_random_Xs=7)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(GradientBoostingClassifier())
    clf.add_layer(pc.ChannelConcatenator())
    clf.add_layer(pc.HardVotingClassifier())
    pc.cross_val_score(clf, Xs, y, cv=3)
    >>>[0.7352941176470589, 0.8198529411764706, 0.6911764705882353]

    # alternative use style:
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(GradientBoostingClassifier())
    clf.add_layer(pc.MultichannelPredictor(pc.HardVotingClassifier()))
    pc.cross_val_score(clf, Xs, y, cv=3)
    >>>[0.7647058823529411, 0.7904411764705883, 0.8455882352941176]
    """
    state_variables = ['classes_']

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
    Currently only supports single output regressors.

    Example
    -------
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    import pipecaster as pc

    Xs, y, _ = pc.make_multi_input_regression(n_informative_Xs=5,
                                              n_random_Xs=5)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(GradientBoostingRegressor(n_estimators=5))
    clf.add_layer(pc.ChannelConcatenator())
    clf.add_layer(pc.AggregatingRegressor(np.mean))
    pc.cross_val_score(clf, Xs, y, cv=3)
    >>>[0.014826094226239817, 0.0048964785149892, 0.0023546593895911183]

    # alternative use style:
    Xs, y, _ = pc.make_multi_input_regression(n_informative_Xs=5,
                                          n_random_Xs=5)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(GradientBoostingRegressor(n_estimators=5))
    clf.add_layer(pc.MultichannelPredictor(pc.AggregatingRegressor(np.mean)))
    pc.cross_val_score(clf, Xs, y, cv=3)
    >>>[0.019732744693820692, 0.011203683351741489, 0.011748087942798469]
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


class EnsemblePredictor(Cloneable, Saveable):
    """
    Ensemble predictor with multiple models for a single input channel.

    EnsemblePredictor makes inferences using a set of base predictors
    and a meta-predictor that uses their inferences as input features.  The
    meta-predictor may be a voting or aggregating method (e.g.
    SoftVotingClassifier, AggregatingRegressor) or a scikit-learn conformant
    predictor.  In the latter case, it is standard practice to use internal
    cross validation training of the base classifiers to prevent them from
    making inferences on training samples.  To enable internal cross validation
    training, set the internal_cv constructor argument of EnsemblePredictor.

    EnsemblePredictor also takes advantage of internal cross validation
    to enable in-pipeling screening of base predictors during model
    fitting. To enable model selection, provide a score_selector (e.g.
    those found in the score_selection module) to the contructor.

    Parameters
    ----------
    base_predictors : predictor instance or list of predictor instances,
    default=None
        Ensemble of scikit-learn conformant base predictors (either all
        classifiers or all regressors).  One predictor is trained per input
        channel.  If a single predictor, it will be broadcast across all
        channels.  If a list of predictors, the list length must be the same
        as the number of input channels.
    meta_predictor : predictor instance, default=None
        Scikit-learn conformant classifier or regresor that makes predictions
        from the inference of the base predictors.
    internal_cv : int, None, sklearn cross-validation generator, default=None
        Method for estimating performance of base classifiers and ensuring that
        they not generate predictions from their training samples during
        training of a meta-predictor.
        If int i: If classifier, StratifiedKfold(n_splits=i), if regressor
            KFold(n_splits=i) for
        If None: default value of 5
        If not int or None: Will be treated as a scikit-learn conformant split
            generator like KFold
    scorer : callable or 'auto', default='auto'
        - Figure of merit score used for model selection:
        - If callable: should return a scalar figure of merit with signature:
          score = scorer(y_true, y_pred).
        - If 'auto': balanced_accuracy_score for classifiers,
          explained_variance_score for regressors.
    score_selector : callable, default=None
        A callable with the pattern: selected_indices = callable(scores)
        If None, all models will be retained in the ensemble.
    disable_cv_train : bool, default=False
        If True, internal cv splits will not be used to generate features for
        the meta-predictor.  Instead, the whole training set will be used.
    base_predict_methods : str or list, default='auto'
        Set the base predictor methods to use. If 'auto', the precedence order
        specified in the transform_wrappers module will be used.  If a single
        string, the value will be broadcast over all channels.
    base_processes : int or 'max', default=1
        The number of parallel processes to run for base predictor fitting.
        If 'max', all available CPUs will be used.
    cv_processes : int or 'max', default=1
        The number of parallel processes to run for internal cross validation.
        If 'max', all available CPUs will be used.

    Example
    -------
    ::

        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        import pipecaster as pc

        X, y = make_classification(n_classes=2, n_samples=500, n_features=100,
                                   n_informative=5, class_sep=0.6)

        predictors = [MLPClassifier(), LogisticRegression(),
                      KNeighborsClassifier(),  GradientBoostingClassifier(),
                      RandomForestClassifier(), GaussianNB()]

        clf = pc.EnsemblePredictor(
                         base_predictors=predictors,
                         meta_predictor=pc.SoftVotingClassifier(),
                         internal_cv=5, scorer='auto',
                         score_selector=pc.RankScoreSelector(k=2),
                         base_processes=pc.count_cpus())
        pc.cross_val_score(clf, X, y)
        # output: [0.7066838783706253, 0.7064687320711417, 0.6987951807228916]

        clf.fit(X, y)
        # Models selected by the EnsemblePredictor:
        [p for i, p in enumerate(predictors) if i in clf.get_support()]
        # output: [GradientBoostingClassifier(), RandomForestClassifier()]
        """
    state_variables = ['classes_', 'scores_', 'selected_indices_']

    def __init__(self, base_predictors=None, meta_predictor=None,
                 internal_cv=None, scorer='auto',
                 score_selector=None,
                 disable_cv_train=False,
                 base_predict_methods='auto',
                 base_processes=1, cv_processes=1):
        self._params_to_attributes(EnsemblePredictor.__init__, locals())

        if isinstance(base_predictors, (tuple, list, np.ndarray)):
            estimator_types = [p._estimator_type for p in base_predictors]
            if len(set(estimator_types)) != 1:
                raise TypeError('base_predictors must be of uniform type '
                                '(e.g. all classifiers or all regressors)')
            self._estimator_type = estimator_types[0]
        else:
            self._estimator_type = base_predictors._estimator_type

        if scorer == 'auto':
            if utils.is_classifier(self):
                self.scorer = balanced_accuracy_score
            elif utils.is_regressor(self):
                self.scorer = explained_variance_score
            else:
                raise AttributeError('predictor type required for automatic '
                                     'assignment of scoring metric')
        self._expose_predictor_interface(meta_predictor)
        if internal_cv is None and score_selector is not None:
            raise ValueError('Must choose an internal cv method when channel '
                             'selection is activated')

    def _expose_predictor_interface(self, meta_predictor):
        for method_name in utils.recognized_pred_methods:
            if hasattr(meta_predictor, method_name):
                prediction_method = functools.partial(self.predict_with_method,
                                                      method_name=method_name)
                setattr(self, method_name, prediction_method)

    @staticmethod
    def _fit_job(predictor, X, y, internal_cv, base_predict_method,
                 cv_processes, scorer, fit_params):
        model = transform_wrappers.SingleChannel(predictor,
                                                 base_predict_method)
        model = utils.get_clone(model)
        predictions = model.fit_transform(X, y, **fit_params)

        cv_predictions, score = None, None
        if internal_cv is not None:
            model_cv = utils.get_clone(predictor)
            model_cv = transform_wrappers.SingleChannelCV(
                                                model_cv, base_predict_method,
                                                internal_cv, cv_processes,
                                                scorer)
            cv_predictions = model_cv.fit_transform(X, y, **fit_params)
            score = model_cv.score_

        return model, predictions, cv_predictions, score

    def fit(self, X, y=None, **fit_params):

        if self._estimator_type == 'classifier' and y is not None:
            self.classes_, y = np.unique(y, return_inverse=True)

        if isinstance(self.base_predict_methods, (tuple, list, np.ndarray)):
            if len(self.base_predict_methods) != len(self.base_predictors):
                raise utils.FitError('Number of base predict methods did not '
                                     'match number of base predictors.')
            methods = self.base_predict_methods
        else:
            methods = [self.base_predict_methods for p in self.base_predictors]

        args_list = [(p, X, y, self.internal_cv, m, self.cv_processes,
                      self.scorer, fit_params)
                     for p, m in zip(self.base_predictors, methods)]

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
                                EnsemblePredictor._fit_job, args_list,
                                n_cpus=n_processes,
                                shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'
                      .format(e))
                print('defaulting to single processor')
                n_processes = 1

        if type(n_processes) == int and n_processes <= 1:
            fit_results = [EnsemblePredictor._fit_job(*args)
                           for args in args_list]

        models, predictions, cv_predictions, scores = zip(*fit_results)

        if scores is not None:
            self.scores_ = scores

        if self.score_selector is not None:
            self.selected_indices_ = self.score_selector(scores)
        else:
            self.selected_indices_ = list(range(len(Xs)))

        if self.internal_cv is not None and self.disable_cv_train is False:
            predictions = cv_predictions

        self.base_models = [m for i, m in enumerate(models)
                            if i in self.selected_indices_]
        predictions = [p for i, p in enumerate(predictions)
                       if i in self.selected_indices_]

        meta_X = np.concatenate(predictions, axis=1)
        self.meta_model = utils.get_clone(self.meta_predictor)
        self.meta_model.fit(meta_X, y, **fit_params)
        if hasattr(self.meta_model, 'classes_'):
            self.classes_ = self.meta_model.classes_

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
        Make channel ensemble predictions.

        Parameters
        ----------
        X: ndarray.shape(n_samples, n_features)
            Feature matrix.
        method_name: str
            Name of the meta-predictor prediction method to invoke.

        Returns
        -------
        Ensemble predictions.
        if method_name is 'predict': ndarray(n_samples,)
        if method_name is 'predict_proba', 'decision_function', or
            'predict_log_proba':
            ndarray(n_sample, n_classes)
        """
        if (hasattr(self, 'base_models') is False or
                hasattr(self, 'meta_model') is False):
            raise utils.FitError('prediction attempted before model fitting')
        predictions_list = [m.transform(X) for m in self.base_models]
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
            clone.base_models = [utils.get_clone(m) if m is not None else None
                                 for m in self.base_models]
        if hasattr(self, 'meta_model'):
            clone.meta_model = utils.get_clone(self.meta_model)
        return clone


class GridSearchEnsemble(EnsemblePredictor):
    """
    Ensemble predictor stack that screens base predictor parameters and
    makes an ensemble of base models with the best parameters.  Supports the
    scikit-learn estimator/predictor interface.

    Parameters
    ----------
    param_dict: dict, default=None
        A dict containing lists of parameters indexed by the parameter name.
        The lists indicate parameter values to be screened in a grid search
        (Cartesian product of the lists).  E.g.:
        {'param1':[value1, value2, value3], 'param2':[value1, value2, value3]}
    base_predictor_cls: class, default=None
        Class to be used for base prediction.  Must implement the
        scikit-learn estimator and predictor interfaces.
    meta_predictor: class instance, default=None
        Class instance to be used for meta-prediction using concatenated base
        predictor outputs as meta-features.  Must implement the scikit-learn
        estimator and predictor interfaces.
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

    Example
    -------
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    import pipecaster as pc

    screen = {
         'learning_rate':[0.1, 10],
         'n_estimators':[5, 25],
    }

    X, y_single = make_classification()
    clf = pc.GridSearchEnsemble(
                     param_dict=screen,
                     base_predictor_cls=GradientBoostingClassifier,
                     meta_predictor=pc.SoftVotingClassifier(),
                     internal_cv=5, scorer='auto',
                     score_selector=pc.RankScoreSelector(k=2),
                     base_processes=pc.count_cpus())
    clf.fit(X, y_single)
    clf.get_results_df()
    >>> (outputs a dataframe with the screen results)

    cross_val_score(clf, X, y_single, scoring='balanced_accuracy', cv=5)
    >>>array([0.9 , 0.85, 0.85, 0.85, 0.65])
    """
    state_variables = ['classes_', 'scores_',
                       'selected_indices_', 'params_list_']

    def __init__(self, param_dict=None, base_predictor_cls=None,
                 meta_predictor=None, internal_cv=5, scorer='auto',
                 score_selector=RankScoreSelector(k=3),
                 base_transform_method=None,
                 base_processes=1, cv_processes=1):
        self._params_to_attributes(GridSearchEnsemble.__init__, locals())

    def fit(self, X, y=None, **fit_params):
        self.params_list_ = list(ParameterGrid(self.param_dict))
        base_predictors = [self.base_predictor_cls(**ps)
                           for ps in self.params_list_]
        super().__init__(base_predictors, self.meta_predictor,
                         self.internal_cv, self.scorer,
                         self.score_selector, self.base_transform_method,
                         self.base_processes, self.cv_processes)
        super().fit(X, y, **fit_params)

    def get_results(self):
        """
        Get the results of the grid search screen if available.
        """
        if hasattr(self, 'base_models') is True:
            selections = ['+' if i in self.selected_indices_ else '-'
                          for i, p in enumerate(self.params_list_)]
            return selections, self.params_list_, self.scores_

    def get_results_df(self):
        """
        Get a Pandas DataFrame with the results of the grid search screen if
        available.
        """
        selections, params_list, scores = self.get_results()
        df = pd.DataFrame({'selections':selections,
                           'parameters':params_list, 'score':scores})
        df.sort_values('score', ascending=False, inplace=True)
        return df.set_index('score')


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

    Example
    -------
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    import pipecaster as pc

    Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                  n_random_Xs=7)
    clf = pc.MultichannelPipeline(n_channels=10)
    clf.add_layer(GradientBoostingClassifier(), pipe_processes='max')
    clf.add_layer(pc.MultichannelPredictor(SVC()))
    pc.cross_val_score(clf, Xs, y, cv=3)
    >>>[0.9411764705882353, 0.8768382352941176, 0.8823529411764706]
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

    def get_descriptor(self, verbose=0, params=None):
        return utils.get_descriptor(self.predictor, verbose, params) + '_MC'


class ChannelEnsemblePredictor(Cloneable, Saveable):
    """
    Ensemble predictor with one model per input channel.

    ChannelEnsemblePredictor makes inferences using a set of base predictors
    and a meta-predictor that uses their inferences as input features.  The
    meta-predictor may be a voting or aggregating method (e.g.
    SoftVotingClassifier, AggregatingRegressor) or a scikit-learn conformant
    predictor.  In the latter case, it is standard practice to use internal
    cross validation training of the base classifiers to prevent them from
    making inferences on training samples.  To enable internal cross validation
    training, set the internal_cv constructor argument of
    ChannelEnsemblePredictor.

    ChannelEnsemblePredictor also takes advantage of internal cross validation
    to enable in-pipeling screening of base predictors during model
    fitting. To enable model selection, provide a score_selector (e.g.
    those found in the score_selection module) to the contructor.

    Parameters
    ----------
    base_predictors : predictor instance or list of predictor instances,
    default=None
        Ensemble of scikit-learn conformant base predictors (either all
        classifiers or all regressors).  One predictor is trained per input
        channel.  If a single predictor, it will be broadcast across all
        channels.  If a list of predictors, the list length must be the same
        as the number of input channels.
    meta_predictor : predictor instance, default=None
        Scikit-learn conformant classifier or regresor that makes predictions
        from the inference of the base predictors.
    internal_cv : int, None, sklearn cross-validation generator, default=None
        Method for estimating performance of base classifiers and ensuring that
        they not generate predictions from their training samples during
        training of a meta-predictor.
        If int i: If classifier, StratifiedKfold(n_splits=i), if regressor
            KFold(n_splits=i) for
        If None: default value of 5
        If not int or None: Will be treated as a scikit-learn conformant split
            generator like KFold
    scorer : callable or 'auto', default='auto'
        Figure of merit score used for model selection.
        If callable: should return a scalar figure of merit with signature:
            score = scorer(y_true, y_pred) and .
        If 'auto': balanced_accuracy_score for classifiers,
            explained_variance_score for regressors
    score_selector : callable, default=None
        A callable with the pattern: selected_indices = callable(scores)
        If None, all models will be retained in the ensemble.
    disable_cv_train : bool, default=False
        If True, internal cv splits will not be used to generate features for
        the meta-predictor.  Instead, the whole training set will be used.
    base_predict_methods : str or list, default='auto'
        Set the base predictor methods to use. If 'auto', the precedence order
        specified in the transform_wrappers module will be used.  If a single
        string, the value will be broadcast over all channels.
    base_processes : int or 'max', default=1
        The number of parallel processes to run for base predictor fitting.
        If 'max', all available CPUs will be used.
    cv_processes : int or 'max', default=1
        The number of parallel processes to run for internal cross validation.
        If 'max', all available CPUs will be used.

    Example
    -------
    ::

        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        Xs, y, X_types = pc.make_multi_input_classification(n_informative_Xs=3,
                                                            n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        clf.add_layer(pc.ChannelEnsemblePredictor(
                        KNeighborsClassifier(), SVC(),
                        internal_cv=5,
                        score_selector=pc.RankScoreSelector(3)),
                      pipe_processes='max')

        pc.cross_val_score(clf, Xs, y)
        # output=> [0.794116, 0.8805, 0.8768]

        import pandas as pd
        clf.fit(Xs, y)
        selected_indices = clf.get_model(1,0).get_support()
        selection_mask = [True if i in selected_indices else False
                          for i, X in enumerate(Xs)]
        pd.DataFrame({'selections':selection_mask, 'input type':X_types})
    """
    state_variables = ['classes_', 'scores_', 'selected_indices_']

    def __init__(self, base_predictors=None, meta_predictor=None,
                 internal_cv=None, scorer='auto',
                 score_selector=None,
                 disable_cv_train=False,
                 base_predict_methods='auto',
                 base_processes=1, cv_processes=1):
        self._params_to_attributes(ChannelEnsemblePredictor.__init__, locals())

        if isinstance(base_predictors, (tuple, list, np.ndarray)):
            estimator_types = [p._estimator_type for p in base_predictors]
            if len(set(estimator_types)) != 1:
                raise TypeError('base_predictors must be of uniform type '
                                '(e.g. all classifiers or all regressors)')
            self._estimator_type = estimator_types[0]
        else:
            self._estimator_type = base_predictors._estimator_type

        if scorer == 'auto':
            if utils.is_classifier(self):
                self.scorer = balanced_accuracy_score
            elif utils.is_regressor(self):
                self.scorer = explained_variance_score
            else:
                raise AttributeError('predictor type required for automatic '
                                     'assignment of scoring metric')
        self._expose_predictor_interface(meta_predictor)
        if internal_cv is None and score_selector is not None:
            raise ValueError('Must choose an internal cv method when channel '
                             'selection is activated')

    def _expose_predictor_interface(self, meta_predictor):
        for method_name in utils.recognized_pred_methods:
            if hasattr(meta_predictor, method_name):
                prediction_method = functools.partial(self.predict_with_method,
                                                      method_name=method_name)
                setattr(self, method_name, prediction_method)

    @staticmethod
    def _fit_job(predictor, X, y, internal_cv, base_predict_method,
                 cv_processes, scorer, fit_params):
        model = transform_wrappers.SingleChannel(predictor,
                                                 base_predict_method)
        model = utils.get_clone(model)
        predictions = model.fit_transform(X, y, **fit_params)

        cv_predictions, score = None, None
        if internal_cv is not None:
            model_cv = utils.get_clone(predictor)
            model_cv = transform_wrappers.SingleChannelCV(
                                                model_cv, base_predict_method,
                                                internal_cv, cv_processes,
                                                scorer)
            cv_predictions = model_cv.fit_transform(X, y, **fit_params)
            score = model_cv.score_

        return model, predictions, cv_predictions, score

    def fit(self, Xs, y=None, **fit_params):

        if self._estimator_type == 'classifier' and y is not None:
            self.classes_, y = np.unique(y, return_inverse=True)

        if isinstance(self.base_predictors, (tuple, list, np.ndarray)):
            if len(self.base_predictors) != len(Xs):
                raise utils.FitError('Number of base predictor did not match '
                                     'number of input channels')
            predictors = self.base_predictors
        else:
            predictors = [self.base_predictors for X in Xs]

        if isinstance(self.base_predict_methods, (tuple, list, np.ndarray)):
            if len(self.base_predict_methods) != len(Xs):
                raise utils.FitError('Number of base predict methods did not '
                                     'match number of input channels')
            methods = self.base_predict_methods
        else:
            methods = [self.base_predict_methods for X in Xs]

        args_list = [(p, X, y, self.internal_cv, m, self.cv_processes,
                      self.scorer, fit_params)
                     for p, X, m in zip(predictors, Xs, methods)]

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
                                ChannelEnsemblePredictor._fit_job, args_list,
                                n_cpus=n_processes,
                                shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'
                      .format(e))
                print('defaulting to single processor')
                n_processes = 1

        if type(n_processes) == int and n_processes <= 1:
            fit_results = [ChannelEnsemblePredictor._fit_job(*args)
                           for args in args_list]

        models, predictions, cv_predictions, scores = zip(*fit_results)
        self.base_models = models
        if scores is not None:
            self.scores_ = scores

        if self.score_selector is not None:
            self.selected_indices_ = self.score_selector(scores)
        else:
            self.selected_indices_ = list(range(len(Xs)))

        if self.internal_cv is not None and self.disable_cv_train is False:
            predictions = cv_predictions

        self.base_models = [m if i in self.selected_indices_ else None
                            for i, m in enumerate(self.base_models)]
        predictions = [p for i, p in enumerate(predictions)
                       if i in self.selected_indices_]

        meta_X = np.concatenate(predictions, axis=1)
        self.meta_model = utils.get_clone(self.meta_predictor)
        self.meta_model.fit(meta_X, y, **fit_params)
        if hasattr(self.meta_model, 'classes_'):
            self.classes_ = self.meta_model.classes_

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
        Make channel ensemble predictions.

        Parameters
        ----------
        Xs: list of (ndarray.shape(n_samples, n_features) or None)
        method_name: str
            Name of the meta-predictor prediction method to invoke.

        Returns
        -------
        Ensemble predictions.
        if method_name is 'predict': ndarray(n_samples,)
        if method_name is 'predict_proba', 'decision_function', or
            'predict_log_proba':
            ndarray(n_sample, n_classes)
        """
        if (hasattr(self, 'base_models') is False or
                hasattr(self, 'meta_model') is False):
            raise utils.FitError('prediction attempted before model fitting')
        predictions_list = [m.transform(X) for i, (m, X) in
                            enumerate(zip(self.base_models, Xs))
                            if i in self.selected_indices_]
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
            clone.base_models = [utils.get_clone(m) if m is not None else None
                                 for m in self.base_models]
        if hasattr(self, 'meta_model'):
            clone.meta_model = utils.get_clone(self.meta_model)
        return clone
