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
from pipecaster.utils import Cloneable, Saveable
import pipecaster.transform_wrappers as transform_wrappers
from pipecaster.score_selection import RankScoreSelector
import pipecaster.parallel as parallel

__all__ = ['SoftVotingClassifier', 'HardVotingClassifier',
           'AggregatingRegressor', 'Ensemble', 'GridSearchEnsemble',
           'MultichannelPredictor', 'ChannelEnsemble']


class SoftVotingClassifier(Cloneable, Saveable):
    """
    Predict using mean predictions of a classifier ensemble.

    This pipeline component takes marginal probabilities from a prior pipeline
    stage and averages them to make an ensemble prediction.  Can be used alone
    or as a meta-predictor within MultichannelPredictor, Ensemble, and
    ChannelEnsemble pipeline components.

    The inputs, which must be concatenated into a single meta-feature matrix in
    a prior stage, are decatenated and predicted classes inferred from the
    order of the meta-feature matrix columns.

    For binary classifiers, the predicted probabilies must sum to 1.0 over the
    classes for each sample so the dropped negative class probabilites can be
    inferred from the positive class.

    Examples
    --------
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        # use style 1:
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = GradientBoostingClassifier()
        meta_clf = pc.SoftVotingClassifier()
        clf.add_layer(pc.ChannelEnsemble(base_clf, meta_clf))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.9117647058823529, 0.8180147058823529, 0.9117647058823529]

        # alternative use style 1:
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.transform_wrappers.SingleChannel(
            GradientBoostingClassifier())
        meta_clf = pc.SoftVotingClassifier()
        clf.add_layer(base_clf)
        clf.add_layer(pc.MultichannelPredictor(meta_clf))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.9117647058823529, 0.8180147058823529, 0.9117647058823529]

        # alternative use style 2:
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.transform_wrappers.SingleChannel(
            GradientBoostingClassifier())
        meta_clf = pc.SoftVotingClassifier()
        clf.add_layer(base_clf)
        clf.add_layer(pc.ChannelConcatenator())
        clf.add_layer(1, meta_clf)
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.9117647058823529, 0.8180147058823529, 0.9117647058823529]
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


class HardVotingClassifier(Cloneable, Saveable):
    """
    Predict using most frequently predicted class in an ensemble.

    This pipeline component takes marginal probabilities from a prior pipeline
    stage and uses them them to make an ensemble prediction.  Predictions are
    made for each classifier in the ensemble by taking the class with the
    highest probability, then an ensemble prediction is made by taking the
    modal prediction.  Can be used alone or as a meta-predictor within
    MultichannelPredictor, Ensemble, and ChannelEnsemble pipeline components.

    The inputs, which must be concatenated into a single meta-feature matrix in
    a prior stage, are decatenated and predicted classes inferred from the
    order of the meta-feature matrix columns.

    For binary classifiers, the predicted probabilies must sum to 1.0 over the
    classes for each sample so the dropped negative class probabilites can be
    inferred from the positive class.

    Examples
    --------
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        # use style 1:
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = GradientBoostingClassifier()
        meta_clf = pc.HardVotingClassifier()
        clf.add_layer(pc.ChannelEnsemble(base_clf, meta_clf))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.8235294117647058, 0.7849264705882353, 0.6911764705882353]

        # alternative use style:
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.transform_wrappers.SingleChannel(
            GradientBoostingClassifier())
        meta_clf = pc.HardVotingClassifier()
        clf.add_layer(base_clf)
        clf.add_layer(pc.MultichannelPredictor(meta_clf))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.8235294117647058, 0.7849264705882353, 0.6911764705882353]

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.transform_wrappers.SingleChannel(
            GradientBoostingClassifier())
        meta_clf = pc.HardVotingClassifier()
        clf.add_layer(base_clf)
        clf.add_layer(pc.ChannelConcatenator())
        clf.add_layer(1, meta_clf)
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.8235294117647058, 0.8161764705882353, 0.6911764705882353]
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
        if n_classes == 2:
            # expand binary classification probs to 2 columns
            n_rows, n_cols = meta_X.shape[0], 2 * meta_X.shape[1]
            Xs_expanded = np.empty((n_rows, n_cols))
            Xs_expanded[:, range(0, n_cols, 2)] =  1.0 - meta_X
            Xs_expanded[:, range(1, n_cols + 1, 2)] = meta_X
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

    def predict(self, X):
        """
        Return the modal class predicted by the base classifiers.
        """
        Xs = self._decatenate(X)
        input_decisions = np.stack([np.argmax(X, axis=1) for X in Xs])
        decisions = scipy.stats.mode(input_decisions, axis=0)[0][0]
        predictions = self.classes_[decisions]
        return predictions


class AggregatingRegressor(Cloneable, Saveable):
    """
    Predict with aggregated outputs of a regressor ensemble.

    This pipeline component takes predictions from a prior pipeline stage that
    have been concatenated into a single meta-feature matrix and converts
    multiple predictions into a single prediction with an aggregator function
    (e.g. np.mean).  Can be used alone or as a meta-predictor within
    MultichannelPredictor, Ensemble, and ChannelEnsemble
    pipeline components.

    Notes
    -----
    Only supports single output regressors.

    Parameters
    ----------
    aggregator : callable, default=np.mean
        Function for converting multiple regression predictions into a single
        prediction.

    Examples
    --------
    ::

        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_regression(n_informative_Xs=5,
                                                  n_random_Xs=5)

        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = GradientBoostingRegressor()
        meta_clf = pc.AggregatingRegressor(np.mean)
        clf.add_layer(pc.ChannelEnsemble(base_clf, meta_clf))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.05944403104941709, 0.08425323185871114, 0.067995808]

        # alternative use style 1:
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.transform_wrappers.SingleChannel(
            GradientBoostingRegressor())
        clf.add_layer(base_clf)
        clf.add_layer(pc.ChannelConcatenator())
        clf.add_layer(1, pc.AggregatingRegressor(np.mean))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.05845840783307943, 0.08014277920579282, 0.0686751928]

        # alternative use style 2:
        Xs, y, _ = pc.make_multi_input_regression(n_informative_Xs=5,
                                              n_random_Xs=5)
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.transform_wrappers.SingleChannel(
            GradientBoostingRegressor())
        clf.add_layer(base_clf)
        clf.add_layer(
            pc.MultichannelPredictor(pc.AggregatingRegressor(np.mean)))
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.01633148118462313, 0.03953337266754531, 0.04450143]
    """
    state_variables = []

    def __init__(self, aggregator=np.mean):
        self._params_to_attributes(AggregatingRegressor.__init__, locals())
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
    SoftVotingClassifier, AggregatingRegressor) or a scikit-learn conformant ML
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
    internal_cv : int, None, or callable, default=None
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int : StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If None : default value of 5.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    scorer : callable or 'auto', default='auto'
        - Performance metric used for model selection.
        - If callable : should return a scalar figure of merit with
          signature: score = scorer(y_true, y_pred).
        - If 'auto' : balanced_accuracy_score if classifier,
          explained_variance_score if regressor.
    score_selector : callable or None, default=None
        - Method for selecting models from the ensemble.
        - If callable : Selector with signature:
          selected_indices = callable(scores).
        - If None :  All models will be retained in the ensemble.
    disable_cv_train : bool, default=False
        - If False : cv predictions will be used to train the meta-predictor.
        - If True : cv predictions not used to train the meta-predictor.
    base_predict_methods : str or list, default='auto'
        - Set the name of the base predictor methods to use.
        - If 'auto' : Use the precedence order specified in
          :mod:`pipecaster.transform_wrappers` to select a predict method.
        - If str other than 'auto' : Name is broadcast over all base
          predictors.
        - If list : List of names, one per base predictor.
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
    Voting ensemble:
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
                         meta_predictor=pc.SoftVotingClassifier(),
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
                         meta_predictor=pc.SoftVotingClassifier(),
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

        pc.cross_val_score(clf, X, y)
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
    state_variables = ['classes_', 'scores_', 'selected_indices_']

    def __init__(self, base_predictors, meta_predictor=None,
                 internal_cv=None, scorer='auto',
                 score_selector=None,
                 disable_cv_train=False,
                 base_predict_methods='auto',
                 base_processes=1, cv_processes=1):
        self._params_to_attributes(Ensemble.__init__, locals())

        if internal_cv is None and score_selector is not None:
            raise ValueError('Must choose an internal cv method when channel '
                             'selection is activated')

        # set the estimator type
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

        if meta_predictor is None and score_selector is None:
            self.score_selector = RankScoreSelector(k=1)

        # expose availbable predictor interface (may change after fit)
        if meta_predictor is not None:
            if meta_predictor._estimator_type != self._estimator_type:
                raise ValueError('Base- and meta- predictors must be the same '
                                 'type (e.g. classifier or regressor).')
            predict_methods = utils.get_predict_methods(meta_predictor)
        else:
            if isinstance(base_predictors, (tuple, list, np.ndarray)):
                predict_methods = [utils.get_predict_methods(p)
                                        for p in base_predictors]
                predict_methods = set.intersection(
                    *[set(pms) for pms in predict_methods])
                predict_methods = list(predict_methods)
            else:
                predict_methods = utils.get_predict_methods(base_predictors)
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
        return self

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
            self.scores_ = scores

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
            predict_methods = utils.get_predict_methods(self.base_models[0])
        elif self.meta_predictor is not None:
            meta_X = np.concatenate(predictions, axis=1)
            self.meta_model = utils.get_clone(self.meta_predictor)
            self.meta_model.fit(meta_X, y, **fit_params)
            if hasattr(self.meta_model, 'classes_'):
                self.classes_ = self.meta_model.classes_
            predict_methods = utils.get_predict_methods(self.meta_model)

        self._set_predictor_interface(predict_methods)
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
            - If method_name is 'predict': ndarray(n_samples,)
            - If method_name is 'predict_proba', 'decision_function', or
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

    def _more_tags(self):
        return {'multichannel': False}

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'base_models'):
            clone.base_models = [utils.get_clone(m) if m is not None else None
                                 for m in self.base_models]
        if hasattr(self, 'meta_model'):
            clone.meta_model = utils.get_clone(self.meta_model)
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
    algorithm (e.g. SoftVotingClassifier, AggregatingRegressor) or a
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
    internal_cv : int, None, or callable, default=None
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int : StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If None : default value of 5.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    scorer : callable or 'auto', default='auto'
        - Performance metric used for model selection.
        - If callable : should return a scalar figure of merit with
          signature: score = scorer(y_true, y_pred).
        - If 'auto' : balanced_accuracy_score if classifier,
          explained_variance_score if regressor.
    score_selector : callable or None, default=None
        - Method for selecting models from the ensemble.
        - If callable : Selector with signature:
          selected_indices = callable(scores).
        - If None :  All models will be retained in the ensemble.
    disable_cv_train : bool, default=False
        - If False : cv predictions will be used to train the meta-predictor.
        - If True : cv predictions not used to train the meta-predictor.
    base_predict_methods : str or list, default='auto'
        - Set the name of the base predictor methods to use.
        - If 'auto' : Use the precedence order specified in
          :mod:`pipecaster.transform_wrappers` to select a predict method.
        - If str other than 'auto' : Name is broadcast over all base
          predictors.
        - If list : List of names, one per base predictor.
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
                             meta_predictor=pc.SoftVotingClassifier(),
                             internal_cv=5, scorer='auto',
                             score_selector=pc.RankScoreSelector(k=2),
                             base_processes=pc.count_cpus())
            clf.fit(X, y)
            clf.get_results_df()
            # output: (outputs a dataframe with the screen results)

            cross_val_score(clf, X, y, scoring='balanced_accuracy', cv=5)
            # output: array([0.9 , 0.85, 0.85, 0.85, 0.65])
    """
    state_variables = ['classes_', 'scores_',
                       'selected_indices_', 'params_list_']

    def __init__(self, param_dict=None, base_predictor_cls=None,
                 meta_predictor=None, internal_cv=5, scorer='auto',
                 score_selector=RankScoreSelector(k=3),
                 disable_cv_train=False,
                 base_transform_method='auto',
                 base_processes=1, cv_processes=1):
        self._params_to_attributes(GridSearchEnsemble.__init__, locals())

    def fit(self, X, y=None, **fit_params):
        self.params_list_ = list(ParameterGrid(self.param_dict))
        base_predictors = [self.base_predictor_cls(**ps)
                           for ps in self.params_list_]
        super().__init__(base_predictors, self.meta_predictor,
                         self.internal_cv, self.scorer,
                         self.score_selector, self.disable_cv_train,
                         self.base_transform_method,
                         self.base_processes, self.cv_processes)

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
            selections = ['+' if i in self.selected_indices_ else '-'
                          for i, p in enumerate(self.params_list_)]
            return selections, self.params_list_, self.scores_

    def get_results_df(self):
        """
        Get pandas DataFrame with screen results.
        """
        selections, params_list, scores = self.get_results()
        df = pd.DataFrame({'selections':selections,
                           'parameters':params_list, 'score':scores})
        df.sort_values('score', ascending=False, inplace=True)
        return df.set_index('score')


class MultichannelPredictor(Cloneable, Saveable):
    """
    Predict with mutliple intput channels.

    This pipeline component concatenates mutliple inputs to create a single
    feature matrix used as input for an ML or voting/aggregating algorithm.
    It outputs a single prediction into the first channel.  Can be used for
    either normal prediction or meta-prediction.

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
    """
    state_variables = ['classes_']

    def __init__(self, predictor):
        self._params_to_attributes(MultichannelPredictor.__init__, locals())
        utils.enforce_fit(predictor)
        utils.enforce_predict(predictor)
        self._estimator_type = utils.detect_predictor_type(predictor)
        if self._estimator_type is None:
            raise TypeError('could not detect predictor type')
        predict_method_names = utils.get_predict_methods(predictor)
        self._set_predictor_interface(predict_method_names)

    def _set_predictor_interface(self, predict_method_names):
        for method_name in utils.recognized_pred_methods:
            is_available = method_name in predict_method_names
            is_exposed = hasattr(self, method_name)
            if is_available and (is_exposed is False):
                prediction_method = functools.partial(self.predict_with_method,
                                                      method_name=method_name)
                setattr(self, method_name, prediction_method)
            elif is_exposed and (is_available is False):
                delattr(self, method_name)
        return self

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
        predict_method_names = utils.get_predict_methods(self.model)
        self._set_predictor_interface(predict_method_names)
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
        if hasattr(self, 'model'):
            clone.model = utils.get_clone(self.model)
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
    SoftVotingClassifier, AggregatingRegressor) or a scikit-learn conformant ML
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
    base_predictors : predictor instance or list of predictor instances
        Ensemble of scikit-learn conformant base predictors (either all
        classifiers or all regressors).
    meta_predictor : predictor instance, default=None
        Scikit-learn conformant classifier or regressor that makes predictions
        from the base predictor inferences.  This parameter is optional when
        the internal_cv and score_selector parameters are set, in which case
        predictions from the top performing model will be used in the absence
        of a meta-predictor.
    internal_cv : int, None, or callable, default=None
        - Function for train/test subdivision of the training data.  Used to
          estimate performance of base classifiers and ensure they do not
          generate predictions from their training samples during
          meta-predictor training.
        - If int : StratifiedKfold(n_splits=internal_cv) if classifier or
          KFold(n_splits=internal_cv) if regressor.
        - If None : default value of 5.
        - If callable: Assumed to be split generator like scikit-learn KFold.
    scorer : callable or 'auto', default='auto'
        - Performance metric used for model selection.
        - If callable : should return a scalar figure of merit with
          signature: score = scorer(y_true, y_pred).
        - If 'auto' : balanced_accuracy_score if classifier,
          explained_variance_score if regressor.
    score_selector : callable or None, default=None
        - Method for selecting models from the ensemble.
        - If callable : Selector with signature:
          selected_indices = callable(scores).
        - If None :  All models will be retained in the ensemble.
    disable_cv_train : bool, default=False
        - If False : cv predictions will be used to train the meta-predictor.
        - If True : cv predictions not used to train the meta-predictor.
    base_predict_methods : str or list, default='auto'
        - Set the name of the base predictor methods to use.
        - If 'auto' : Use the precedence order specified in
          :mod:`pipecaster.transform_wrappers` to select a predict method.
        - If str other than 'auto' : Name is broadcast over all channels.
        - If list : List of names, one per channel.
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
    Measure accuracy of KNeighborsClassifier on each of 10 input channels using
    internal cross validation, select the top 3 performers, and metapredict
    with SVC:
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
        clf.add_layer(pc.ChannelEnsemble(
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

    def __init__(self, base_predictors, meta_predictor=None,
                 internal_cv=None, scorer='auto',
                 score_selector=None,
                 disable_cv_train=False,
                 base_predict_methods='auto',
                 base_processes=1, cv_processes=1):
        self._params_to_attributes(ChannelEnsemble.__init__, locals())

        if internal_cv is None and score_selector is not None:
            raise ValueError('Must choose an internal cv method when channel '
                             'selection is activated')

        # set the estimator type
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

        if meta_predictor is None and score_selector is None:
            self.score_selector = RankScoreSelector(k=1)

        # expose availbable predictor interface (may change after fit)
        if meta_predictor is not None:
            if meta_predictor._estimator_type != self._estimator_type:
                raise ValueError('Base- and meta- predictors must be the same '
                                 'type (e.g. classifier or regressor).')
            predict_methods = utils.get_predict_methods(meta_predictor)
        else:
            if isinstance(base_predictors, (tuple, list, np.ndarray)):
                predict_methods = [utils.get_predict_methods(p)
                                        for p in base_predictors]
                predict_methods = set.intersection(
                    *[set(pms) for pms in predict_methods])
                predict_methods = list(predict_methods)
            else:
                predict_methods = utils.get_predict_methods(base_predictors)
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
        return self

    @staticmethod
    def _fit_job(predictor, X, y, internal_cv, base_predict_method,
                 cv_processes, scorer, fit_params):
        if X is None:
            return None, None, None, None
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
            self.scores_ = scores

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
            predict_methods = utils.get_predict_methods(model)
        elif self.meta_predictor is not None:
            meta_X = np.concatenate(predictions, axis=1)
            self.meta_model = utils.get_clone(self.meta_predictor)
            self.meta_model.fit(meta_X, y, **fit_params)
            if hasattr(self.meta_model, 'classes_'):
                self.classes_ = self.meta_model.classes_
            predict_methods = utils.get_predict_methods(self.meta_model)

        self._set_predictor_interface(predict_methods)
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

    def _more_tags(self):
        return {'multichannel': False}

    def get_clone(self):
        """
        Get a stateful clone.
        """
        clone = super().get_clone()
        if hasattr(self, 'base_models'):
            clone.base_models = [utils.get_clone(m) if m is not None else None
                                 for m in self.base_models]
        if hasattr(self, 'meta_model'):
            clone.meta_model = utils.get_clone(self.meta_model)
        return clone
