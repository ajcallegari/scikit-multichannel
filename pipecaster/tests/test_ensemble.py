import numpy as np
import unittest
import random
import warnings
from scipy.stats import pearsonr

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    StackingClassifier, StackingRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, explained_variance_score, \
    balanced_accuracy_score

import pipecaster.utils as utils
from pipecaster.testing_utils import make_multi_input_classification, \
    make_multi_input_regression
import pipecaster.transform_wrappers as transform_wrappers
from pipecaster.transform_wrappers import make_transformer
from pipecaster.score_selection import RankScoreSelector
from pipecaster.multichannel_pipeline import MultichannelPipeline
from pipecaster.ensemble_learning import Ensemble, SoftVotingClassifier, \
    SoftVotingDecision, MultichannelPredictor, HardVotingClassifier, \
    AggregatingRegressor
from pipecaster.cross_validation import cross_val_score

class TestVoting(unittest.TestCase):

    def test_soft_voting(self, verbose=0, seed=42):
        Xs, y, _ = make_multi_input_classification(n_informative_Xs=5,
                                              n_random_Xs=2, random_state=seed)
        clf = MultichannelPipeline(n_channels=7)
        clf.add_layer(StandardScaler())
        base_clf = KNeighborsClassifier()
        base_clf = transform_wrappers.SingleChannel(base_clf)
        clf.add_layer(base_clf)
        clf.add_layer(SoftVotingClassifier())
        scores = cross_val_score(clf, Xs, y, score_method='predict',
                                scorer=balanced_accuracy_score)
        score = np.mean(scores)
        if verbose > 0:
            print('accuracy = {}'.format(score))

        self.assertTrue(score > 0.80)

    def test_soft_voting_decision(self, verbose=0, seed=42):

        Xs, y, _ = make_multi_input_classification(n_informative_Xs=6,
                                                   n_random_Xs=3,
                                                   random_state=seed)

        clf = MultichannelPipeline(n_channels=9)
        clf.add_layer(StandardScaler())
        base_clf = make_transformer(SVC(),
                                    transform_method='decision_function')
        clf.add_layer(base_clf)
        meta_clf1 = SoftVotingDecision()
        clf.add_layer(3, meta_clf1, 3, meta_clf1, 3, meta_clf1)
        meta_clf2 = MultichannelPredictor(GradientBoostingClassifier())
        clf.add_layer(meta_clf2)
        scores = cross_val_score(clf, Xs, y, score_method='predict',
                                scorer=balanced_accuracy_score)
        score = np.mean(scores)
        if verbose > 0:
            print('accuracy = {}'.format(score))

        self.assertTrue(score > 0.85)

    def test_hard_voting(self, verbose=0, seed=42):
        Xs, y, _ = make_multi_input_classification(
                        n_informative_Xs=10, n_random_Xs=0,
                        class_sep=2, random_state=seed)
        clf = MultichannelPipeline(n_channels=10)
        clf.add_layer(StandardScaler())
        base_clf = KNeighborsClassifier()
        base_clf = make_transformer(base_clf, transform_method='predict')
        clf.add_layer(base_clf)
        clf.add_layer(HardVotingClassifier())
        scores = cross_val_score(clf, Xs, y, score_method='predict',
                                scorer=balanced_accuracy_score)
        score = np.mean(scores)
        if verbose > 0:
            print('accuracy = {}'.format(score))

        self.assertTrue(score > 0.90)


class TestAggragation(unittest.TestCase):

    def test_aggregating_regressor(self, verbose=0, seed=42):
        Xs, y, _ = make_multi_input_regression(n_informative_Xs=3,
                                               random_state=seed)

        clf = MultichannelPipeline(n_channels=3)
        base_clf = GradientBoostingRegressor(n_estimators=50)
        clf.add_layer(make_transformer(base_clf))
        clf.add_layer(AggregatingRegressor(np.mean))
        cross_val_score(clf, Xs, y, cv=3)
        scores = cross_val_score(clf, Xs, y, score_method='predict',
                                scorer=explained_variance_score)
        score = np.mean(scores)
        if verbose > 0:
            print('accuracy = {}'.format(score))

        self.assertTrue(score > 0.3)

class TestEnsembleMetaprediction(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore')

    def tearDown(self):
        warnings.resetwarnings()

    def test_discrimination_cls(self, verbose=0, seed=42):
        """
        Determine if Ensemble can discriminate between dummy classifiers and LogisticRegression classifiers
        """
        X, y = make_classification(n_samples=500, n_features=20, n_informative=15, class_sep=1, random_state=seed)

        base_classifiers = [DummyClassifier(strategy='stratified') for i in range(5)]
        base_classifiers.extend([LogisticRegression() for i in range(5)])
        random.shuffle(base_classifiers)
        informative_mask = [True if type(c) == LogisticRegression else False for c in base_classifiers]

        mclf = MultichannelPipeline(n_channels=1)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(Ensemble(base_classifiers, SVC(), internal_cv=5, score_selector=RankScoreSelector(k=5)))
        mclf.fit([X], y)
        selected_indices = mclf.get_model(layer_index=1, model_index=0).get_support()
        selection_mask = [True if i in selected_indices else False for i in range(len(base_classifiers))]
        if verbose > 0:
            n_correct = sum([1 for i, s in zip(informative_mask, selection_mask) if i and s])
            print('\n\ncorrectly selected {}/5 LogigistRegression classifiers'.format(n_correct))
            print('incorrectly selected {}/5 DummyClassifiers\n\n'.format(5- n_correct))
        self.assertTrue(np.array_equal(selection_mask, informative_mask),
                        'Ensemble failed to discriminate between dummy classifiers and LogisticRegression')

    def test_compare_to_StackingClassifier(self, verbose=0, seed=42):
        """
        Determine if Ensemble with dummies correctly selects the real predictors and gives similar
        performance to scikit-learn StackingClassifier trained without dummies.
        """

        X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, class_sep=0.5, random_state=seed)

        classifiers = [LogisticRegression(random_state=seed),
                       KNeighborsClassifier(),
                       RandomForestClassifier(random_state=seed)]
        dummy_classifiers = [DummyClassifier(strategy='stratified', random_state=seed) for repeat in range(100)]
        all_classifiers = classifiers + dummy_classifiers
        random.shuffle(all_classifiers)

        mclf = MultichannelPipeline(n_channels=1)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(Ensemble(all_classifiers, SVC(random_state=seed), internal_cv=5, score_selector=RankScoreSelector(k=3)))
        pc_score_all = np.mean(cross_val_score(mclf, [X], y, cv=5, n_processes=5))

        mclf.fit([X], y)
        selected_classifiers = mclf.get_model(1,0).get_base_models()
        self.assertTrue(len(selected_classifiers) == 3,
                        'Ensemble picked the {} classifiers instead of 3.'.format(len(selected_classifiers)))
        self.assertFalse(DummyClassifier in [c.__class__ for c in selected_classifiers],
                         'Ensemble chose a dummy classifier over a real one')

        mclf = MultichannelPipeline(n_channels=1)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(Ensemble(classifiers, SVC(random_state=seed), internal_cv=5, score_selector=RankScoreSelector(k=3)))
        pc_score_informative = np.mean(cross_val_score(mclf, [X], y, cv=5, n_processes=5))

        base_classifier_arg = [(str(i), c) for i, c in enumerate(classifiers)]
        clf = StackingClassifier(base_classifier_arg, SVC(random_state=seed), cv=StratifiedKFold(n_splits=3))
        sk_score_informative = np.mean(cross_val_score(clf, X, y, cv=5, n_processes=5))

        if verbose > 0:
            base_classifier_arg = [(str(i), c) for i, c in enumerate(all_classifiers)]
            clf = StackingClassifier(base_classifier_arg, SVC(random_state=seed), cv=StratifiedKFold(n_splits=3))
            sk_score_all = np.mean(cross_val_score(clf, X, y, cv=5, n_processes=5))
            print('\nBalanced accuracy scores')
            print('Ensemble informative predictors: {}'.format(pc_score_informative))
            print('Ensemble all predictors: {}'.format(pc_score_all))
            print('StackingClassifier informative predictors: {}'.format(sk_score_informative))
            print('StackingClassifier all predictors: {}'.format(sk_score_all))

        self.assertTrue(np.round(pc_score_all, 2) == np.round(pc_score_informative, 2),
                        'Ensemble accuracy is not same for all classifiers and informative classifiers.')
        tolerance_pct = 5
        self.assertTrue(pc_score_all >= sk_score_informative * (1 - tolerance_pct / 100.0),
                        '''Ensemble with random inputs did not perform within accepted tolerance of StackingClassifier with no dummy classifiers.''')

    def test_discrimination_rgr(self, verbose=0, seed=42):
        """
        Determine if Ensemble can discriminate between dummy regressors and LinearRegression classifiers
        """
        X, y = make_regression(n_samples=500, n_features=20, n_informative=10, random_state=seed)

        base_regressors = [DummyRegressor(strategy='mean') for i in range(5)]
        base_regressors.extend([LinearRegression() for i in range(5)])
        random.shuffle(base_regressors)
        informative_mask = [True if type(c) == LinearRegression else False for c in base_regressors]

        mclf = MultichannelPipeline(n_channels=1)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(Ensemble(base_regressors, SVR(), internal_cv=5, score_selector=RankScoreSelector(k=5)))
        mclf.fit([X], y)
        selected_indices = mclf.get_model(layer_index=1, model_index=0).get_support()
        selection_mask = [True if i in selected_indices else False for i in range(len(base_regressors))]
        if verbose > 0:
            n_correct = sum([1 for i, s in zip(informative_mask, selection_mask) if i and s])
            print('\n\ncorrectly selected {}/5 LinearRegression regressors'.format(n_correct))
            print('incorrectly selected {}/5 DummyRegressors\n\n'.format(5- n_correct))
        self.assertTrue(np.array_equal(selection_mask, informative_mask),
                        'Ensemble failed to discriminate between dummy regressors and LinearRegression')

    def test_compare_to_StackingRegressor(self, verbose=0, seed=42):
        """
        Determine if Ensemble with dummies correctly selects the real predictors and gives similar
        performance to scikit-learn StackingRegressor trained without dummies.
        """
        X, y = make_regression(n_samples=500, n_features=20, n_informative=10, random_state=seed)

        regressors = [LinearRegression(),
                       KNeighborsRegressor(),
                       RandomForestRegressor(random_state=seed)]
        dummy_regressors = [DummyRegressor(strategy='mean') for repeat in range(100)]
        all_regressors = regressors + dummy_regressors
        random.shuffle(all_regressors)

        mclf = MultichannelPipeline(n_channels=1)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(Ensemble(all_regressors, SVR(), internal_cv=5, score_selector=RankScoreSelector(k=3)))
        pc_score_all = np.mean(cross_val_score(mclf, [X], y, cv=5, n_processes=5))

        mclf.fit([X], y)
        selected_regressors = mclf.get_model(1,0).get_base_models()
        self.assertTrue(len(selected_regressors) == 3,
                        'Ensemble picked the {} regressors instead of 3.'.format(len(selected_regressors)))
        self.assertFalse(DummyRegressor in [c.__class__ for c in selected_regressors],
                         'Ensemble chose a dummy regressors over a real one')

        mclf = MultichannelPipeline(n_channels=1)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(Ensemble(regressors, SVR(), internal_cv=5, score_selector=RankScoreSelector(k=3)))
        pc_score_informative = np.mean(cross_val_score(mclf, [X], y, cv=5, n_processes=5))

        base_arg = [(str(i), c) for i, c in enumerate(regressors)]
        clf = StackingRegressor(base_arg, SVR(), cv=KFold(n_splits=3))
        sk_score_informative = np.mean(cross_val_score(clf, X, y, cv=5, n_processes=5))

        if verbose > 0:
            base_arg = [(str(i), c) for i, c in enumerate(all_regressors)]
            clf = StackingRegressor(base_arg, SVR(), cv=KFold(n_splits=3))
            sk_score_all = np.mean(cross_val_score(clf, X, y, cv=5, n_processes=5))
            print('\nExplained variance scores')
            print('Ensemble informative predictors: {}'.format(pc_score_informative))
            print('Ensemble all predictors: {}'.format(pc_score_all))
            print('StackingRegressor informative predictors: {}'.format(sk_score_informative))
            print('StackingRegressor all predictors: {}'.format(sk_score_all))

        self.assertTrue(np.round(pc_score_all, 2) == np.round(pc_score_informative, 2),
                        'Ensemble accuracy is not same for all regressors and informative regressors.')
        tolerance_pct = 5
        self.assertTrue(pc_score_all >= sk_score_informative * (1 - tolerance_pct / 100.0),
                        '''Ensemble with dummy regressors did not perform within accepted tolerance of StackingClassifier with no dummy regressors.''')

class TestEnsembleSelection(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings('ignore')

    def tearDown(self):
        warnings.resetwarnings()

    def test_discrimination_cls(self, verbose=0, seed=42):
        """
        Determine if Ensemble can pick real classifier over dummy and
        test performance.
        """
        X, y = make_classification(n_samples=500, n_features=20,
                                   n_informative=15, class_sep=1,
                                   random_state=seed)

        base_classifiers = [DummyClassifier(strategy='stratified')
                            for i in range(5)]
        base_classifiers.append(LogisticRegression())
        random.shuffle(base_classifiers)

        mclf = MultichannelPipeline(n_channels=1)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(Ensemble(base_classifiers, internal_cv=5,
                                score_selector=RankScoreSelector(k=1)))
        mclf.fit([X], y)

        c = mclf.get_model(1, 0).get_base_models()[0]
        c = transform_wrappers.unwrap_model(c)

        self.assertTrue(type(c) == LogisticRegression,
                        'Ensemble failed to pick LogisticRegression '
                        'over dummies')

        acc = np.mean(cross_val_score(mclf, [X], y))
        if verbose > 0:
            print('cross val accuracy: {}'.format(acc))

        self.assertTrue(acc > 0.70, 'Accuracy tolerance failure.')

    def test_discrimination_rgr(self, verbose=0, seed=42):
        """
        Determine if Ensemble can pick real regressor over dummy and
        test performance.
        """
        X, y = make_regression(n_samples=500, n_features=20, n_informative=10,
                               random_state=seed)

        base_regressors = [DummyRegressor(strategy='mean') for i in range(5)]
        base_regressors.append(LinearRegression())
        random.shuffle(base_regressors)

        mclf = MultichannelPipeline(n_channels=1)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(Ensemble(base_regressors, internal_cv=5,
                                score_selector=RankScoreSelector(k=1)))
        mclf.fit([X], y)

        ensemble = mclf.get_model(1, 0)
        selected_model = ensemble.get_base_models()[0]
        selected_model = transform_wrappers.unwrap_model(selected_model)

        if verbose > 0:
            print(ensemble.get_screen_results())

        self.assertTrue(type(selected_model) == LinearRegression,
                        'Ensemble failed to pick LinearRegression '
                        'over dummies')

        acc = np.mean(cross_val_score(mclf, [X], y))
        if verbose > 0:
            print('cross val accuracy: {}'.format(acc))

        self.assertTrue(acc > 0.9, 'Accuracy tolerance failure.')


if __name__ == '__main__':
    unittest.main()
