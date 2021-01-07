import numpy as np
import unittest
import random
import warnings
from scipy.stats import pearsonr

from pipecaster.ensemble_learning import SelectivePredictorStack

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, explained_variance_score

import pipecaster.utils as utils
from pipecaster.score_selection import RankScoreSelector
from pipecaster.multichannel_pipeline import MultichannelPipeline
from pipecaster.cross_validation import cross_val_score

class TestSelectivePredictorStack(unittest.TestCase):
    
    def setUp(self):
        warnings.filterwarnings('ignore')
        
    def tearDown(self):
        warnings.resetwarnings()
    
    def test_discrimination_cls(self, verbose=0, seed=42):
        """
        Determine if SelectivePredictorStack can discriminate between dummy classifiers and LogisticRegression classifiers
        """
        X, y = make_classification(n_samples=500, n_features=20, n_informative=15, class_sep=1, random_state=seed)
        
        base_classifiers = [DummyClassifier(strategy='stratified') for i in range(5)]
        base_classifiers.extend([LogisticRegression() for i in range(5)])
        random.shuffle(base_classifiers)
        informative_mask = [True if type(c) == LogisticRegression else False for c in base_classifiers]
        
        mclf = MultichannelPipeline(n_channels=1, internal_cv=None)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(SelectivePredictorStack(base_classifiers, SVC(), score_selector=RankScoreSelector(k=5)))
        mclf.fit([X], y)
        selected_indices = mclf.get_model(layer_index=1, model_index=0).get_support()
        selection_mask = [True if i in selected_indices else False for i in range(len(base_classifiers))]
        if verbose > 0:
            n_correct = sum([1 for i, s in zip(informative_mask, selection_mask) if i and s])
            print('\n\ncorrectly selected {}/5 LogigistRegression classifiers'.format(n_correct))
            print('incorrectly selected {}/5 DummyClassifiers\n\n'.format(5- n_correct))
        self.assertTrue(np.array_equal(selection_mask, informative_mask), 
                        'SelectivePredictorStack failed to discriminate between dummy classifiers and LogisticRegression')
        
    def test_compare_to_StackingClassifier(self, verbose=0, seed=42):
        """
        Determine if SelectivePredictorStack with dummies correctly selects the real predictors and gives similar
        performance to scikit-learn StackingClassifier trained without dummies.
        """
        
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, class_sep=0.5, random_state=seed)
        
        classifiers = [LogisticRegression(random_state=seed), 
                       KNeighborsClassifier(), 
                       RandomForestClassifier(random_state=seed)]
        dummy_classifiers = [DummyClassifier(strategy='stratified', random_state=seed) for repeat in range(100)]
        all_classifiers = classifiers + dummy_classifiers
        random.shuffle(all_classifiers)
        
        mclf = MultichannelPipeline(n_channels=1, internal_cv=3)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(SelectivePredictorStack(all_classifiers, SVC(random_state=seed), score_selector=RankScoreSelector(k=3)))
        pc_score_all = np.mean(cross_val_score(mclf, [X], y, cv=5, n_processes=5))
        
        mclf.fit([X], y)
        selected_classifiers = mclf.get_model(1,0).get_base_models()
        self.assertTrue(len(selected_classifiers) == 3, 
                        'SelectivePredictorStack picked the {} classifiers instead of 3.'.format(len(selected_classifiers)))
        self.assertFalse(DummyClassifier in [c.__class__ for c in selected_classifiers], 
                         'SelectivePredictorStack chose a dummy classifier over a real one') 
        
        mclf = MultichannelPipeline(n_channels=1, internal_cv=3)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(SelectivePredictorStack(classifiers, SVC(random_state=seed), score_selector=RankScoreSelector(k=3)))
        pc_score_informative = np.mean(cross_val_score(mclf, [X], y, cv=5, n_processes=5))

        base_classifier_arg = [(str(i), c) for i, c in enumerate(classifiers)]
        clf = StackingClassifier(base_classifier_arg, SVC(random_state=seed), cv=StratifiedKFold(n_splits=3))
        sk_score_informative = np.mean(cross_val_score(clf, X, y, cv=5, n_processes=5))
        
        if verbose > 0:
            base_classifier_arg = [(str(i), c) for i, c in enumerate(all_classifiers)]
            clf = StackingClassifier(base_classifier_arg, SVC(random_state=seed), cv=StratifiedKFold(n_splits=3))
            sk_score_all = np.mean(cross_val_score(clf, X, y, cv=5, n_processes=5))
            print('\nBalanced accuracy scores')
            print('SelectivePredictorStack informative predictors: {}'.format(pc_score_informative))
            print('SelectivePredictorStack all predictors: {}'.format(pc_score_all))
            print('StackingClassifier informative predictors: {}'.format(sk_score_informative))
            print('StackingClassifier all predictors: {}'.format(sk_score_all))
            
        self.assertTrue(np.round(pc_score_all, 2) == np.round(pc_score_informative, 2), 
                        'SelectivePredictorStack accuracy is not same for all classifiers and informative classifiers.')
        tolerance_pct = 5
        self.assertTrue(pc_score_all >= sk_score_informative * (1 - tolerance_pct / 100.0), 
                        '''SelectivePredictorStack with random inputs did not perform within accepted tolerance of StackingClassifier with no dummy classifiers.''')
        
    def test_discrimination_rgr(self, verbose=0, seed=42):
        """
        Determine if SelectivePredictorStack can discriminate between dummy regressors and LinearRegression classifiers
        """
        X, y = make_regression(n_samples=500, n_features=20, n_informative=10, random_state=seed)
        
        base_regressors = [DummyRegressor(strategy='mean') for i in range(5)]
        base_regressors.extend([LinearRegression() for i in range(5)])
        random.shuffle(base_regressors)
        informative_mask = [True if type(c) == LinearRegression else False for c in base_regressors]
        
        mclf = MultichannelPipeline(n_channels=1, internal_cv=None)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(SelectivePredictorStack(base_regressors, SVR(), score_selector=RankScoreSelector(k=5)))
        mclf.fit([X], y)
        selected_indices = mclf.get_model(layer_index=1, model_index=0).get_support()
        selection_mask = [True if i in selected_indices else False for i in range(len(base_regressors))]
        if verbose > 0:
            n_correct = sum([1 for i, s in zip(informative_mask, selection_mask) if i and s])
            print('\n\ncorrectly selected {}/5 LinearRegression regressors'.format(n_correct))
            print('incorrectly selected {}/5 DummyRegressors\n\n'.format(5- n_correct))
        self.assertTrue(np.array_equal(selection_mask, informative_mask), 
                        'SelectivePredictorStack failed to discriminate between dummy regressors and LinearRegression')
        
    def test_compare_to_StackingRegressor(self, verbose=0, seed=42):
        """
        Determine if SelectivePredictorStack with dummies correctly selects the real predictors and gives similar
        performance to scikit-learn StackingRegressor trained without dummies.
        """
        
        X, y = make_regression(n_samples=500, n_features=20, n_informative=10, random_state=seed)
        
        regressors = [LinearRegression(), 
                       KNeighborsRegressor(), 
                       RandomForestRegressor(random_state=seed)]
        dummy_regressors = [DummyRegressor(strategy='mean') for repeat in range(100)]
        all_regressors = regressors + dummy_regressors
        random.shuffle(all_regressors)
        
        mclf = MultichannelPipeline(n_channels=1, internal_cv=3)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(SelectivePredictorStack(all_regressors, SVR(), score_selector=RankScoreSelector(k=3)))
        pc_score_all = np.mean(cross_val_score(mclf, [X], y, cv=5, n_processes=5))
        
        mclf.fit([X], y)
        selected_regressors = mclf.get_model(1,0).get_base_models()
        self.assertTrue(len(selected_regressors) == 3, 
                        'SelectivePredictorStack picked the {} regressors instead of 3.'.format(len(selected_regressors)))
        self.assertFalse(DummyRegressor in [c.__class__ for c in selected_regressors], 
                         'SelectivePredictorStack chose a dummy regressors over a real one') 
        
        mclf = MultichannelPipeline(n_channels=1, internal_cv=3)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(SelectivePredictorStack(regressors, SVR(), score_selector=RankScoreSelector(k=3)))
        pc_score_informative = np.mean(cross_val_score(mclf, [X], y, cv=5, n_processes=5))

        base_arg = [(str(i), c) for i, c in enumerate(regressors)]
        clf = StackingRegressor(base_arg, SVR(), cv=KFold(n_splits=3))
        sk_score_informative = np.mean(cross_val_score(clf, X, y, cv=5, n_processes=5))
        
        if verbose > 0:
            base_arg = [(str(i), c) for i, c in enumerate(all_regressors)]
            clf = StackingRegressor(base_arg, SVR(), cv=KFold(n_splits=3))
            sk_score_all = np.mean(cross_val_score(clf, X, y, cv=5, n_processes=5))
            print('\nExplained variance scores')
            print('SelectivePredictorStack informative predictors: {}'.format(pc_score_informative))
            print('SelectivePredictorStack all predictors: {}'.format(pc_score_all))
            print('StackingRegressor informative predictors: {}'.format(sk_score_informative))
            print('StackingRegressor all predictors: {}'.format(sk_score_all))
            
        self.assertTrue(np.round(pc_score_all, 2) == np.round(pc_score_informative, 2), 
                        'SelectivePredictorStack accuracy is not same for all regressors and informative regressors.')
        tolerance_pct = 5
        self.assertTrue(pc_score_all >= sk_score_informative * (1 - tolerance_pct / 100.0), 
                        '''SelectivePredictorStack with dummy regressors did not perform within accepted tolerance of StackingClassifier with no dummy regressors.''')
        
if __name__ == '__main__':
    unittest.main()