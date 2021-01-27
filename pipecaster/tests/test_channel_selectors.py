import unittest
import random
import multiprocessing
import numpy as np
import warnings

from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from pipecaster.multichannel_pipeline import MultichannelPipeline
from pipecaster.channel_selection import SelectKBestScores, SelectKBestProbes, SelectKBestModels
from pipecaster.testing_utils import make_multi_input_classification, make_multi_input_regression
import pipecaster.parallel as parallel

n_cpus = multiprocessing.cpu_count()

class TestChannelSelectors(unittest.TestCase):

    ### TEST CHANNEL SELECTORS USING SYNTHETIC CLASSIFICATION DATA ###

    def setUp(self):
        warnings.filterwarnings("ignore")

    def tearDown(self):
        warnings.resetwarnings()

    @staticmethod
    def _select_synthetic_classification(channel_selector, n_informative_Xs=3, n_weak_Xs=0, n_random_Xs=0,
                                         weak_noise_sd=1.0, verbose = 0, seed=None, **sklearn_params):

        n_Xs = n_informative_Xs + n_weak_Xs + n_random_Xs

        Xs, y, X_types = make_multi_input_classification(n_informative_Xs, n_weak_Xs,
                                                         n_random_Xs, weak_noise_sd, seed, **sklearn_params)

        clf = MultichannelPipeline(n_channels=n_Xs)
        clf.add_layer(StandardScaler())
        clf.add_layer(channel_selector)
        clf.fit(Xs, y)
        Xs_t = clf.transform(Xs)
        Xs_selected = ['selected' if X is not None else 'not selected' for X in Xs_t]

        n_informative_hits, n_random_hits, n_weak_hits = 0, 0, 0
        for X, t in zip(Xs_selected, X_types):
            if X == 'selected' and t == 'informative':
                n_informative_hits +=1
            if X == 'not selected' and t == 'random':
                n_random_hits +=1
            if X == 'selected' and t == 'weak':
                n_weak_hits +=1

        if verbose > 0:
            print('InputSelector selected {} out of {} informative inputs'
                  .format(n_informative_hits, n_informative_Xs))
            print('InputSelector filtered out {} out of {} random inputs'
                  .format(n_random_hits, n_Xs - n_informative_Xs - n_weak_Xs))
            print('InputSelector selected out {} out of {} weakly informative inputs'
                  .format(n_weak_hits, n_weak_Xs))

        return n_informative_hits, n_random_hits, n_weak_hits

    @staticmethod
    def _test_weak_strong_cls_input_discrimination(channel_selector, n_weak = 5, n_strong = 5,
                                                   weak_noise_sd = 0.25, verbose=0, seed = None, **sklearn_params):
        n_random = n_weak + n_strong
        n_Xs = n_weak + n_strong + n_random
        n_informative_hits, n_random_hits, n_weak_hits = TestChannelSelectors._select_synthetic_classification(channel_selector,
                                                                             n_informative_Xs=n_strong,
                                                                             n_weak_Xs=n_weak,
                                                                             n_random_Xs=n_random,
                                                                             weak_noise_sd=weak_noise_sd,
                                                                             verbose=verbose, seed=seed, **sklearn_params)
        passed = True
        if n_informative_hits != n_strong:
            passed = False
        if n_weak_hits != 0:
            passed = False
        if n_random_hits != (n_Xs - n_weak - n_strong):
            passed = False
        return passed

    @staticmethod
    def _test_weak_cls_input_detection(channel_selector, n_weak = 5, n_strong = 5, weak_noise_sd = 0.25,
                                       verbose=0, seed = None, **sklearn_params):
        n_random = n_weak + n_strong
        n_Xs = n_weak + n_strong + n_random
        n_informative_hits, n_random_hits, n_weak_hits = TestChannelSelectors._select_synthetic_classification(channel_selector,
                                                                             n_informative_Xs=n_strong,
                                                                             n_weak_Xs=n_weak,
                                                                             n_random_Xs=n_random,
                                                                             weak_noise_sd=weak_noise_sd,
                                                                             verbose=verbose, seed=seed, **sklearn_params)
        passed = True
        if n_informative_hits != n_strong:
            passed = False
        if n_weak_hits != n_weak:
            passed = False
        if n_random_hits != (n_Xs - n_weak - n_strong):
            passed = False
        return passed

    def test_SelectKBestScores_weak_strong_cls_input_discrimination(self, verbose=0, seed=42):
        k = 5
        sklearn_params = {'n_classes':2,
                  'n_samples':1000,
                  'n_features':100,
                  'n_informative':20,
                  'n_redundant':0,
                  'n_repeated':0,
                  'class_sep':2.0}
        channel_selector = SelectKBestScores(feature_scorer=f_classif, aggregator=np.mean, k=k)
        passed = TestChannelSelectors._test_weak_strong_cls_input_discrimination(channel_selector, n_weak = k,
                                                                           n_strong = k, weak_noise_sd = 30,
                                                                           verbose=verbose, seed = seed, **sklearn_params)
        self.assertTrue(passed, 'SelectKBestScores failed to discriminate between weak & strong classification input matrices')

    def test_SelectKBestScores_weak_cls_input_detection(self, verbose=0, seed=42):
        k = 10
        sklearn_params = {'n_classes':2,
                  'n_samples':1000,
                  'n_features':100,
                  'n_informative':20,
                  'n_redundant':0,
                  'n_repeated':0,
                  'class_sep':2.0}
        channel_selector = SelectKBestScores(feature_scorer=f_classif, aggregator=np.mean, k=k)
        passed = TestChannelSelectors._test_weak_cls_input_detection(channel_selector, n_weak = int(k/2),
                                                               n_strong = k - int(k/2), weak_noise_sd = 0.2,
                                                               verbose=verbose, seed=seed, **sklearn_params)
        self.assertTrue(passed, 'SelectKBestScores failed to detect all weak clasification input matrices')

    def test_SelectKBestProbes_weak_strong_cls_input_discrimination(self, verbose=0, seed=42):
        k = 5
        sklearn_params = {'n_classes':2,
                  'n_samples':2000,
                  'n_features':20,
                  'n_informative':10,
                  'n_redundant':0,
                  'n_repeated':0,
                  'class_sep':2.0}
        channel_selector = SelectKBestProbes(predictor_probe=KNeighborsClassifier(n_neighbors=5, weights='uniform'),
                                                 cv=3, scorer=accuracy_score, k=k, channel_processes=n_cpus, cv_processes=1)
        passed = TestChannelSelectors._test_weak_strong_cls_input_discrimination(channel_selector, n_weak=k,
                                                                           n_strong=k, weak_noise_sd=50,
                                                                           verbose=verbose, seed=seed, **sklearn_params)
        self.assertTrue(passed, 'SelectKBestProbes failed to discriminate between weak & strong classification input matrices')

    def test_SelectKBestProbes_weak_cls_input_detection(self, verbose=0, seed=42):
        k = 10
        sklearn_params = {'n_classes':2,
                  'n_samples':2000,
                  'n_features':20,
                  'n_informative':15,
                  'n_redundant':0,
                  'n_repeated':0,
                  'class_sep':2.0}
        channel_selector = SelectKBestProbes(predictor_probe=KNeighborsClassifier(n_neighbors=5, weights='uniform'),
                                                 cv=3, scorer=accuracy_score, k=k, channel_processes=n_cpus, cv_processes=1)
        passed = TestChannelSelectors._test_weak_cls_input_detection(channel_selector, n_weak = int(k/2),
                                                               n_strong = k - int(k/2), weak_noise_sd = 1,
                                                               verbose=verbose, seed=seed, **sklearn_params)
        self.assertTrue(passed, 'KBestPerformers failed to detect all weak clasification input matrices')

    def test_SelectKBestModels_weak_strong_cls_input_discrimination(self, verbose=0, seed=42):
        k = 5
        sklearn_params = {'n_classes':2,
                  'n_samples':2000,
                  'n_features':20,
                  'n_informative':10,
                  'n_redundant':0,
                  'n_repeated':0,
                  'class_sep':2.0}

        channel_selector = SelectKBestModels(predictors=KNeighborsClassifier(n_neighbors=10, weights='distance'),
                                             cv=3, scorer=accuracy_score, k=k, channel_processes=n_cpus, cv_processes=1)
        passed = TestChannelSelectors._test_weak_strong_cls_input_discrimination(channel_selector, n_weak=k,
                                                                           n_strong=k, weak_noise_sd=50,
                                                                           verbose=verbose, seed=seed, **sklearn_params)
        self.assertTrue(passed, 'SelectKBestModels failed to discriminate between weak & strong classification input matrices')

    def test_SelectKBestModels_weak_cls_input_detection(self, verbose=0, seed=42):
        k = 10
        sklearn_params = {'n_classes':2,
                  'n_samples':2000,
                  'n_features':20,
                  'n_informative':15,
                  'n_redundant':0,
                  'n_repeated':0,
                  'class_sep':2.0}
        channel_selector = SelectKBestModels(predictors=KNeighborsClassifier(n_neighbors=5, weights='distance'),
                                             cv=3, scorer=accuracy_score, k=k, channel_processes=n_cpus, cv_processes=1)
        passed = TestChannelSelectors._test_weak_cls_input_detection(channel_selector, n_weak = int(k/2),
                                                               n_strong = k - int(k/2), weak_noise_sd = 1,
                                                               verbose=verbose, seed=seed, **sklearn_params)
        self.assertTrue(passed, 'SelectKBestModels failed to detect all weak clasification input matrices')

    ### TEST CHANNEL SELECTORS USING SYNTHETIC REGRESSION DATA ###

    @staticmethod
    def _select_synthetic_regression(channel_selector, n_informative_Xs=5, n_weak_Xs=0, n_random_Xs=0,
                                     weak_noise_sd=None, verbose = 0, seed = None, **rgr_params):

        n_Xs =  n_informative_Xs + n_weak_Xs + n_random_Xs
        Xs, y, X_types = make_multi_input_regression(n_informative_Xs, n_weak_Xs,
                                                     n_random_Xs, weak_noise_sd,
                                                     seed, **rgr_params)
        clf = MultichannelPipeline(n_channels=n_Xs)
        clf.add_layer(StandardScaler())
        clf.add_layer(channel_selector)
        Xs_t = clf.fit_transform(Xs, y)
        Xs_selected = ['selected' if X is not None else 'not selected' for X in Xs_t]

        n_informative_hits, n_random_hits, n_weak_hits = 0, 0, 0
        for X, t in zip(Xs_selected, X_types):
            if X == 'selected' and t == 'informative':
                n_informative_hits +=1
            if X == 'not selected' and t == 'random':
                n_random_hits +=1
            if X == 'selected' and t == 'weak':
                n_weak_hits +=1

        if verbose > 0:
            print('InputSelector selected {} out of {} informative inputs'
                  .format(n_informative_hits, n_informative_Xs))
            print('InputSelector filtered out {} out of {} random inputs'
                  .format(n_random_hits, n_Xs - n_informative_Xs - n_weak_Xs))
            print('InputSelector selected out {} out of {} weakly informative inputs'
                  .format(n_weak_hits, n_weak_Xs))

        return n_informative_hits, n_random_hits, n_weak_hits

    @staticmethod
    def _test_weak_strong_rgr_input_discrimination(channel_selector, n_weak=5, n_strong=5,
                                                   weak_noise_sd=0.25, verbose=0,  seed=None, **rgr_params):
        n_random = n_weak + n_strong
        n_Xs = n_weak + n_strong + n_random
        n_informative_hits, n_random_hits, n_weak_hits = TestChannelSelectors._select_synthetic_regression(channel_selector,
                                                                             n_informative_Xs=n_strong,
                                                                             n_weak_Xs=n_weak,
                                                                             n_random_Xs=n_random,
                                                                             weak_noise_sd=weak_noise_sd,
                                                                             verbose=verbose, seed=seed,
                                                                             **rgr_params)
        passed = True
        if n_informative_hits != n_strong:
            passed = False
        if n_weak_hits != 0:
            passed = False
        if n_random_hits != (n_Xs - n_weak - n_strong):
            passed = False
        return passed

    @staticmethod
    def _test_weak_rgr_input_detection(channel_selector, n_weak=5, n_strong=5,
                                       weak_noise_sd=0.25, verbose=0, seed = None, **rgr_params):
        n_random = n_weak + n_strong
        n_Xs = n_weak + n_strong + n_random
        n_informative_hits, n_random_hits, n_weak_hits = TestChannelSelectors._select_synthetic_regression(channel_selector,
                                                                              n_informative_Xs=n_strong,
                                                                             n_weak_Xs=n_weak,
                                                                             n_random_Xs=n_random,
                                                                             weak_noise_sd=weak_noise_sd,
                                                                             verbose=verbose, seed = seed,
                                                                             **rgr_params)
        passed = True
        if n_informative_hits != n_strong:
            passed = False
        if n_weak_hits != n_weak:
            passed = False
        if n_random_hits != (n_Xs - n_weak - n_strong):
            passed = False
        return passed

    def test_SelectKBestScores_weak_strong_rgr_input_discrimination(self, verbose=0, seed=42):
        k = 5
        rgr_params = {'n_samples':2000,
                      'n_features':30,
                      'n_informative':20
                      }
        channel_selector = SelectKBestScores(feature_scorer=f_regression, aggregator=np.mean, k=k)
        passed = TestChannelSelectors._test_weak_strong_rgr_input_discrimination(channel_selector, n_weak=k,
                                                            n_strong=k, weak_noise_sd=10,
                                                            verbose=verbose, seed=seed, **rgr_params)
        self.assertTrue(passed, 'SelectKBestScores failed to discriminate between weak & strong regression input matrices')

    def test_SelectKBestScores_weak_rgr_input_detection(self, verbose=0, seed=42):
        k = 10
        rgr_params = {'n_samples':2000,
                      'n_features':30,
                      'n_informative':20
                      }
        channel_selector = SelectKBestScores(feature_scorer=f_regression, aggregator=np.mean, k=k)
        passed = TestChannelSelectors._test_weak_rgr_input_detection(channel_selector, n_weak=int(k/2),
                                                n_strong=k - int(k/2), weak_noise_sd=0.2,
                                                verbose=verbose, seed=seed, **rgr_params)
        self.assertTrue(passed, 'SelectKBestScores failed to detect all week regression input matrices')

    def test_SelectKBestProbes_weak_strong_rgr_input_discrimination(self, verbose=0, seed=42):
        k = 5
        rgr_params = {'n_samples':2000,
                      'n_features':10,
                      'n_informative':5
                      }
        channel_selector = SelectKBestProbes(predictor_probe=RandomForestRegressor(n_estimators=20, max_depth=2),
                                                 cv=3, scorer=explained_variance_score, k=k,
                                                 channel_processes=n_cpus, cv_processes=1)

        passed = TestChannelSelectors._test_weak_strong_rgr_input_discrimination(channel_selector, n_weak=k,
                                                            n_strong=k, weak_noise_sd=30,
                                                            verbose=verbose, seed=seed, **rgr_params)
        self.assertTrue(passed, 'SelectKBestProbes failed to discriminate between weak & strong regression input matrices')

    def test_SelectKBestProbes_weak_rgr_input_detection(self, verbose=0, seed=42):
        k = 10
        rgr_params = {'n_samples':2000,
                      'n_features':10,
                      'n_informative':5
                      }
        channel_selector = SelectKBestProbes(predictor_probe=RandomForestRegressor(n_estimators=25, max_depth=2),
                                                 cv=3, scorer=explained_variance_score, k=k, channel_processes=n_cpus, cv_processes=1)
        passed = TestChannelSelectors._test_weak_rgr_input_detection(channel_selector, n_weak=int(k/2),
                                                n_strong=k - int(k/2), weak_noise_sd=0.2,
                                                verbose=verbose, seed=seed, **rgr_params)
        self.assertTrue(passed, 'SelectKBestProbes failed to detect all week regression input matrices')

    def test_SelectKBestModels_weak_strong_rgr_input_discrimination(self, verbose=0, seed=42):
        k = 5
        rgr_params = {'n_samples':2000,
                      'n_features':10,
                      'n_informative':5
                      }

        channel_selector = SelectKBestModels(predictors=KNeighborsRegressor(n_neighbors=300, weights='uniform'),
                                             cv=5, scorer=explained_variance_score, k=k,
                                             channel_processes=n_cpus, cv_processes=1)
        passed = TestChannelSelectors._test_weak_strong_rgr_input_discrimination(channel_selector, n_weak=k,
                                                                           n_strong=k, weak_noise_sd=10,
                                                                           verbose=verbose, seed=seed, **rgr_params)
        self.assertTrue(passed, 'SelectKBestModels failed to discriminate between weak & strong classification input matrices')

    def test_SelectKBestModels_weak_rgr_input_detection(self, verbose=0, seed=42):
        k = 10
        rgr_params = {'n_samples':2000,
                      'n_features':10,
                      'n_informative':7
                      }
        from sklearn.linear_model import LinearRegression
        channel_selector = SelectKBestModels(predictors=KNeighborsRegressor(n_neighbors=300, weights='uniform'),
                                             cv=5, scorer=explained_variance_score, k=k,
                                             channel_processes=n_cpus, cv_processes=1)
        passed = TestChannelSelectors._test_weak_rgr_input_detection(channel_selector, n_weak=int(k/2),
                                                               n_strong=k - int(k/2), weak_noise_sd=0.1,
                                                               verbose=verbose, seed=seed, **rgr_params)
        self.assertTrue(passed, 'SelectKBestModels failed to detect all weak regression input matrices')

if __name__ == '__main__':
    unittest.main()
