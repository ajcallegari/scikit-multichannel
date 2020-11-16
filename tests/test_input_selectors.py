import unittest
import random

import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

import pipecaster as pc
import synthetic_data

class TestInputSelectors(unittest.TestCase):

    @staticmethod
    def _select_synthetic(input_selector, n_Xs=20, n_informative_Xs=5, n_weak_Xs=5, weak_noise_sd=10, verbose = 0, seed = None):
        
        Xs, y, X_types = synthetic_data.make_multi_input_classification(n_classes = 2, 
                                            n_Xs=n_Xs, 
                                            n_informative_Xs=n_informative_Xs, 
                                            n_weak_Xs=n_weak_Xs,
                                            n_samples=1000, 
                                            n_features=100, 
                                            n_informative=20,
                                            n_redundant=0,
                                            n_repeated=0,
                                            class_sep=2.0,
                                            weak_noise_sd=weak_noise_sd,
                                            seed=seed)

        clf = pc.Pipeline(n_inputs = n_Xs)
        layer0 = clf.get_next_layer()
        layer0[:] = StandardScaler()
        layer1 = clf.get_next_layer()
        layer1[:] = input_selector
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
    def _test_weak_strong_input_discrimination(input_selector, n_weak = 5, n_strong = 5, weak_noise_sd = 0.25, seed = None):
        n_Xs = 2*(n_weak + n_strong)
        n_informative_hits, n_random_hits, n_weak_hits = TestInputSelectors._select_synthetic(input_selector, 
                                                                             n_Xs =  n_Xs,
                                                                             n_informative_Xs = n_strong,
                                                                             n_weak_Xs = n_weak, 
                                                                             weak_noise_sd = weak_noise_sd,
                                                                             seed = seed)
        passed = True
        if n_informative_hits != n_strong:
            passed = False
        if n_weak_hits != 0:
            passed = False
        if n_random_hits != (n_Xs - n_weak - n_strong):
            passed = False
        return passed
    
    @staticmethod
    def _test_weak_input_detection(input_selector, n_weak = 5, n_strong = 5, weak_noise_sd = 0.25, seed = None):
        n_Xs = 2*(n_weak + n_strong)
        n_informative_hits, n_random_hits, n_weak_hits = TestInputSelectors._select_synthetic(input_selector, 
                                                                             n_Xs =  n_Xs,
                                                                             n_informative_Xs = n_strong,
                                                                             n_weak_Xs = n_weak, 
                                                                             weak_noise_sd = weak_noise_sd,
                                                                             seed = seed)
        passed = True
        if n_informative_hits != n_strong:
            passed = False
        if n_weak_hits != n_weak:
            passed = False
        if n_random_hits != (n_Xs - n_weak - n_strong):
            passed = False
        return passed    
    
    def test_SelectKBestInputs_weak_strong_input_discrimination(self):
        k = 5
        input_selector = pc.SelectKBestInputs(score_func=f_classif, aggregator=np.mean, k=k)
        passed = TestInputSelectors._test_weak_strong_input_discrimination(input_selector, n_weak = k, 
                                                                           n_strong = k, weak_noise_sd = 30, seed = 42)
        self.assertTrue(passed)
        
    def test_SelectKBestInputs_weak_input_detection(self):
        k = 10
        input_selector = pc.SelectKBestInputs(score_func=f_classif, aggregator=np.mean, k=k)
        passed = TestInputSelectors._test_weak_input_detection(input_selector, n_weak = int(k/2), 
                                                               n_strong = k - int(k/2), weak_noise_sd = 0.2, seed = 42)
        self.assertTrue(passed)     

if __name__ == '__main__':
    unittest.main()