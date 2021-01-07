import ray
import numpy as np
import unittest
import warnings
import timeit
import multiprocessing

from sklearn.datasets import make_classification, make_regression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import roc_auc_score, explained_variance_score, make_scorer
from sklearn.model_selection import KFold

import sklearn.model_selection as sk_model_selection
import pipecaster.cross_validation as pc_cross_validation
from pipecaster.multichannel_pipeline import MultichannelPipeline
import pipecaster.parallel as parallel
from pipecaster.testing_utils import DummyClassifier

test_seed = None
n_cpus = multiprocessing.cpu_count()
    
class TestCrossValScore(unittest.TestCase):
    
    def setUp(self):
        # get positive control values from sklearn cross_val_score selection
        
        # classification
        self.cv = KFold(n_splits=5)
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        self.clf = clf
        self.X_cls, self.y_cls = make_classification(n_classes=2, n_samples=500, n_features=40, 
                                                     n_informative=20, random_state=test_seed)
        self.cls_scores = sk_model_selection.cross_val_score(clf, self.X_cls, self.y_cls, 
                                                             scoring=make_scorer(roc_auc_score), 
                                                             cv=self.cv, n_jobs=1)
        
        rgr = KNeighborsRegressor(n_neighbors=5, weights='uniform')
        self.rgr = rgr
        self.X_rgr, self.y_rgr = make_regression(n_targets=1, n_samples = 500, n_features=40, 
                                                 n_informative=20, random_state=test_seed)
        self.rgr_scores = sk_model_selection.cross_val_score(rgr, self.X_rgr, self.y_rgr, 
                                                             scoring=make_scorer(explained_variance_score), 
                                                             cv=self.cv, n_jobs=1)
        
    def test_single_input_classification(self):
        pc_scores = pc_cross_validation.cross_val_score(self.clf, self.X_cls, self.y_cls, scorer=roc_auc_score,
                                                       cv=self.cv, n_processes=1)  
        self.assertTrue(np.array_equal(self.cls_scores, pc_scores), 'classifier scores from pipecaster.cross_validation.cross_val_score did not match sklearn control (single input predictor)')
        
    def test_multi_input_classification(self):
        mclf = MultichannelPipeline(n_channels=1)
        mclf.add_layer(self.clf)
        pc_scores = pc_cross_validation.cross_val_score(mclf, [self.X_cls], self.y_cls, scorer=roc_auc_score,
                                                       cv=self.cv, n_processes=1) 
        self.assertTrue(np.array_equal(self.cls_scores, pc_scores), 'classifier scores from pipecaster.cross_validation.cross_val_score did not match sklearn control (multi input predictor)')
        
    def test_multi_input_classification_parallel(self):
        if n_cpus > 1:
            warnings.filterwarnings("ignore")
            parallel.start_if_needed()
            mclf = MultichannelPipeline(n_channels=1)
            mclf.add_layer(self.clf)
            pc_scores = pc_cross_validation.cross_val_score(mclf, [self.X_cls], self.y_cls, scorer=roc_auc_score,
                                                           cv=self.cv, n_processes=n_cpus) 
            self.assertTrue(np.array_equal(self.cls_scores, pc_scores), 'classifier scores from pipecaster.cross_validation.cross_val_score did not match sklearn control (multi input predictor)')
            warnings.resetwarnings()
                                                                    
    def test_single_input_regression(self):
        pc_scores = pc_cross_validation.cross_val_score(self.rgr, self.X_rgr, self.y_rgr, scorer=explained_variance_score,
                                                       cv=self.cv, n_processes=1)  
        self.assertTrue(np.array_equal(self.rgr_scores, pc_scores), 'regressor scores from pipecaster.cross_validation.cross_val_score did not match sklearn control (single input predictor)')

    def test_multi_input_regression(self):
        mrgr = MultichannelPipeline(n_channels=1)
        mrgr.add_layer(self.rgr)  
        pc_scores = pc_cross_validation.cross_val_score(mrgr, [self.X_rgr], self.y_rgr, scorer=explained_variance_score, 
                                                       cv=self.cv, n_processes=1) 
        self.assertTrue(np.array_equal(self.rgr_scores, pc_scores), 'regressor scores from pipecaster.cross_validation.cross_val_score did not match sklearn control (multi input predictor)')
        
    def test_multi_input_regression_parallel(self):
        if n_cpus > 1:
            warnings.filterwarnings("ignore")
            parallel.start_if_needed(n_cpus=n_cpus)        
            mrgr = MultichannelPipeline(n_channels=1)
            mrgr.add_layer(self.rgr)
            pc_scores = pc_cross_validation.cross_val_score(mrgr, [self.X_rgr], self.y_rgr, scorer=explained_variance_score,
                                                           cv=self.cv, n_processes=n_cpus) 
            self.assertTrue(np.array_equal(self.rgr_scores, pc_scores), 'regressor scores from pipecaster.cross_validation.cross_val_score did not match sklearn control (multi input predictor)')
            warnings.resetwarnings()
       
    def test_multiprocessing_speedup(self, verbose=0):
 

        if n_cpus > 1:
            warnings.filterwarnings("ignore")
            parallel.start_if_needed(n_cpus=n_cpus)
            X, y = self.X_cls, self.y_cls = make_classification(n_classes=2, n_samples=500, n_features=40, 
                                             n_informative=20, random_state=test_seed)
            mclf = MultichannelPipeline(n_channels=1)
            mclf.add_layer(DummyClassifier(futile_cycles_fit=2000000, futile_cycles_pred=10))
            
            # shut off warnings because ray and redis generate massive numbers
            SETUP_CODE = ''' 
import pipecaster.cross_validation'''
            TEST_CODE = ''' 
pipecaster.cross_validation.cross_val_score(mclf, [X], y, cv = 5, n_processes = 1)'''
            t_serial = timeit.timeit(setup = SETUP_CODE, 
                                  stmt = TEST_CODE, 
                                  globals = locals(), 
                                  number = 5) 
            TEST_CODE = ''' 
pipecaster.cross_validation.cross_val_score(mclf, [X], y, cv = 5, n_processes = {})'''.format(n_cpus)
            t_parallel = timeit.timeit(setup = SETUP_CODE, 
                                  stmt = TEST_CODE, 
                                  globals = locals(), 
                                  number = 5) 
            
            warnings.resetwarnings()
            
            if verbose > 0:
                print('serial run mean time = {} s'.format(t_serial))
                print('parallel run mean time = {} s'.format(t_parallel))
    
            if t_serial <= t_parallel:
                warnings.warn('mulitple cpus detected, but parallel cross_val_score not faster than serial, possible problem with multiprocessing')
                
if __name__ == '__main__':
    unittest.main()