import ray
import numpy as np
import unittest
import multiprocessing
import warnings
import timeit

from sklearn.datasets import make_classification, make_regression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold

import sklearn.model_selection as sk_model_selection
import pipecaster.cross_validation as pc_cross_validation
from pipecaster.pipeline import Pipeline

test_seed = None

try:
    ray.nodes()
except RuntimeError:
    ray.init()
    
n_cpus = multiprocessing.cpu_count()

class TestCrossValPredict(unittest.TestCase):
    
    def setUp(self):
        # get positive control values from sklearn cross_val_predict selection
        
        # classification
        self.cv = KFold(n_splits=5)
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        self.clf = clf
        self.X_cls, self.y_cls = make_classification(n_classes=2, n_samples=500, n_features=40, 
                                                     n_informative=20, random_state=test_seed)
        self.cls_predictions = sk_model_selection.cross_val_predict(clf, self.X_cls, self.y_cls, cv=self.cv, n_jobs=1)
        
        rgr = KNeighborsRegressor(n_neighbors=5, weights='uniform')
        self.rgr = rgr
        self.X_rgr, self.y_rgr = make_regression(n_targets=1, n_samples = 500, n_features=40, 
                                                 n_informative=20, random_state=test_seed)
        self.rgr_predictions = sk_model_selection.cross_val_predict(rgr, self.X_rgr, self.y_rgr, cv=self.cv, n_jobs=1)
        
    def test_single_input_classification(self):
        pc_predictions = pc_cross_validation.cross_val_predict(self.clf, self.X_cls, self.y_cls, cv=self.cv, n_jobs=1)  
        self.assertTrue(np.array_equal(self.cls_predictions, pc_predictions), 'classifier predictions from pipecaster.cross_validation.cross_val_predict did not match sklearn control (single input predictor)')
        
    def test_multi_input_classification(self):
        mclf = Pipeline(n_inputs=1)
        mclf.get_next_layer()[:] = self.clf   
        pc_predictions = pc_cross_validation.cross_val_predict(mclf, [self.X_cls], self.y_cls, cv=self.cv, n_jobs=1) 
        self.assertTrue(np.array_equal(self.cls_predictions, pc_predictions), 'classifier predictions from pipecaster.cross_validation.cross_val_predict did not match sklearn control (multi input predictor)')
        
    def test_multi_input_classification_parallel(self):
        mclf = Pipeline(n_inputs=1)
        mclf.get_next_layer()[:] = self.clf   
        pc_predictions = pc_cross_validation.cross_val_predict(mclf, [self.X_cls], self.y_cls, cv=self.cv, n_jobs=n_cpus) 
        self.assertTrue(np.array_equal(self.cls_predictions, pc_predictions), 'classifier predictions from pipecaster.cross_validation.cross_val_predict did not match sklearn control (multi input predictor)')
                                                                    
    def test_single_input_regression(self):
        pc_predictions = pc_cross_validation.cross_val_predict(self.rgr, self.X_rgr, self.y_rgr, cv=self.cv, n_jobs=1)  
        self.assertTrue(np.array_equal(self.rgr_predictions, pc_predictions), 'regressor predictions from pipecaster.cross_validation.cross_val_predict did not match sklearn control (single input predictor)')

    def test_multi_input_regression(self):
        mrgr = Pipeline(n_inputs=1)
        mrgr.get_next_layer()[:] = self.rgr  
        pc_predictions = pc_cross_validation.cross_val_predict(mrgr, [self.X_rgr], self.y_rgr, cv=self.cv, n_jobs=1) 
        self.assertTrue(np.array_equal(self.rgr_predictions, pc_predictions), 'regressor predictions from pipecaster.cross_validation.cross_val_predict did not match sklearn control (multi input predictor)')
        
    def test_multi_input_regression_parallel(self):
        mrgr = Pipeline(n_inputs=1)
        mrgr.get_next_layer()[:] = self.rgr  
        pc_predictions = pc_cross_validation.cross_val_predict(mrgr, [self.X_rgr], self.y_rgr, cv=self.cv, n_jobs=n_cpus) 
        self.assertTrue(np.array_equal(self.rgr_predictions, pc_predictions), 'regressor predictions from pipecaster.cross_validation.cross_val_predict did not match sklearn control (multi input predictor)')
       
    def test_multiprocessing_speedup(self):
        X, y = self.X_cls, self.y_cls
        
        mclf = Pipeline(n_inputs=1)
        mclf.get_next_layer()[:] = self.clf   

        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")
            
            SETUP_CODE = ''' 
import pipecaster.cross_validation'''
            TEST_CODE = ''' 
pipecaster.cross_validation.cross_val_predict(mclf, [X], y, cv = 5, n_jobs = 1)'''
            t_serial = timeit.timeit(setup = SETUP_CODE, 
                                  stmt = TEST_CODE, 
                                  globals = locals(), 
                                  number = 5) 
            TEST_CODE = ''' 
pipecaster.cross_validation.cross_val_predict(mclf, [X], y, cv = 5, n_jobs = {})'''.format(n_cpus)
            t_parallel = timeit.timeit(setup = SETUP_CODE, 
                                  stmt = TEST_CODE, 
                                  globals = locals(), 
                                  number = 5) 
            
            warnings.resetwarnings()
    
            if t_serial <= t_parallel:
                warnings.warn('mulitple cpus detected, but parallel cross_val_predict not faster than serial, possible problem with multiprocessing')
                
    
if __name__ == '__main__':
    unittest.main()