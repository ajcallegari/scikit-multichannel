import ray
import numpy as np
import unittest
import multiprocessing
import warnings
import timeit

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

import sklearn.model_selection as sk_model_selection
import skmultichannel.cross_validation as sm_cross_validation
from skmultichannel.multichannel_pipeline import MultichannelPipeline
import skmultichannel.parallel as parallel
from skmultichannel.testing_utils import DummyClassifier

test_seed = 42

n_cpus = multiprocessing.cpu_count()

class TestCrossValPredict(unittest.TestCase):

    def setUp(self):
        # get positive control values from sklearn cross_val_predict selection
        # classification
        self.cv = KFold(n_splits=5)
        clf = RandomForestClassifier(n_estimators=10, random_state=test_seed, n_jobs=1)
        self.clf = clf
        self.X_cls, self.y_cls = make_classification(n_classes=2, n_samples=500, n_features=40,
                                                     n_informative=20, random_state=test_seed)
        self.cls_predictions = sk_model_selection.cross_val_predict(clf, self.X_cls, self.y_cls, cv=self.cv, n_jobs=1)

        rgr = RandomForestRegressor(n_estimators=50, random_state=test_seed, n_jobs=1)
        self.rgr = rgr
        self.X_rgr, self.y_rgr = make_regression(n_targets=1, n_samples = 500, n_features=40,
                                                 n_informative=20, random_state=test_seed)
        self.rgr_predictions = sk_model_selection.cross_val_predict(rgr, self.X_rgr, self.y_rgr, cv=self.cv, n_jobs=1)

    def test_single_input_classification(self):
        sm_predictions = sm_cross_validation.cross_val_predict(
            self.clf, self.X_cls, self.y_cls, cv=self.cv, n_processes=1)
        self.assertTrue(np.array_equal(self.cls_predictions,
                                       sm_predictions['predict']['y_pred']),
                        'skmultichannel predictions did not match sklearn control')

    def test_multi_input_classification(self):
        mclf = MultichannelPipeline(n_channels=1)
        mclf.add_layer(self.clf)
        sm_predictions = sm_cross_validation.cross_val_predict(
            mclf, [self.X_cls], self.y_cls, cv=self.cv, n_processes=1)
        self.assertTrue(np.array_equal(self.cls_predictions,
                                       sm_predictions['predict']['y_pred']),
                        'skmultichannel predictions did not match sklearn control')

    def test_multi_input_classification_parallel(self):
        if n_cpus > 1:
            warnings.filterwarnings("ignore")
            parallel.start_if_needed(n_cpus=n_cpus)
            mclf = MultichannelPipeline(n_channels=1)
            mclf.add_layer(self.clf)
            sm_predictions = sm_cross_validation.cross_val_predict(
                mclf, [self.X_cls], self.y_cls, cv=self.cv, n_processes=n_cpus)
            self.assertTrue(np.array_equal(
                self.cls_predictions, sm_predictions['predict']['y_pred']),
                'skmultichannel predictions did not match sklearn control')
            warnings.resetwarnings()

    def test_multi_input_classification_parallel_chunked(self):
        if n_cpus > 1:
            warnings.filterwarnings("ignore")
            parallel.start_if_needed(n_cpus=n_cpus)
            mclf = MultichannelPipeline(n_channels=1)
            mclf.add_layer(self.clf)
            sm_predictions = sm_cross_validation.cross_val_predict(
                mclf, [self.X_cls], self.y_cls, cv=self.cv,
                n_processes=n_cpus-1)
            self.assertTrue(np.array_equal(self.cls_predictions,
                               sm_predictions['predict']['y_pred']),
                               'skmultichannel predictions did not match sklearn '
                               'control')
            warnings.resetwarnings()

    def test_single_input_regression(self):
        sm_predictions = sm_cross_validation.cross_val_predict(
            self.rgr, self.X_rgr, self.y_rgr, cv=self.cv, n_processes=1)
        self.assertTrue(np.array_equal(self.rgr_predictions,
                                       sm_predictions['predict']['y_pred']),
                        'skmultichannel predictions did not match sklearn control')

    def test_multi_input_regression(self):
        mrgr = MultichannelPipeline(n_channels=1)
        mrgr.add_layer(self.rgr)
        sm_predictions = sm_cross_validation.cross_val_predict(
            mrgr, [self.X_rgr], self.y_rgr, cv=self.cv, n_processes=1)
        self.assertTrue(np.array_equal(self.rgr_predictions,
                                       sm_predictions['predict']['y_pred']),
                        'skmultichannel predictions did not match sklearn control')

    def test_multi_input_regression_parallel_get(self):
        if n_cpus > 1:
            warnings.filterwarnings("ignore")
            parallel.start_if_needed(n_cpus=n_cpus)
            mrgr = MultichannelPipeline(n_channels=1)
            mrgr.add_layer(self.rgr)
            sm_predictions = sm_cross_validation.cross_val_predict(
                mrgr, [self.X_rgr], self.y_rgr, cv=self.cv, n_processes=n_cpus)
            self.assertTrue(np.array_equal(
                                self.rgr_predictions,
                                sm_predictions['predict']['y_pred']),
                            'skmultichannel predictions did not match sklearn '
                            'control')
            warnings.resetwarnings()

    def test_multi_input_regression_parallel_starmap(self):
        if n_cpus > 2:
            warnings.filterwarnings("ignore")
            parallel.start_if_needed(n_cpus=n_cpus)
            mrgr = MultichannelPipeline(n_channels=1)
            mrgr.add_layer(self.rgr)
            sm_predictions = sm_cross_validation.cross_val_predict(
                mrgr, [self.X_rgr], self.y_rgr,
                cv=self.cv, n_processes=n_cpus-1)
            self.assertTrue(np.array_equal(
                                self.rgr_predictions,
                                sm_predictions['predict']['y_pred']),
                            'skmultichannel predictions did not match sklearn '
                            'control')
            warnings.resetwarnings()

    def test_multiprocessing_speedup(self, verbose=0):

        if n_cpus > 1:
            warnings.filterwarnings("ignore")
            parallel.start_if_needed(n_cpus=n_cpus)
            X, y = self.X_cls, self.y_cls = make_classification(
                n_classes=2, n_samples=500, n_features=40, n_informative=20,
                random_state=test_seed)
            mclf = MultichannelPipeline(n_channels=1)
            mclf.add_layer(DummyClassifier(futile_cycles_fit=2000000, futile_cycles_pred=10))

            SETUP_CODE = '''
import skmultichannel.cross_validation'''
            TEST_CODE = '''
skmultichannel.cross_validation.cross_val_predict(mclf, [X], y, cv = {}, n_processes = 1)'''.format(n_cpus)
            t_serial = timeit.timeit(setup = SETUP_CODE,
                                  stmt = TEST_CODE,
                                  globals = locals(),
                                  number = 5)
            TEST_CODE = '''
skmultichannel.cross_validation.cross_val_predict(mclf, [X], y, cv = {}, n_processes = {})'''.format(n_cpus, n_cpus)
            t_parallel = timeit.timeit(setup = SETUP_CODE,
                                  stmt = TEST_CODE,
                                  globals = locals(),
                                  number = 5)

            warnings.resetwarnings()

            if verbose > 0:
                print('number of CPUs detected and parallel jobs requested: {}'.format(n_cpus))
                print('duration of serial cross cross_val_predict task: {} s'.format(t_serial))
                print('duration of parallel cross cross_val_predict task: {} s'.format(t_parallel))

            if t_serial <= t_parallel:
                warnings.warn('multiple cpus detected, but parallel cross_val_predict not faster than serial using ray.get(), possible problem with multiprocessing')

    def test_throttled_multiprocessing_speedup(self, verbose=0):

        if n_cpus > 1:
            warnings.filterwarnings("ignore")
            parallel.start_if_needed(n_cpus=n_cpus)
            X, y = self.X_cls, self.y_cls = make_classification(n_classes=2, n_samples=500, n_features=40,
                                             n_informative=20, random_state=test_seed)
            mclf = MultichannelPipeline(n_channels=1)
            mclf.add_layer(DummyClassifier(futile_cycles_fit=2000000, futile_cycles_pred=10))

            SETUP_CODE = '''
import skmultichannel.cross_validation'''
            TEST_CODE = '''
skmultichannel.cross_validation.cross_val_predict(mclf, [X], y, cv = {}, n_processes = 1)'''.format(n_cpus - 1)
            t_serial = timeit.timeit(setup = SETUP_CODE,
                                  stmt = TEST_CODE,
                                  globals = locals(),
                                  number = 5)
            TEST_CODE = '''
skmultichannel.cross_validation.cross_val_predict(mclf, [X], y, cv = {}, n_processes = {})'''.format(n_cpus - 1, n_cpus - 1)
            t_parallel = timeit.timeit(setup = SETUP_CODE,
                                  stmt = TEST_CODE,
                                  globals = locals(),
                                  number = 5)

            warnings.resetwarnings()

            if verbose > 0:
                print('number of CPUs detected and parallel jobs requested: {}'.format(n_cpus))
                print('duration of serial cross cross_val_predict task: {} s'.format(t_serial))
                print('duration of parallel cross cross_val_predict task (ray pool.starmap): {} s'.format(t_parallel))

            if t_serial <= t_parallel:
                warnings.warn('multiple cpus detected, but parallel cross_val_predict not faster than serial using ray.multiprocessing.starmap(), possible problem with multiprocessing')

if __name__ == '__main__':
    unittest.main()
