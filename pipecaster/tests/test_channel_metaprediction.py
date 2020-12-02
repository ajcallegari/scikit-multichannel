import timeit
import multiprocessing
import ray
import numpy as np
import unittest
import warnings 

from scipy.stats import pearsonr

from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, explained_variance_score

from pipecaster import synthetic_data
from pipecaster.pipeline import Pipeline
from pipecaster.channel_metaprediction import ChannelClassifier, ChannelRegressor
from pipecaster.cross_validation import cross_val_score

try:
    ray.nodes()
except RuntimeError:
    ray.init()
    
n_cpus = multiprocessing.cpu_count()

class TestChannelClassifier(unittest.TestCase):
    
    def test_single_matrix_soft_voting(self):
        """Determine if KNN->ChannelClassifier(soft voting) in a pipecaster pipeline gives identical predictions to sklearn KNN on training data
        """
        X, y = make_classification(n_samples=100, n_features=20, n_informative=10, class_sep=5, random_state=42)
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        clf.fit(X, y)
        clf_predictions = clf.predict(X)
        n_inputs = 1
        mclf = Pipeline(n_inputs)
        layer1 = mclf.get_next_layer()
        layer1[:] = clf
        layer2 = mclf.get_next_layer()
        layer2[:] = ChannelClassifier('soft vote')
        mclf.fit([X], y)
        mclf_predictions = mclf.predict([X])
        self.assertTrue(np.array_equal(clf_predictions, mclf_predictions), 
                        'soft voting metaclassifier did not reproduce sklearn result on single matrix prediction task')
        
    def test_single_matrix_hard_voting(self):
        """Determine if KNN->ChannelClassifier(hard voting) in a pipecaster pipeline gives identical predictions to sklearn KNN on training data
        """
        X, y = make_classification(n_samples=100, n_features=20, n_informative=10, class_sep=5, random_state=42)
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        clf.fit(X, y)
        clf_predictions = clf.predict(X)
        n_inputs = 1
        mclf = Pipeline(n_inputs)
        layer1 = mclf.get_next_layer()
        layer1[:] = clf
        layer2 = mclf.get_next_layer()
        layer2[:] = ChannelClassifier('hard vote')
        mclf.fit([X], y)
        mclf_predictions = mclf.predict([X])
        self.assertTrue(np.array_equal(clf_predictions, mclf_predictions), 
                        'hard voting metaclassifier did not reproduce sklearn result on single matrix prediction task')
        
    def test_multi_matrix_voting(self):
        """Test if KNN->ChannelClassifier(soft voting) in a pipecaster pipeline gives monotonically increasing accuracy with increasing number of inputs in concordance with Condorcet's jury theorem, and also test hard voting with same pass criterion. Test if accuracy is > 80%.
        """
        
        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")
        
        n_inputs = 5
        soft_accuracies, hard_accuracies = [], []
        
        sklearn_params = {'n_classes':2, 
                  'n_samples':500, 
                  'n_features':100, 
                  'n_informative':30, 
                  'n_redundant':0, 
                  'n_repeated':0, 
                  'class_sep':3.0}

        for i in range(0, n_inputs + 1):
            
            Xs, y, _ = synthetic_data.make_multi_input_classification(n_informative_Xs=i, 
                                    n_weak_Xs=0,
                                    n_random_Xs=n_inputs - i,
                                    weak_noise_sd=None,
                                    seed = 42,
                                    **sklearn_params                                   
                                    )

            mclf = Pipeline(n_inputs)
            layer0 = mclf.get_next_layer()
            layer0[:] = StandardScaler()
            layer1 = mclf.get_next_layer()
            layer1[:] = KNeighborsClassifier(n_neighbors=5, weights='uniform')
            layer2 = mclf.get_next_layer()
            layer2[:] = ChannelClassifier('soft vote')

            split_accuracies = cross_val_score(mclf, Xs, y, predict_method='predict', 
                                     scorer=roc_auc_score, cv=3, n_jobs=1)
            soft_accuracies.append(np.mean(split_accuracies))
            
            layer2.clear()
            layer2[:] = ChannelClassifier('hard vote')
            split_accuracies = cross_val_score(mclf, Xs, y, predict_method='predict', 
                                     scorer=roc_auc_score, cv=3, n_jobs=1)
            hard_accuracies.append(np.mean(split_accuracies))
            
        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.resetwarnings()
            
        n_informative = range(0, n_inputs + 1)
        accuracy = soft_accuracies[-1]
        self.assertTrue(accuracy > 0.80, 'soft voting accuracy of {} below acceptable threshold of 0.80'.format(accuracy))
        linearity = pearsonr(soft_accuracies, n_informative)[0]
        self.assertTrue(linearity > 0.80, 
                        'hard voting linearity of {} below acceptable threshold of 0.80 pearsonr'.format(linearity))
        accuracy = hard_accuracies[-1]
        self.assertTrue(accuracy > 0.80, 'soft voting accuracy of {} below acceptable threshold of 0.80'.format(accuracy))
        linearity = pearsonr(hard_accuracies, n_informative)[0]
        self.assertTrue(linearity > 0.80, 
                        'hard voting linearity of {} below acceptable threshold of 0.80 pearsonr'.format(linearity))
        
    def test_multi_matrices_svm_metaclassifier(self):
        """Test if KNN classifier->ChannelClassifier(SVC) in a pipecaster pipeline gives monotonically increasing accuracy with increasing number of inputs, and test if accuracy is > 80%.
        """        
        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")

        n_inputs = 5
        accuracies = []
        
        sklearn_params = {'n_classes':2, 
                          'n_samples':500, 
                          'n_features':100, 
                          'n_informative':20, 
                          'n_redundant':0, 
                          'n_repeated':0, 
                          'class_sep':3.0}

        for i in range(0, n_inputs + 1):
            Xs, y, _ = synthetic_data.make_multi_input_classification(n_informative_Xs=i, 
                                    n_weak_Xs=0,
                                    n_random_Xs=n_inputs - i,
                                    weak_noise_sd=None,
                                    seed = 42,
                                    **sklearn_params                                   
                                    )
            mclf = Pipeline(n_inputs)
            layer0 = mclf.get_next_layer()
            layer0[:] = StandardScaler()
            layer1 = mclf.get_next_layer()
            layer1[:] = KNeighborsClassifier(n_neighbors=5, weights='uniform')
            layer2 = mclf.get_next_layer()
            layer2[:] = ChannelClassifier(SVC())

            split_accuracies = cross_val_score(mclf, Xs, y, predict_method='predict', 
                                     scorer=roc_auc_score, cv=3, n_jobs=n_cpus)
            accuracies.append(np.mean(split_accuracies))
            
        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.resetwarnings()
        
        n_informative = range(0, n_inputs + 1)
        self.assertTrue(accuracies[-1] > 0.80, 
                        'SVC metaclassification accuracy of {} below acceptable threshold of 0.80'.format(accuracies[-1]))
        linearity = pearsonr(accuracies, n_informative)[0]
        self.assertTrue(linearity > 0.80, 
                        'SVC metaclassification linearity of {} below acceptable threshold of 0.80 pearsonr'.format(linearity))
        
        
class TestChannelRegressor(unittest.TestCase):
    
    def test_single_matrix_mean_voting(self, seed=42):
        """Determine if KNN->ChannelRegressor(mean voting) in a pipecaster pipeline gives identical predictions to sklearn KNN on training data
        """
        X, y = make_regression(n_samples=100, n_features=20, n_informative=10, random_state=seed)
        
        rgr = KNeighborsRegressor(n_neighbors=5, weights='uniform')
        rgr.fit(X, y)
        rgr_predictions = rgr.predict(X)
        
        n_inputs = 1
        mrgr = Pipeline(n_inputs)
        layer1 = mrgr.get_next_layer()
        layer1[:] = rgr
        layer2 = mrgr.get_next_layer()
        layer2[:] = ChannelRegressor('mean voting')
        mrgr.fit([X], y)
        mrgr_predictions = mrgr.predict([X])
        self.assertTrue(np.array_equal(rgr_predictions, mrgr_predictions), 
                        'mean voting ChannelRegressor failed to reproduce sklearn result on single matrix prediction task')
        
    def test_single_matrix_median_voting(self, seed=42):
        """Determine if KNN->ChannelRegressor(median voting) in a pipecaster pipeline gives identical predictions to sklearn KNN on training data
        """
        X, y = make_regression(n_samples=100, n_features=20, n_informative=10, random_state=seed)
        
        rgr = KNeighborsRegressor(n_neighbors=5, weights='uniform')
        rgr.fit(X, y)
        rgr_predictions = rgr.predict(X)
        
        n_inputs = 1
        mrgr = Pipeline(n_inputs)
        layer1 = mrgr.get_next_layer()
        layer1[:] = rgr
        layer2 = mrgr.get_next_layer()
        layer2[:] = ChannelRegressor('median voting')
        mrgr.fit([X], y)
        mrgr_predictions = mrgr.predict([X])
        self.assertTrue(np.array_equal(rgr_predictions, mrgr_predictions), 
                        'median voting ChannelRegressor failed to reproduce sklearn result on single matrix prediction task')
        
    def test_multi_matrix_voting(self, seed = 42, verbose=0):
        """Determine if KNN->ChannelRegressor(voting) in a pipecaster pipeline gives monotonically increasing accuracy with increasing number of inputs and exceeds an accuracy cutoff
        """
        
        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")
        
        n_inputs = 5
        mean_accuracies, median_accuracies = [], []
        
        sklearn_params = {'n_targets':1, 
                  'n_samples':500, 
                  'n_features':10, 
                  'n_informative':5}

        for i in range(0, n_inputs + 1):
            
            Xs, y, _ = synthetic_data.make_multi_input_regression(n_informative_Xs=i, 
                                    n_weak_Xs=0,
                                    n_random_Xs=n_inputs - i,
                                    weak_noise_sd=None,
                                    seed = seed,
                                    **sklearn_params                                   
                                    )

            mrgr = Pipeline(n_inputs)
            layer0 = mrgr.get_next_layer()
            layer0[:] = StandardScaler()
            layer1 = mrgr.get_next_layer()
            layer1[:] = KNeighborsRegressor(n_neighbors=5, weights='uniform')
            layer2 = mrgr.get_next_layer()
            layer2[:] = ChannelRegressor('mean voting')

            split_accuracies = cross_val_score(mrgr, Xs, y, predict_method='predict', 
                                     scorer=explained_variance_score, cv=3, n_jobs=n_cpus)
            mean_accuracies.append(np.mean(split_accuracies))
            
            layer2.clear()
            layer2[:] = ChannelRegressor('median voting')
            split_accuracies = cross_val_score(mrgr, Xs, y, predict_method='predict', 
                                     scorer=explained_variance_score, cv=3, n_jobs=n_cpus)
            median_accuracies.append(np.mean(split_accuracies))
            
        n_informatives = range(0, n_inputs + 1)    
        if verbose > 0:
            print('explained variance scores')
            print('informative Xs\t\t mean voting\t\t median voting')
            for n_informative, mean_ev, median_ev in zip(n_informatives, mean_accuracies, median_accuracies):
                print('{}\t\t {}\t\t {}'.format(n_informative, mean_ev, median_ev))
                
        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.resetwarnings()
            
        mean_ev = mean_accuracies[-1]
        mean_linearity = pearsonr(mean_accuracies, n_informatives)[0]
        median_ev = median_accuracies[-1]
        median_linearity = pearsonr(median_accuracies, n_informatives)[0]
        
        if verbose > 0:
            print('mean voting pearsonr = {}'.format(mean_linearity))
            print('median voting pearsonr = {}'.format(median_linearity))
        
        self.assertTrue(mean_ev > 0.1, 
                        'mean voting explained variance of {} is below acceptable threshold of 0.80'.format(mean_ev))
        linearity = pearsonr(mean_accuracies, n_informatives)[0]
        self.assertTrue(mean_linearity > 0.9, 
                        'mean voting linearity of {} below acceptable threshold of 0.80 pearsonr'.format(mean_linearity))
        accuracy = median_accuracies[-1]
        self.assertTrue(median_ev > 0.1, 
                        'median voting explained variance of {} is below acceptable threshold of 0.80'.format(median_ev))
        linearity = pearsonr(median_accuracies, n_informatives)[0]
        self.assertTrue(median_linearity > 0.9, 
                        'median voting linearity of {} below acceptable threshold of 0.80 pearsonr'.format(median_linearity))
        
    def test_multi_matrix_SVR_stacking(self, seed = 42, verbose=0):
        """Determine if KNN->ChannelRegressor(SVR()) in a pipecaster pipeline gives monotonically 
           increasing accuracy with increasing number of inputs and exceeds a minimum accuracy cutoff.
        """
        
        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")
        
        n_inputs = 5
        accuracies = []
        
        sklearn_params = {'n_targets':1, 
                  'n_samples':1000, 
                  'n_features':10, 
                  'n_informative':10}

        for i in range(0, n_inputs + 1):
            
            Xs, y, _ = synthetic_data.make_multi_input_regression(n_informative_Xs=i, 
                                    n_weak_Xs=0,
                                    n_random_Xs=n_inputs - i,
                                    weak_noise_sd=None,
                                    seed = seed,
                                    **sklearn_params                                   
                                    )

            mrgr = Pipeline(n_inputs)
            layer0 = mrgr.get_next_layer()
            layer0[:] = StandardScaler()
            layer1 = mrgr.get_next_layer()
            layer1[:] = LinearRegression()
            layer2 = mrgr.get_next_layer()
            layer2[:] = ChannelRegressor(SVR())

            split_accuracies = cross_val_score(mrgr, Xs, y, predict_method='predict', 
                                     scorer=explained_variance_score, cv=3, n_jobs=n_cpus)
            accuracies.append(np.mean(split_accuracies))

            
        n_informatives = range(0, n_inputs + 1)    
        if verbose > 0:
            print('explained variance scores')
            print('informative Xs\t\t svr stacking')
            for n_informative, ev in zip(n_informatives, accuracies):
                print('{}\t\t {}'.format(n_informative, ev))
                
        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.resetwarnings()
            
        final_ev = accuracies[-1]
        linearity = pearsonr(accuracies, n_informatives)[0]
        
        if verbose > 0:
            print('SVR stacking pearsonr = {}'.format(linearity))
        
        self.assertTrue(final_ev > 0.1, 
                        'SVR stacking explained variance of {} is below acceptable threshold of 0.80'.format(final_ev))
        linearity = pearsonr(accuracies, n_informatives)[0]
        self.assertTrue(linearity > 0.0, 
                        'SVR stacking linearity of {} below acceptable threshold of 0.80 pearsonr'.format(linearity))
            
if __name__ == '__main__':
    unittest.main()