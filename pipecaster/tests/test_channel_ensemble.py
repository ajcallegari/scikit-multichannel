import timeit
import multiprocessing
import numpy as np
import unittest
import warnings

from scipy.stats import pearsonr

from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, explained_variance_score

import pipecaster.utils as utils
from pipecaster.testing_utils import make_multi_input_classification
from pipecaster.testing_utils import make_multi_input_regression

from pipecaster.multichannel_pipeline import MultichannelPipeline
from pipecaster.ensemble_learning import SoftVotingClassifier
from pipecaster.ensemble_learning import HardVotingClassifier
from pipecaster.ensemble_learning import AggregatingRegressor
from pipecaster.ensemble_learning import MultichannelPredictor
from pipecaster.ensemble_learning import ChannelEnsemble
from pipecaster.cross_validation import cross_val_score
from pipecaster.score_selection import RankScoreSelector
import pipecaster.transform_wrappers as transform_wrappers

n_cpus = multiprocessing.cpu_count()

class TestMultichannelClassification(unittest.TestCase):

    def test_single_matrix_soft_voting(self):
        """
        Determine if KNN->ChannelClassifier(soft voting) in a pipecaster
        pipeline gives identical predictions to sklearn KNN on training data.
        """
        X, y = make_classification(n_samples=100, n_features=20,
                                   n_informative=10, class_sep=5,
                                   random_state=42)

        # control
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        clf.fit(X, y)
        clf_predictions = clf.predict(X)

        # implemenation 1
        mclf = MultichannelPipeline(n_channels=1)
        base_clf = transform_wrappers.SingleChannel(KNeighborsClassifier(
                                            n_neighbors=5, weights='uniform'))
        mclf.add_layer(base_clf)
        mclf.add_layer(MultichannelPredictor(SoftVotingClassifier()))
        mclf.fit([X], y)
        mclf_predictions = mclf.predict([X])
        self.assertTrue(np.array_equal(clf_predictions, mclf_predictions),
                        'soft voting metaclassifier did not reproduce sklearn '
                        'result on single matrix prediction task')

        # implementation 2
        mclf = MultichannelPipeline(n_channels=1)
        base_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        mclf.add_layer(ChannelEnsemble(base_clf, SoftVotingClassifier()))
        mclf.fit([X], y)
        mclf_predictions = mclf.predict([X])
        self.assertTrue(np.array_equal(clf_predictions, mclf_predictions),
                        'soft voting metaclassifier did not reproduce sklearn '
                        'result on single matrix prediction task')

    def test_single_matrix_hard_voting(self):
        """
        Determine if KNN->ChannelClassifier(hard voting) in a pipecaster
        pipeline gives identical predictions to sklearn KNN on training data.
        """
        X, y = make_classification(n_samples=100, n_features=20,
                                   n_informative=10, class_sep=5,
                                   random_state=42)

        # control
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        clf.fit(X, y)
        clf_predictions = clf.predict(X)

        # implementation 1
        mclf = MultichannelPipeline(n_channels=1)
        base_clf = transform_wrappers.SingleChannel(KNeighborsClassifier(
                                            n_neighbors=5, weights='uniform'))
        mclf.add_layer(base_clf)
        mclf.add_layer(MultichannelPredictor(HardVotingClassifier()))
        mclf.fit([X], y)
        mclf_predictions = mclf.predict([X])
        self.assertTrue(np.array_equal(clf_predictions, mclf_predictions),
                        'hard voting metaclassifier did not reproduce sklearn '
                        'result on single matrix prediction task')

        # implementation 2
        mclf = MultichannelPipeline(n_channels=1)
        base_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        mclf.add_layer(ChannelEnsemble(base_clf, HardVotingClassifier()))
        mclf.fit([X], y)
        mclf_predictions = mclf.predict([X])
        self.assertTrue(np.array_equal(clf_predictions, mclf_predictions),
                        'hard voting metaclassifier did not reproduce sklearn '
                        'result on single matrix prediction task')

    def test_multi_matrix_voting(self, verbose=0):
        """
        Test if KNN->ChannelClassifier(soft voting) in a pipecaster pipeline
        gives monotonically increasing accuracy with increasing number of
        inputs in concordance with Condorcet's jury theorem, and also test hard
        voting with same pass criterion. Test if accuracy is > 80%.
        """

        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")

        n_channels = 5

        sklearn_params = {'n_classes':2,
                  'n_samples':500,
                  'n_features':100,
                  'n_informative':30,
                  'n_redundant':0,
                  'n_repeated':0,
                  'class_sep':3.0}

        # implementation 1
        soft_accuracies, hard_accuracies = [], []
        for i in range(0, n_channels + 1):

            Xs, y, _ = make_multi_input_classification(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed = 42,
                                    **sklearn_params
                                    )

            mclf = MultichannelPipeline(n_channels)
            mclf.add_layer(StandardScaler(), pipe_processes=n_cpus)
            clf = transform_wrappers.SingleChannel(
                        KNeighborsClassifier(n_neighbors=5, weights='uniform'))
            mclf.add_layer(clf, pipe_processes=n_cpus)
            mclf.add_layer(MultichannelPredictor(SoftVotingClassifier()))

            split_accuracies = cross_val_score(mclf, Xs, y,
                                               scorer=roc_auc_score,
                                               cv=3, n_processes=1)
            soft_accuracies.append(np.mean(split_accuracies))

            mclf = MultichannelPipeline(n_channels)
            mclf.add_layer(StandardScaler(), pipe_processes=n_cpus)
            clf = transform_wrappers.SingleChannel(
                        KNeighborsClassifier(n_neighbors=5, weights='uniform'))
            mclf.add_layer(clf, pipe_processes=n_cpus)
            mclf.add_layer(MultichannelPredictor(HardVotingClassifier()))

            split_accuracies = cross_val_score(mclf, Xs, y,
                                               scorer=roc_auc_score, cv=3,
                                               n_processes=1)
            hard_accuracies.append(np.mean(split_accuracies))

        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.resetwarnings()

        if verbose > 0:
            print('soft voting results:')
            print('n_informative, accuray')
            for i in range(0, n_channels + 1):
                print(i, soft_accuracies[i])
            print('hard voting results:')
            print('n_informative, accuray')
            for i in range(0, n_channels + 1):
                print(i, hard_accuracies[i])
        n_informative = range(0, n_channels + 1)
        accuracy = soft_accuracies[-1]
        self.assertTrue(accuracy > 0.80, 'soft voting accuracy of {} below '
                        'acceptable threshold of 0.80'.format(accuracy))
        linearity = pearsonr(soft_accuracies, n_informative)[0]
        self.assertTrue(linearity > 0.80,
                        'hard voting linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(linearity))
        accuracy = hard_accuracies[-1]
        self.assertTrue(accuracy > 0.80, 'hard voting accuracy of {} below '
                        'acceptable threshold of 0.80'.format(accuracy))
        linearity = pearsonr(hard_accuracies, n_informative)[0]
        self.assertTrue(linearity > 0.80,
                        'hard voting linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(linearity))

        # implementation 2
        soft_accuracies, hard_accuracies = [], []
        for i in range(0, n_channels + 1):

            Xs, y, _ = make_multi_input_classification(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed = 42,
                                    **sklearn_params
                                    )

            mclf = MultichannelPipeline(n_channels)
            mclf.add_layer(StandardScaler(), pipe_processes=n_cpus)
            base_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
            mclf.add_layer(
                ChannelEnsemble(
                    base_clf, SoftVotingClassifier(), base_processes=n_cpus))
            split_accuracies = cross_val_score(mclf, Xs, y,
                                               scorer=roc_auc_score,
                                               cv=3, n_processes=1)
            soft_accuracies.append(np.mean(split_accuracies))

            mclf = MultichannelPipeline(n_channels)
            mclf.add_layer(StandardScaler(), pipe_processes='max')
            clf = transform_wrappers.SingleChannel(
                        KNeighborsClassifier(n_neighbors=5, weights='uniform'))
            mclf.add_layer(clf, pipe_processes='max')
            mclf.add_layer(MultichannelPredictor(HardVotingClassifier()))

            split_accuracies = cross_val_score(mclf, Xs, y,
                                               scorer=roc_auc_score, cv=3,
                                               n_processes=1)
            hard_accuracies.append(np.mean(split_accuracies))

        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.resetwarnings()

        if verbose > 0:
            print('soft voting results:')
            print('n_informative, accuray')
            for i in range(0, n_channels + 1):
                print(i, soft_accuracies[i])
            print('hard voting results:')
            print('n_informative, accuray')
            for i in range(0, n_channels + 1):
                print(i, hard_accuracies[i])
        n_informative = range(0, n_channels + 1)
        accuracy = soft_accuracies[-1]
        self.assertTrue(accuracy > 0.80, 'soft voting accuracy of {} below '
                        'acceptable threshold of 0.80'.format(accuracy))
        linearity = pearsonr(soft_accuracies, n_informative)[0]
        self.assertTrue(linearity > 0.80,
                        'hard voting linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(linearity))
        accuracy = hard_accuracies[-1]
        self.assertTrue(accuracy > 0.80, 'hard voting accuracy of {} below '
                        'acceptable threshold of 0.80'.format(accuracy))
        linearity = pearsonr(hard_accuracies, n_informative)[0]
        self.assertTrue(linearity > 0.80,
                        'hard voting linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(linearity))

    def test_multi_matrices_svm_metaclassifier(self, seed=42, verbose=0):
        """
        Test if KNN classifier->ChannelClassifier(SVC) in a pipecaster
        pipeline gives monotonically increasing accuracy with increasing number
        of inputs, and test if accuracy is > 75%.
        """
        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")

        n_channels = 5
        accuracies = []

        sklearn_params = {'n_classes':2,
                          'n_samples':500,
                          'n_features':100,
                          'n_informative':5,
                          'n_redundant':10,
                          'n_repeated':5,
                          'class_sep':1.0}

        # implementation 1
        for i in range(0, n_channels + 1):
            Xs, y, _ = make_multi_input_classification(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed = seed,
                                    **sklearn_params
                                    )
            mclf = MultichannelPipeline(n_channels)
            mclf.add_layer(StandardScaler(), pipe_processes='max')
            clf = transform_wrappers.SingleChannel(
                        KNeighborsClassifier(n_neighbors=5, weights='uniform'))
            mclf.add_layer(clf, pipe_processes='max')
            mclf.add_layer(MultichannelPredictor(SVC()))

            split_accuracies = cross_val_score(mclf, Xs, y,
                                               scorer=roc_auc_score, cv=3,
                                               n_processes=1)
            accuracies.append(np.mean(split_accuracies))

        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.resetwarnings()

        if verbose > 0:
            print('SVC meta-classification results:')
            print('n_informative, accuray')
            for i in range(0, n_channels + 1):
                print(i, accuracies[i])

        n_informative = range(0, n_channels + 1)
        self.assertTrue(accuracies[-1] > 0.75,
                        'SVC metaclassification accuracy of {} below \
                        acceptable threshold of 0.75'.format(accuracies[-1]))
        linearity = pearsonr(accuracies, n_informative)[0]
        self.assertTrue(linearity > 0.75,
                        'SVC metaclassification linearity of {} below \
                        acceptable threshold of 0.75 pearsonr'
                        .format(linearity))

        # implementation 2
        accuracies = []
        for i in range(0, n_channels + 1):
            Xs, y, _ = make_multi_input_classification(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed = seed,
                                    **sklearn_params
                                    )
            mclf = MultichannelPipeline(n_channels)
            mclf.add_layer(StandardScaler(), pipe_processes='max')
            base_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
            mclf.add_layer(
                ChannelEnsemble(
                    base_clf, SVC(), internal_cv=5, base_processes='max'))
            split_accuracies = cross_val_score(mclf, Xs, y,
                                               scorer=roc_auc_score, cv=3,
                                               n_processes=1)
            accuracies.append(np.mean(split_accuracies))

        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.resetwarnings()

        if verbose > 0:
            print('SVC meta-classification results:')
            print('n_informative, accuray')
            for i in range(0, n_channels + 1):
                print(i, accuracies[i])

        n_informative = range(0, n_channels + 1)
        self.assertTrue(accuracies[-1] > 0.75,
                        'SVC metaclassification accuracy of {} below \
                        acceptable threshold of 0.75'.format(accuracies[-1]))
        linearity = pearsonr(accuracies, n_informative)[0]
        self.assertTrue(linearity > 0.75,
                        'SVC metaclassification linearity of {} below \
                        acceptable threshold of 0.75 pearsonr'
                        .format(linearity))

    def test_multi_matrices_no_metaclassifier(self, verbose=0, seed=42):
        """
        Determine if ChannelEnsemble works without a meta-predictor.

        Determine if it can pick informative input over random and
        test its performance.
        """

        Xs, y, types = make_multi_input_classification(n_informative_Xs=1,
                                n_weak_Xs=0, n_random_Xs=4, weak_noise_sd=None,
                                seed = seed, n_samples=500, n_features=20,
                                n_informative=10, class_sep=3)

        mclf = MultichannelPipeline(n_channels=5)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(ChannelEnsemble(LogisticRegression(), internal_cv=5,
                                score_selector=RankScoreSelector(k=1)))
        mclf.fit(Xs, y)

        selection = types[mclf.get_model(1, 0).get_support()[0]]

        self.assertTrue(selection == 'informative',
                        'Ensemble failed to pick informative channel')

        acc = np.mean(cross_val_score(mclf, Xs, y))
        if verbose > 0:
            print('cross val accuracy: {}'.format(acc))

        self.assertTrue(acc > 0.90, 'Accuracy tolerance failure.')


class TestMultiChannelRegression(unittest.TestCase):

    def test_single_matrix_mean_voting(self, seed=42):
        """
        Determine if KNN->ChannelRegressor(mean voting) in a pipecaster
        pipeline gives identical predictions to sklearn KNN on training data
        """
        X, y = make_regression(n_samples=100, n_features=20, n_informative=10,
                               random_state=seed)

        # control
        rgr = KNeighborsRegressor(n_neighbors=5, weights='uniform')
        rgr.fit(X, y)
        rgr_predictions = rgr.predict(X)

        # implementation 1
        mrgr = MultichannelPipeline(n_channels=1)
        rgr = transform_wrappers.SingleChannel(
                        KNeighborsRegressor(n_neighbors=5, weights='uniform'))
        mrgr.add_layer(rgr, pipe_processes=n_cpus)
        mrgr.add_layer(MultichannelPredictor(AggregatingRegressor(np.mean)))
        mrgr.fit([X], y)
        mrgr_predictions = mrgr.predict([X])
        self.assertTrue(np.array_equal(rgr_predictions, mrgr_predictions),
                        'mean voting ChannelRegressor failed to reproduce '
                        'sklearn result on single matrix prediction task')

        # implementation 2
        mrgr = MultichannelPipeline(n_channels=1)
        base_rgr = KNeighborsRegressor(n_neighbors=5, weights='uniform')
        mrgr.add_layer(ChannelEnsemble(
            base_rgr, AggregatingRegressor(np.mean), base_processes='max'))
        mrgr.fit([X], y)
        mrgr_predictions = mrgr.predict([X])
        self.assertTrue(np.array_equal(rgr_predictions, mrgr_predictions),
                        'mean voting ChannelRegressor failed to reproduce '
                        'sklearn result on single matrix prediction task')

    def test_multi_matrix_voting(self, verbose=0, seed=42):
        """
        Determine if KNN->ChannelRegressor(voting) in a MultichannelPipeline
        gives monotonically increasing accuracy with increasing number of
        inputs and exceeds an accuracy cutoff.
        """

        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")

        n_channels = 5

        rgr_params = {'n_samples':500,
                      'n_features':10,
                      'n_informative':5}

        # implementation 1
        mean_accuracies, median_accuracies = [], []
        for i in range(0, n_channels + 1):

            Xs, y, _ = make_multi_input_regression(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed=seed,
                                    **rgr_params
                                    )
            # mean aggregation
            mrgr = MultichannelPipeline(n_channels)
            mrgr.add_layer(StandardScaler(), pipe_processes=n_cpus)
            rgr = transform_wrappers.SingleChannel(
                KNeighborsRegressor(n_neighbors=20, weights='distance'))
            mrgr.add_layer(rgr, pipe_processes=n_cpus)
            mrgr.add_layer(
                MultichannelPredictor(AggregatingRegressor(np.mean)))
            split_accuracies = cross_val_score(
                mrgr, Xs, y, scorer=explained_variance_score,
                cv=3, n_processes=1)
            mean_accuracies.append(np.mean(split_accuracies))

            # median aggregation
            mrgr = MultichannelPipeline(n_channels)
            mrgr.add_layer(StandardScaler(), pipe_processes=n_cpus)
            rgr = transform_wrappers.SingleChannel(
                KNeighborsRegressor(n_neighbors=20, weights='distance'))
            mrgr.add_layer(rgr, pipe_processes=n_cpus)
            mrgr.add_layer(
                MultichannelPredictor(AggregatingRegressor(np.median)))

            split_accuracies = cross_val_score(mrgr, Xs, y,
                                               scorer=explained_variance_score,
                                               cv=3, n_processes=1)
            median_accuracies.append(np.mean(split_accuracies))

        n_informatives = range(0, n_channels + 1)
        if verbose > 0:
            print('explained variance scores')
            print('informative Xs\t\t mean voting\t\t median voting')
            for n_informative, mean_ev, median_ev in zip(n_informatives,
                                                         mean_accuracies,
                                                         median_accuracies):
                print('{}\t\t {}\t\t {}'.format(n_informative, mean_ev,
                                                median_ev))

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
                        'mean voting explained variance of {} is below '
                        'acceptable threshold of 0.80'.format(mean_ev))
        linearity = pearsonr(mean_accuracies, n_informatives)[0]
        self.assertTrue(mean_linearity > 0.9,
                        'mean voting linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(mean_linearity))
        accuracy = median_accuracies[-1]
        self.assertTrue(median_ev > 0.1,
                        'median voting explained variance of {} is below '
                        'acceptable threshold of 0.80'.format(median_ev))
        linearity = pearsonr(median_accuracies, n_informatives)[0]
        self.assertTrue(median_linearity > 0.9,
                        'median voting linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(median_linearity))

        # implementation 2
        mean_accuracies, median_accuracies = [], []
        for i in range(0, n_channels + 1):

            Xs, y, _ = make_multi_input_regression(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed=seed,
                                    **rgr_params
                                    )

            # mean aggregation
            mrgr = MultichannelPipeline(n_channels)
            mrgr.add_layer(StandardScaler(), pipe_processes=n_cpus)
            base_rgr = KNeighborsRegressor(n_neighbors=20, weights='distance')
            mrgr.add_layer(ChannelEnsemble(
                base_rgr, AggregatingRegressor(np.mean), base_processes='max'))
            split_accuracies = cross_val_score(
                mrgr, Xs, y, scorer=explained_variance_score,
                cv=3, n_processes=1)
            mean_accuracies.append(np.mean(split_accuracies))

            # median aggregation
            mrgr = MultichannelPipeline(n_channels)
            mrgr.add_layer(StandardScaler(), pipe_processes=n_cpus)
            rgr = KNeighborsRegressor(n_neighbors=20, weights='distance')
            mrgr.add_layer(ChannelEnsemble(
                base_rgr, AggregatingRegressor(np.median),
                base_processes='max'))
            split_accuracies = cross_val_score(mrgr, Xs, y,
                                               scorer=explained_variance_score,
                                               cv=3, n_processes=1)
            median_accuracies.append(np.mean(split_accuracies))

        n_informatives = range(0, n_channels + 1)
        if verbose > 0:
            print('explained variance scores')
            print('informative Xs\t\t mean voting\t\t median voting')
            for n_informative, mean_ev, median_ev in zip(n_informatives,
                                                         mean_accuracies,
                                                         median_accuracies):
                print('{}\t\t {}\t\t {}'.format(n_informative, mean_ev,
                                                median_ev))

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
                        'mean voting explained variance of {} is below '
                        'acceptable threshold of 0.80'.format(mean_ev))
        linearity = pearsonr(mean_accuracies, n_informatives)[0]
        self.assertTrue(mean_linearity > 0.9,
                        'mean voting linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(mean_linearity))
        accuracy = median_accuracies[-1]
        self.assertTrue(median_ev > 0.1,
                        'median voting explained variance of {} is below '
                        'acceptable threshold of 0.80'.format(median_ev))
        linearity = pearsonr(median_accuracies, n_informatives)[0]
        self.assertTrue(median_linearity > 0.9,
                        'median voting linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(median_linearity))

    def test_multi_matrix_SVR_stacking(self, verbose=0, seed=42):
        """
        Determine if KNN->ChannelRegressor(SVR()) in a pipecaster pipeline
        gives monotonically increasing accuracy with increasing number of
        inputs and exceeds a minimum accuracy cutoff.
        """

        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")

        n_channels = 5
        accuracies = []

        rgr_params = {'n_samples':1000,
                      'n_features':10,
                      'n_informative':10}

        # implementation 1
        for i in range(0, n_channels + 1):

            Xs, y, _ = make_multi_input_regression(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed=seed,
                                    **rgr_params
                                    )
            mrgr = MultichannelPipeline(n_channels)
            mrgr.add_layer(StandardScaler(), pipe_processes=n_cpus)
            rgr = transform_wrappers.SingleChannelCV(LinearRegression())
            mrgr.add_layer(rgr, pipe_processes=n_cpus)
            mrgr.add_layer(MultichannelPredictor(SVR()))
            split_accuracies = cross_val_score(
                mrgr, Xs, y, scorer=explained_variance_score,
                cv=3, n_processes=1)
            accuracies.append(np.mean(split_accuracies))

        n_informatives = range(0, n_channels + 1)
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
                        'SVR stacking explained variance of {} is below '
                        'acceptable threshold of 0.80'.format(final_ev))
        linearity = pearsonr(accuracies, n_informatives)[0]
        self.assertTrue(linearity > 0.0,
                        'SVR stacking linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(linearity))

        # implementation 2
        accuracies = []
        for i in range(0, n_channels + 1):

            Xs, y, _ = make_multi_input_regression(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed=seed,
                                    **rgr_params
                                    )
            mrgr = MultichannelPipeline(n_channels)
            mrgr.add_layer(StandardScaler(), pipe_processes=n_cpus)
            base_rgr = LinearRegression()
            mrgr.add_layer(ChannelEnsemble(
                base_rgr, SVR(), internal_cv=5, base_processes='max'))
            split_accuracies = cross_val_score(
                mrgr, Xs, y, scorer=explained_variance_score,
                cv=3, n_processes=1)
            accuracies.append(np.mean(split_accuracies))

        n_informatives = range(0, n_channels + 1)
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
                        'SVR stacking explained variance of {} is below '
                        'acceptable threshold of 0.80'.format(final_ev))
        linearity = pearsonr(accuracies, n_informatives)[0]
        self.assertTrue(linearity > 0.0,
                        'SVR stacking linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(linearity))

    def test_add_layer_interface_broadcast(self, verbose=0, seed=42):
        """
        Functional test of multichannel regression using the
        MultichannnelPipeline.add_layer() method to create the pipeline.
        Single argument broadcasting is used in pipeline construction.
        """

        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")

        n_channels = 5
        accuracies = []

        rgr_params = {'n_samples':1000,
                      'n_features':10,
                      'n_informative':10}

        # implementation 1
        for i in range(0, n_channels + 1):

            Xs, y, _ = make_multi_input_regression(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed = seed,
                                    **rgr_params
                                    )

            mrgr = MultichannelPipeline(n_channels)
            mrgr.add_layer(StandardScaler(), pipe_processes=n_cpus)
            rgr = transform_wrappers.SingleChannelCV(LinearRegression())
            mrgr.add_layer(rgr, pipe_processes=n_cpus)
            mrgr.add_layer(MultichannelPredictor(SVR()))

            split_accuracies = cross_val_score(mrgr, Xs, y,
                                               scorer=explained_variance_score,
                                               cv=3, n_processes=1)
            accuracies.append(np.mean(split_accuracies))


        n_informatives = range(0, n_channels + 1)
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
                        'SVR stacking explained variance of {} is below '
                        'acceptable threshold of 0.80'.format(final_ev))
        linearity = pearsonr(accuracies, n_informatives)[0]
        self.assertTrue(linearity > 0.0,
                        'SVR stacking linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(linearity))

        # implementation 2
        accuracies = []
        for i in range(0, n_channels + 1):

            Xs, y, _ = make_multi_input_regression(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed = seed,
                                    **rgr_params
                                    )

            mrgr = MultichannelPipeline(n_channels)
            mrgr.add_layer(StandardScaler(), pipe_processes=n_cpus)
            base_rgr = LinearRegression()
            mrgr.add_layer(ChannelEnsemble(
                base_rgr, SVR(), internal_cv=5, base_processes='max'))
            split_accuracies = cross_val_score(mrgr, Xs, y,
                                               scorer=explained_variance_score,
                                               cv=3, n_processes=1)
            accuracies.append(np.mean(split_accuracies))

        n_informatives = range(0, n_channels + 1)
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
                        'SVR stacking explained variance of {} is below '
                        'acceptable threshold of 0.80'.format(final_ev))
        linearity = pearsonr(accuracies, n_informatives)[0]
        self.assertTrue(linearity > 0.0,
                        'SVR stacking linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(linearity))

    def test_add_layer_interface_mapping(self, verbose=0, seed=42):
        """
        Functional test of the MultichannelPipeline channel mapping interface.
        """

        if n_cpus > 1:
            # shut off warnings because ray and redis generate massive numbers
            warnings.filterwarnings("ignore")

        n_channels = 5
        accuracies = []

        rgr_params = {'n_samples':1000,
                      'n_features':10,
                      'n_informative':10}

        # implementation 1
        for i in range(0, n_channels + 1):

            Xs, y, _ = make_multi_input_regression(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed = seed,
                                    **rgr_params
                                    )

            mrgr = MultichannelPipeline(n_channels)
            mrgr.add_layer(2, StandardScaler(), 3, StandardScaler(),
                           pipe_processes=n_cpus)
            base_rgr = transform_wrappers.SingleChannelCV(LinearRegression())
            mrgr.add_layer(2, base_rgr, 3, base_rgr, pipe_processes=n_cpus)
            mrgr.add_layer(5, MultichannelPredictor(SVR()))
            split_accuracies = cross_val_score(
                mrgr, Xs, y, scorer=explained_variance_score,
                cv=3, n_processes=1)
            accuracies.append(np.mean(split_accuracies))

        n_informatives = range(0, n_channels + 1)
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
                        'SVR stacking explained variance of {} is below '
                        'acceptable threshold of 0.80'.format(final_ev))
        linearity = pearsonr(accuracies, n_informatives)[0]
        self.assertTrue(linearity > 0.0,
                        'SVR stacking linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(linearity))

        # implementation 2
        accuracies = []
        for i in range(0, n_channels + 1):

            Xs, y, _ = make_multi_input_regression(n_informative_Xs=i,
                                    n_weak_Xs=0,
                                    n_random_Xs=n_channels - i,
                                    weak_noise_sd=None,
                                    seed = seed,
                                    **rgr_params
                                    )

            mrgr = MultichannelPipeline(n_channels)
            mrgr.add_layer(2, StandardScaler(), 3, StandardScaler(),
                           pipe_processes='max')
            base_rfrs = [LinearRegression() for i in range(2)]
            base_rfrs += [LinearRegression() for i in range(3)]
            mrgr.add_layer(ChannelEnsemble(
                base_rfrs, SVR(), base_processes='max', internal_cv=5))
            split_accuracies = cross_val_score(
                mrgr, Xs, y, scorer=explained_variance_score,
                cv=3, n_processes=1)
            accuracies.append(np.mean(split_accuracies))

        n_informatives = range(0, n_channels + 1)
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
                        'SVR stacking explained variance of {} is below '
                        'acceptable threshold of 0.80'.format(final_ev))
        linearity = pearsonr(accuracies, n_informatives)[0]
        self.assertTrue(linearity > 0.0,
                        'SVR stacking linearity of {} below acceptable '
                        'threshold of 0.80 pearsonr'.format(linearity))

    def test_multi_matrices_no_metapredictor(self, verbose=0, seed=42):
        """
        Determine if ChannelEnsemble works without a meta-predictor.

        Determine if it can pick informative input over random and
        test its performance.
        """

        Xs, y, types = make_multi_input_regression(n_informative_Xs=1,
                                n_weak_Xs=0, n_random_Xs=4, weak_noise_sd=None,
                                seed = seed, n_samples=500, n_features=20,
                                n_informative=20)

        mclf = MultichannelPipeline(n_channels=5)
        mclf.add_layer(StandardScaler())
        mclf.add_layer(ChannelEnsemble(LinearRegression(), internal_cv=5,
                                       score_selector=RankScoreSelector(k=1)))
        mclf.fit(Xs, y)

        selected_type = types[mclf.get_model(1, 0).get_support()[0]]

        self.assertTrue(selected_type == 'informative',
                        'Ensemble failed to pick informative channel')

        acc = np.mean(cross_val_score(mclf, Xs, y))
        if verbose > 0:
            print('cross val accuracy: {}'.format(acc))

        self.assertTrue(acc > 0.10, 'Accuracy tolerance failure.')


if __name__ == '__main__':
    unittest.main()
