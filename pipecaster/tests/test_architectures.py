import numpy as np
import unittest

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import balanced_accuracy_score

from pipecaster.multichannel_pipeline import MultichannelPipeline
from pipecaster.channel_selection import SelectKBestScores
from pipecaster.ensemble_learning import ChannelEnsemble
from pipecaster.cross_validation import cross_val_score
import pipecaster.transform_wrappers as transform_wrappers

class TestArchitectures(unittest.TestCase):

    def test_architecture_01(self, verbose=0, seed=42):
        """
        Test the accuracy and hygiene (shuffle control) of a complex pipeline
        with feature selection, matrix selection, model selection, and
        model stacking.
        """
        X_rand = np.random.rand(500, 30)
        X_inf, y = make_classification(n_samples=500, n_features=30,
                                       n_informative=15, class_sep=3,
                                       random_state=seed)

        Xs = [X_rand, X_rand, X_inf, X_rand, X_inf, X_inf]

        clf = MultichannelPipeline(n_channels=6)
        clf.add_layer(SimpleImputer())
        clf.add_layer(StandardScaler())
        clf.add_layer(SelectPercentile(percentile=25))
        clf.add_layer(5, SelectKBestScores(feature_scorer=f_classif,
                                           aggregator=np.mean, k=2))
        LR = transform_wrappers.SingleChannelCV(LogisticRegression())
        clf.add_layer(
            5, ChannelEnsemble(predictors=KNeighborsClassifier(), k=1),
            1, LR)
        clf.add_layer(MultichannelPredictor(SVC()))

        score = np.mean(
            cross_val_score(clf, Xs, y, scorer=balanced_accuracy_score))
        if verbose > 0:
            print('accuracy score: {}'.format(score))
        self.assertTrue(score > 0.95, 'Accuracy score of {} did not exceed '
                        'tolerance value of 95%'.format(score))

        clf.fit(Xs, y)
        score_selector = clf.get_model(3,0)
        if verbose > 0:
            print('indices selected by SelectKBestScores: {}'
                  .format(score_selector.get_support()))
            print('correct indices: [2, 4]')
        self.assertTrue(np.array_equal(score_selector.get_support(), [2, 4]),
                        'SelectKBestScores selected the wrong channels.')

        model_selector = clf.get_model(4,0)
        if verbose > 0:
            print('indices selected by SelectKBestModels: {}'
                  .format(model_selector.get_support()))
            print('correct indices: [2, 4]')
        self.assertTrue(model_selector.get_support()[0] in [2, 4],
                        'SelectKBestModels selected the wrong model')

        score = np.mean(
            cross_val_score(clf, Xs, y[np.random.permutation(len(y))],
                            scorer=balanced_accuracy_score))
        if verbose > 0:
            print('shuffle control accuracy score: {}'.format(score))
        self.assertTrue(score < 0.55, 'Accuracy score of shuffle control, {}, '
                        'exceeded tolerance value of 55%'.format(score))

if __name__ == '__main__':
    unittest.main()
