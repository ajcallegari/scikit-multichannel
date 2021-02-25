"""
Selectors that choose items from a list based on figure of merit scores.

(Use for feature selection and channel selection).

Signature:
    selection_indices = score_selector(scores)
"""

import numpy as np

from pipecaster.utils import Cloneable, Saveable



__all__ = ['RankScoreSelector', 'PctRankScoreSelector',
           'HighPassScoreSelector', 'VarianceHighPassScoreSelector']


class RankScoreSelector(Cloneable, Saveable):
    """
    Select items with the highest scores.

    Parameters
    ----------
    k: int
        The number of items to select.
    """
    def __init__(self, k):
        self.k = k if k >= 1 else k

    def __call__(self, scores):
        """
        Select items and return their indices.

        Parameters:
        ----------
        scores: list or nd.array.shape(n_samples,)
            Ordered list of scalar figure of merit scores for each item of
            selection.

        Returns:
        --------
        selection_indices: list of ndarray.shape(n_selections,)
            List of int indices for the selected items.
        """
        scores = [np.NINF if s is None else s for s in scores]
        scores = np.array(scores, dtype=object)
        n_live = sum([1 for s in scores if s is not None])
        k = n_live if self.k > n_live else self.k
        return np.argsort(scores)[-k:]


class PctRankScoreSelector(Cloneable, Saveable):
    """
    Select a specified percentage of the top-scoring items.

    Parameters
    ----------
    pct: float
        The percentage of items to select.
    n_min: int
        The minimum number of items to select.  Takes precedence over pct.
    """
    def __init__(self, pct, n_min=1):
        self._params_to_attributes(PctRankScoreSelector.__init__, locals())

    def __call__(self, scores):
        """
        Select items and return their indices.

        Parameters:
        ----------
        scores: list or nd.array.shape(n_samples,)
            Ordered list of scalar figure of merit scores for each item of
            selection.

        Returns:
        --------
        selection_indices: list of ndarray.shape(n_selections,)
            List of int indices for the selected items.
        """
        scores = [np.NINF if s is None else s for s in scores]
        scores = np.array(scores, dtype=object)
        k = int(len(scores) * self.pct/100.0)
        k = 1 if k < 1 else k
        n_live = sum([1 for s in scores if s is not None])
        k = n_live if k > n_live else k
        selected_indices = np.argsort(scores)[-k:]
        if len(selected_indices) < self.n_min:
            selected_indices = RankScoreSelector(self.n_min)(scores)
        return selected_indices


class HighPassScoreSelector(Cloneable, Saveable):
    """
    Select items with scores above a specified cutoff value.

    Parameters
    ----------
    cutoff : float, default=0.0
        Score that defines the selection.  Items with scores above this value
        are selected.
    n_min : int, default=1
        The minimum number of items to select.  Takes precedence over cutoff.
    """

    def __init__(self, cutoff=0.0, n_min=1):
        self._params_to_attributes(HighPassScoreSelector.__init__, locals())

    def __call__(self, scores):
        """
        Select items and return their indices.

        Parameters:
        ----------
        scores: list or nd.array.shape(n_samples,)
            Ordered list of scalar figure of merit scores for each item of
            selection.

        Returns:
        --------
        selection_indices: list of ndarray.shape(n_selections,)
            List of int indices for the selected items.
        """
        scores = [np.NINF if s is None else s for s in scores]
        scores = np.array(scores, dtype=object)
        selected_indices = np.flatnonzero(np.array(scores) > self.cutoff)
        if len(selected_indices) < self.n_min:
            selected_indices = RankScoreSelector(self.n_min)(scores)
        return selected_indices


class VarianceHighPassScoreSelector(Cloneable, Saveable):
    """
    Select items with scores above a relative cutoff.

    Cutoff value is definied relative to a statistic describing score variance
    and a statistic describing the baseline:

    cutoff = baseline + variance_cutoff * variance.

    Parameters
    ----------
    variance_cutoff : float, default=2.0
        The number of units of variance used to define the cutoff.  Items with
        scores above variance_cutoff * variance will will be selected.
    get_variance : callable, default=np.nanstd
        Callable that provides a scalar measure of the variability of the
        scores (e.g. np.nanstd, scipy.stats.iqr).
        Pattern: variance = get_variance(scores)
    get_baseline : callable, default=np.nanmean
        Callable that provides a scalar baseline score (e.g. np.nanmean or
        np.nanmedian).
        Pattern : baseline = get_baseline(scores)
    n_min : int, default=1
        The minimum number of items to select.  Takes precedence over other
        parameters.
    """

    def __init__(self, variance_cutoff=2.0, get_variance=np.nanstd,
                 get_baseline=np.nanmean, n_min=1):
        self._params_to_attributes(VarianceHighPassScoreSelector.__init__,
                                   locals())

    def __call__(self, scores):
        """
        Select items and return their indices.

        Parameters:
        ----------
        scores : list or nd.array.shape(n_samples,)
            Ordered list of scalar figure of merit scores for each item of
            selection.

        Returns:
        --------
        selection_indices: list of ndarray.shape(n_selections,)
            List of int indices for the selected items.
        """
        scores = [np.NINF if s is None else s for s in scores]
        finite_scores = [s for s in scores if np.isfinite(s)]
        scores = np.array(scores, dtype=object)
        variance = self.get_variance(finite_scores)
        baseline = self.get_baseline(finite_scores)
        selected_indices = np.flatnonzero(
                          scores >= baseline + self.variance_cutoff * variance)
        if len(selected_indices) < self.n_min:
            selected_indices = RankScoreSelector(self.n_min)(scores)
        return selected_indices
