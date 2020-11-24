import numpy as np
import ray
from sklearn.model_selection._split import check_cv
from sklearn.metrics import r2_score
from pipecaster.utils import is_classifier, get_clone, is_multi_input

def split_Xs(Xs, train_indices, test_indices):
    X_trains = [X[train_indices] for X in Xs]
    X_tests = [X[test_indices] for X in Xs]
    return X_trains, X_tests

def fit_and_predict(multipredictor, Xs, y, train_indices, test_indices, verbose, fit_params, prediction_method):
    X_trains, X_tests = split_Xs(Xs, train_indices, test_indices)
    
    fit_params = {} if fit_params is None else fit_params
    try:
        if y is None:
            multipredictor.fit(X_trains, **fit_params)
        else:
            multipredictor.fit(X_trains, y[train_indices], **fit_params)

    except Exception as e:
        print('fit failed with predictor: ' + multipredictor.__class__.__name__)
        return None
    
    predictions = getattr(multipredictor, prediction_method)(X_tests)

    return predictions, test_indices

@ray.remote
def ray_fit_and_predict(multipredictor, Xs, y, train_indices, test_indices, verbose, fit_params, prediction_method):
    return fit_and_predict(multipredictor, Xs, y, train_indices, test_indices, verbose, fit_params, prediction_method)
   
def cv_pred_splits(multipredictor, Xs, y=None, *, groups=None, prediction_method = 'predict',
                    cv=3, n_jobs=1, verbose=0, fit_params=None):
    
    if is_multi_input(multipredictor) == False:
        raise TypeError('predictor does not take multiple inputs')
    
    live_Xs = [X for X in Xs if X is not None]
    cv = check_cv(cv, y, classifier=is_classifier(multipredictor))
    splits = list(cv.split(live_Xs[0], y, groups))
    
    if n_jobs == 1:
        prediction_splits = [fit_and_predict(get_clone(multipredictor), live_Xs, y, train_indices, test_indices, verbose, 
                                             fit_params, prediction_method) 
                             for train_indices, test_indices in splits] 
    elif n_jobs > 1:
        try:
            ray.nodes()
        except RuntimeError:
            ray.init()
        live_Xs = ray.put(live_Xs)
        y = ray.put(y)
        fit_params = ray.put(fit_params)
        jobs = [ray_fit_and_predict.remote(ray.put(get_clone(multipredictor)), live_Xs, y, train_indices, test_indices, verbose, 
                                           fit_params, prediction_method) 
                for train_indices, test_indices in splits] 
        prediction_splits = ray.get(jobs)
    else:
        raise ValueError('invalid n_jobs value: {}.  must be int greater than 0'.format(n_jobs))
        
    return prediction_splits
    
def cross_val_score(multipredictor, Xs, y=None, *, sample_weights=None, groups=None, prediction_method='predict', 
                    scoring_metric=r2_score, cv=3, n_jobs=1, verbose=0, fit_params=None, error_score=np.nan):
    
    prediction_splits = cv_pred_splits(multipredictor, Xs, y, groups=groups, prediction_method=prediction_method,
                    cv=cv, n_jobs=n_jobs, verbose=verbose, fit_params=fit_params)
        
    
    if sample_weights is None:
        split_scores = [scoring_metric(y[indices], predictions) for predictions, indices in prediction_splits]
    else:
        split_scores = [scoring_metric(y[indices], predictions, sample_weights=sample_weights) 
                        for predictions, indices in prediction_splits]

    return split_scores
    
    
    
        