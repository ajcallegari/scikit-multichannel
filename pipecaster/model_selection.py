import ray
from sklearn.model_selection._split import check_cv
from sklearn.metrics import r2_score

def split_Xs(Xs, train_indices, test_indices):
    X_trains = [X[train_indices] for X in Xs]
    X_tests = [X[test_indices] for X in Xs]
    return X_trains, X_tests

def fit_and_predict(multipredictor, Xs, y, train_indices, test_indices, verbose, fit_params, prediction_method):
    X_trains, X_tests = split_Xs(Xs, train_indices, test_indices)
    
    try:
        if y is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y[train_indices], **fit_params)

    except Exception as e:
        return None

    return getattr(multipredictor, prediction_method)(Xs)

@ray.remote
def ray_fit_and_predict(multipredictor, Xs, y, train_indices, test_indices, verbose, fit_params, prediction_method):
    return fit_and_predict(multipredictor, Xs, y, train_indices, test_indices, verbose, fit_params, prediction_method)
   
def cv_pred_splits(multipredictor, Xs, y=None, *, groups=None, prediction_method = 'predict',
                    cv=3, n_jobs=None, verbose=0, fit_params=None):

    cv = check_cv(cv, y, classifier=is_classifier(predictor))
    
    if n_jobs == 1:
        prediction_splits = [fit_and_predict(get_clone(multipredictor), Xs, y, train_indices, test_indices, verbose, 
                                             fit_params, prediction_method) 
                             for train_indices, test_indices in cv.split(X, y, groups)] 
    elif n_jobs > 1:
        try:
            ray.nodes()
        except RuntimeError:
            ray.init()
        splits = list(cv.split(X, y, groups))
        X = ray.put(X)
        y = ray.put(y)
        fit_params = ray.put(fit_params)
        jobs = [ray_fit_and_predict.remote(ray.put(get_clone(multipredictor)), Xs, y, train_indices, test_indices, verbose, 
                                           fit_params, prediction_method) 
                for train_indices, test_indices in splits] 
        prediction_splits = ray.get(jobs)
        X = ray.get(X)
        y = ray.get(y)
        fit_params = ray.get(fit_params)
    else:
        raise ValueError('invalid n_jobs value: {}.  must be int greater than 0'.format(n_jobs))
        
    return prediction_splits
    
def cross_val_score(multipredictor, Xs, y=None, *, sample_weights = None, groups=None, prediction_method='predict', 
                    scoring_metric=r2_score, cv=3, n_jobs=None, verbose=0, fit_params=None, error_score=np.nan):
    
    prediction_splits = cv_pred_splits(multipredictor, Xs, y, groups=groups, prediction_method=prediction_method,
                    cv=cv, n_jobs=n_jobs, verbose=verbose, fit_params=fit_params)
    
    if sample_weights is None:
        split_scores = [scoring_metric(y[test_indices], predictions) for predictions, indices in prediction_splits]
    else:
        split_scores = [scoring_metric(y[test_indices], predictions, sample_weights = sample_weights) 
                        for predictions, indices in prediction_splits]

    return split_scores
    
    
    
        