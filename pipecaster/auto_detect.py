"""Functions for automatically detecting prediction methods after model
fitting.

"""


def _predict(model, Xs, test_indices=None, method_name):
    predict_method = getattr(model, method_name)
    if test_indices is None:
        y_pred = predict_method(Xs)
    else:
        if utils.is_multichannel(model):
            X_tests = [X[test_indices] if X is not None else None
                       for X in Xs]
            y_pred = predict_method(X_tests)
        else:
            y_pred = predict_method(Xs[test_indices])

    return y_pred

def _detect_predict_methods(model, Xs, y, fit_params=None):
        """
        Do forward pass on fit model to detect functional predict methods.
        """
        fit_params = {} if fit_params is None else fit_params

        active_predict_methods = []

        for predict_method in config.recognized_pred_methods:
            try:
                y_pred = getattr(model, predict_method)(Xs, y, **fit_params)
                if y_pred is not None:
                    active_predict_methods.apped(predict_method)
            except:
                pass

        return active_predict_methods
