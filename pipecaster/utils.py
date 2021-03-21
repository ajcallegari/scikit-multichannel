"""
Utilities for characterizing and defining pipeline components.

Note: The term "pipes" is used to describe objects with the scikit-learn
estimator/transformer or estimator/predictor interfaces, or multichannel
analog of these interfaces.
"""

import numpy as np
from inspect import signature, getfullargspec
import sklearn.base
import joblib
import tempfile
import os
import ray

import pipecaster.config as config

__all__ = ['get_transform_method', 'get_score_method', 'is_classifier',
           'is_regressor', 'is_predictor',
           'is_transformer', 'detect_predictor_type', 'is_multichannel',
           'get_clone', 'get_sklearn_clone', 'get_clones',
           'save_pipe', 'load_pipe', 'get_predict_methods',
           'is_predictor', 'FitError', 'PredictError',
           'ParallelBackendError', 'get_descriptor', 'get_param_names',
           'get_param_clone', 'Cloneable', 'Saveable', 'encode_labels',
           'decode_labels', 'classify_probs', 'classify_decision_function',
           'detect_predict_methods']

def get_transform_method(pipe):
    """
    Get a predictor method for transforming using
    config.transform_method_precedence.

    Parameters
    ----------
    pipe: pipe instance

    Returns
    -------
    Name (str) of a pipe method that can be used for transforming.
    """
    for method_name in config.transform_method_precedence:
        if hasattr(pipe, method_name):
            return method_name
    return None

def get_score_method(pipe):
    """
    Get prediction method name using score_method_precedence.

    Parameters
    ----------
    pipe: pipe instance

    Returns
    -------
    Name (str) of a pipe method that can be used for making predictons for
    performance estimation.
    """
    if is_classifier(pipe):
        for method_name in config.score_method_precedence:
            if hasattr(pipe, method_name):
                return method_name
    else:
        return 'predict'

def get_clone(pipe, stateless=False):
    """
    Get a new copy of a pipe instance.

    Parameters
    ----------
    pipe : pipe instance
    stateless : bool, default=False
        - If False: Use the pipe's get_clone() method or fall back on
          scikit-learn stateless clone sklearn.base.clone(pipe).
        - If True: Force scikit-learn stateless clone:
          sklearn.base.clone(pipe).

    Returns
    -------
    Copy of the pipe argument.

    Notes
    -----
    Custom cloning with get_clone() class methods has been introduced in
    pipecaster to enable neural net warm starts.
    """
    if hasattr(pipe, 'get_clone') and stateless is False:
        return pipe.get_clone()
    else:
        return sklearn.base.clone(pipe)


def get_sklearn_clone(pipe):
    """
    Get a scikit-learn style stateless parameter clone of a pipe.
    """
    return get_clone(pipe, stateless=True)


def get_clones(pipes):
    """
    Clone a pipe or list of pipes.
    """
    if isinstance(pipes, (list, tuple, np.ndarray)):
        return [get_clone(p) if p is not None else None for p in pipes]
    else:
        return get_clone(pipes) if p is not None else None


def is_classifier(pipe):
    """
    Determine if a pipe is a classifier.
    """
    if hasattr(pipe, '_estimator_type'):
        if getattr(pipe, '_estimator_type') == 'classifier':
            return True
    elif hasattr(pipe, 'classes_'):
        return True
    else:
        return False


def is_regressor(pipe):
    """
    Determine if a pipe is a regressor.
    """
    if hasattr(pipe, '_estimator_type'):
        if getattr(pipe, '_estimator_type', None) == 'regressor':
            return True
    elif hasattr(pipe, 'classes_'):
        return False
    else:
        return False


def is_predictor(pipe):
    """
    Determine if a pipe is a predictor.
    """
    for method in config.recognized_pred_methods:
        if hasattr(pipe, method):
            return True
    return False


def is_transformer(pipe):
    """
    Determine if a pipe is a transformer.
    """
    if hasattr(pipe, 'transform'):
        return True
    else:
        return False


def detect_predictor_type(pipe):
    """
    Detect the predictor type of a pipe (classifier, regressor, or None).
    """
    if is_classifier(pipe):
        predictor_type = 'classifier'
    elif is_regressor(pipe):
        predictor_type = 'regressor'
    else:
        predictor_type = None
    return predictor_type


def enforce_fit(pipe):
    """
    Raise TypeError if pipe lacks a fit method.
    """
    if hasattr(pipe, 'fit'):
        return
    else:
        raise TypeError('{} lacks a required fit method'
                        .format(pipe.__class__.__name__))


def enforce_predict(pipe):
    """
    Raise TypeError if pipe lacks a recognized prediction method.
    """
    if is_predictor(pipe):
        return
    else:
        raise TypeError('{} lacks a recognized method for prediction'
                        .format(pipe.__class__.__name__))


def enforce_output(pipe):
    """
    Raise TypeError if pipe lacks recognized methods for generating
        outputs (predicting or transforming).
    """
    if is_predictor(pipe) or is_transformer(pipe):
        return
    else:
        raise TypeError('{} lacks a required transform or prediction method'
                        .format(pipe.__class__.__name__))


def check_pipe_interface(pipe):
    """
    Raise TypeError if if pipe lack a fit method or a recognized method
        for generating output.
    """
    enforce_fit(pipe)
    enforce_output(pipe)


def is_multichannel(pipe):
    """
    Determine if a pipe takes multiple inputs by determining if the first
        argument to its fit() method is 'Xs'.
    """
    first_param = list(signature(pipe.fit).parameters.keys())[0]
    return first_param == 'Xs'


def get_predict_methods(pipe):
    """
    Return a list of the pipe's recongized prediction methods or None.
    """
    return [m for m in config.recognized_pred_methods if hasattr(pipe, m)]


def save_pipe(pipe, filepath):
    """
    Save a pipe to disk (atomic).
    """
    tf = tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(filepath),
                                     delete=False)
    joblib.dump(pipe, tf)
    tempname = tf.name
    tf.close()
    os.rename(tempname, filepath)

def load_pipe(filepath):
    """
    Load a pipe from disk.
    """
    return joblib.load(filepath)

class FitError(Exception):
    """
    Exception to raise when calls to fit() fail.
    """
    def __init__(self, message="call to fit() method failed"):
        self.message = message
        super().__init__(self.message)


class PredictError(Exception):
    """
    Exception to raise when calls to predict() fail.
    """
    def __init__(self, message="call to predict() method failed"):
        self.message = message
        super().__init__(self.message)


class ParallelBackendError(Exception):
    """
    Exception to raise when a parallel backend function fails.
    """
    def __init__(self, message='request to the parallel backend failed'):
        self.message = message
        super().__init__(self.message)


def get_descriptor(obj, verbose=0, params=None):
    """
    Get a text description of on object.

    Parameters
    ----------
    class_name : string
        Name of the class to describe.
    params : dict, default=None
        Dict of parameter value mappings to be included in description.
    verbose : int, default=0
        - If 0: return a string with no parameters,
          e.g.: "RandomForestClassifier"
        - If 1: return a string including parameters in params argument.
          e.g. "RandomForestClassifier(n_estimators=50)"
        - If -1: return a condensed class name, e.g. "RanForCla",
          not implemented)
    """
    if hasattr(obj, 'get_descriptor'):
        return obj.get_descriptor(verbose)
    else:
        if verbose == 0:
            return obj.__class__.__name__
        elif verbose == 1:
            string_ = obj.__class__.__name__ + '('
            argstrings = []
            for k, v in params.items():
                argstring = k + '='
                if hasattr(v, '__name__'):
                    argstring += v.__name__
                elif hasattr(v, '__str__'):
                    argstring += v.__str__()
                elif type(v) in [str, int, float]:
                    argstring += v
                else:
                    argstring += 'NA'
                argstrings.append(argstring)
            string_ += ', '.join(argstrings)
            return string_ + ')'
        elif verbose == -1:
            raise NotImplmentedError('Condensed names not implemented yet.')
        else:
            raise ValueError('Unsupported value for verbose.')


def get_param_names(callable_, omit_self=True):
    """
    Get names of the arguments of a function or method.
    """
    param_names = set(getfullargspec(callable_)[0])
    if omit_self:
        param_names.remove('self')
    return param_names


def get_param_clone(pipe):
    """
    Clone pipe instance by getting/setting parameters (scikit-learn pattern).
    """
    return pipe.__class__(**pipe.get_params())


class Cloneable:

    """
    Base class that provides stateless cloning and text descriptors.

    Eliminates the scikit-learn estimator boilerplate code that converts
    initialization parameters to class attributes.

    Notes
    -----
    Include this line in sublass __init__() to automatically store params:
        _params_to_attributes(self, callable_, locals())

    To clone attributes not present in the signature of __init__, override the
        get_clone() method in derived classes, e.g.:
        ::

            def get_clone(self):
                clone = super().get_clone()
                clone.my_list = my_list.copy()
                return clone
    """

    @property
    def param_names(self):
        return get_param_names(self.__init__)

    def _params_to_attributes(self, callable_, locals_):
        for param_name in get_param_names(callable_):
            setattr(self, param_name, locals_[param_name])

    def get_params(self, deep=False):
        return {p: getattr(self, p) for p in self.param_names}

    def set_params(self, params):
        for key, value in params.items():
            if key in self.__class__.params:
                setattr(self, key, value)
            else:
                raise AttributeError('invalid parameter name')
        return self

    def get_clone(self):
        clone = get_param_clone(self)
        return clone

    def to_str(self, verbose=1):
        return get_descriptor(self, verbose, self.get_params())

    def __str__(self):
        return self.to_str(verbose=1)

    def __repr__(self):
        return self.to_str(verbose=1)


class Saveable:
    """
    Base class for saveable objects, including pipes and pipelines.
    """
    def save(self, filepath):
        """
        Save the model to a filepath using joblib pickling.
        """
        joblib.dump(self, filepath)
        return self

    @staticmethod
    def load(filepath):
        """
        Load the model from a filepath using joblib pickling.
        """
        return joblib.load(filepath)


def encode_labels(y):
    """
    Encode labels as integers.
    """
    if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
        raise NotImplementedError('Multilabel and multi-output \
                                  meta-classification not supported.')
    classes_, y = np.unique(y, return_inverse=True)
    return classes_, y_encoded


def decode_labels(y, classes_):
    """
    Decode integer labels into class labels.
    """
    return classes_[y]


def classify_probs(class_probs, class_names=None,
                    operating_characteristic=None):
    """
    Choose a class based on marginal probabilities.

    Parameters
    ----------
    class_probs : ndarray.shape(n_classes) or list of len n_classes
        Predicted marginal probility of each class.
    class_names : ndarray, default=None
        Ordered array of class names for decoding.
    operating_characteristic : float, default=None
        - Classification threshold for binary classification:
        - If None : choose the class with the greatest marginal probability
        - If float : classify positive class marginal prob values of
          operating_characteristic or above as positive, else negative.
    """
    if operating_characteristic is None:
        class_number = np.argmax(class_probs)
        if class_names is None:
            return class_number
        else:
            return decode_labels(class_number, class_names)
    else:
        if len(class_probs) > 2:
            raise NotImplmentedError('Operating characteristic not \
                                     implemented for more than 2 classes.')
        elif len(class_probs) == 2:
            if class_names is None:
                return 1 if class_probs[1] >= operating_characteristic else 0
            else:
                if class_probs[1] >= operating_characteristic:
                    return decode_labels(1, class_names)
                else:
                    return decode_labels(0, class_names)

def classify_decision_function(decision_function_value, class_names=None,
                    operating_characteristic=0):
    """
    Choose a class based on marginal probabilities.

    Parameters
    ----------
    decision_function_value : scalar
        Decision function value for binary classifier (e.g. output from scikit-
        learn's SVC and LogisticRegression)
    class_names : ndarray, default=None
        Ordered array of class names for decoding.
    operating_characteristic : float, default=None
        - Classification threshold for binary classification:
        - If None : choose the class with the greatest marginal probability
        - If float : classify positive class marginal prob values of
          operating_characteristic or above as positive, else negative.
    """

    if class_names is None:
        return 1 if decision_function_value >= operating_characteristic else 0
    else:
        if decision_function_value >= operating_characteristic:
            return decode_labels(1, class_names)
        else:
            return decode_labels(0, class_names)


def detect_predict_methods(model, Xs):
        """
        Detect functional prediction methods on a fitted model using a forward
        pass.

        This method allows you to detect prediction methods are active on a
        scikit-learn predictor.  Some scikit-learn predictors, like Pipeline,
        have prection method attributes that are non-functional so you can't
        simply read the object's attributes to determine its prediction
        interface.
        """

        if is_multichannel(model):
            Xs_test = [X[0:1,:] if X is not None else None for X in Xs]
        else:
            Xs_test = Xs[0:1,:]

        active_methods = []

        for predict_method in config.recognized_pred_methods:
            if hasattr(model, predict_method):
                try:
                    y_pred = getattr(model, predict_method)(Xs_test)
                    if y_pred is not None:
                        active_methods.append(predict_method)
                except:
                    pass

        return active_methods
