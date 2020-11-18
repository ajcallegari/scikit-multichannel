import numpy as np

from pipecaster.utility import get_clone, is_multi_input
from pipecaster.metaprediction import TransformingPredictor

__all__ = ['Layer', 'Pipeline']
       
class Layer:
    """A list of pipe instances with input mappings to support multi-input transformers. Also supports construction via slice assignment and broadcasting.
    
    Examples
    --------
    
    """
    
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.pipe_list = []
        self.all_inputs = set(range(n_inputs))
        self.mapped_inputs = set()
        
    def _get_slice_indices(self, slice_):
        return np.arange(self.n_inputs)[slice_]
        
    def __setitem__(self, slice_, val):
        
        is_listlike = isinstance(val, (list, tuple, np.ndarray))
        
        if type(slice_) == slice:
            if slice_.step not in [None, 1]:
                raise ValueError('Invalid slice step; must be exactly 1 (Pipes may only accept contiguous inputs)')
            input_indices = self._get_slice_indices(slice_)
            if len(input_indices) <= 0:
                raise ValueError('Invalid slice: no inputs')
        elif type(slice_) == int:
            input_indices = [slice_]
        else:
            raise ValueError('unrecognized slice format')
            
        for i in input_indices:
            if i not in self.all_inputs:
                raise ValueError('Slice index out of bounds') 
            if i in self.mapped_inputs:
                raise ValueError('Two pipes are mapped to input {}.  Max allowed is 1'.format(i)) 
                
        if is_listlike == False:
            n = len(input_indices)
            if is_multi_input(val) == True:
                self.pipe_list.append((val, slice_, input_indices))
            else:
                for i in input_indices:
                    self.pipe_list.append((val, i, [i]))          
        elif is_listlike == True:
            n = len(val)
            if n != len(input_indices):
                raise ValueError('List of pipe objects does not match slice dimension during assignment')
            else:
                for pipe, i in zip(val, input_indices):
                    self.pipe_list.append((val, i, [i]))
            
        self.mapped_inputs = self.mapped_inputs.union(input_indices)
        
        return self
    
    def clear(self):
        self.pipe_list = []
        self.all_inputs = set(range(self.n_inputs))
        self.mapped_inputs = set()
        
    def get_pipe_from_input(self, input_index):
        for pipe, slice_ in layer.pipe_list:
            if type(slice_) == int:
                if slice_ == input_index:
                    return pipe
            elif input_index in self._get_slice_indices(slice_):
                return pipe
        return None

class Pipeline:
    
    """Machine learning pipeline that accepts multiple inputs.
    
    """
    
    def __init__(self, n_inputs = 3):
        self.n_inputs = n_inputs
        self.layers = []
        
    def get_next_layer(self):
        layer = Layer(self.n_inputs)
        self.layers.append(layer)
        return layer
    
    @staticmethod
    def _get_live_inputs(input_indices, Xs):
        if type(input_indices) == int:
            live_inputs = [] if Xs[input_indices] is None else input_indices
        else:
            live_inputs = [i for i in input_indices if Xs[i] is not None]   
        return live_inputs
            
    def _fit_layer(self, layer_index, Xs, y, **fit_params):
        is_last = True if layer_index + 1 == len(self.layers) else False
        layer = self.layers[layer_index]
        for i, (pipe, slice_, input_indices) in enumerate(layer.pipe_list):
            if hasattr(pipe, 'fit'):
                live_inputs = Pipeline._get_live_inputs(input_indices, Xs)
                if len(live_inputs) > 0:
                    input_ = [Xs[slice_]] if (is_multi_input(pipe) and len(input_indices) == 1) else Xs[slice_]
                    pipe = get_clone(pipe)
                    if hasattr(pipe, 'transform') == False:
                        pipe = TransformingPredictor(pipe, method='auto', internal_cv = 5, n_jobs = 1)
                    pipe.fit(input_, y, **fit_params)
                    self.layers[layer_index].pipe_list[i] = (pipe, slice_, input_indices)
                else:
                    del self.layers[layer_index].pipe_list[i]
            else:
                raise TypeError('{} in layer {} is missing a fit() method'.format(pipe.__class__.__name__, layer_index))
            
    def _transform_layer(self, layer_index, Xs):
        Xs = Xs.copy() 
        layer = self.layers[layer_index]
        for i, (pipe, slice_, input_indices) in enumerate(layer.pipe_list):
            input_ = [Xs[slice_]] if (is_multi_input(pipe) and len(input_indices) == 1) else Xs[slice_]
            
            if hasattr(pipe, 'transform'):
                Xs[slice_] = pipe.transform(input_)                
            elif hasattr(pipe, 'predict_proba'):
                Xs[slice_] = pipe.predict_proba(input_)
            elif hasattr(pipe, 'decision_function'):
                Xs[slice_] = pipe.decision_function(input_)                
            elif hasattr(pipe, 'predict'):
                Xs[slice_] = pipe.predict(input_)
            else:
                raise TypeError('{} in layer {} lacks a method for transforming data'
                                    .format(pipe.__class__.__name__, layer_index)) 
                
            for i in input_indices:
                # when regressors or other pipe components return a 1D array, convert to a 1 column matrix
                if Xs[i] is not None and len(Xs[i].shape) == 1:
                    Xs[i] = Xs[i].reshape(-1,1)
        return Xs
    
    def _predict_layer(self, layer_index, Xs, proba = False):
        predictions = [None for X in Xs] 
        layer = self.layers[layer_index]
        
        for i, (pipe, slice_, input_indices) in enumerate(layer.pipe_list):
            input_ = [Xs[slice_]] if (is_multi_input(pipe) and len(input_indices) == 1) else Xs[slice_]
            if proba == True:
                if hasattr(pipe, 'predict_proba'):
                    predictions[slice_] = pipe.predict_proba(input_)
                else: 
                    raise TypeError('{} in layer {} lacks a required predict_proba() method'
                                    .format(pipe.__class__.__name__, layer_index)) 
            else:
                if hasattr(pipe, 'predict'):
                    predictions[slice_] = pipe.predict(Xs[input_])
                else:
                    raise TypeError('{} in layer {} lacks a required predict() method'
                                    .format(pipe.__class__.__name__, layer_index))  
                    
        outputs = [p for p in predictions if p is not None]
        
        if len(outputs) == 1:
            # typical pattern: pipeline has converged to a single y
            return outputs[0]
        else:
            # atypical pattern: pipeline has not converged and final layer makes multiple predictions
            return predictions
        
    def fit(self, Xs, y, **fit_params):
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            self._fit_layer(i, Xs, y, **fit_params)
            Xs = self._transform_layer(i, Xs)
        # fit the last layer without transforming:
        self._fit_layer(n_layers - 1, Xs, y, **fit_params)
    
    def transform(self, Xs, y=None):
        for i in range(len(self.layers)):
            Xs = self._transform_layer(i, Xs)
        return Xs
    
    def fit_transform(self, Xs, y, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)
    
    def predict(self, Xs):
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self._transform_layer(i, Xs)
        return self._predict_layer(n_layers - 1, Xs)
    
    def predict_proba(self, Xs):
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self._transform_layer(i, Xs)
        return self._predict_layer(n_layers - 1, Xs, proba = True)
    
    def get_pipe(self, input_index, layer_index):
        return self.layers[layer_index].get_pipe_from_input(input_index)
    