import numpy as np
import pipecaster.utils as utils
from pipecaster.metaprediction import TransformingPredictor

__all__ = ['Layer', 'Pipeline']
       
class Layer:
    """A list of pipe instances with input channel mappings supporting both single and multi-input transformers. Supports construction via slice assignment and broadcasting.
    
    Examples
    --------
    
    """
    
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.pipe_list = []
        self.all_inputs = set(range(n_inputs))
        self.mapped_inputs = set()
        
    def _get_slice_indices(self, slice_):
        return list(range(self.n_inputs)[slice_])
        
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
            raise TypeError('unrecognized slice format')
            
        for i in input_indices:
            if i not in self.all_inputs:
                raise IndexError('Slice index out of bounds') 
            if i in self.mapped_inputs:
                raise ValueError('Two pipes are mapped to input {}.  Max allowed is 1'.format(i)) 
                
        if is_listlike == False:
            n = len(input_indices)
            if utils.is_multi_input(val) == True:
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
    
    def fit(self, Xs, y=None, fit_params):
        for i, (pipe, slice_, input_indices) in enumerate(self.pipe_list):
            input_ = [Xs[slice_]] if (type(slice_) == int and utils.is_multi_input(pipe)) else Xs[slice_]
            if hasattr(pipe, 'fit'):
                n_live_inputs = len(Pipeline._get_live_inputs(input_indices, Xs))
                if n_live_inputs == 0:
                    del self.layers[layer_index].pipe_list[i]
                else:
                    input_ = [Xs[slice_]] if (type(slice_) == int and utils.is_multi_input(pipe)) else Xs[slice_]
                    pipe = utils.get_clone(pipe)
                    try:
                        if y is not None:
                            pipe.fit(input_, **fit_params)
                        else:
                            pipe.fit(input_, y, **fit_params)
                    except:
                        raise FitError('pipe {} raised an error on fit()'.format(pipe.__class__.__name__))
                    self.layers[layer_index].pipe_list[i] = (pipe, slice_, input_indices)
            else:
                raise FitError('missing fit() method in {}'.format(pipe.__class__.__name__))
                
            for i in input_indices:
                # when regressors or other pipe components return a 1D array, convert to a 1 column matrix
                if Xs[i] is not None and len(Xs[i].shape) == 1:
                    Xs[i] = Xs[i].reshape(-1,1)
        return Xs
    
    def fit_transform(self, Xs, y=None, fit_params):
        """Replace the transformers in this layer with clones, then call fit_transform() if the 
           method is available or fall back on fit() then transform(). If the input channel is dead (None value),
           the transformer is deleted along with its channel mapping info. 
        """
        Xs = Xs.copy() 
        for i, (pipe, slice_, input_indices) in enumerate(self.pipe_list):
            input_ = [Xs[slice_]] if (type(slice_) == int and utils.is_multi_input(pipe)) else Xs[slice_]
            n_live_inputs = len(Pipeline._get_live_inputs(input_indices, Xs))
            
            if n_live_inputs == 0:
                del self.layers[layer_index].pipe_list[i]
            else:
                pipe = utils.get_clone(pipe)
                if hasattr(pipe, 'fit_transform'):
                    try:
                        Xs[slice_] = pipe.fit_transform(input_, y, **fit_params)
                    except:
                        raise FitError('pipe {} raised an error on fit_transform()'.format(pipe.__class__.__name__))
                elif hasattr(pipe, 'fit'):
                    try:
                        if y is not None:
                            pipe.fit(input_, **fit_params)
                        else:
                            pipe.fit(input_, y, **fit_params)
                    except:
                        raise FitError('pipe {} raised an error on fit()'.format(pipe.__class__.__name__))
                    else:
                        raise AttributeError('missing fit() method in {}'.format(pipe.__class__.__name__))

                    transform_method = utils.get_transform_method(pipe)
                    if transform_method is not None:
                        Xs[slice_] = transform_method(input_)  
                    else:
                        raise AttributeError('missing tranform method in {}'.format(pipe.__class__.__name__)
                                         
                self.layers[layer_index].pipe_list[i] = (pipe, slice_, input_indices)   
                                         
        for i in input_indices:
            # when regressors or other pipe components return a 1D array, convert to a 1 column matrix
            if Xs[i] is not None and len(Xs[i].shape) == 1:
                Xs[i] = Xs[i].reshape(-1,1)
                
        return Xs
                                     
    def transform(self, Xs):
        Xs = Xs.copy() 
        for i, (pipe, slice_, input_indices) in enumerate(self.pipe_list):
            input_ = [Xs[slice_]] if (type(slice_) == int and utils.is_multi_input(pipe)) else Xs[slice_]
            transform_method = utils.get_transform_method(pipe)
            if transform_method is not None:
                Xs[slice_] = transform_method(input_)  
            else:
                raise AttributeError('missing tranform method in {}'.format(pipe.__class__.__name__)
                
        for i in input_indices:
            # when regressors or other pipe components return a 1D array, convert to a 1 column matrix
            if Xs[i] is not None and len(Xs[i].shape) == 1:
                Xs[i] = Xs[i].reshape(-1,1)
                
        return Xs
                                     
    def predict(self, Xs, method = 'auto'):
        predictions = [None for X in Xs] 
                                     
        for i, (pipe, slice_, input_indices) in enumerate(self.pipe_list):
            input_ = [Xs[slice_]] if (type(slice_) == int and utils.is_multi_input(pipe)) else Xs[slice_]
            if method == 'auto':
                predict_method = utils.get_predict_method(pipe)
            else:
                 if hasattr(pipe, method):
                     predict_method = getattr(pipe, method)
                 else:
                     raise AttributeError('predict method {} not found in attributes of pipeline component'.format(method))
            predictions[slice_] = predict_method(input_)
                    
        outputs = [p for p in predictions if p is not None]
        if len(outputs) == 0:
            raise ValueError('no predictions made by call to predict()')        
        elif len(outputs) == 1:
            # typical pattern: pipeline has converged to a single y
            return outputs[0]
        else:
            # atypical pattern: pipeline has not converged and final layer makes multiple predictions
            return predictions
    
    def get_clone(self):
        new_layer = Layer(self.n_inputs)
        new_layer.all_inputs = self.all_inputs
        new_layer.mapped_inputs = self.mapped_inputs        
        new_layer.pipe_list = [(utils.get_clone(p), s, i.copy()) for p, s, i in self.pipe_list]
        
        return new_layer

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
            
    def fit(self, Xs, y=None, fit_params=None):
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            try:
                Xs = self.layers[i].fit_transform(Xs, y, fit_params)
            except Exception as e::
                raise utils.FitError('Error raised during fit() call on layer {}: {}'.format(i, e))
        # fit the last layer without transforming:
        self.layers[-1].fit(Xs, y, fit_params)
    
    def transform(self, Xs, y=None):
        for layer in self.layers:
            Xs = layer.transform(i, Xs)
        return Xs
    
    def fit_transform(self, Xs, y, **fit_params):
        self.fit(Xs, y, fit_params)
        return self.transform(Xs)
    
    def predict(self, Xs):
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self._transform_layer(i, Xs)
        return self._predict_layer(n_layers - 1, Xs, method = 'predict')
    
    def predict_proba(self, Xs):
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self._transform_layer(i, Xs)
        return self._predict_layer(n_layers - 1, Xs, method = 'predict_proba')
    
    def decision_function(self, Xs):
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self._transform_layer(i, Xs)
        return self._predict_layer(n_layers - 1, Xs, method = 'decision_function')
    
    def get_pipe(self, input_index, layer_index):
        return self.layers[layer_index].get_pipe_from_input(input_index)
    
    def get_clone(self):
        new_pipline = Pipeline(self.n_inputs)
        new_pipline.layers = [layer.get_clone() for layer in self.layers]
        return new_pipline