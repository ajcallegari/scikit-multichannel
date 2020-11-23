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
        
    @staticmethod
    def _get_live_channels(input_indices, Xs):
        if type(input_indices) == int:
            live_channels = [] if Xs[input_indices] is None else [input_indices]
        else:
            live_channels = [i for i in input_indices if Xs[i] is not None]   
        return live_channels
    
    @staticmethod
    def _has_live_channels(input_indices, Xs):
        live_channels = Layer._get_live_channels(input_indices, Xs)
        return True if len(live_channels) > 0 else False
    
    def _get_slice_indices(self, slice_):
        if type(slice_) == int:
            return [slice_]
        else:
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
                    self.pipe_list.append((val, slice(i, i+1, 1), [i]))          
        elif is_listlike == True:
            n = len(val)
            if n != len(input_indices):
                raise ValueError('List of pipe objects does not match slice dimension during assignment')
            else:
                for pipe, i in zip(val, input_indices):
                    self.pipe_list.append((val, slice(i, i+1, 1), [i]))
            
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
    
    def fit(self, Xs, y=None, **fit_params):
        """Clone the pipes in this layer, invoke their fit() methods, then replace the current pipe 
           instances with the fitted clone.  If all the input channels for a pipe are dead (None value), 
           the transformer is deleted along with its channel mapping info. 
        """
        for i, (pipe, slice_, input_indices) in enumerate(self.pipe_list):
            if Layer._has_live_channels(input_indices, Xs) == False:
                del self.pipe_list[i] 
            else:
                input_ = Xs[slice_] if utils.is_multi_input(pipe) else Xs[slice_][0]
                if hasattr(pipe, 'fit'):
                    pipe = utils.get_clone(pipe)
                    try:
                        if y is not None:
                            pipe.fit(input_, **fit_params)
                        else:
                            pipe.fit(input_, y, **fit_params)
                    except:
                        raise FitError('pipe {} raised an error on fit()'.format(pipe.__class__.__name__))
                    self.pipe_list[i] = (pipe, slice_, input_indices)
                else:
                    raise FitError('missing fit() method in {}'.format(pipe.__class__.__name__))
    
    def fit_transform(self, Xs, y=None, **fit_params):
        """Clone pipes this layer, invoke their fit_transform() if available or fall back on fit() then transform(). 
           If all input channels for a pipe are dead (None value), pipe and mapping info are deleted.
           Predictors are automatically converted to transformers by wrapping them in the TransformingPredictor class.
        """
        Xs = Xs.copy() 
        for i, (pipe, slice_, input_indices) in enumerate(self.pipe_list):
            if Layer._has_live_channels(input_indices, Xs) == False:
                del self.pipe_list[i] 
            else:
                input_ = Xs[slice_] if utils.is_multi_input(pipe) else Xs[slice_][0]                
                pipe = utils.get_clone(pipe)
                if hasattr(pipe, 'fit_transform'):
                    try:
                        if utils.is_multi_input(pipe):
                            Xs[slice_] = pipe.fit_transform(input_, y, **fit_params)
                        else:
                            Xs[input_indices[0]] = pipe.fit_transform(input_, y, **fit_params)
                    except Exception as e:
                        raise FitError('pipe {} raised an error on fit_transform(): {}'.format(pipe.__class__.__name__, e))
                elif hasattr(pipe, 'fit') and hasattr(pipe, 'transform'):
                    try:
                        if y is not None:
                            pipe.fit(input_, **fit_params)
                        else:
                            pipe.fit(input_, y, **fit_params)
                    except Exception as e:
                        raise FitError('pipe {} raised an error on fit(): {}'.format(pipe.__class__.__name__, e))
                    if utils.is_multi_input(pipe):
                        Xs[slice_] = pipe.transform(input_)
                    else:
                        Xs[input_indices[0]] = pipe.transform(input_)                
                elif hasattr(pipe, 'fit') and utils.is_predictor(pipe):
                    pipe = TransformingPredictor(pipe)
                    try:
                        Xs[input_indices[0]] = pipe.fit_transform(input_, y, **fit_params) 
                    except Exception as e:
                        raise FitError('pipe {} raised an error on fit_transform(): {}'
                                       .format(pipe.__class__.__name__, e))
                else:
                    raise AttributeError('valid transform method not found (fit_transform, transform, \
                                          predict, predict_proba, or decision_function)')
                    
                self.pipe_list[i] = (pipe, slice_, input_indices)   
                
        return Xs
                                     
    def transform(self, Xs):
        Xs = Xs.copy() 
        for i, (pipe, slice_, input_indices) in enumerate(self.pipe_list):
            input_ = Xs[slice_] if utils.is_multi_input(pipe) else Xs[slice_][0]
            transform_method = utils.get_transform_method(pipe)
            if transform_method is not None:
                if utils.is_multi_input(pipe):
                    Xs[slice_] = transform_method(input_)
                else:
                    Xs[input_indices[0]] = transform_method(input_)            
            else:
                raise AttributeError('missing tranform method in {}'.format(pipe.__class__.__name__))
                
        return Xs
                                     
    def predict(self, Xs):
        """Call the predict() methods in this layer
        
        arguments
        ---------
        Xs: list, input channels
        
        returns
        -------
        If one predict() method is found, returns a single prediction array of length n_samples.
        If more than one predict() method is found, returns a list with either the predictions or None for each input channel.
        """
        predictions = [None for X in Xs] 
                                     
        for i, (pipe, slice_, input_indices) in enumerate(self.pipe_list):
            input_ = Xs[slice_] if utils.is_multi_input(pipe) else Xs[slice_][0]
            if hasattr(pipe, 'predict'):
                if utils.is_multi_input(pipe):
                    predictions[slice_] = pipe.predict(input_)
                else:
                    predictions[input_indices[0]] = pipe.predict(input_)        
                    
        outputs = [p for p in predictions if p is not None]
        if len(outputs) == 0:
            raise ValueError('missing predict{} method in Layer')        
        elif len(outputs) == 1:
            # typical pattern: pipeline has converged to a single y
            return outputs[0]
        else:
            # atypical pattern: pipeline has not converged and final layer makes multiple predictions
            return predictions
                                     
    def predict_proba(self, Xs):
        """Call the predict_proba() methods in this layer
        
        arguments
        ---------
        Xs: list, input channels
        
        returns
        -------
        If one predict_proba() method is found, returns a single prediction array of length n_samples.
        If more than one predict_proba() method is found, 
        returns a list with either the predicted probabilities or None for each input channel.
        """
        predictions = [None for X in Xs] 
                                     
        for i, (pipe, slice_, input_indices) in enumerate(self.pipe_list):
            input_ = Xs[slice_] if utils.is_multi_input(pipe) else Xs[slice_][0]
            if hasattr(pipe, 'predict_proba'):
                if utils.is_multi_input(pipe):
                    predictions[slice_] = pipe.predict_proba(input_)
                else:
                    predictions[input_indices[0]] = pipe.predict_proba(input_)        
                    
        outputs = [p for p in predictions if p is not None]
        if len(outputs) == 0:
            raise ValueError('missing predict_proba{} method in Layer')        
        elif len(outputs) == 1:
            # typical pattern: pipeline has converged to a single y
            return outputs[0]
        else:
            # atypical pattern: pipeline has not converged and final layer makes multiple predictions
            return predictions 
                                     
    def decision_function(self, Xs):
        """Call the decision_function() methods in this layer
        
        arguments
        ---------
        Xs: list, input channels
        
        returns
        -------
        If one decision_function() method is found, returns a single prediction array of length n_samples.
        If more than one decision_function() method is found, 
        returns a list with either the decision_function values or None for each input channel.
        """
        predictions = [None for X in Xs] 
                                     
        for i, (pipe, slice_, input_indices) in enumerate(self.pipe_list):
            input_ = Xs[slice_] if utils.is_multi_input(pipe) else Xs[slice_][0]
            if hasattr(pipe, 'decision_function'):
                if utils.is_multi_input(pipe):
                    predictions[slice_] = pipe.decision_function(input_)
                else:
                    predictions[input_indices[0]] = pipe.decision_function(input_)        
                    
        outputs = [p for p in predictions if p is not None]
        if len(outputs) == 0:
            raise ValueError('missing decision_function{} method in Layer')        
        elif len(outputs) == 1:
            # typical pattern: pipeline has converged to a single y
            return outputs[0]
        else:
            # atypical pattern: pipeline has not converged and final layer makes multiple predictions
            return predictions 
                                     
    def get_clone(self):
        clone = Layer(self.n_inputs)
        clone.all_inputs = self.all_inputs
        clone.mapped_inputs = self.mapped_inputs        
        clone.pipe_list = [(utils.get_clone(p), s, i.copy()) for p, s, i in self.pipe_list]
        
        return clone

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
            
    def fit(self, Xs, y=None, **fit_params):
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            try:
                Xs = self.layers[i].fit_transform(Xs, y, **fit_params)
            except Exception as e:
                raise utils.FitError('Error raised during fit() call on layer {}: {}'.format(i, e))
        # fit the last layer without transforming:
        self.layers[-1].fit(Xs, y, **fit_params)
    
    def transform(self, Xs, y=None):
        for layer in self.layers:
            Xs = layer.transform(Xs)
        return Xs
    
    def fit_transform(self, Xs, y, **fit_params):
        for layer in self.layers:
            Xs = layer.fit_transform(Xs, y, **fit_params)
        return Xs
    
    def predict(self, Xs):
        Xs = [X.reshape(1, -1) if len(X.shape) == 1 else X for X in Xs]
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self.layers[i].transform(Xs)
        return self.layers[-1].predict(Xs)
    
    def predict_proba(self, Xs):
        Xs = [X.reshape(1, -1) if len(X.shape) == 1 else X for X in Xs]
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self.layers[i].transform(Xs)
        return self.layers[-1].predict_proba(Xs)
    
    def decision_function(self, Xs):
        Xs = [X.reshape(1, -1) if len(X.shape) == 1 else X for X in Xs]
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self.layers[i].transform(Xs)
        return self.layers[-1].decision_function(Xs)
    
    def get_pipe(self, input_index, layer_index):
        return self.layers[layer_index].get_pipe_from_input(input_index)
    
    def get_clone(self):
        new_pipline = Pipeline(self.n_inputs)
        new_pipline.layers = [layer.get_clone() for layer in self.layers]
        return new_pipline