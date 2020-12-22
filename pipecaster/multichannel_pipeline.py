import numpy as np

import pipecaster.utils as utils
from pipecaster.utils import Cloneable, Saveable, FitError
import pipecaster.transform_wrappers as transform_wrappers

__all__ = ['Layer', 'MultichannelPipeline']

        
def get_live_channels(channel_indices, Xs):
    if type(channel_indices) == int:
        live_channels = [] if Xs[channel_indices] is None else [channel_indices]
    else:
        live_channels = [i for i in channel_indices if Xs[i] is not None]   
    return live_channels

def has_live_channels(channel_indices, Xs):
    return True if get_live_channels(channel_indices, Xs) > 0 else False
       
class Layer(Cloneable, Saveable):
    """A list of transformer and/or predictor instances with channel mappings to con
    
    Examples
    --------
    
    """
    
    state_variables = [] 
    
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.pipe_list = []
        self._all_channels = set(range(n_channels))
        self._mapped_channels = set()
        
    def get_estimator_types(self):
        pipe_list = self.model_list if hasattr(self, 'model_list') else self.pipe_list
        estimator_types = []
        for pipe, _, _ in pipe_list:
            if hasattr(pipe, '_etimator_type'):
                estimator_types.append(pipe._estimator_type)
        return estimator_types
    
    def _get_slice_indices(self, slice_):
        if type(slice_) == int:
            return [slice_]
        else:
            return list(range(self.n_channels)[slice_])
        
    def __setitem__(self, slice_, val):
        
        is_listlike = isinstance(val, (list, tuple, np.ndarray))
        
        if is_listlike:
            for pipe in val:
                utils.check_pipe_interface(pipe)
        else:
            utils.check_pipe_interface(val)
        
        if type(slice_) == slice:
            if slice_.step not in [None, 1]:
                raise ValueError('Invalid slice step; must be exactly 1 (Pipes may only accept contiguous inputs)')
            channel_indices = self._get_slice_indices(slice_)
            if len(channel_indices) <= 0:
                raise ValueError('Invalid slice: no inputs')
        elif type(slice_) == int:
            channel_indices = [slice_]
        else:
            raise TypeError('unrecognized slice format')
            
        for i in channel_indices:
            if i not in self._all_channels:
                raise IndexError('Slice index out of bounds') 
            if i in self._mapped_channels:
                raise ValueError('Two pipes are mapped to channel {}.  Max allowed is 1'.format(i)) 
                
        if is_listlike == False:
            n = len(channel_indices)
            if utils.is_multichannel(val) == True:
                self.pipe_list.append((val, slice_, channel_indices))
            else:
                for i in channel_indices:
                    self.pipe_list.append((val, slice(i, i+1, 1), [i]))          
        elif is_listlike == True:
            n = len(val)
            if n != len(channel_indices):
                raise ValueError('List of pipe objects does not match slice dimension during assignment')
            else:
                for pipe, i in zip(val, channel_indices):
                    self.pipe_list.append((val, slice(i, i+1, 1), [i]))
            
        self._mapped_channels = self._mapped_channels.union(channel_indices)
        
        return self
        
    def get_pipe_from_input(self, input_index):
        for pipe, slice_ in layer.pipe_list:
            if type(slice_) == int:
                if slice_ == input_index:
                    return pipe
            elif input_index in self._get_slice_indices(slice_):
                return pipe
        return None
    
    def fit_transform(self, Xs, y=None, internal_cv=5, **fit_params):
        """
        For each pipe in this layer, call fit_transform() if available, fall back on fit() then transform(), 
        or automatically convert predictors into transformers to enable meta-prediction.  
        
        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
        y: targets, default=None
        internal_cv: None, int, or cv split generator, default=5
            Control the autoconversion of predictors into transformers for meta-prediction.
            If an integer above 1 or a split generator, wrap pipes lacking transform methods in CvTransformer
            If None or int below 2, wrap pipes lacking transform methods in PredictingTransformer
        fit_params: dict, default={}
            Auxiliary parameters to be sent to the fit_transform or fit methods of the pipes.
            Currently there is not support for pipe-specific parameters, but this is on the short-list.
            
        Returns
        -------
        Xs_t: list of [ndarray.shape(n_samples, n_features) or None] with length len(Xs)
            Transformed matrices or passthrough from inputs.  A None value indicates that the transformation 
            inactivated (selected against) a channel or this channel was inactivated in a prior layer.
        
        Notes
        -----
        Models are only fit/transformed for active channels.
        Classes that lack transform methods and have a prediction method are automatically converted 
        to transformers by wrapping them in either the PredictingTransformer or CvTransformer class.
        The wrapper and internal cv training is set by the 'internal_cv' parameter in fit_params if available.  
        If there is no 'internal_cv' parameter in fit_params, 
        """
        
        Xs_t = Xs.copy() 
        model_list = []
        for i, (pipe, slice_, channel_indices) in enumerate(self.pipe_list):
            if has_live_channels(channel_indices, Xs):
                input_ = Xs[slice_] if utils.is_multichannel(pipe) else Xs[slice_][0]                
                model = utils.get_clone(pipe)
                
                if hasattr(model, 'fit_transform'):
                    try:
                        if utils.is_multichannel(pipe):
                            if y is None:
                                Xs_t[slice_] = model.fit_transform(input_, **fit_params)
                            else:
                                Xs_t[slice_] = model.fit_transform(input_, y, **fit_params)
                        else:
                            if y is None:
                                Xs_t[channel_indices[0]] = model.fit_transform(input_, **fit_params)
                            else:
                                Xs_t[channel_indices[0]] = model.fit_transform(input_, y, **fit_params)
                    except Exception as e:
                        raise FitError('pipe {} raised an error on fit_transform(): {}'.format(model.__class__.__name__, e))
                        
                elif hasattr(model, 'fit') and hasattr(pipe, 'transform'):
                    try:
                        if y is not None:
                            model.fit(input_, **fit_params)
                        else:
                            model.fit(input_, y, **fit_params)
                    except Exception as e:
                        raise FitError('pipe {} raised an error on fit(): {}'.format(model.__class__.__name__, e))
                    if utils.is_multichannel(model):
                        Xs_t[slice_] = model.transform(input_)
                    else:
                        Xs_t[channel_indices[0]] = model.transform(input_)       
                        
                elif utils.is_predictor(model):
                    
                    if utils.is_multichannel(model):
                        if internal_cv is None or (type(internal_cv) == int and internal_cv < 2):
                            model = transform_wrappers.Multichannel(model)
                        else:
                            model = transform_wrappers.MultichannelCV(model, internal_cv, cv_processes=1)
                        try:
                            if y is None:
                                Xs_t[channel_indices[0]] = model.fit_transform(input_, **fit_params) 
                            else:
                                Xs_t[channel_indices[0]] = model.fit_transform(input_, y, **fit_params)
                        except Exception as e:
                            raise FitError('pipe {} raised an error on fit_transform(): {}'
                                           .format(model.__class__.__name__, e))
                    else:
                        if internal_cv is None or (type(internal_cv) == int and internal_cv < 2):
                            model = transform_wrappers.SingleChannel(model)
                        else:
                            model = transform_wrappers.SingleChannelCv(model, internal_cv, cv_processes=1)
                        try:
                            if y is None:
                                Xs_t[channel_indices[0]] = model.fit_transform(input_, **fit_params) 
                            else:
                                Xs_t[channel_indices[0]] = model.fit_transform(input_, y, **fit_params)
                        except Exception as e:
                            raise FitError('pipe {} raised an error on fit_transform(): {}'
                                           .format(model.__class__.__name__, e))
                    
                self.model_list.append((model, slice_, channel_indices))
                
        return Xs_t
    
    def fit_final(self, Xs, y=None, internal_cv=5, **fit_params):
        """
        Fit the final layer of a MultiChannelPipeline. Exposes available prediction and transform 
        methods on the layer and returns a list of prediction methods.
        
        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
        y: targets, default=None
        internal_cv: None, int, or cv split generator, default=5
            Control the autoconversion of predictors into transformers for meta-prediction.
            If an integer above 1 or a split generator, wrap predictors to provide internal cv traning & transform method
            If None or int below 2, wrap predictors to provide transform method without internal cv training
        fit_params: dict, default={}
            Auxiliary parameters to be sent to the fit_transform or fit methods of the pipes.
            Currently there is not support for pipe-specific parameters, but this is on the short-list.
        
        Notes
        -----
        Models are only fit/transformed for active channels.
        Classes that lack transform methods and have a prediction method are automatically converted 
            to transformers by wrapping them in the following classes from the transform_wrappers module:
            SingleChannel, SingleChennelCV, Multichannel, MultichannelCV).  The conversion process is controlled
            by the transform_method_precedence variable in the transform_wrappers module.

        """
        
        model_list = []
        prediction_method_names = []
        for pipe, slice_, channel_indices in self.pipe_list:
            if has_live_channels(channel_indices, Xs):
                model = utils.get_clone(pipe)
                input_ = Xs[slice_] if utils.is_multichannel(model) else Xs[slice_][0]                
                
                if hasattr(model, 'transform') == False:
                    if utils.is_multichannel(model):
                        model = transform_wrappers.MultiChannel(model)
                    else:
                        model = transform_wrappers.SingleChannel(model)
                        
                if y is not None:
                    model.fit(input_, **fit_params)
                else:
                    model.fit(input_, y, **fit_params)
            
                prediction_method_names.extend(utils.get_prediction_method_names(model))
                self.model_list.append((model, slice_, channel_indices))
                
        prediction_method_names = set(prediction_method_names)
        # expose predictor interface
        for method_name in prediction_method_names:
            prediction_method = lambda self, Xs : self.predict_with_method(Xs, method_name)
            setattr(self, method_name, prediction_method)
                
        return prediction_method_names
    
    def transform(self, Xs):
        """
        Call transform() method of each pipe in the layer. 
        
        Parameters
        ----------
        Xs: list of [ndarray.shape(n_sample, n_feature) of None]
        
        Returns
        -------
        list of [ndarray.shape(n_sample, n_feature) of None]
            contains values received in Xs argument unless transformed by a pipe in this layer.
        """
        if hasattr(self, model_list) == False:
            raise utils.FitError('tranform attempted before fitting')
        Xs_t = Xs.copy()
        for model, slice_, channel_indices in self.model_list:
            input_ = Xs[slice_] if utils.is_multichannel(model) else Xs[slice_][0]
            if utils.is_multichannel(pipe):
                Xs_t[slice_] = model.transform(input_)
            else:
                Xs_t[channel_indices[0]] = model.transform(input_)           
                
        return Xs_t
                                     
    def predict_with_method(self, Xs, method_name):
        """
        Call the predict() methods in this layer that match the method_name. 
        
        arguments
        ---------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
        method_name: str
            name of the method to use for prediction
        
        returns
        -------
        ndarray.shape(n_sample,s) or list
            If one predict() method is found, returns a single prediction array of length n_samples.
            If more than one predict() method is found, returns a list with either the predictions or 
            None for each input channel.
        """
        if hasattr(self, 'model_list') == False:
            raise utils.FitError('prediction attempted before model fitting')
        
        predictions = [None for X in Xs] 
                                     
        for model, slice_, channel_indices in self.model_list:
            input_ = Xs[slice_] if utils.is_multichannel(model) else Xs[slice_][0]
            if hasattr(model, method_name):
                prediction_method = getattr(self.model, method_name)
                if utils.is_multichannel(model):
                    predictions[slice_] = prediction_method(input_)
                else:
                    predictions[channel_indices[0]] = prediction_method(input_)        
                    
        outputs = [p for p in predictions if p is not None]        
        if len(outputs) == 1:
            # typical pattern: pipeline has converged to a single y
            return outputs[0]
        else:
            # atypical pattern: pipeline has not converged and final layer makes multiple predictions
            return predictions

class MultichannelPipeline(Cloneable, Saveable):
    """
    Machine learning or data processing pipeline that accepts multiple inputs and outputs transormed data or predictions.
    
    Notes:
    ------
    
    Fitting, Predicting, and Transforming
    --------------------------------------
    
        Fitting:
        Call pipeline.fit(Xs_train, y_train) to fit the transformers and predictors in the pipeline to 
        training data.  This method invokes layer.fit_transform() on each layer and then layer.final_fit() 
        on the last layer, and exposes all prediction methods found in the final layer (i.e. predict(), 
        predict_proba(), or decision_function()) so that they can be called directly on the pipeline itself.
            The layer.fit_tranform() method calls fit_transform() - or falls back on fit()/transform() - on each
                transformer in the layer.  The method also automatically wrap predictors in tranform_wrappers to add a 
                fit_transform() method and to provide internal cv training for meta-prediction (default is 
                5-fold KFold for regressor or StratifiedKFold for classifiers; custom cv available by specifying 
                internal_cv in fit_params).
            The layer.final_fit() method fits the final transformer or predictor.  Predictors are automatically wrapped
                to give them transform methods.  A list of all prediction methods identified are returned to 
                MultichannelPipeline.
        
        Predicting:
        Make inferences by calling pipeline.predict().  This call results in NameError if layer.fit() did 
            not detect a predict() method  in the final layer.  Otherwise it invokes transform() on all 
            layers except the final layer and then calls predict() on the final layer.  Returns either a 
            single prediction array.shape(n_samples,) in the typical use case or a list of arrays if more 
            than one prediction() method is present in the final layer's live channels.
        Make inferences by calling pipeline.predict_proba().  This call results in NameError if layer.fit() did 
            not detect a predict_proba() method  in the final layer.  Otherwise it invokes transform() on all 
            layers except the final layer and then calls predict_proba() on the final layer.  Returns either a 
            single prediction array.shape(n_samples, n_classes) in the typical use case or a list of arrays if more 
            than one predict_proba() method is present in the final layer's live channels.
        Make inferences by calling pipeline.decision_function().  This call results in NameError if layer.fit() did 
            not detect a decision_function() method  in the final layer.  Otherwise it invokes transform() on all 
            layers except the final layer and then calls decision_function() on the final layer.  Returns either a 
            single prediction array.shape(n_samples, n_classes) in the typical use case or a list of arrays if more 
            than one decision_function() method is present in the final layer's live channels.
        
        Transforming:
        Transform data by calling pipeline.transform().  This call invokes layer.transform() on each layer, 
            including the last layer.
    
    """
    
    def __init__(self, n_channels=1, internal_cv=5):
        super()._init_params(locals())
        self.layers = []
        
    def get_new_layer(self):
        return Layer(self.n_channels)
            
    def add_layer(self, *pipe_mapping):
        """
        Add a layer of pipes to the pipeline.
        
        Parameters
        ----------
        pipe_mapping: layer, single pipe, iterable, or multiple arguments in format int, pipe, int, pipe etc ...
            If pipe_mapping is a layer, add the layer and return.
            If pipe_mapping is a single argument, the pipe_mapping argument will be atomatically repeated to 
                fill all channels if it is a single channel pipe, or set to receive all channels as inputs 
                into a single pipe if it is a multichannel pipe.
            If pipe_mapping contains multiple arguments, it must be a list of alternating integers / pipes.  The integer
                sets how many continguous channels are mapped to the pipe.  Single channel pipes are automatically 
                repeated for each input channel specified by the int argument, and multichannel pipes are 
                automatically set to receive the number of inputs specified by the int arugment.  Input channels
                are mapped sequentially in the order in which the arguments are entered in the function call.

        returns:
            self (MultiChannelPipeline)
            
        Examples
        --------
        import pipecaster as pc
        clf = pc.MultichanelPipeline(n_channels=6)
        clf.add_layer(3, LogisticRegression(), 3, KNeightborsClassifier())
        clf.add_layer(pc.MultichannelPredictor(SVC()))
        
        """
        if len(pipe_mapping) == 1 and type(pipe_mapping) == Layer:
            if pipe_mapping.n_channels <= self.n_channels:
                self.layers.append(pipe_mapping[0])
                return self
            else:
                raise ValueError('Added layer has more channels than the pipeline')
                
        if len(pipe_mapping) == 1 and type(pipe_mapping) != Layer:
            n_channels = [self.n_channels]
            pipes = [pipe_mapping[0]]
        elif len(pipe_mapping) > 1:
            if len(pipe_mapping) % 2 != 0:
                raise TypeError('even number of arguments required when the number of arguments is > 1') 
            n_channels = pipe_mapping[::2]
            pipes = pipe_mapping[1::2]
            if len(pipes) > self.n_channels:
                raise TypeError('too many arguments: more pipe mappings than pipeline channels')

        new_layer = Layer(self.n_channels)
        first_index = 0
        for n, pipe in zip(n_channels, pipes):
            last_index = first_index + n
            new_layer[first_index:last_index] = pipe
            first_index = last_index
                    
        self.layers.append(new_layer)
        return self
            
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
        Xs = [np.array(X, dtype=float) for X in Xs]
        Xs = [X.reshape(1, -1) if len(X.shape) == 1 else X for X in Xs]
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self.layers[i].transform(Xs)
        return self.layers[-1].predict(Xs)
    
    def predict_proba(self, Xs):
        Xs = [np.array(X, dtype=float) for X in Xs]
        Xs = [X.reshape(1, -1) if len(X.shape) == 1 else X for X in Xs]
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self.layers[i].transform(Xs)
        return self.layers[-1].predict_proba(Xs)
    
    def decision_function(self, Xs):
        Xs = [np.array(X, dtype=float) for X in Xs]
        Xs = [X.reshape(1, -1) if len(X.shape) == 1 else X for X in Xs]
        n_layers = len(self.layers)
        for i in range(n_layers - 1):
            Xs = self.layers[i].transform(Xs)
        return self.layers[-1].decision_function(Xs)
    
    def get_pipe(self, input_index, layer_index):
        return self.layers[layer_index].get_pipe_from_input(input_index)
    
    def get_clone(self):
        new_pipline = Pipeline(self.n_channels)
        new_pipline.layers = [layer.get_clone() for layer in self.layers]
        return new_pipline
    