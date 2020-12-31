import numpy as np
import functools

import pipecaster.utils as utils
import pipecaster.parallel as parallel
from pipecaster.utils import Cloneable, Saveable, FitError
import pipecaster.transform_wrappers as transform_wrappers

__all__ = ['Layer', 'MultichannelPipeline', 'ChannelConcatenator']

        
def get_live_channels(Xs, channel_indices=None):
    channel_indices = range(len(Xs)) if channel_indices is None else channel_indices
    if type(channel_indices) == int:
        live_channels = [] if Xs[channel_indices] is None else [channel_indices]
    else:
        live_channels = [i for i in channel_indices if Xs[i] is not None]   
    return live_channels

def has_live_channels(channel_indices, Xs):
    return True if len(get_live_channels(Xs, channel_indices)) > 0 else False
       
class Layer(Cloneable, Saveable):
    """A list of transformer and/or predictor instances with channel mappings to con
    
    Examples
    --------
        
        
    Notes
    -----
    This class uses reflection to expose the predictor methods found in the last layer, so 
        the method attributes in a MultichannelPipeline instance are not identical to the method attributes of the 
        MultichannelPipeline class.     
    """
    
    state_variables = ['_all_channels', '_mapped_channels', '_estimator_type'] 
    
    def __init__(self, n_channels, pipe_processes=1):
        self._params_to_attributes(Layer.__init__, locals())
        self.pipe_list = []
        self._all_channels = set(range(n_channels))
        self._mapped_channels = set()
    
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
                self.expose_predictor_type(pipe)
        else:
            utils.check_pipe_interface(val)
            self.expose_predictor_type(val)
        
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
    
    def expose_predictor_type(self, pipe):
        if hasattr(pipe, '_estimator_type') == True:
            predictor_type = pipe._estimator_type
            if hasattr(self, '_estimator_type') == False:
                self._estimator_type = predictor_type
            else:
                if self._estimator_type != predictor_type:
                    raise ValueError('All predictors in a layer must have the same type (e.g. classifier or regressor)')
                    
    def get_pipe(self, index, unwrap=True):
        pipe = self.pipe_list[index][0]
        return transform_wrappers.unwrap_predictor(pipe) if unwrap else pipe
    
    def get_model(self, index, unwrap=True):
        model = self.model_list[index][0]
        return transform_wrappers.unwrap_model(model) if unwrap else model
        
    def get_pipe_from_channel(self, channel_index, unwrap=True):
        for pipe, slice_, indices in self.pipe_list:
            if type(slice_) == int and slice_ == channel_index:
                return transform_wrappers.unwrap_predictor(pipe) if unwrap else pipe
            elif channel_index in indices:
                return transform_wrappers.unwrap_predictor(pipe) if unwrap else pipe
        return None
    
    def get_model_from_channel(self, channel_index, unwrap=True):
        for model, slice_, indices in self.model_list:
            if type(slice_) == int and slice_ == channel_index:
                return transform_wrappers.unwrap_model(model) if unwrap else model
            elif channel_index in indices:
                return transform_wrappers.unwrap_model(model) if unwrap else model
        return None
    
    @staticmethod
    def _fit_transform_job(pipe, Xs, y, fit_params, slice_, channel_indices, 
                           transform_method_name, internal_cv, cv_processes):
        
        input_ = Xs[slice_] if utils.is_multichannel(pipe) else Xs[slice_][0]                
        model = utils.get_clone(pipe)

        if hasattr(model, 'fit_transform'):
            try:
                if utils.is_multichannel(model):
                    if y is None:
                        Xs_t = model.fit_transform(input_, **fit_params)
                    else:
                        Xs_t = model.fit_transform(input_, y, **fit_params)
                else:
                    if y is None:
                        Xs_t = [model.fit_transform(input_, **fit_params)]
                    else:
                        Xs_t = [model.fit_transform(input_, y, **fit_params)]
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
                Xs_t = model.transform(input_)
            else:
                Xs_t = [model.transform(input_)]     

        elif utils.is_predictor(model):

            if utils.is_multichannel(model):
                if type(internal_cv) == int and internal_cv < 2:
                    model = transform_wrappers.Multichannel(model, transform_method_name)
                else:
                    model = transform_wrappers.MultichannelCV(model, transform_method_name, 
                                                              internal_cv, cv_processes)
                if y is None:
                    Xs_t = model.fit_transform(input_, **fit_params) 
                else:
                    Xs_t = model.fit_transform(input_, y, **fit_params)
            else:
                if type(internal_cv) == int and internal_cv < 2:
                    model = transform_wrappers.SingleChannel(model, transform_method_name)
                else:
                    model = transform_wrappers.SingleChannelCV(model, transform_method_name, 
                                                               internal_cv, cv_processes)
                if y is None:
                    Xs_t = [model.fit_transform(input_, **fit_params)]
                else:
                    Xs_t = [model.fit_transform(input_, y, **fit_params)] 
                    
        return Xs_t, model, slice_, channel_indices
    
    def fit_transform(self, Xs, y=None, transform_method_name=None, internal_cv=5, cv_processes=1,
                      **fit_params):
        """
        For each pipe in this layer, call fit_transform() if available, fall back on fit() then transform(), 
        or automatically convert predictors into transformers to enable meta-prediction.  
        
        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
        y: targets, default=None
        transform_method_name: string or None, default=None
            Set the name of the method to be used when transforming with a predictor. If None, the method
            will be selected automatically by the precedence defined in the transform_wrapper module.
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
        args_list = []
        live_pipes = []
        for i, (pipe, slice_, channel_indices) in enumerate(self.pipe_list):
            if has_live_channels(channel_indices, Xs):
                args_list.append((pipe, Xs, y, fit_params, slice_, channel_indices, 
                                  transform_method_name, internal_cv, cv_processes))
                live_pipes.append(i)
                
        n_jobs = len(args_list)
        n_processes = 1 if self.pipe_processes is None else self.pipe_processes
        n_processes = n_jobs if (type(n_processes) == int and n_jobs < n_processes) else n_processes
        if n_processes == 'max' or n_processes > 1:
            try:
                shared_mem_objects = [Xs, y, fit_params, internal_cv, cv_processes]
                fit_results = parallel.starmap_jobs(Layer._fit_transform_job, args_list, n_cpus=n_processes, 
                                                    shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'.format(e))
                print('defaulting to single processor')
                n_processes = 1       
        if n_processes is None or n_processes <= 1:
            # print('running a single process with {} jobs'.format(len(args_list)))
            fit_results = [Layer._fit_transform_job(*args) for args in args_list]
    
        self.model_list = [(model, slice_, channel_indices) for _, model, slice_, channel_indices in fit_results]
        Xs_t = [None for X in Xs]
        for pipe_Xs_t, _, slice_, _ in fit_results:
            Xs_t[slice_] = pipe_Xs_t
            
        return Xs_t
    
    def fit_last(self, Xs, y=None, transform_method_name=None, internal_cv=5, cv_processes=1, **fit_params):
        """
        Fit the last layer of a MultiChannelPipeline. Exposes available prediction and transform 
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
        
        self.model_list = []
        prediction_method_names = []
        estimator_types = []
        for pipe, slice_, channel_indices in self.pipe_list:
            if has_live_channels(channel_indices, Xs):
                model = utils.get_clone(pipe)
                input_ = Xs[slice_] if utils.is_multichannel(model) else Xs[slice_][0]                
                if hasattr(model, 'transform') == False:
                    if utils.is_multichannel(model):
                        model = transform_wrappers.Multichannel(model, transform_method_name)
                    else:
                        model = transform_wrappers.SingleChannel(model, transform_method_name)
                        
                if y is None:
                    model.fit(input_, **fit_params)
                else:
                    model.fit(input_, y, **fit_params)
            
                prediction_method_names.extend(utils.get_prediction_method_names(model))
                estimator_type = utils.detect_predictor_type(model)
                if estimator_type is not None:
                    estimator_types.append(estimator_type)
                    
                self.model_list.append((model, slice_, channel_indices))
                
        prediction_method_names = set(prediction_method_names)
        # expose predictor interface
        for method_name in prediction_method_names:
            prediction_method = functools.partial(self.predict_with_method, method_name=method_name)
            setattr(self, method_name, prediction_method)
            
        estimator_types = set(estimator_types)
        if len(estimator_types) > 1:
            raise TypeError('more than 1 predictor type found')
        elif len(estimator_types) == 1:
            self._estimator_type = list(estimator_types)[0]
    
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
        if hasattr(self, 'model_list') == False:
            raise utils.FitError('tranform attempted before fitting')
        Xs_t = Xs.copy()
        for model, slice_, channel_indices in self.model_list:
            input_ = Xs[slice_] if utils.is_multichannel(model) else Xs[slice_][0]
            if utils.is_multichannel(model):
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
                prediction_method = getattr(model, method_name)
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
        
    def get_clone(self):
        clone = super().get_clone()
        clone.pipe_list = [(utils.get_clone(p), s, i.copy()) for p, s, i in self.pipe_list]
        if hasattr(self, 'model_list'):
            clone.model_list = [(utils.get_clone(p), s, i.copy()) for p, s, i in self.model_list]
        return clone

class MultichannelPipeline(Cloneable, Saveable):
    """
    Machine learning or data processing pipeline that accepts multiple inputs and returns transormed data 
    or predictions.
    
    Parameters
    ----------
    n_channels: int, default=1
        The number of separate i/o channels throughout the pipeline (except last output, which is a single
        matrix when the pipeline outputs predictions to a single channel).  The number of live channels
        is reduced by concatenation and selection operations, but the channel depth remains constant internally 
        with dead channels indicated by their None value transform outputs.
    transform_method_name: string, default=None 
        **Predictors not located in the last layer of the pipeline must have a transform method to output
        features for meta-prediction or further processing. transform_method_name sets a method 
        to use for output thoughout the pipeline. If None, pipecaster will automatically assign a method 
        using the precedence defined in the transform_wrapper module.  
    internal_cv: None, int, or scikit-learn cv splitter instance, default=5
        **Set a global internal cross validation training method used for preditors (except predictors last 
        layer of pipeline).  If 1, internal cv training is inactivated.  If int > 1 then KFold() 
        for regressors and StratifiedKFold() for classifiers. When providing a cv splitter instance,
        set the random state to an integer if you want to make the same splits throughout the pipeline 
        and ensure that sub-predictors don't make inferences from their training samples during meta-predictor
        training.
    cv_processes: 'max' or int, default=1
        **Number of parallel processes to use for internal cv training.  If 'max', all available CPUs will be 
        available for use.
        
    **internal_cv parameters: transform_method_name, internal_cv, and cv_processes
        These parameters are ignored for predictors that have already have a tranform method when added 
        to the pipeline.  Local cv parameters can be set for each predictor by wrapping them with 
        transform_wrappers.  SingleChannelCV or transform_wrapper.MultichannelCV before addition to 
        the pipeline.  When manually wrapping predictors, parameters should be uniform throughout the pipeline
        if you want to ensure that no meta-predcitors are trained on their own training examples.
    
    
    Fitting, Predicting, and Transforming
    --------------------------------------
    
        Fitting: pipeline.fit(Xs_train, y_train)
        
        Call pipeline.fit(Xs_train, y_train) to fit the transformers and predictors in the pipeline to 
        training data.  This method invokes layer.fit_transform() on each layer and then layer.final_fit() 
        on the last layer, and exposes all prediction methods found in the final layer (i.e. predict, 
        predict_proba, decision_function, or predict_log_proba) so that they can be called directly on the pipeline itself.
            The layer.fit_tranform() method calls fit_transform() - or falls back on fit()/transform() - on each
                transformer in the layer.  The method also automatically wraps predictors in tranform_wrappers to add a 
                fit_transform() method and to provide internal cv training for meta-prediction (default is 
                5-fold KFold for regressor or StratifiedKFold for classifiers; custom cv available by specifying 
                internal_cv in fit_params).
            The layer.final_fit() method fits the final transformer or predictor.  Predictors are automatically wrapped
                to give them transform methods.  A list of all prediction methods identified are returned to 
                MultichannelPipeline.
        To override or inactivate default 5-fold internal cv training, pass an 'internal_cv' argument to fit() 
            or fit_transform():
            pipeline.fit(Xs, y, internal_cv=my_cv)
            pipeline.fit_transform(Xs, y, internal_cv=my_cv)
            If my_cv == 1 then internal cv training is inactivated.
        
        Predicting: pipeline.predict(), pipeline.predict_proba(), 
                    pipeline.predict_log_proba(), pipeline.decision_function()
        
        Calls to prediction methods results in NameError if layer.fit_last() did not detect a the method 
            in the final layer.  Otherwise it invokes transform() on all 
            layers except the final layer and calls the prediction method on the final layer.  Returns either a 
            single prediction array.shape(n_samples,) in the typical use case or a list of arrays if more 
            than one prediction() method is present in the final layer's live channels.
        
        Transforming:
        Transform data by calling pipeline.transform().  This call invokes layer.transform() on each layer, 
            including the last layer.
            
        CvTransforming:
        Calling fit_transform() on a pipeline results in outputs where the inferences are made on training
            samples.  In specific cases were the outputs of a pipeline are to be used in meta-prediction
            external to the pipeline, this can be avoided by calling cv_fit_transform() on 
            the pipeline instead of fit_transform().
            
    Notes:
    ------
    This class uses reflection to expose the predictor methods found in the last layer, so 
        the method attributes in a MultichannelPipeline instance are not identical to the method 
        attributes of the MultichannelPipeline class.
    Internally (i.e. on calls to layer.fit_tranform()), fit_transform() is a signal that predictors 
        need to be wrapped with transform_wrappers with cross validation training.  However, calling
        fit_transform on a MultichannelPipeline will not wrap
    """
    
    def __init__(self, n_channels=1, transform_method_name=None, internal_cv=5, cv_processes=1):
        self._params_to_attributes(MultichannelPipeline.__init__, locals())
        self.layers = []
        
    def get_new_layer(self, pipe_processes=1):
        return Layer(self.n_channels, pipe_processes)
    
    def get_next_layer(self, pipe_processes=1):
        layer = self.get_new_layer(pipe_processes)
        self.layers.append(layer)
        return layer
            
    def add_layer(self, *pipe_mapping, pipe_processes=1):
        """
        Add a layer of pipes to the pipeline.
        
        Parameters
        ----------
        pipe_mapping: layer, single pipe, or multiple arguments in format int, pipe, int, pipe etc ...
            If pipe_mapping is a layer, add the layer and return.
            If pipe_mapping is a single argument, the argument will be atomatically repeated to 
                fill all channels if it is a single channel pipe, or set to receive all channels as inputs 
                into a single pipe if it is a multichannel pipe.
            If pipe_mapping contains multiple arguments, it must be a list of alternating integers / pipes.  The integer
                sets how many continguous channels are mapped to the pipe.  Single channel pipes are automatically 
                repeated for each input channel specified by the int argument, and multichannel pipes are 
                automatically set to receive the number of inputs specified by the int arugment.  Input channels
                are mapped sequentially in the order in which the arguments are entered in the function call.
        pipe_processes: int or 'max'
            The number of parallel processes to allow for the layer.

        returns:
            self (MultiChannelPipeline)
            
        Examples
        --------
        import pipecaster as pc
        clf = pc.MultichanelPipeline(n_channels=6)
        clf.add_layer(3, LogisticRegression(), 3, KNeightborsClassifier())
        clf.add_layer(pc.MultichannelPredictor(SVC()))
        
        Note:
        -----
        There is no stateless sklearn-like clone implemented because MultiChannelPipeline arguments
        are not sufficient to reproduce the pipeline.  Use MultiChannelPipeline.get_clone() to
        get a stateful clone or rebuild pipeline from scratch to get a stateless clone(). 
        
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

        new_layer = Layer(self.n_channels, pipe_processes)
        first_index = 0
        for n, pipe in zip(n_channels, pipes):
            last_index = first_index + n
            new_layer[first_index:last_index] = pipe
            first_index = last_index
                    
        self.layers.append(new_layer)
        return self
    
    def set_pipe_processes(self, pipe_processes):
        """
        Set the number of parallel processes used during fitting of the model.
        
        Paramteters
        -----------
        pipe_processes: int, 'max', or iterable of int/'max'
            int: Every layer is set to use the same number of parallel processes.
            'max': Every layer is set to use up to the maximum number of available processes.
            iterable: The number of processes is set for each layer according to the values in the list
        """
        if isinstance(pipe_processes, (tuple, list, np.array)):
            if len(pipe_processes) != len(self.layers):
                raise ValueError('Number of pipe process values do not match the number of layers.')
            for n, layer in zip(pipe_processes, self.layers):
                layer.pipe_processes = n
        else:
            for layer in self.layers:
                layer.pipe_processes = pipe_processes
                
    def fit(self, Xs, y=None, **fit_params):
        if hasattr(self.layers[-1], '_estimator_type'):
            self._estimator_type = self.layers[-1]._estimator_type
            # encode labels as integers
            if self._estimator_type == 'classifier':
                if y is not None:
                    self.classes_, y = np.unique(y, return_inverse=True)
            
        for layer in self.layers[:-1]:
            Xs = layer.fit_transform(Xs, y, self.transform_method_name, self.internal_cv, 
                                     self.cv_processes, **fit_params)
        # fit the last layer without transforming:
        self.layers[-1].fit_last(Xs, y, self.transform_method_name, self.internal_cv, 
                                 self.cv_processes, **fit_params)
        
        # expose the prediction methods found in the last layer
        for method_name in utils.get_prediction_method_names(self.layers[-1]):
            prediction_method = functools.partial(self.predict_with_method, method_name=method_name)
            setattr(self, method_name, prediction_method)
    
    def transform(self, Xs, y=None):
        for layer in self.layers:
            Xs = layer.transform(Xs)
        return Xs
    
    def fit_transform(self, Xs, y, **fit_params):
        for layer in self.layers[:-1]:
            Xs = layer.fit_transform(Xs, y, self.transform_method_name, self.internal_cv, 
                                     self.cv_processes, **fit_params)
        self.layers[-1].fit_last(Xs, y, self.transform_method_name, self.internal_cv, 
                                 self.cv_processes, **fit_params)
        return self.layers[-1].transform(Xs)
    
    def cv_fit_transform(self, Xs, y, **fit_params):
        fit_params['internal_cv'] = internal_cv
        for layer in self.layers[:-1]:
            Xs = layer.fit_transform(Xs, y, self.transform_method_name, self.internal_cv, 
                                     self.cv_processes, **fit_params)
        return self.layers[-1].fit_transform(Xs, y, self.transform_method_name, self.internal_cv, 
                                     self.cv_processes, **fit_params)
    
    def predict_with_method(self, Xs, method_name):
        Xs = [np.array(X, dtype=float) for X in Xs]
        for layer in self.layers[:-1]:
            Xs = layer.transform(Xs)
        prediction_method = getattr(self.layers[-1], method_name)
        predictions = prediction_method(Xs)
        # decode class names
        if utils.is_classifier(self) and method_name == 'predict':
            prediction = self.classes_[predictions]
        return predictions
    
    def get_pipe(self, layer_index, pipe_index, unwrap=True):
        return self.layers[layer_index].get_pipe(pipe_index, unwrap)
    
    def get_model(self, layer_index, model_index, unwrap=True):
        return self.layers[layer_index].get_model(model_index, unwrap)
    
    def get_pipe_from_channel(self, layer_index, channel_index, unwrap=True):
        return self.layers[layer_index].get_pipe_from_channel(channel_index, unwrap)
    
    def get_model_from_channel(self, layer_index, channel_index, unwrap=True):
        return self.layers[layer_index].get_model_from_channel(channel_index, unwrap)
    
    def get_clone(self):
        clone = super().get_clone()
        clone.layers = [layer.get_clone() for layer in self.layers]
        return clone
    
class ChannelConcatenator(Cloneable, Saveable):
    
    def fit(self, Xs, y=None, **fit_params):
        pass
    
    def transform(self, Xs):
        live_Xs = [X for X in Xs if X is not None]
        Xs_t = [None for X in Xs]
        Xs_t[0] = np.concatenate(live_Xs, axis=1) if len(live_Xs) > 0 else None
        return Xs_t
    
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)
    