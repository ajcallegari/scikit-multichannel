import numpy as np
import pandas as pd
import functools

import pipecaster.utils as utils
import pipecaster.parallel as parallel
from pipecaster.utils import Cloneable, Saveable, FitError
import pipecaster.transform_wrappers as transform_wrappers

__all__ = ['Layer', 'MultichannelPipeline', 'ChannelConcatenator']


def get_live_channels(Xs, channel_indices=None):
    """
    Determine the indices of the matrix list Xs have values other than None.

    Parameters
    ----------
    Xs: list of [nd.array(n_samples, n_features) or None]
    channel_indices: int, list/array of ints, or None, default=None
        If int or list/array of int: Indices of the matrices in Xs to query.
        If None: Query all input matrices in Xs

    Returns
    -------
    List of indices into Xs where the value is not None.
    """
    if channel_indices is None:
        channel_indices = range(len(Xs))
    if type(channel_indices) == int:
        if Xs[channel_indices] is None:
            live_channels = []
        else:
            live_channels = [channel_indices]
    else:
        live_channels = [i for i in channel_indices if Xs[i] is not None]
    return live_channels


def has_live_channels(Xs, channel_indices=None):
    """
    Determine if a list of matrices contains one or more live (not None value)
    matrices.

    Parameters
    ----------
    Xs: list of nd.array(n_samples, n_features)
    channel_indices: int, list/array of ints, or None, default=None
        If int or list/array of int: Indices of the matrices in Xs to query.
        If None: Query all input matrices in Xs

    Returns
    -------
    True if a value other than None is found, otherwise False.
    """
    return True if len(get_live_channels(Xs, channel_indices)) > 0 else False


class Layer(Cloneable, Saveable):
    """
    A list of pipe instances with io channel mappings.

    Parameters
    ----------
    n_channels: int
        The number of input matrices accepted by the layer, must be >=1.
    pipe_processes: 'max' or int, default=1
        If 1: Run all fitting computations with a single process.
        If max: Run fitting computations in parallel, using all available CPUs.
        If >1: Running fitting computationsin parallel with up to
            pipe_processes number of CPUs

    Examples
    --------
    # create a layer with 10 inputs and outputs, add a single scaler for each
    # (broadcasting):

    layer = Layer(n_channels=10)[StandardScaler()]
    out_Xs = layer.fit_transform(in_Xs)

    # Create a layer with 10 inputs that are mapped to a multichannel
    # classifier that will concatenate the inputs and classify them using a
    # support vector machine. Output predicions to a single matrix.

    layer = Layer(n_channels=10)[MultichannelPredictor(SVC())]
    layer.fit(Xs_train, y_train)
    predictions = layer.predict(Xs_test)

    # Slicing example.  Create a layer that selects the 100 best features in
    # first 5 channels and 200 best features in the last 5 channels.

    layer = Layer(n_channels=10)
    layer[0:5] = SelectKBest(k=100)
    layer[5:10] = SelectKBest(k=200)
    out_Xs = layer.fit_transform(in_Xs)

    Note
    ----
    This class may use reflection to expose the prediction methods found in the
    layer, so method attributes in a Layer instance may not be identical to the
    method attributes of the Layer class.
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
                raise ValueError('Invalid slice step; must be exactly 1 \
                                 (Pipes may only accept contiguous inputs)')
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
                raise ValueError('Two pipes are mapped to channel {}.  \
                                 Max allowed is 1'.format(i))

        if is_listlike is False:
            n = len(channel_indices)
            if utils.is_multichannel(val) is True:
                self.pipe_list.append((val, slice_, channel_indices))
            else:
                for i in channel_indices:
                    self.pipe_list.append((val, slice(i, i+1, 1), [i]))
        elif is_listlike is True:
            n = len(val)
            if n != len(channel_indices):
                raise ValueError('List of pipe objects does not match slice \
                                 dimension during assignment')
            else:
                for pipe, i in zip(val, channel_indices):
                    self.pipe_list.append((val, slice(i, i+1, 1), [i]))

        self._mapped_channels = self._mapped_channels.union(channel_indices)

        return self

    def expose_predictor_type(self, pipe):
        if hasattr(pipe, '_estimator_type') is True:
            predictor_type = pipe._estimator_type
            if hasattr(self, '_estimator_type') is False:
                self._estimator_type = predictor_type
            else:
                if self._estimator_type != predictor_type:
                    raise ValueError('All predictors in a layer must have the \
                                     same type (e.g. classifier or regressor)')

    def get_pipe(self, index, unwrap=True):
        pipe = self.pipe_list[index][0]
        return transform_wrappers.unwrap_predictor(pipe) if unwrap else pipe

    def get_model(self, index, unwrap=True):
        model = self.model_list[index][0]
        return transform_wrappers.unwrap_model(model) if unwrap else model

    def get_pipe_from_channel(self, channel_index, unwrap=True):
        for pipe, slice_, indices in self.pipe_list:
            if type(slice_) == int and slice_ == channel_index:
                return (transform_wrappers.unwrap_predictor(pipe) if unwrap
                        else pipe)
            elif channel_index in indices:
                return (transform_wrappers.unwrap_predictor(pipe) if unwrap
                        else pipe)
        return None

    def get_model_from_channel(self, channel_index, unwrap=True):
        for model, slice_, indices in self.model_list:
            if type(slice_) == int and slice_ == channel_index:
                return (transform_wrappers.unwrap_model(model) if unwrap
                        else model)
            elif channel_index in indices:
                return (transform_wrappers.unwrap_model(model) if unwrap
                        else model)
        return None

    @staticmethod
    def _fit_transform_job(pipe, Xs, y, fit_params, slice_, channel_indices,
                           transform_method_name, internal_cv, cv_processes):

        input_ = Xs[slice_] if utils.is_multichannel(pipe) else Xs[slice_][0]
        model = utils.get_clone(pipe)

        if hasattr(model, 'fit_transform'):
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

        elif hasattr(model, 'fit') and hasattr(pipe, 'transform'):
            if y is not None:
                model.fit(input_, **fit_params)
            else:
                model.fit(input_, y, **fit_params)
            if utils.is_multichannel(model):
                Xs_t = model.transform(input_)
            else:
                Xs_t = [model.transform(input_)]

        elif utils.is_predictor(model):

            if utils.is_multichannel(model):
                if type(internal_cv) == int and internal_cv < 2:
                    model = transform_wrappers.Multichannel(
                                                        model,
                                                        transform_method_name)
                else:
                    model = transform_wrappers.MultichannelCV(
                                                model, transform_method_name,
                                                internal_cv, cv_processes)
                if y is None:
                    Xs_t = model.fit_transform(input_, **fit_params)
                else:
                    Xs_t = model.fit_transform(input_, y, **fit_params)
            else:
                if type(internal_cv) == int and internal_cv < 2:
                    model = transform_wrappers.SingleChannel(
                                                        model,
                                                        transform_method_name)
                else:
                    model = transform_wrappers.SingleChannelCV(
                                                model, transform_method_name,
                                                internal_cv, cv_processes)
                if y is None:
                    Xs_t = [model.fit_transform(input_, **fit_params)]
                else:
                    Xs_t = [model.fit_transform(input_, y, **fit_params)]

        return Xs_t, model, slice_, channel_indices

    def fit_transform(self, Xs, y=None, transform_method_name=None,
                      internal_cv=5, cv_processes=1, **fit_params):
        """
        Clone each pipe in this layer, call fit_transform() if available,
        or fall back on fit() then transform().  Pipes that have a prediction
        method but lack a transform method are given transform functionality
        and internal cross validation training functionality by wrapping them
        with classes defined in the transform_wrappers module.  Leaves the
        original pipe_list untouched for reference and creates a model_list
        attribute with fit versions of the pipes.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.
        y: list/array of length n_samples, default=None
            Targets for supervised ML.

        transform_method_name, internal_cv, and cv_processes override the
        default parameters for adding transform functionality to the
        predictors in this layer:

        transform_method_name: str or None, default=None
            Set the name of the prediction method used when transform or
            fit_transform are called. If None, the method will be selected
            automatically by the precedence defined in the transform_wrapper
            module.
        internal_cv: None, int, or callable, default=5
            Set the internal cv training method for predictors.
            If None or 5: Use 5 splits with the default split generator
            if 1: Inactivate internal cv traning.
            if int > 1: Use StratifiedKfold(n_splits=internal_cv) for
                classifiers or Kfold(n_splits=internal_cv) for regressors.
            If callable: Assumes scikit-learn interface like Kfold
        cv_processes: 'max' or int, default=1
            If 1: run all internal cv splits on a single CPU
            If 'max': Run internal splits in parallel, using up
                to all available CPUs.
            If int > 1: Run internal cv splits in parallel, using up to
                cv_processes number of CPUs.
        fit_params: dict, default={}
            Auxiliary parameters to be sent to the fit_transform or fit methods
            of the pipes. Pipe-specific parameters not supported yet, but
            there will probably be an index-based reference system soon.

        Returns
        -------
        Xs_t: list of [ndarray.shape(n_samples, n_features) or None]
            Transformed matrices or passthrough from inputs.  A None value
            indicates that the transformation inactivated (selected against) a
            channel or the channel had been inactivated in a prior layer.
        """
        args_list = []
        live_pipes = []
        for i, (pipe, slice_, channel_indices) in enumerate(self.pipe_list):
            if has_live_channels(Xs, channel_indices):
                args_list.append((pipe, Xs, y, fit_params, slice_,
                                  channel_indices, transform_method_name,
                                  internal_cv, cv_processes))
                live_pipes.append(i)

        n_jobs = len(args_list)
        n_processes = 1 if self.pipe_processes is None else self.pipe_processes
        n_processes = (n_jobs
                       if (type(n_processes) == int and n_jobs < n_processes)
                       else n_processes)
        if n_processes == 'max' or n_processes > 1:
            try:
                shared_mem_objects = [Xs, y, fit_params, internal_cv,
                                      cv_processes]
                fit_results = parallel.starmap_jobs(
                                        Layer._fit_transform_job,
                                        args_list,
                                        n_cpus=n_processes,
                                        shared_mem_objects=shared_mem_objects)
            except Exception as e:
                print('parallel processing request failed with message {}'
                      .format(e))
                print('defaulting to single processor')
                n_processes = 1
        if type(n_processes) == int and n_processes <= 1:
            fit_results = [Layer._fit_transform_job(*args)
                           for args in args_list]

        self.model_list = [(model, slice_, channel_indices)
                           for _, model, slice_, channel_indices
                           in fit_results]
        Xs_t = Xs.copy()
        for pipe_Xs_t, _, slice_, _ in fit_results:
            Xs_t[slice_] = pipe_Xs_t

        self.output_mask = [True if X_t is not None else False for X_t in Xs_t]

        return Xs_t

    def fit_last(self, Xs, y=None, transform_method_name=None, **fit_params):
        """
        Method for fitting the last layer of a MultiChannelPipeline. Clones
        each pipe, fits them to the training data, and wraps all predictors
        to add transform functionality to the pipeline.  Exposes available
        prediction and transform mothods as attributes of the layer instance.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.
        y: list/array of length n_samples, default=None
            Targets for supervised ML.
        transform_method_name: str or None, default=None
            Set the name of the prediction method used when transform or
            fit_transform are called. If None, the method will be selected
            automatically by the precedence defined in the transform_wrapper
            module.
        fit_params: dict, default={}
            Auxiliary parameters to be sent to the fit_transform or fit methods
            of the pipes. Pipe-specific parameters not supported yet, but
            there will probably be an index-based reference system soon.
        """
        self.model_list = []
        prediction_method_names = []
        estimator_types = []
        for pipe, slice_, channel_indices in self.pipe_list:
            if has_live_channels(Xs, channel_indices):
                model = utils.get_clone(pipe)
                input_ = (Xs[slice_] if utils.is_multichannel(model)
                          else Xs[slice_][0])
                if hasattr(model, 'transform') is False:
                    if utils.is_multichannel(model):
                        model = transform_wrappers.Multichannel(
                                                model, transform_method_name)
                    else:
                        model = transform_wrappers.SingleChannel(
                                                model, transform_method_name)

                if y is None:
                    model.fit(input_, **fit_params)
                else:
                    model.fit(input_, y, **fit_params)

                prediction_method_names.extend(
                                    utils.get_prediction_method_names(model))
                estimator_type = utils.detect_predictor_type(model)
                if estimator_type is not None:
                    estimator_types.append(estimator_type)

                self.model_list.append((model, slice_, channel_indices))

        prediction_method_names = set(prediction_method_names)
        # expose predictor interface
        for method_name in prediction_method_names:
            prediction_method = functools.partial(self.predict_with_method,
                                                  method_name=method_name)
            setattr(self, method_name, prediction_method)

        estimator_types = set(estimator_types)
        if len(estimator_types) > 1:
            raise TypeError('more than 1 predictor type found')
        elif len(estimator_types) == 1:
            self._estimator_type = list(estimator_types)[0]

        self.output_mask = [True if X_t is not None else False
                            for X_t in self.transform(Xs)]

    def transform(self, Xs):
        """
        Call transform() method of each pipe in the layer and return
        multichannel transformed output.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.

        Returns
        -------
        list of [ndarray.shape(n_sample, n_feature) of None]
            Returns transformed matrices for mapped channels and passthrough
            matrices for unmapped values.  None values are returned to mark
            dead channels.
        """
        if hasattr(self, 'model_list') is False:
            raise utils.FitError('transform attempted before fitting')
        Xs_t = Xs.copy()
        for model, slice_, channel_indices in self.model_list:
            input_ = (Xs[slice_] if utils.is_multichannel(model)
                      else Xs[slice_][0])
            if utils.is_multichannel(model):
                Xs_t[slice_] = model.transform(input_)
            else:
                Xs_t[channel_indices[0]] = model.transform(input_)

        return Xs_t

    def predict_with_method(self, Xs, method_name):
        """
        Call the predict() methods in this layer that match the method_name.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.
        method_name: str
            Name of method to use for prediction.

        Returns
        -------
        ndarray.shape(n_samples,) or list of ndarrays
            If one matching prediction method is found in the layer, returns a
            single prediction array of length n_samples (typical use). If more
            than one prediction methods is found, returns a list with either
            the predictions or None for each input channel.
        """
        if hasattr(self, 'model_list') is False:
            raise utils.FitError('prediction attempted before model fitting')

        predictions = [None for X in Xs]

        for model, slice_, channel_indices in self.model_list:
            input_ = (Xs[slice_] if utils.is_multichannel(model)
                      else Xs[slice_][0])
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
            # atypical pattern: pipeline has not converged and final layer
            # makes multiple predictions
            return predictions

    def get_clone(self):
        """
        Return a new instance of this layer using Cloneable to copy parameters
        and state variables, and utils.get_clone() to copy pipes and models.
        """
        clone = super().get_clone()
        clone.pipe_list = [(utils.get_clone(p), s, i.copy())
                           for p, s, i in self.pipe_list]
        if hasattr(self, 'model_list'):
            clone.model_list = [(utils.get_clone(p), s, i.copy())
                                for p, s, i in self.model_list]
        return clone


class MultichannelPipeline(Cloneable, Saveable):
    """
    Machine learning or data processing pipeline that takes multiple inputs
    and outputs transormed data or predictions.

    Parameters
    ----------
    n_channels: int, default=1
        The number of separate i/o channels throughout the pipeline (except
        last output, which is a single matrix when the pipeline outputs
        predictions to a single channel).  The number of live channels is
        reduced by concatenation and selection operations, but the channel
        depth remains constant internally with dead channels indicated by None
        values.

    The parameters below are global internal cross validation parameters
    applied only when an internal predictor lacks a transformer interface
    (which occurs during model stacking).  Internal cross validation prevents
    predictors from make predictions on their training data during
    meta-predictor training.  These global parameters can be overriden locally
    by wrapping pipes with classes found in the transform_wrappers module
    before adding them to the pipeline, or by making custom predictors with
    their own transform and fit_transform methods.  When overriding global
    internal cv parameters, the user must ensure identical splits throughout
    the pipeline to prevent base predictors from make predictions on their
    training data.

    transform_method_name: str, default=None
        Prediction method name to use when transform is called on a predictor
        that lacks a transform method.  This occurs whenever a predictor is not
        located in the last layer of the pipeline.  If None, pipecaster will
        automatically assign a method using the precedence defined in the
        transform_wrapper module.
    internal_cv: int, scikit-learn cv splitter instance, or None, default=5
        Set a global internal cross validation training method to be used for
        internal predictors that lack a transformer interface (typicaally
        all predictors except those located in the last layer of pipeline).
        If 1: Internal cv training is inactivated.
        If int > 1: StratifiedKFold(n_splits=internal_cv) for classifiers and
            KFold(n_splits=internal_cv) for regressors.
        If None: The default value of 5 is used.
        If callable: Assumes scikit-learn interface like KFold.
    cv_processes: int or 'max', default=1
        If 1: Run all split computations in a single process.
        If 'max': Run each split in a different process, using all available
            CPUs
        If int > 1: Run each split in a different process, using up to
            n_processes number of CPUs

    Notes:
    ------
    *There is no stateless scikit-learn-like clone implemented because
        MultiChannelPipeline arguments are not sufficient to reproduce the
        pipeline.  Use MultiChannelPipeline.get_clone() to get a stateful clone
        or rebuild pipeline from scratch to get a stateless clone().
    * This class uses reflection to expose the predictor methods found in the
        last pipeline layer, so the method attributes in a MultichannelPipeline
        instance are not usually identical to the method attributes of the
        MultichannelPipeline class.
    * Fit failures are currently not handled / allowed.
    * Multi-output prediction not yet supported.
    * Groups for internal cv not yet supported.
    * Sample weights for internal cv performance metrics not yet supported.
    * Caching of intermediate results not yet implemented.
    * fit_params targeted to specific pipes not yet implemented.
    * GPUs not yet supported.
    """

    state_variables = ['classes_']

    def __init__(self, n_channels=1, transform_method_name=None, internal_cv=5,
                 cv_processes=1):
        self._params_to_attributes(MultichannelPipeline.__init__, locals())
        self.layers = []

    def add_layer(self, *pipe_mapping, pipe_processes=1):
        """
        Add a layer of pipes to the pipeline.

        Parameters
        ----------
        pipe_mapping: layer, single pipe, or multiple arguments in format int,
        pipe, int, pipe etc ...
            If layer: add the layer and return.
            If a single argument: the argument will be atomatically repeated to
                fill all channels if it is a single channel pipe, or set to
                receive all channels as inputs into a single pipe if it is a
                multichannel pipe.
            If multiple arguments: read as a list of alternating
                int, pipes, int, pipe...  The int sets how many continguous
                channels are mapped to the pipe.  Single channel pipes are
                automatically repeated for each input channel specified by the
                int argument, and multichannel pipes are automatically set to
                receive the number of inputs specified by the int arugment.
                Input channels are mapped sequentially in the order in which
                the arguments are entered in the function call.
        pipe_processes: int or 'max', default=1
            The number of parallel processes to allow for the layer.

        returns:
            self

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
                raise ValueError('Added layer has more channels than the \
                                 pipeline')

        if len(pipe_mapping) == 1 and type(pipe_mapping) != Layer:
            n_channels = [self.n_channels]
            pipes = [pipe_mapping[0]]
        elif len(pipe_mapping) > 1:
            if len(pipe_mapping) % 2 != 0:
                raise TypeError('even number of arguments required when the \
                                number of arguments is > 1')
            n_channels = pipe_mapping[::2]
            pipes = pipe_mapping[1::2]
            if len(pipes) > self.n_channels:
                raise TypeError('too many arguments: more pipe mappings than \
                                pipeline channels')

        new_layer = Layer(self.n_channels, pipe_processes)
        first_index = 0
        for n, pipe in zip(n_channels, pipes):
            last_index = first_index + n
            new_layer[first_index:last_index] = pipe
            first_index = last_index

        if hasattr(new_layer, '_estimator_type'):
            self._estimator_type = new_layer._estimator_type
        self.layers.append(new_layer)
        return self

    def set_pipe_processes(self, pipe_processes):
        """
        Set the number of parallel processes used during fitting of the model.

        Parameters
        ----------
        pipe_processes: int, 'max', or iterable of int/'max'
            If int: Every layer is set to use the same number of parallel
                processes.
            If 'max': Every layer is set to use up to the maximum number of
                available processes.
            If iterable: The number of processes is set for each layer
                according to the values in the list.
        """
        if isinstance(pipe_processes, (tuple, list, np.array)):
            if len(pipe_processes) != len(self.layers):
                raise ValueError('Number of pipe process values do not match \
                                 the number of layers.')
            for n, layer in zip(pipe_processes, self.layers):
                layer.pipe_processes = n
        else:
            for layer in self.layers:
                layer.pipe_processes = pipe_processes

    def fit(self, Xs, y=None, **fit_params):
        """
        Clone and fit all of the pipes in the pipeline.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.
        y: list/array of length n_samples, default=None
            Targets for supervised ML.
        fit_params: dict, defualt=None
            Auxiliary parameters to pass to the fit method of the predictor.

        Notes
        -----
        All pipes are cloned before fitting so that the original pipeline
        architecture is maintained for reference until the final model is
        exported (model exporting not yet implemented).

        Calling pipeline.fit(Xs_train, y_train) invokes layer.fit_transform()
        on each layer and then layer.final_fit() on the last layer.  The call
        also exposes all prediction methods found in the final layer (i.e.
        predict, predict_proba, decision_function, or predict_log_proba) so
        that they can be called directly on the pipeline itself.

        Calls to layer.fit_transform() in turn call the fit_transform() method
        of each cloned pipe in the layer, falling back on fit() then
        transform() when fit_transform() is not found.  Layer.fit_transform()
        also automatically wraps predictors in transform_wrappers to add a
        transformer interface and internal cv training.

        Calls to layer.fit_last() clones and fit the models in the last layer
        of the pipeline and wraps all models that lack transform methods to add
        transform() and fit_transform().  The current implementation also calls
        transform() on the last layer to map live outputs which so that
        they can be visualized graphically.
        """
        if hasattr(self.layers[-1], '_estimator_type'):
            self._estimator_type = self.layers[-1]._estimator_type
            # encode labels as integers
            if self._estimator_type == 'classifier':
                if y is not None:
                    self.classes_, y = np.unique(y, return_inverse=True)

        for layer in self.layers[:-1]:
            Xs = layer.fit_transform(Xs, y, self.transform_method_name,
                                     self.internal_cv, self.cv_processes,
                                     **fit_params)
        # fit the last layer without transforming:
        self.layers[-1].fit_last(Xs, y, self.transform_method_name,
                                 **fit_params)

        # expose the prediction methods found in the last layer
        for method_name in utils.get_prediction_method_names(self.layers[-1]):
            prediction_method = functools.partial(self.predict_with_method,
                                                  method_name=method_name)
            setattr(self, method_name, prediction_method)
        return self

    def transform(self, Xs):
        """
        Transform data by invoking layer.transform() on each layer.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.
        """
        for layer in self.layers:
            Xs = layer.transform(Xs)
        return Xs

    def fit_transform(self, Xs, y, **fit_params):
        """
        Invoke fit_transform() on each pipeline layer, and fit_last() and
        transform() on the last layer.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.
        y: list/array of length n_samples, default=None
            Targets for supervised ML.
        fit_params: dict, defualt=None
            Auxiliary parameters to pass to the fit method of the predictor.
        """
        for layer in self.layers[:-1]:
            Xs = layer.fit_transform(Xs, y, self.transform_method_name,
                                     self.internal_cv, self.cv_processes,
                                     **fit_params)
        self.layers[-1].fit_last(Xs, y, self.transform_method_name,
                                 **fit_params)
        return self.layers[-1].transform(Xs)

    def cv_fit_transform(self, Xs, y, internal_cv=5, **fit_params):
        """
        Generate pipeline outputs from the training data without allowing
        individual predictors to generate outputs from their training data.
        Use if pipeline outputs are to be used in meta-prediction external to
        the pipeline.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.
        y: list/array of length n_samples, default=None
            Targets for supervised ML.
        internal_cv: None, int, or callable, default=5
            Set the internal cv training method for predictors.
            If 1: Internal cv training is inactivated.
            If int > 1: StratifiedKFold(n_splits=internal_cv) for classifiers
                and KFold(n_splits=internal_cv) for regressors.
            If None: The default value of 5 is used.
            If callable: Assumes scikit-learn interface like KFold.
        fit_params: dict, defualt=None
            Auxiliary parameters to pass to the fit method of the predictor.
        """
        fit_params['internal_cv'] = internal_cv
        for layer in self.layers[:-1]:
            Xs = layer.fit_transform(Xs, y, self.transform_method_name,
                                     self.internal_cv,
                                     self.cv_processes, **fit_params)
        return self.layers[-1].fit_transform(Xs, y, self.transform_method_name,
                                             self.internal_cv,
                                             self.cv_processes, **fit_params)

    def predict_with_method(self, Xs, method_name):
        """
        Base method for predicting.  Users will not generally need to call
        this method because it is exposed through the standard scikit-learn
        prediction interface:
            pipeline.predict()
            pipeline.predict_proba()
            pipeline.predict_log_proba()
            pipeline.decision_function()

        Parameters
        ----------
        Xs: list of [nd.array(n_samples, n_features) or None]
        method_name: str
            Name of the prediction method to call on the last layer of the
            pipeline.

        Notes
        -----
        Invokes transform() on all layers except the last layer. Calls the
        prediction method on the last layer.  Returns either a single
        prediction array.shape(n_samples,) in the typical use case or a list of
        arrays if more than one prediction() method is present in the final
        layer's live channels. Calls to predict_with_method raise NameError if
        layer.fit_last() did not detect a the method in the final layer.
        """
        Xs = [np.array(X, dtype=float) for X in Xs]
        for layer in self.layers[:-1]:
            Xs = layer.transform(Xs)
        prediction_method = getattr(self.layers[-1], method_name)
        predictions = prediction_method(Xs)
        # decode class names
        if utils.is_classifier(self) and method_name == 'predict':
            predictions = [self.classes_[p] if p is not None else None
                           for p in predictions]
        live_predictions = [p for p in predictions if p is not None]

        return (predictions if len(live_predictions) > 1
                else live_predictions[0])

    def get_pipe(self, layer_index, pipe_index, unwrap=True):
        """
        Get a pipe (not fitted) from the pipeline using integer indexing. To
        view the index numbers, visualize the pipeline using get_dataframe(),
        get_html(), or by entering the pipeline instance name in the last line
        of a jupyter notebook cell.

        Parameters
        ----------
        layer_index: int
            Index of the layer where the pipe is located.
        pipe_index: int
            Index of the pipe within the layer's list of pipes.
        unwrap: bool
            If True: Remove wrapper classes found in the transform_wrappers
                module
            If False: Do not remove wrapper classes

        Returns
        -------
        The pipe (unfitted) specified by the index arguments.
        """
        return self.layers[layer_index].get_pipe(pipe_index, unwrap)

    def get_model(self, layer_index, model_index, unwrap=True):
        """
        Get a fitted model from the pipeline using integer indexing. To
        view the index numbers, visualize the pipeline using get_dataframe(),
        get_html(), or by entering the pipeline instance name in the last line
        of a jupyter notebook cell.

        Parameters
        ----------
        layer_index: int
            Index of the layer where the pipe is located.
        model_index: int
            Index of the model within the layer's list of pipes.
        unwrap: bool
            If True: Remove wrapper classes found in the transform_wrappers
                module
            If False: Do not remove wrapper classes

        Returns
        -------
        The model (fitted pipe) specified by the index arguments.
        """
        return self.layers[layer_index].get_model(model_index, unwrap)

    def get_pipe_from_channel(self, layer_index, channel_index, unwrap=True):
        """
        Get a pipe (not fitted) from the pipeline using integer indexing. To
        view the index numbers, visualize the pipeline using get_dataframe(),
        get_html(), or by entering the pipeline instance name in the last line
        of a jupyter notebook cell.

        Parameters
        ----------
        layer_index: int
            Index of the layer where the pipe is located.
        channel_index: int
            Index of the pipe within the pipeline's list of channels.
        unwrap: bool
            If True: Remove wrapper classes found in the transform_wrappers
                module
            If False: Do not remove wrapper classes

        Returns
        -------
        The pipe (unfitted) specified by the index arguments.
        """
        return self.layers[layer_index].get_pipe_from_channel(channel_index,
                                                              unwrap)

    def get_model_from_channel(self, layer_index, channel_index, unwrap=True):
        """
        Get a fitted model from the pipeline using integer indexing. To
        view the index numbers, visualize the pipeline using get_dataframe(),
        get_html(), or by entering the pipeline instance name in the last line
        of a jupyter notebook cell.

        Parameters
        ----------
        layer_index: int
            Index of the layer where the pipe is located.
        channel_index: int
            Index of the model within the pipeline's list of channels.
        unwrap: bool
            If True: Remove wrapper classes found in the transform_wrappers
                module
            If False: Do not remove wrapper classes

        Returns
        -------
        The model (fitted pipe) specified by the index arguments.
        """
        return self.layers[layer_index].get_model_from_channel(channel_index,
                                                               unwrap)

    def get_clone(self):
        """
        Make a clone of the pipeline instance by copying the instances's
        initialization parameters and state variables, and then calling
        utils.get_clone() on each layer and on each unfitted pipe and fitted
        model within each later.
        """
        clone = super().get_clone()
        clone.layers = [layer.get_clone() for layer in self.layers]
        return clone

    def get_dataframe(self, verbose=0, show_fit=True):
        """
        Get a dataframe visual representation of the pipeline.

        Parameters
        ----------
        verbose: int, default=0
            Set the length of the pipe's text descriptors.
            If 0: return a string with no parameters,
                e.g.: "RandomForestClassifier"
            If 1: return a string including parameters in params argument.
                e.g. "RandomForestClassifier(n_estimators=50)"
            If -1: return a condensed class name, e.g. "RanForCla",
                (not implemented)
        show_fit: bool, default=True
            If True: Show the fit version of the pipeline. Dead channels and
                counterselected models will not appear.
            If False: Show the original, unfit version of the pipeline

        Returns
        -------
        A dataframe visualization of the pipeline.
        """

        def _get_pre_fit_descriptors(layer, verbose=0):
            right_arrow = '\u2192'
            down_arrow = '\u25BD'
            descriptors = [right_arrow for channel in range(self.n_channels)]
            for pipe, slice_, indices in layer.pipe_list:
                descriptor = utils.get_descriptor(pipe, verbose)
                descriptors[slice_] = [down_arrow for i in indices]
                descriptors[indices[0]] = descriptor
            outputs = [right_arrow for i in range(self.n_channels)]
            return descriptors, outputs

        def _get_post_fit_descriptors(layer, verbose=0):
            right_arrow = '\u2192'
            down_arrow = '\u25BD'
            descriptors = [right_arrow for channel in range(self.n_channels)]
            for model, slice_, indices in layer.model_list:
                descriptor = utils.get_descriptor(model, verbose)
                descriptors[slice_] = [down_arrow for i in indices]
                descriptors[indices[0]] = descriptor
            outputs = [right_arrow if flag is True else ' ' for flag in
                       layer.output_mask]
            return descriptors, outputs

        dataframe = pd.DataFrame({'channel': range(self.n_channels)})
        dataframe = dataframe.set_index('channel')

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'model_list') and show_fit:
                descriptors, outputs = _get_post_fit_descriptors(layer,
                                                                 verbose=False)
            else:
                descriptors, outputs = _get_pre_fit_descriptors(layer,
                                                                verbose=False)
            dataframe['layer_{}'.format(i)] = descriptors
            dataframe['out_{}'.format(i)] = outputs

        return dataframe

    def get_html(self, verbose=0, show_fit=True):
        """
        Get an html visual representation of the pipeline.

        Parameters
        ----------
        verbose: int, default=0
            Set the length of the pipe's text descriptors.
            If 0: return a string with no parameters,
                e.g.: "RandomForestClassifier"
            If 1: return a string including parameters in params argument.
                e.g. "RandomForestClassifier(n_estimators=50)"
            If -1: return a condensed class name, e.g. "RanForCla",
                (not implemented)
        show_fit: bool, default=True
            If True: Show the fit version of the pipeline. Dead channels and
                counterselected models will not appear.
            If False: Show the original, unfit version of the pipeline

        Returns
        -------
        An html visualization of the pipeline.
        """
        df = self.get_dataframe(verbose, show_fit)
        styler = df.style.set_properties(
            **{'text-align': 'center'}).set_table_styles(
            [dict(selector='th', props=[('text-align', 'center')])])
        return styler._repr_html_()

    def _repr_html_(self):
        return self.get_html(verbose=0, show_fit=True)


class ChannelConcatenator(Cloneable, Saveable):
    """
    Concatenate a block of contiguous channels into a single matrix,
        outputting the concatemer in the first i/o channel and None into the
        remaining channels.
    """

    def fit(self, Xs, y=None, **fit_params):
        """
        For compatabilitiy only.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.
        y: list/array of length n_samples, default=None
            Targets for supervised ML.
        fit_params: dict, defualt=None
            Auxiliary parameters to pass to the fit method of the predictor.
        """
        pass

    def transform(self, Xs):
        """
        Concatenate input matrices.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.
        """
        live_Xs = [X for X in Xs if X is not None]
        Xs_t = [None for X in Xs]
        Xs_t[0] = np.concatenate(live_Xs, axis=1) if len(live_Xs) > 0 else None
        return Xs_t

    def fit_transform(self, Xs, y=None, **fit_params):
        """
        Concatenate input matrices.

        Parameters
        ----------
        Xs: list of [ndarray.shape(n_samples, n_features) or None]
            List of feature matrix inputs.
        y: list/array of length n_samples, default=None
            Targets for supervised ML.
        fit_params: dict, defualt=None
            Auxiliary parameters to pass to the fit method of the predictor.
        """
        self.fit(Xs, y, **fit_params)
        return self.transform(Xs)
