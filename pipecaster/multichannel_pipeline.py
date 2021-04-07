"""
ML pipeline that takes multiple feature matrix inputs.
"""

import numpy as np
import pandas as pd
import functools

import pipecaster.utils as utils
import pipecaster.config as config
import pipecaster.parallel as parallel
from pipecaster.utils import Cloneable, Saveable, FitError

__all__ = ['Layer', 'MultichannelPipeline', 'ChannelConcatenator']


def _get_live_channels(Xs, channel_indices=None):
    """
    Determine which channels are not None.

    Parameters
    ----------
    Xs : list
        List of feature matrices and None placeholders.
    channel_indices : int, list/array of ints, or None, default=None
        - If int or list/array of int: Indices of the matrices in Xs to query.
        - If None: Query all input matrices in Xs.

    Returns
    -------
    list
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


def _has_live_channels(Xs, channel_indices=None):
    """
    Determine if channels contain a value other than None.

    Parameters
    ----------
    Xs : list of nd.array(n_samples, n_features)
    channel_indices : int, list/array of ints, or None, default=None
        - If int or list/array of int : Indices of the matrices in Xs to query.
        - If None : Query all input matrices in Xs

    Returns
    -------
    bool
        True if a value other than None is found, otherwise False.
    """
    return True if len(_get_live_channels(Xs, channel_indices)) > 0 else False


class Layer(Cloneable, Saveable):
    """
    Stage in a multi-stage pipeline.

    Layers objects are generally instantiated and handled internally by the
    MultichannelPipeline class.  In special cases, the user may want to create
    a Layer manually to acces the slicing interface, as illustrated in the
    examples below.

    Parameters
    ----------
    n_channels : int
        The number of input matrices accepted by the layer. Must be >=1.
    pipe_processes : 'max' or int, default=1
        - If 1 : Run all fitting computations with a single process.
        - If 'max' : Run fitting computations in parallel, using all available
          CPUs.
        - If >1 : Run fitting computationsin parallel with up to pipe_processes
          number of CPUs.

    Notes
    -----
    This class uses reflection to expose the prediction methods found in the
    layer's pipes, so method attributes in a Layer instance typically will not
    be identical to the method attributes of the Layer class.

    Examples
    --------
    ::

        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        import pipecaster as pc

        # broadcast a single pipe across a slice of channels
        clf = pc.MultichannelPipeline(n_channels=3)
        layer = pc.Layer(n_channels=3)
        layer[:] = LogisticRegression()
        clf.add_layer(layer)

        # specify a pipe for each channel using a list
        clf = pc.MultichannelPipeline(n_channels=3)
        layer = pc.Layer(n_channels=3)
        layer[:] = [LogisticRegression(), SVC(), RandomForestClassifier()]
        clf.add_layer(layer)

        # specify pipes using mix of slices and indices
        clf = pc.MultichannelPipeline(n_channels=3)
        layer = pc.Layer(n_channels=3)
        layer[:2] = [LogisticRegression(), SVC()]
        layer[2] = [RandomForestClassifier()]
        clf.add_layer(layer)

        # use slicing to map 3 channels to a multichannel pipe
        clf = pc.MultichannelPipeline(n_channels=3)
        layer = pc.Layer(n_channels=3)
        layer[:] = pc.MultichannelPredictor(SVC())
        clf.add_layer(layer)

        # use slicing to map 2 channel to a multichannel pipe and 3rd to a
        # single channel pipe
        clf = pc.MultichannelPipeline(n_channels=3)
        layer = pc.Layer(n_channels=3)
        layer[:2] = pc.MultichannelPredictor(SVC())
        layer[2] = LogisticRegression()
        clf.add_layer(layer)
    """

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

    def _set_estimator_type(self, pipe):
        if hasattr(pipe, '_estimator_type') is True:
            estimator_type = pipe._estimator_type
            if hasattr(self, '_estimator_type') is False:
                self._estimator_type = estimator_type
            else:
                if self._estimator_type != estimator_type:
                    raise ValueError('All predictors in a layer must have the \
                                     same type (e.g. classifier or regressor)')

    def _add_predictor_interface(self, predictor):
        for method_name in config.recognized_pred_methods:
            if hasattr(predictor, method_name):
                prediction_method = functools.partial(self.predict_with_method,
                                                      method_name=method_name)
                setattr(self, method_name, prediction_method)

    def _remove_predictor_interface(self):
        for method_name in config.recognized_pred_methods:
            if hasattr(self, method_name):
                delattr(self, method_name)

    def __setitem__(self, slice_, val):

        is_listlike = isinstance(val, (list, tuple, np.ndarray))

        # verify and expose pipe interface
        if is_listlike:
            for pipe in val:
                utils.check_pipe_interface(pipe)
                self._set_estimator_type(pipe)
                self._add_predictor_interface(pipe)
        else:
            utils.check_pipe_interface(val)
            self._set_estimator_type(val)
            self._add_predictor_interface(val)

        # get channel indices
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

        # validate channel index values
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
                    self.pipe_list.append((pipe, slice(i, i+1, 1), [i]))

        self._mapped_channels = self._mapped_channels.union(channel_indices)

        return self

    def get_pipe(self, index):
        """
        Get unfit pipe for given index (orderer of pipe addition).
        """
        return self.pipe_list[index][0]

    def get_model(self, index):
        """
        Get fit pipe for given index (orderer of pipe addition).
        """
        return self.model_list[index][0]

    def get_pipe_from_channel(self, channel_index):
        """
        Get unfit pipe taking input from given channel.
        """
        for pipe, slice_, indices in self.pipe_list:
            if type(slice_) == int and slice_ == channel_index:
                return pipe
            elif channel_index in indices:
                return pipe
        return None

    def get_model_from_channel(self, channel_index):
        """
        Get fit pipe taking input from given channel.
        """
        for model, slice_, indices in self.model_list:
            if type(slice_) == int and slice_ == channel_index:
                return model
            elif channel_index in indices:
                return model
        return None

    @staticmethod
    def _fit_transform_job(pipe, Xs, y, fit_params, slice_, channel_indices):

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
        else:
            raise utils.FitError('pipe lacks required methods for \
                                 fit_transform')

        return Xs_t, model, slice_, channel_indices

    def fit_transform(self, Xs, y=None, **fit_params):
        """
        Fit & transform each pipe in layer.

        This method clones each pipe in the layer, calls  its fit_transform()
        method if available or falls back on fit() then transform(). It leaves
        the original pipe_list untouched for reference and creates a model_list
        attribute containing cloned and fit versions of the pipes.

        Parameters
        ----------
        Xs : list
            List of feature matrices and None spaceholders.
        y : list/array, default=None
            Optional targets for supervised ML.
        fit_params : dict, default={}
            Auxiliary parameters to be sent to the fit_transform or fit methods
            of the pipes. Pipe-specific parameters not yet supported.

        Returns
        -------
        Xs_t : list
            Transformed outputs.  Ordered list of values, one per channel.
            Value can be either a transformed matrix, a passthrough from the
            input matrix, or None. A None value indicates that the
            transformation inactivated the channel (via selection or diversion
            through concatenation), or that the input was None.
        """
        args_list = []
        live_pipes = []
        for i, (pipe, slice_, channel_indices) in enumerate(self.pipe_list):
            if _has_live_channels(Xs, channel_indices):
                args_list.append((pipe, Xs, y, fit_params, slice_,
                                  channel_indices))
                live_pipes.append(i)

        n_jobs = len(args_list)
        n_processes = 1 if self.pipe_processes is None else self.pipe_processes
        n_processes = (n_jobs
                       if (type(n_processes) == int and n_jobs < n_processes)
                       else n_processes)
        if n_processes == 'max' or n_processes > 1:
            try:
                shared_mem_objects = [Xs, y, fit_params]
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

        self.output_mask_ = [True if X_t is not None else False
                             for X_t in Xs_t]

        self._remove_predictor_interface()
        for model, _, _  in self.model_list:
            self._add_predictor_interface(model)

        return Xs_t

    def fit_last(self, Xs, y=None, **fit_params):
        """
        Fit the last layer of a MultichannelPipeline.

        This method clones each pipe in the Layer, fits it to the training
        data, and exposes its prediction and transform mothods as attributes of
        the layer instance.  It leaves the original pipe_list untouched for
        reference and creates a model_list attribute containing cloned and fit
        versions of the pipes.

        Parameters
        ----------
        Xs: list
            List of feature matrix inputs (or None value placeholders).
        y: list/array of length n_samples, default=None
            Optional targets for supervised ML.
        fit_params: dict, default={}
            Auxiliary parameters for the fit_transform or fit methods of the
            pipes. Pipe-specific parameters not yet supported.

        Returns
        -------
        self
        """
        self.model_list = []
        prediction_method_names = []
        estimator_types = []
        output_mask = [False for X in Xs]
        for pipe, slice_, channel_indices in self.pipe_list:
            if _has_live_channels(Xs, channel_indices):
                model = utils.get_clone(pipe)
                input_ = (Xs[slice_] if utils.is_multichannel(model)
                          else Xs[slice_][0])
                if y is None:
                    model.fit(input_, **fit_params)
                else:
                    model.fit(input_, y, **fit_params)

                prediction_method_names.extend(
                                    utils.get_predict_methods(model))
                estimator_type = utils.detect_predictor_type(model)

                if estimator_type is not None:
                    estimator_types.append(estimator_type)

                self.model_list.append((model, slice_, channel_indices))
                self._set_estimator_type(model)

                if utils.is_predictor(model):
                    output_mask[channel_indices[0]] = True
                if utils.is_transformer(model):
                    if utils.is_multichannel(model):
                        outputs = [True if X_t is not None else False
                                            for X_t in model.transform(input_)]
                        output_mask[slice_] = outputs
                    else:
                        output = (True if model.transform(input_) is not None
                                  else False)
                        output_mask[slice_] = output

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
        self.output_mask_ = output_mask

        self._remove_predictor_interface()
        for model, _, _  in self.model_list:
            self._add_predictor_interface(model)

        return self

    def transform(self, Xs):
        """
        Call transform() method of each pipe in the layer.

        Parameters
        ----------
        Xs: list
            List of feature matrix inputs (or None value placeholders).

        Returns
        -------
        Xs_t : list
            Transformed outputs.  Ordered list of values, one per channel.
            Value can be either a transformed matrix, a passthrough from the
            input matrix, or None. A None value indicates that the
            transformation inactivated the channel (via selection or diversion
            through concatenation), or that the input was None.
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
        Predict with each pipe using methods that match a specified name.

        Users will not generally call this method directly because available
        preditors are exposed through reflection for scikit-learn compliant
        prediction methods:
            - pipeline.predict()
            - pipeline.predict_proba()
            - pipeline.predict_log_proba()
            - pipeline.decision_function()

        Parameters
        ----------
        Xs: list
            List of feature matrix inputs (or None value placeholders).
        method_name: str
            Name of method to use for prediction.

        Returns
        -------
        ndarray.shape(n_samples,) or list of ndarrays
            If one matching prediction method is found in the layer, returns a
            single prediction array (typical use). If more than one pipe has
            a matching prediction method, this method returns a list with
            either a prediction array or None value for each input channel.
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
        Get a stateful clone of the Layer.
        """
        clone = super().get_clone()
        if hasattr(self, '_all_channels'):
            clone._all_channels = self._all_channels.copy()
        if hasattr(self, '_mapped_channels'):
            clone._mapped_channels = self._mapped_channels.copy()
        if hasattr(self, '_estimator_type'):
            clone._estimator_type = self._estimator_type
        if hasattr(self, 'output_mask_'):
            clone.output_mask_ = self.output_mask_.copy()
        clone.pipe_list = [(utils.get_clone(p), s, i.copy())
               for p, s, i in self.pipe_list]
        for pipe in clone.pipe_list:
            clone._set_estimator_type(pipe)
            clone._add_predictor_interface(pipe)
        if hasattr(self, 'model_list'):
            clone.model_list = [(utils.get_clone(p), s, i.copy())
                                for p, s, i in self.model_list]
            clone._remove_predictor_interface()
            for model in clone.model_list:
                clone._set_estimator_type(model)
                clone._add_predictor_interface(model)
        return clone


class MultichannelPipeline(Cloneable, Saveable):
    """
    ML pipeline with multiple feature matrix inputs.

    MultichannelPipeline implements a multichannel analog of the scikit-learn
    Pipeline class.  For instance, a scikit-learn pipeline (which has the
    estimator and predictor interfaces) is fit by calling
    pipeline.fit(X_train, y_train) and predicts with pipeline.predict(X), while
    a MultichannelPipeline is fit by calling pipeline.fit(Xs_train, y_train)
    and predicts with pipeline.predict(Xs).

    **Visual feedback**

    In Jupyter Notebook interactive mode, pipelines are depicted graphically
    when the instance name is on the last line of a cell.  In command line
    interactive mode, my_pipeline.get_dataframe() can be used to visualize a
    pipeline instance.  Visual outputs can be useful for inspecting the
    topology of complex pipelines or for finding the coordinates
    (layer, channel) of a pipeline component of interest.

    **Under the hood**

    MultichannelPipelines consist of a stack of Layer objects representing the
    sequential stages of a pipeline.  Each layer is a list of "pipes", the
    general term used here to refer to scikit-learn estimator/transformers,
    estimator/predictors, and pipecaster's multichannel analogs of these
    components.  Layer objects also track which channels the pipes use for
    input and output.  During calls to pipeline.fit(), each pipe in the
    internal layers is used to fit and transform the data.  Predictors can't
    be directly included in internal layers as they lack a required transform()
    or fit_transform() method, but must first be converte into tranformers
    using the make_transformer and make_cv_transformer functions
    as shown in the example below.  These wrappers can be used to generate
    outputs for voting, aggregation, or model stacking.

    MultichannelPipelines have a constant number of I/O channels internally,
    but some channels may be set to None when they are inactivated by selection
    processes or when diverted by concatenation processes.  In a typical use
    case the final layer of the pipeline will be a single predictor that pools
    information from all of the channels in some way.  When this is the case,
    all of the prediction methods of the final pipe will be exposed on the
    MultichannelPipeline object.  For instance, you can use
    pipeline.predict_proba(Xs) or pipeline.predict(Xs) if your last layer is a
    classifier that predicts both classes and marginal probabilies. In atypical
    use cases where MultichannelPipeline is used to create a transformer rather
    than a predictor or has multiple outputs for external meta-prediction, the
    last layer may not be a single predictor with a single output.  In these
    cases MultichannelPipeline does not output a single matrix but outputs a
    list of matices, one per channel.

    Parameters
    ----------
    n_channels: int, default=1
        The number of separate I/O channels within the pipeline.

    Notes
    -----
    - There is no stateless scikit-learn-like clone implementation because
      MultiChannelPipeline __init__() arguments are not sufficient to
      reproduce the pipeline.  Use :meth:`get_clone` to make a
      stateful clone or rebuild pipeline from scratch to get a stateless
      clone.

    - This class uses reflection to expose the predictor methods found in the
      last pipeline layer, so the method attributes of MultichannelPipeline
      instances are typicall not identical to the method attributes of the
      MultichannelPipeline class.

    - Fit failures are currently not handled / allowed.

    - Multi-output prediction not yet supported.

    - Groups for internal cv not yet supported.

    - Sample weights for internal cv performance metrics not yet supported
      (but sample_weights for model training should work fine).

    - Caching of intermediate results not yet implemented.

    - fit_params targeted to specific pipes not yet implemented.

    Examples
    --------
    **Feature selection and channel selection:**
    ::

        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectPercentile, f_classif
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        # make a synthetic dataset with 10 input matrices
        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)

        # add 10 feature scalers, one per input:
        clf.add_layer(StandardScaler())

        # add 10 feature selectors, one per input:
        clf.add_layer(SelectPercentile(f_classif, percentile=25))

        # add 1 channel selector that selects the 3 best scores
        clf.add_layer(pc.SelectKBestScores(f_classif, np.mean, k=3))

        # concatenates matrices predict with GradientBoostingClassifier
        clf.add_layer(pc.MultichannelPredictor(GradientBoostingClassifier()))

        # test the pipeline on synthetic data
        pc.cross_val_score(clf, Xs, y)
        # output: [0.9705882352941176, 1.0, 0.9099264705882353]

    **Model stacking, style 1:**
    ::

        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        import pipecaster as pc

        # make a synthetic dataset with 10 input matrices
        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)

        # create 10 StandardScaler objs, one per input:
        clf.add_layer(StandardScaler())

        # Create ensemble predictor with 10 LogistRegression base predictors
        # executed in parallel processes and a SVC meta-classifier.
        base_clf, meta_clf = LogisticRegression(), SVC()
        clf.add_layer(
            pc.ChannelEnsemble(base_clf, meta_clf, internal_cv=5),
            pipe_processes='max')

        # test the pipeline on synthetic data
        pc.cross_val_score(clf, Xs, y)

    **Model stacking, style 2:**
    ::

        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        import pipecaster as pc

        # make a synthetic dataset with 10 input matrices
        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)

        # create 10 StandardScaler objs, one per input:
        clf.add_layer(StandardScaler())

        # add internal cross validation training for stacked generalizion:
        base_clf = pc.make_cv_transformer(LogisticRegression())

        # create 10 LogisticRegression models, one per input,
        # that train in parallel using all available CPUs:
        clf.add_layer(base_clf, pipe_processes='max')

        # add a SVC meta-predictor that takes all channels as inputs:
        clf.add_layer(pc.MultichannelPredictor(SVC()))

        # test the pipeline on synthetic data
        pc.cross_val_score(clf, Xs, y)

        # output: [0.9117647058823529, 0.8805147058823529, 0.7886029411764706]

    **Model stacking, style 3:**
    ::

        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        import pipecaster as pc

        # make a synthetic dataset with 10 input matrices
        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)

        clf = pc.MultichannelPipeline(n_channels=10)

        # create 10 StandardScaler objs, one per input:
        clf.add_layer(StandardScaler())

        # add internal cross validation training for stacked generalizion:
        base_clf = pc.make_cv_transformer(LogisticRegression())

        # create 10 LogisticRegression models, one per input,
        # that train in parallel using all available CPUs:
        clf.add_layer(base_clf, pipe_processes='max')

        clf.add_layer(pc.ChannelConcatenator())

        # add a SVC meta-predictor that takes all channels as inputs:
        clf.add_layer(SVC())

        # test the pipeline on synthetic data
        pc.cross_val_score(clf, Xs, y)

        # output: [0.9117647058823529, 0.96875, 0.9393382352941176]
    """

    def __init__(self, n_channels=1):
        self._params_to_attributes(MultichannelPipeline.__init__, locals())
        self.layers = []

    def _set_estimator_type(self, layer):
        if hasattr(layer, '_estimator_type'):
            self._estimator_type = layer._estimator_type

    def _add_predictor_interface(self, layer):
        for method_name in config.recognized_pred_methods:
            if hasattr(layer, method_name):
                prediction_method = functools.partial(self.predict_with_method,
                                                      method_name=method_name)
                setattr(self, method_name, prediction_method)

    def _remove_predictor_interface(self):
        for method_name in config.recognized_pred_methods:
            if hasattr(self, method_name):
                delattr(self, method_name)

    def add_layer(self, *pipe_mapping, pipe_processes=1):
        """
        Add a new stage to the pipeline.

        Parameters
        ----------
        pipe_mapping : Layer, pipe, or (int, pipe, int, pipe ...)
            - If Layer : add the Layer instance and return.
            - If a single argument : the argument will be atomatically repeated
              to fill all channels if it is a single channel pipe, or set to
              receive all channels as inputs into a single pipe if it is a
              multichannel pipe.
            - If multiple arguments : read as a list of alternating int, pipes,
              int, pipe...  The int sets how many continguous channels are
              mapped to the pipe.  Single channel pipes are automatically
              repeated for each input channel specified by the int argument,
              and multichannel  pipes are set to receive the number of inputs
              specified by the int. Input channels are mapped sequentially in
              the order in which the arguments are received.
        pipe_processes : int or 'max', default=1
            - If int : The number of parallel processes to run on call to
              fit_transform().
            - If 'max' : Use all available CPUs for fit_transform().

        Returns
        -------
        self

        Examples
        --------
        ::

            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier
            import pipecaster as pc

            # (broadcasting) duplicate a single pipe to make 1 for
            # each channel:
            clf = pc.MultichannelPipeline(n_channels=3)
            clf.add_layer(GradientBoostingClassifier())

            # apply a different pipe to each channel:
            clf = pc.MultichannelPipeline(n_channels=3)
            clf.add_layer([GradientBoostingClassifier(),
                          LogisticRegression(),
                          KNeighborsClassifier()])

            # use channel mapping notation to set the first 2 channels
            # to duplicates of GradientBoostingClassifier and the 3rd channel
            # to LogisticRegression:
            clf = pc.MultichannelPipeline(n_channels=3)
            clf.add_layer(2, GradientBoostingClassifier(),
                          1, LogisticRegression())

            # add a single multichannel pipe that takes all channels as inputs:
            clf = pc.MultichannelPipeline(n_channels=3)
            clf.add_layer(
                pc.MultichannelPredictor(GradientBoostingClassifier()))

            # map the first two channels to a single multichannel pipe and the
            # 3rd channel to a single channel pipe
            clf = pc.MultichannelPipeline(n_channels=3)
            m_pipe = pc.MultichannelPredictor(GradientBoostingClassifier())
            clf.add_layer(2, m_pipe, 1, LogisticRegression())
        """
        if len(pipe_mapping) == 1 and type(pipe_mapping[0]) == Layer:
            if pipe_mapping[0].n_channels <= self.n_channels:
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

        self.layers.append(new_layer)

        self._set_estimator_type(new_layer)
        self._remove_predictor_interface()
        self._add_predictor_interface(new_layer)

        return self

    def set_pipe_processes(self, pipe_processes):
        """
        Set the number of parallel processes to use during model fitting.

        Parameters
        ----------
        pipe_processes : int, 'max', or iterable
            - If int : Every layer is set to use the same number of parallel
              processes.
            - If 'max' : Every layer is set to use up to the maximum
              number of available CPUs.
            - If iterable: The number of processes is set for each layer
              according to the values in the list {int or 'max'}.

        Returns
        -------
        self
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
        return self

    def fit(self, Xs, y=None, **fit_params):
        """
        Fit all pipes in the pipeline.

        All pipes are cloned before fitting so that the original pipeline
        architecture is maintained for reference until the final model is
        exported (model exporting not yet implemented).

        Calling pipeline.fit(Xs_train, y_train) invokes layer.fit_transform()
        on each layer and then layer.fit_last() on the last layer.  The call
        also exposes all prediction methods found in the final layer (i.e.
        predict, predict_proba, decision_function, or predict_log_proba) so
        that they can be called directly on the pipeline itself.

        Calls to layer.fit_transform() in turn calls the fit_transform() method
        of each cloned pipe in the layer, falling back on fit() then
        transform() when a fit_transform() method is not available.

        Calls to layer.fit_last() clones and fits the models in the last layer
        of the pipeline.

        Parameters
        ----------
        Xs : list
            List of feature matrices and None spaceholders.
        y : list/array, default=None
            Optional targets for supervised ML.
        fit_params : dict, default=None
            Auxiliary parameters to pass to the fit method of the predictors.

        Returns
        -------
        self
        """

        if y is not None and utils.is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)

        for layer in self.layers[:-1]:
            Xs = layer.fit_transform(Xs, y, **fit_params)
        # fit the last layer without transforming:
        self.layers[-1].fit_last(Xs, y, **fit_params)
        self._set_estimator_type(self.layers[-1])
        self._remove_predictor_interface()
        self._add_predictor_interface(self.layers[-1])

        return self

    def transform(self, Xs):
        """
        Transform the inputs.

        Parameters
        ----------
        Xs : list
            List of input feature matrices (or None value placeholders).

        Returns
        -------
        Xs_t : list
            Transformed matrices.  Ordered list of values, one per channel.
            Value can be either a transformed matrix, a passthrough from the
            input matrix, or None.
        """
        for layer in self.layers:
            Xs = layer.transform(Xs)
        return Xs

    def fit_transform(self, Xs, y, **fit_params):
        """
        Fit and transform inputs.

        Parameters
        ----------
        Xs: list
            List of feature matrix inputs (or None value placeholders).
        y: list/array of length n_samples, default=None
            Optional targets for supervised ML.
        fit_params: dict, default=None
            Auxiliary parameters passed to the fit method of the predictor.

        Returns
        -------
        Xs_t : list
            Transformed matrices.  Ordered list of values, one per channel.
            Value can be either a transformed matrix, a passthrough from the
            input matrix, or None.
        """
        if y is not None and utils.is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)

        for layer in self.layers[:-1]:
            Xs = layer.fit_transform(Xs, y,  **fit_params)
        self.layers[-1].fit_last(Xs, y, **fit_params)
        self._set_estimator_type(self.layers[-1])
        self._remove_predictor_interface()
        self._add_predictor_interface(self.layers[-1])

        return self.layers[-1].transform(Xs)

    def predict_with_method(self, Xs, method_name):
        """
        Base method for predicting.

        Users will not generally call this method directly because available
        preditors are exposed through reflection for scikit-learn compliant
        prediction methods:
            - pipeline.predict()
            - pipeline.predict_proba()
            - pipeline.predict_log_proba()
            - pipeline.decision_function()

        Parameters
        ----------
        Xs: list
            List of input feature matrices (or None value placeholders).
        method_name: str
            Name of method to use for prediction.

        Returns
        -------
        ndarray.shape(n_samples,) or list
            If one matching prediction method is found in the last layer,
            returns a single prediction array (typical use). If more than one
            matching prediction method is found, a list with values for each
            input channel (or None placeholders) is returned.
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

    def get_pipe(self, layer_index, pipe_index):
        """
        Get the pipe from the specified layer index and pipe index.

        Parameters
        ----------
        layer_index : int
            Index of the layer where the pipe is located.
        pipe_index : int
            Index of the pipe within the layer's list of pipes (order of
            addition).

        Returns
        -------
        pipe
            The pipe (unfitted) specified by the index arguments.
        """
        return self.layers[layer_index].get_pipe(pipe_index)

    def get_model(self, layer_index, model_index):
        """
        Get the fitted pipe from the specified layer index and pipe index.

        Parameters
        ----------
        layer_index : int
            Index of the layer where the model is located.
        pipe_index : int
            Index of the model within the layer's list of pipes (order of
            addition).

        Returns
        -------
        model
            The model (fitted pipe) specified by the index arguments.
        """
        return self.layers[layer_index].get_model(model_index)

    def get_pipe_from_channel(self, layer_index, channel_index):
        """
        Get pipe with input from the specified channel.

        Parameters
        ----------
        layer_index : int
            Index of the layer where the pipe is located.
        channel_index : int
            Index of channel in question.

        Returns
        -------
        pipe or None
            The pipe (unfitted) specified by the index arguments or None.
        """
        return self.layers[layer_index].get_pipe_from_channel(channel_index)

    def get_model_from_channel(self, layer_index, channel_index):
        """
        Get the fitted pipe that takes input from the specified channel.

        Parameters
        ----------
        layer_index: int
            Index of the layer where the pipe is located.
        channel_index: int
            Index of a channel to which the pipe is mapped.

        Returns
        -------
        pipe
            The pipe (fitted) specified by the index arguments.
        """
        return self.layers[layer_index].get_model_from_channel(channel_index)

    def get_clone(self):
        """
        Get a stateful clone of the MultichannelPipeline.

        Uses methods inherited from utils.Cloneable to copy parameters
        and state variables, and utils.get_clone() to copy pipes and models.
        """
        clone = super().get_clone()
        if hasattr(self, 'classes_'):
            clone.classes_ = self.classes_.copy()
        clone.layers = [layer.get_clone() for layer in self.layers]
        clone._set_estimator_type(self.layers[-1])
        clone._add_predictor_interface(self.layers[-1])

        return clone

    def get_dataframe(self, verbose=0, show_fit=True):
        """
        Get a pandas DataFrame representation of the pipeline.

        Parameters
        ----------
        verbose: int, default=0
            - Set the length of the pipe's text descriptors:
            - If 0 : return a string with no parameters,
              e.g.: "RandomForestClassifier"
            - If 1 : return a string including parameters in params argument.
              e.g. "RandomForestClassifier(n_estimators=50)"
            - If -1 : return a condensed class name, e.g. "RanForCla",
              (not yet implemented)
        show_fit: bool, default=True
            - If True: Show the fit version of the pipeline with channel
              outputs. Dead channels and counterselected models will not
              appear.
            - If False: Show the original, unfit version of the pipeline.

        Returns
        -------
        pandas DataFrame
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
            return descriptors

        def _get_post_fit_descriptors(layer, verbose=0):
            right_arrow = '\u2192'
            down_arrow = '\u25BD'
            descriptors = [' ' for channel in range(self.n_channels)]
            for model, slice_, indices in layer.model_list:
                descriptor = utils.get_descriptor(model, verbose)
                descriptors[slice_] = [down_arrow for i in indices]
                descriptors[indices[0]] = descriptor
            outputs = [right_arrow if flag is True else ' ' for flag in
                       layer.output_mask_]
            return descriptors, outputs

        dataframe = pd.DataFrame({'channel': range(self.n_channels)})
        dataframe = dataframe.set_index('channel')

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'model_list') and show_fit:
                descriptors, outputs = _get_post_fit_descriptors(layer,
                                                                 verbose=False)
                dataframe['layer_{}'.format(i)] = descriptors
                dataframe['out_{}'.format(i)] = outputs
            else:
                descriptors = _get_pre_fit_descriptors(layer, verbose=False)
                dataframe['layer_{}'.format(i)] = descriptors

        return dataframe

    def get_html(self, verbose=0, show_fit=True):
        """
        Get an html representation of the pipeline.

        Parameters
        ----------
        verbose: int, default=0
            - Set the length of the pipe's text descriptors:
            - If 0 : return a string with no parameters,
              e.g.: "RandomForestClassifier"
            - If 1 : return a string including parameters in params argument.
              e.g. "RandomForestClassifier(n_estimators=50)"
            - If -1 : return a condensed class name, e.g. "RanForCla",
              (not implemented)
        show_fit: bool, default=True
            - If True: Show the fit version of the pipeline with channel
              outputs.  Dead channels and counterselected models will not
              appear.
            - If False: Show the original, unfit version of the pipeline.

        Returns
        -------
        html
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
    Concatenate multiple channel outputs into a single matrix.

    Concatenate a block of contiguous channel outputs into a single matrix and
    output the concatemer in the first ouptut channel and None into the
    remaining channels.

    Examples
    --------
    ::

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        import pipecaster as pc

        Xs, y, _ = pc.make_multi_input_classification(n_informative_Xs=3,
                                                      n_random_Xs=7)
        clf = pc.MultichannelPipeline(n_channels=10)
        base_clf = pc.make_cv_transformer(GradientBoostingClassifier())
        clf.add_layer(base_clf)
        clf.add_layer(pc.ChannelConcatenator())
        clf.add_layer(1, SVC())
        pc.cross_val_score(clf, Xs, y, cv=3)
        # output: [0.9411764705882353, 0.9099264705882353, 0.8768382352941176]
    """

    def fit(self, Xs, y=None, **fit_params):
        """
        For compatabilitiy only.

        Parameters
        ----------
        Xs : list
            List of feature matrices and None spaceholders.
        y : list/array, default=None
            Optional targets for supervised ML.
        fit_params: dict, default=None
            Auxiliary parameters to pass to the fit method of the predictor.
        """
        return self

    def transform(self, Xs):
        """
        Concatenate input matrices.

        Parameters
        ----------
        Xs: list
            List of input feature matrices (or None value placeholders).

        Returns
        -------
        list
            List of values, one per channel,  with the concatenated matrix at
            index 0 and all other channel values set to None.
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
        Xs : list
            List of feature matrices and None spaceholders.
        y : list/array, default=None
            Optional targets for supervised ML.
        fit_params : dict, default=None
            Auxiliary parameters to pass to the fit method of the predictor.

        Returns
        -------
        list
            List of values, one per channel, with the concatenated matrix at
            index 0 and all other channel values set to None.
        """
        return self.transform(Xs)
