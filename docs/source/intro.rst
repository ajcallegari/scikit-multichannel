Introduction to pipecaster
==========================

.. figure::  _images/profile.png
   :align:   center

Pipecaster is a Python library for building machine learning pipelines out of
scikit-learn components.  It features:

- a multichannel pipeline architecture
- ensemble learning (voting, aggregating, stacked generalization)
- tools for managing complex pipeline architectures:

    - Keras-like layers
    - Visual feedback during pipeline construction

- in-pipeline workflow automation (or 'semi-auto-ML'):

    - screening of input sources based on aggregate features score or
      performance of a probe ML model
    - screening of ML algorithms
    - screening of model hyperparameters

- fast distributed computing with ray

What is a multichannel pipeline?
--------------------------------
A multichannel pipeline is an ML pipeline that takes multiple input vectors
and processes them in separate channels before combining them through
concatenation, voting, or model stacking to generate a single prediction.

See:
:class:`MultichannelPipeline <pipecaster.multichannel_pipeline.MultichannelPipeline>`

Why use a multichannel architecture?
------------------------------------

- When there are multiple input matrices coming from different data sources or
  feature extraction methods, you can sometimes get better model performance by
  training a separate ML model on each input and then making an ensemble
  prediction.

- You want to automate the selection of input sources, feature extraction
  methods, ML algrithms, and hyperparameters (e.g. when you have a large
  number of related tasks or find yourself screening the "usual suspects" with
  each new task).

- You want to use an enormous number of inputs but don't want the
  computational cost of having your ML model train on all of them.

Distributed computing with ray
------------------------------
Pipecaster uses the `ray <https://docs.ray.io/en/master/>`_ library to speed up
multiprocessing by passing arguments through a distributed in-memory object
store without the usual serialization/deserialization overhead and without
passing the same object multiple times when needed by multiple jobs.  Ray also
enables pipecaster to rapidly distribute jobs among networked computers.

Installation
------------
::

  git clone https://github.com/ajcallegari/pipecaster.git
  cd pipecaster
  pip install .

or:
::
  pip install pipecaster

pipecaster was developed with Python 3.7.5 and tested with the following
dependencies:
::

  numpy==1.17.2
  joblib==0.16.0
  ray==1.1.0
  scipy==1.3.1
  pandas==0.24.2
  scikit-learn==0.23.2


**Thanks** to `Ilya Goldberg <https://github.com/igg/>`_ and
`Josiah Johnston <https://github.com/josiahjohnston>`_ for the fascinating
conversations that inspired pipecaster.

Pipecaster was developed by `A. John Callegari <https://www.linkedin.com/in/ajcallegari/>`_
