# pipecaster
(in progress)

Pipecaster is a Python library for building ensemble machine learning pipelines with multiple input channels (multichannel pipelines) and in-pipeline workflow automation (semi-auto-ML).  The pipeline construction workflow is loosely based on Keras layers: pipelines are built layer by layer and there is visual feedback to help manage complex architectures.  The current version supports algorithms with the scikit-learn estimator/transformer/predictor interfaces.

tutorial: https://github.com/ajcallegari/pipecaster/blob/master/tutorial.ipynb

![Use case 1](/images/tutorial_1.2.svg)

## multichannel machine learning

ML pipelines often combine multiple input feature vectors derived from different data sources or feature extraction/engineering methods.  In these instances, the best performance is not always obtained by concatenating feature vectors into a single input vector.  Better accuracy can sometimes be obtained by keeping the inputs in different silos through a first round of ML (fig. 1), with outputs of the base learners used for ensemble learning (e.g. voting or model stacking).  Silos may improve accuracy by increasing the diversity of important features or via other mechanisms.  Multichannel ML is defined here as ML with a pipeline architecture that takes multiple feature vectors as inputs and keeps their processing siloed through one or more pipeline steps.  Because the inputs are no longer technically still inputs after the first layer of the pipeline, the silos are referred to as "channels."

![figure 1.](/images/performance_comparison.png)  
**figure 1**. Performance results from example 1.1 in [tutorial.ipynb](https://github.com/ajcallegari/pipecaster/blob/master/tutorial.ipynb).  

The **MultichannelPipeline** class simplifies the construction and visualization of multichannel ensemble architectures.  This class makes it easy to create wide pipelines (many inputs) by broadcasting construction operations across multiple input channels, and deep pipelines (many layers) with a layer-by-layer construction workflow and internal cross validation training (1).  
(1) Wolpert, David H. "Stacked generalization." Neural networks 5.2 (1992): 241-259.

MultichannelPipeline has the familiar scikit-learn estimator/transformer/predictor interfaces but its methods take a list of input matrices rather than a single matrix:  

scikit-learn:  
`pipeline.fit(X, y).predict(X)`  
`pipeline.fit(X, y).transform(X)`  

pipecaster MultichannelPipeline:  
`pipeline.fit(Xs, y).predict(Xs)`  
`pipeline.fit(Xs, y).transform(Xs).`  

pipecaster MultichannelPipeline with a single input matrix:  
`pipeline.fit([X], y).predict([X])`  
`pipeline.fit([X], y).transform([X]).`  

## semi-auto-ML
A typical ML workflow involves screening input sources, feature extraction & engineering steps, ML algorithms, and model hyperparameters.  Pipecaster allows you to semi-automate each of these screening tasks by including them in the ML pipeline and executing the screens during calls to pipeline.fit().  This can be useful when you are developing a large number of different pipelines in parallel and don't have time to optimize each one separately, and it may accelerate ML workflows in general.  

Relevant classes: **SelectiveStack**, **GridSearchStack**,  **SelectKBestScorers**, **SelectKBestPerformers**, **SelectKBestModels**, **SelectKBestProbes**

## fast distributed computing
Pipecaster uses the ray library to speed up multiprocessing by passing arguments through a distributed in-memory object store without the usual serialization/deserialization overhead and without passing the same object multiple times when needed by multiple jobs.  Ray also enables pipecaster to rapidly distribute jobs among networked computers.
