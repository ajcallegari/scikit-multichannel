# pipecaster
(in progress)

Pipecaster is a Python library for building multichannel machine learning pipelines and for in-pipeline screening of hyperparameters, models, data sources, and feature engineering steps.  The current version supports algorithms with the scikit-learn transformer and predictor interfaces.

## multichannel machine learning

ML libraries often implicitly encourage concatenation of features from multiple data sources into a single feature matrix (X) prior to feature selection or ML.  In practice, concatenation often reduces performance and greater predictive accuracy can be obtained by siloing the different inputs in separate channels through the initial feature selection and ML steps and then combining inferences at a later stage with voting or model stacking.

Pipecaster provides a *MultichannelPipeline* class to simplify the construction of multichannel architectures, with slice notation for broadcasting operations across multiple input channels and keras-like layer-by-layer construction.  The library encourages input silos by modifying the sklearn predictor interface from:  
`pipeline.fit(X, y).predict(X)`  
to:  
`pipeline.fit(Xs, y).predict(Xs).`

## in-pipeline screening
A typical ML workflow involves screening input sources, feature engineering steps, ML algorithms, and model hyperparameters.  Pipecaster allows you to semi-automate each of these screening tasks by including them in the ML pipeline.  This can be useful when you are developing a large number of different pipelines in parallel and don't have time to optimize each one separately, and it may accelerate ML workflows in general.

## blazing fast distributed computing
Pipecaster uses the ray library to speed up multiprocessing by passing arguments through the plasma in-memory object store without the usual serialization/deserialization overhead.  Ray also enables pipecaster to rapidly distribute jobs among computers in a cluster.

# sample architecture
![Use case 1](/images/architecture_1.png)

This diagram shows a pipecaster classification pipeline taking 5 numerical input matrices (X0 to X4) and 1 text input (X5).  Code for building this pipeline is given below.  SelectKBestChannels computes a score for each input channel by aggregating their feature scores and then selects the k=3 best channels.  SelectKBestPredictors does an internal cross validation run within the training set during the call to pipeline.fit(Xs, y), estimates the accuracy of models trained on inputs 0 to 4, then selects the k=2 best models and sends their inferences on to a meta-classifier.

## sample code:

```
import pipecaster as pc  

clf = pc.Pipeline(n_inputs=6)

layer = clf.get_next_layer()
layer[:5] = SimpleImputer()
layer[5] = CountVectorizer()

layer = clf.get_next_layer()
layer[:5] = StandardScaler()
layer[5] = TfidfTransformer()

clf.get_next_layer()[:] = SelectKBest(f_classif, k = 100)

clf.get_next_layer()[:5] = pc.SelectKBestInputs(scoring=f_classif, aggregator='sum', k=3)

layer = clf.get_next_layer()
predictors = [KNeighborsClassifier() for i in range(5)]
layer[:5] = pc.SelectKBestPredictors(predictors=predictors,
                                     scoring=make_scorer(roc_auc_score), cv=3)
layer[5] = MultinomialNB()

clf.get_next_layer()[:] = pc.MetaClassifier(SVC())

clf.fit(X_trains, y_train)
clf.predict(X_tests)
```


# features

## meta-prediction with input channel ensembles
Inferences generated by siloed input channels can easily be combined through voting and model stacking by layering pipecaster's ChannelClassifier/ChannelRegressor classes onto your pipeline (see example above).  This pipeline architecture can be also be built in scikit-learn by nesting ColumnTransformers within a VotingClassifier/VotingRegressor or within a StackingClassifier/StackingRegressor.  Pipecaster automatically detects predictors with outputs used in meta-classification and provides them with internal cross validation training(1) and transform()/fit_transform() methods.  
(1) Wolpert, David H. "Stacked generalization." Neural networks 5.2 (1992): 241-259.

## in-pipeline screening

**Input screening**   
The different input channels passed to pipecaster pipelines (Xs) may come from different data sources, different transformations of the data (i.e. for feature engineering), or both.  Pipecaster provides three ways to select input channels in order to keep garbage from flowing into and out of your ML models.    

  1. The *ScoreChannelSelector* class selects input channels based on aggregated feature scores.  
  1. The *PerformanceChannelSelector* class selects input channels based on performance of a probe model on an internal cross validation run.
  1. The *ChannelModelSelector* is similar to the PerformanceChannelSelector, but outputs the predictions of selected models rather passing through the values from the previous pipeline step.  

**Model screening**  
Pipecaster allows in-pipeline screening of ML models and their hyperparameters with the *SelectiveEnsemble* class.  A *SelectiveEnsemble*, which operates on a single input, is a voting or concatenating ensemble that selects only the most performant models from within the ensemble. Model performance is assessed with an internal cross validation run within the training set during calls to pipeline.fit().  

## fast distributed computing
Pipecaster uses Ray to distribute parallel computations to multiple processors and computers (.e.g for cross validation runs, hyperparameter screens, etc.).  Apart from enabling cluster computing, Ray dramatically increases the speed of single-computer multiprocessing relative to joblib or the Python standard library's multiprocessing package by reducing or eliminating serialization/de-serialization overhead with an in-memory object store.  Ray's plasma object store is also used by Apache Arrow & Apache Spark.  The plasmas store is also useful for reducing your memory footprint by allowing all processes on a given computer to share a single copy of large objects.  
