# pipecaster
(in progress)

A scikit-learn extension for broadcasting machine learning pipeline construction operations across multiple input sources and for in-pipeline workflow automation.  Provides slice notation and a Keras-like layered workflow for creating complex pipelines, and enables in-pipeline screens to automate the selection of data sources, feature engineering steps, ML algorithms, and model hyperparameters.

## silo ML inputs

ML libraries often implicitly encourage concatenation of features from multiple data sources into a single feature matrix (X) prior to feature selection or ML.  In practice, concatenation often reduces performance and greater predictive accuracy can be obtained by siloing the different inputs through the initial feature selection and ML steps and combining inferences at a later stage using voting or stacked generalization.  Pipecaster encourages input silos by modifying the sklearn interface from:  
`pipeline.fit(X, y).predict(X)`  
to   
`pipeline.fit(Xs, y).predict(Xs).`  

## automate workflows with in-pipeline screening

A typical ML workflow involves screening different input sources, feature engineering steps, models, and model hyperparameters.  Pipecaster allows you to semi-automate each of these screening tasks by including them in the ML pipeline.  This can be useful when you are developing a large number of different pipelines in parallel and don't have time to optimize each one separately, and it may accelerate ML workflows in general.  I wasn't able to find these in-pipeline operations in sklearn, Dask, or Spark ML, which provided some of the motivation for developing this library.

**Channel selection**   
The different input channels passed to pipecaster pipelines (Xs) may come from different data sources, different transformations of the data (i.e. for feature engineering), or both.  Pipecaster provides three ways to select input channels in order to keep garbage from flowing into and out of your ML models.    

  1. The *ScoreChannelSelector* class selects channels based on aggregated feature scores.  
  1. The *PerformanceChannelSelector* class selects channels based on performance on an internal cross validation run with with a probe ML model and passes through the channel values that meet specified performance criteria.  
  1. The *ChannelModelSelector* is similar to the PerformanceChannelSelector, but outputs the predictions of selected models rather passing through the values from the previous pipeline step.  


**Model selection**  
Pipecaster allows in-pipeline screening of ML models and their hyperparameters with the *SelectiveEnsemble* class.  A *SelectiveEnsemble*, which operates on a single input, is a voting or concatenating ensemble that selects only the most performant models from within the ensemble. Model performance is assessed with an internal cross validation run within the training set during calls to pipeline.fit().  

# illustrative example
![Use case 1](/images/architecture_1.png)

This diagram shows a classification pipeline taking 5 numerical input matrices (X0 to X4) and 1 text input (X5).  Code for building this pipeline is given below.  The InputScoreSelector "SelectKBestInputs" computes a matrix score for each input by aggregating feature scores and then selects the k=3 best inputs.  The ModelSelector "SelectKBestPredictors" does an internal cross validation run within the training set during the call to pipeline.fit(Xs, y), estimates the accuracy of models trained on inputs 0 to 4, then selects the k=2 best models and sends their inferences on to a meta-classifier.

## sample code:

```
import pipecaster as pc
clf = pc.Pipeline(n_inputs=6)

layer0 = clf.get_next_layer()
layer0[:5] = SimpleImputer()
layer0[5] = CountVectorizer()

layer1 = clf.get_next_layer()
layer1[:5] = StandardScaler()
layer1[5] = TfidfTransformer()

layer2 = clf.get_next_layer()
layer2[:] = SelectKBest(f_classif, k = 100)

layer3 = clf.get_next_layer()
layer3[:5] = pc.SelectKBestInputs(scoring=f_classif, aggregator='sum', k=3)

layer4 = clf.get_next_layer()
predictors = [KNeighborsClassifier() for i in range(5)]
layer4[:5] = pc.SelectKBestPredictors(predictors=predictors, scoring=make_scorer(roc_auc_score), cv=3)
layer4[5] = MultinomialNB()

layer5 = clf.get_next_layer()
layer5[:] = pc.MetaClassifier(SVC())

clf.fit(X_trains, y_train)
clf.predict(X_tests)
```
