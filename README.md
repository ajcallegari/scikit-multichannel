# pipecaster-sklearn
(in progress)

A Python library for broadcasting machine learning (ML) pipeline construction operations across multiple input sources.  The library provides convenient slice notation and a Keras-like layered workflow with an sklearn interface.

## silo ML inputs

Many ML libraries implicitly encourage concatenation of features from multiple data sources into a single feature matrix (X) prior to feature selection or ML.  In practice, concatenation often reduces performance and greater predictive accuracy can be obtained by siloing the different inputs through the initial feature selection and ML steps and combining inferences at a later stage using voting or stacked generalization.  Pipecaster encourages input silos by modifying the sklearn interface from:  

`model.fit(X, y).predict(X)`  
to   
`model.fit(Xs, y).predict(Xs).`  

## automate workflows with in-pipeline screening

A typical ML workflow involves screening different input sources, different feature engineering steps, different models, and different model hyperparameters.  Pipecaster allows you to semi-automate each of these screening tasks by including them in the ML pipeline.  This can be useful when you are developing a large number of models in parallel and don't have time to optimize each one separately, and may accelerate ML workflows in general.  I wasn't able to find these in-pipeline operations in sklearn, Dask, or Spark ML, which provided some of the motivation for developing pipecaster.

1. **Input sceening** The different inputs to pipecaster pipelines (Xs) may come from different data sources, different transformations of the data (i.e. for feature engineering), or both.  Pipecaster provides two different ways to select inputs in order to keep garbage from flowing into and out of your ML model.  The *InputScoreSelector* class select inputs based on aggregated univariate feature scores.  The *InputPerformanceSelector* class selects inputs based on performance on an internal cross validation run with with a probe ML model.

1. **Model Screening**  Pipecaster allows in-pipeline screening of ML models and their hyperparameters through the *SelectiveEnsemble* class.  SelectiveEnsembles are voting or concatenating ensembles that select only the most performant models from within the ensemble.  Model performance is assessed with an internal cross validation run within the training set.  

# illustrative example
![Use case 1](/images/example_1.png)

This diagram shows a classification pipeline taking 5 numerical input matrices (X0 to X4) and 1 text input (X5).  Code for building this pipeline is given below.  The InputSelector and ModelSelector objects are highlighted in red because they provide functionality not present in sklearn, Dask, or spark-ML. The InputSelector "SelectKBestInputs" computes a matrix score for each input by aggregating univariate feature scores and then selects the k=3 best inputs.  The ModelSelector "SelectKBestPredictors" does an internal cross validation run within the training set during the call to pipeline.fit(Xs, y), estimates the accuracy of models trained on inputs 0 to 4, then selects the k=2 best models and sends their inferences on to a meta-classifier.

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
