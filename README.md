# pipecaster-sklearn
(in progress)

A Python library for broadcasting machine learning (ML) pipeline construction operations across multiple input sources.  The library provides convenient slicing notation and a Keras-like layered workflow with an sklearn interface.

## silo ML inputs

Many ML libraries implicitly encourage concatenation of features from multiple data sources into a single feature matrix prior to feature selection or ML.  In practice, concatenation often reduces performance and greater predictive accuracy can be obtained by siloing the different inputs through the initial feature selection and ML steps and combining inferences at a later stage using voting or stacked generalization.  Pipecaster encourages input silos by modifying the sklearn interface from:    
`model.fit(X, y).predict(X)`  
to   
`model.fit(Xs, y).predict(Xs).`

## keep workflows fast and pipelines clean

In addition, pipecaster adds in-pipeline input selection to keep garbage from flowing into and out of your ML models and in-pipeline model selection to semi-automate the ML workflow for high throughput ML applications.

# use case example 1
![Use case 1](/images/example_1.png)

This diagram shows a classification pipeline taking 5 numerical input matrices (X0 to X4) and 1 text input (X5).  Code for building this pipeline is given below.  The InputSelector and ModelSelector objects are highlighted in red because they provide functionality not present in sklearn or spark-ML. The InputSelector "SelectKBestInputs" computes a matrix score for each input by aggregating univariate feature scores and then selects the k=3 best inputs.  The ModelSelector "SelectKBestPredictors" does an internal cross validation run within the training set during the call to pipeline.fit(Xs, y), estimates the accuracy of models trained on inputs 0 to 4, then selects the k=2 best models and sends their inferences on to a meta-classifier.

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
