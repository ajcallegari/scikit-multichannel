# pipecaster-sklearn
(in progress)

A Python library for broadcasting machine learning pipeline construction operations across multiple input sources.  The library features the sklearn interface, convenient slice notation, and a Keras-like layered workflow.

## silo ML inputs

Many ML workflows implicitly encourage concatenation of features from multiple data sources prior to feature selection or ML.  In practice, concatenation often reduces performance and greater predictive accuracy is obtained by siloing the different inputs through the initial feature selection and ML steps and combining inferences at a later stage using voting or stacked generalization.  Pipecaster encourages input silos by modifying the sklearn interface from:    
`model.fit(X, y).predict(X)`  
to   
`model.fit(Xs, y).predict(Xs).`

## keep workflows fast and pipelines clean

In addition, pipecaster adds in-pipeline input source selection to keep garbage from flowing into and of your ML models and in-pipeline model selection to automate parts of the ML workflow for high throughput ML applications.

# use case example 1
![Use case 1](/images/example_1.png)

This diagram shows a classification pipeline taking 5 numerical input matrices and 1 text input.  Code for creating this pipeline is given below.  The InputSelector and ModelSelector objects are highlighted in red because they provide functionality not present in sklearn or spark-ML. The InputSelector "SelectKBestInputs" computes a matrix score for each input by aggregating univariate feature scores and then selects the 3 best inputs.  The ModelSelector "SelectKBestPredictors" does an internal cross validation run within the training set during the calls to cls.fit(Xs, y), assesses the accuracy of models trained on inputs 0 to 4, then selects the two best models and sends their inferences on to a meta-classifier.

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
