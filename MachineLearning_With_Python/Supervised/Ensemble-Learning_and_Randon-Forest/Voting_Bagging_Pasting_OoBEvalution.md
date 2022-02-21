## Index
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
* [Voting Classifiers](#voting-classifiers)
   * [Voting Classifier Notebook](https://nbviewer.org/github/iAmKankan/MachineLearning_With_Python/blob/master/Supervised/Ensemble-Learning_and_Randon-Forest/votingClassifier.ipynb)
 * [Bagging and Pasting](#bagging-and-pasting)
 * [Out-of-Bag Evaluation](https://nbviewer.org/github/iAmKankan/MachineLearning_With_Python/tree/master/Supervised/Ensemble-Learning_and_Randon-Forest#out-of-bag-evaluation)
   * [Bagging-Pasting, Out-of-Box Evalution Notebook](https://nbviewer.org/github/iAmKankan/MachineLearning_With_Python/blob/master/Supervised/Ensemble-Learning_and_Randon-Forest/bagging_pasting.ipynb)


### _Voting Classifiers_
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* Suppose we have trained a few classifiers, each one achieving about 80% accuracy.Like- _a Logistic Regression classifier, an SVM classifier, a Random Forest classifier, a K-Nearest Neighbors classifier_, and many more.
* A very simple way to create an even better classifier is to aggregate the predictions of each classifier and predict the class that gets the most votes. 
* This majority-vote classifier is called a _**hard voting classifier**_
* **Voting Classifier** Voting classifier has two types. **Hard Voting Classifier** and **Soft Voting Classifier**.
* Voating Classicier works in **Ensemble** or **RandomForest** or in **Deep Learning**.
* Suppose in Random Forest classifier has two classes ***0*** and ***1*** and we have four Decision Trees **[D1,D2,D3,D4]** which gives the results- **[0, 1, 1, 1]**.
    * For **Hard Voting Classifier** the result would be ***1*** since the maximum models gives the output.
    * For **Soft Voting Classifier** it gives us the probablity. As shown in the below table, It take sthe higher probability scores.
    
| Decission Tree |Class **1** |Class **0** |
|----|-------|-------|
| D1 | .95 | .05 |
| D2 | .86 | .14 |
| D3 | .7 | .3 |
| D4 | .6 | .4 |


* The result is (.95+.86+.7+.6)/4 = 0.7775


### Bagging and Pasting
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* One way to get a diverse set of classifiers is to use very **_different training algorithms_**. [See Voting Classifier Notebook](https://nbviewer.org/github/iAmKankan/MachineLearning_With_Python/blob/master/Supervised/Ensemble-Learning_and_Randon-Forest/votingClassifier.ipynb)
* Another approach is to use the ***same training algorithm*** for every predictor and train them on ***different random subsets of the training set***.
* When **_sampling_** is performed **_with replacement_**, this method is called **`bagging`** (short for **`bootstrap aggregating`** ). 
* When **_sampling_** is performed **_without replacement_**, it is called **`pasting`**. 
* In other words, both `bagging` and `pasting` allow **_training instances to be sampled several times across multiple predictors_**, but only bagging allows training instances to be sampled several times for the same predictor.
> #### Note: The BaggingClassifier automatically performs soft voting instead of hard voting if the base classifier can estimate class probabilities (i.e., if it has a `predict_proba()` method), which is the case with Decision Tree classifiers.

> #### Difference
> * _**Bootstrapping**_ introduces a bit more diversity in the subsets that each predictor is trained on, so bagging ends up with **a slightly higher bias than pasting**; 
> * but the extra diversity also means that the predictors end up being **less correlated**, so the **ensemble’s variance is reduced**. 
> * **Overall**, _**bagging often results in better models**_. 
> * However, if you have spare time and CPU power, you can use cross-validation to evaluate both bagging and pasting and select the one that works best.


### Out-of-Bag Evaluation
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* With _bagging_, some instances may be sampled _several times_ for any given predictor, while others may _not be sampled at all_. 
* By default a `BaggingClassifier` samples `m` training instances with `replacement` (`bootstrap=True`), where _m_ is the size of the training set. 
* This means that only about 63% of the training instances are sampled on average for each predictor. 
* The remaining 37% of the training instances that are not sampled are called **_out-of-bag (oob) instances_**.
*  **Note**:  that they are not the same 37% for all predictors. 
*  Since a predictor never sees the oob instances during training, it can be evaluated on these instances, without the need for a separate validation set. 
*  You can evaluate the ensemble itself by averaging out the oob evaluations of each predictor. In Scikit-Learn, you can set oob_score=True when creating a BaggingClassifier to request an automatic oob evaluation after training. The following code demonstrates this. The resulting evaluation score is available through the oob_score_ variable

### Random Patches and Random Subspaces
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* The _`BaggingClassifier`_ class supports sampling the features as well. 
* Sampling is controlled by two hyperparameters: `max_features` and `bootstrap_features`. 
* They work the same way as `max_samples` and `bootstrap`, but for feature sampling instead of instance sampling. 
* Thus, each predictor will be trained on a random subset of the input features.

* Sampling both training instances and features is called the _**Random Patches method**_. 
* Keeping all training instances (by setting bootstrap=False and max_samples=1.0) but sampling features (by setting bootstrap_features to True and/or max_features to a value smaller than 1.0) is called the _**Random Subspaces method**_. 
