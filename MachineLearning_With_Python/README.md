### Index 
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
*  [What is Machine Learning](#what-is-machine-learning)
*  [Machine Learning Life Cycle](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/MachineLearning_LifeCycle/README.md)
* [Supervised Learning](https://github.com/iAmKankan/MachineLearning_With_Python/tree/master/Supervised) 
* [Unsupervised Learning](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/unsupervised/README.md)
* [Terminologies](#terminologies)
   * [Bias and Varience](#bias-and-varience)
     * [Regularization]()
     * [Boosting]()
     * [Bagging]()
   * [Overfitting And Underfitting](#overfitting-and-underfitting)
   * [MODEL INTERPRETATION](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/README.md#model-interpretation)
      * [WHITE BOX Models](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/README.md#white-box-models)
      * [BLACK BOX Models](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/README.md#black-box-models)
   * [Curse of Dimensionlity](https://github.com/iAmKankan/Data-Gathering-And-Preprocessing/blob/main/README.md#curse-of-dimentionlity)
   * [Linear and Non-Linear Classifications](#linear-classification-vs-non-linear-classification)
   * [Stratified Sampling](#stratified-sampling)
   * [Out-of-Core Algorithm](#out-of-core-algorithm)


## What is Machine Learning
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
* Machine Learning is the science (and art) of programming computers so they can _learn from data_.
* Here is a slightly more general definition:
> #### [Machine Learning is the] field of study that gives computers the ability to learn without being explicitly programmed.
_— Arthur Samuel, 1959_
* And a more engineering-oriented one: 
> #### A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. 
— _Tom Mitchell, 1997_
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
### Machine Learning Algorithms Based on Trainging
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* Machine Learning systems can be classified according to the amount and type of supervision they get during training. 
* There are four major categories:

<img src="https://user-images.githubusercontent.com/12748752/150331825-26bc70db-8c11-4d0a-bf54-445d32cc03a8.png" width=100% />

### Based on the Performence
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

<img src="https://user-images.githubusercontent.com/12748752/146688585-e2ecc295-c3e0-4ea8-acb1-ad03a8ec8ce6.png" width=110% />
 

## Terminologies
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)

### Bias and Varience
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* Suppose we have some observations(data) and they are split into train data and test data.
> #### In the first scenario
* Like Linear Regression we draw a regression line through the training observations and obhously that straight line cann't cover each and every observation.
* The minimum of the sum of the squre of the difference between the line and each observation is the error.
* That scenario is called Bias.
> #### In the second scenario
* We draw a squiggly line which connects each and every training observations.
* Now the error is zero.
* But when we run test data, the error for train data and the test data of 1st scenario is almost the same. The error for train data which is zero but the test data of 2nd scenario is different. Here difference between the train and the test is a consequence of variance.
* The 1st scenario is having high Bias but it has low varience because the sum of squre is similer for the different data sets and it performence is consistent.
* Because the squiggly line is fitted in train set but not in test set we say it is **Overfit** and the performance is not consistence.
* In machine learning, the ideal algorithm has low bias and can accurately model the true relationship and it has low variability, by producing consistent predictions across different datasets.
* Three commonly used methods for finding the sweet spot between simple and complicated models are:
  * [**Regularization.**](https://github.com/iAmKankan/Regularization/blob/master/README.md)
  * [**Boosting.**](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/Supervised/Ensemble-Learning_and_Randon-Forest/boosting.md)
  * [**Bagging.**](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/Supervised/Ensemble-Learning_and_Randon-Forest/Voting_Bagging_Pasting_OoBEvalution.md)
### Overfitting And Underfitting
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* Overfitting and Underfitting is direct related to Bias and Varience.
* In simple words-
  * **Overfitting** means you have trained your model havely that it gives good accuracy with training datya but failing miserably with test data.
  * **Underfit** means you have not trained your data good enough to get a decent accuracy.

## MODEL INTERPRETATION
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
* `Decision Trees` are intuitive, and their decisions are easy to interpret. 
* Such models are often called white box models. 
* `Random Forests` or `neural networks` are generally considered black box models. 
* They make great predictions, and you can easily check the calculations that they performed to make these predictions; nevertheless, it is usually hard to explain in simple terms why the predictions were made. 
* For example, if a neural network says that a particular person appears on a picture, it is hard to know what contributed to this prediction: did the model recognize that person’s eyes? Their mouth? Their nose? Their shoes? Or even the couch that they were sitting on? Conversely, Decision Trees provide nice, simple classification rules that can even be applied manually if need be (e.g., for flower classification).
### WHITE BOX Models
* Interpretable models are models who explain themselves, for instance from a decision tree you can easily extract decision rules. 
   * **Linear Regression**
   * **Logistic Regression**
   * **Desission Tree**
* Banking sector and Healthcare sector it is very common.
### BLACK BOX Models
* Every other model than interpretable models are fall under this catagory
  * **Deep Learning** models are fully Black-Box models.
  *  **Random Forest** is fully Black-Box model.
  
  
  

### Linear Classification Vs Non-Linear Classification
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)

#### Linear Classifier
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* Let’s say we have data from two classes (o and  χ ) distributed as shown in the figure below. 
* To discriminate the two classes, one can draw an arbitrary line, s.t. all the ‘o’ are on one side of the line and  χ ’s on the other side of the line. 
* These two classes are called linearly-separable.

<img src="https://user-images.githubusercontent.com/12748752/144843896-495dcb21-0fda-4887-ad92-63581c467e52.gif" width=20%/>

* However, there are infinite number of lines that can be drawn to discriminate two separable classes, as shown below
<img src="https://user-images.githubusercontent.com/12748752/144843889-fc4494a5-7493-4220-99a5-094c7a6ff107.png" width=20%/>

* How you approximate the exact location of this discriminating line (or plane or hyperplane) depends on the type of a classifier called linear classifier.
* Some examples of linear classifier are: `Linear Discriminant Classifier`, `Naive Bayes`, `Logistic Regression`, `Perceptron`, `SVM (with linear kernel)`.


#### Non-Linear Classifier
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* Let’s consider the famous XOR problem. An XOR’s truth table and the data distribution is shown below.
* In this situation, there is no way that a straight line can be drawn to discriminate the two classes (0 and 1)

<img src="https://user-images.githubusercontent.com/12748752/144843899-ea16eeac-2329-4664-8912-60f26d9cfaff.gif"/>

### _Stratified Sampling_
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* _`Random Sampling`_ methods is generally fine if the dataset is large enough (especially relative to the number of attributes), but if it is not, you run the risk of introducing a _**significant sampling bias**_. 
* For example, the US population is 51.3% females and 48.7% males, so a well-conducted survey in the US would try to maintain this ratio in the sample: 513 female and 487 male.
* This is called **_`Stratified Sampling`_**: 
   * The population is divided into homogeneous subgroups called _`strata`_, and the right number of instances are sampled from each _`stratum`_ to guarantee that the test set is representative of the overall population. 
   * If the people running the survey used purely random sampling, there would be about a 12% chance of sampling a skewed test set that was either less than 49% female or more than 54% female. Either way, the survey results would be significantly biased.
### Out-of-Core Algorithm
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* An algorithm which exploits external storage in order to support large data volumes that cannot be supported by primary memory. 
