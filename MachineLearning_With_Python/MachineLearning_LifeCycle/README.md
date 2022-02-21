## Index
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
* [Data Preprocessing](https://github.com/iAmKankan/Data-Gathering-And-Preprocessing/blob/main/README.md)
* [Regularization](https://github.com/iAmKankan/Regularization/blob/master/README.md)
* [Optimizer](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/training/README.md)
## Lifecycle of ML
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
<img src="https://user-images.githubusercontent.com/12748752/141670973-047dd271-d037-4004-81c0-64c5dca810d3.png" width=60% />
## Lifecycle of data and ML 
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)

### 1-Categorize the problem
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* **Categorize by the input:** 
    - If it is a labeled data, it’s a supervised learning problem. 
    - If it’s unlabeled data with the purpose of finding structure, it’s an unsupervised learning problem. 
    - If the solution implies to optimize an objective function by interacting with an environment, it’s a reinforcement learning problem.

* **Categorize by output:**
    - If the output of the model is a number, it’s a regression problem.
    - If the output of the model is a class, it’s a classification problem. 
    - If the output of the model is a set of input groups, it’s a clustering problem.
    
### 2-Understand Your Data
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* The process of understanding the data plays a key role in the process of choosing the right algorithm for the right problem. * Some algorithms can work with smaller sample sets while others require tons and tons of samples.

* **Analyze the Data**
* **Process the data**
* **Transform the data**
    - Transform data and feature engineering may, in fact, be synonyms. 
    - **Feature engineering**
     - Is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data. -By Jason Brownlee.
        
        
### 3-Find the available algorithms     
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

- The accuracy of the model.
- The interpretability of the model.
- The complexity of the model.
- The scalability of the model.
- How long does it take to build, train, and test the model?
- How long does it take to make predictions using the model?
- Does the model meet the business goal?




### 4-Implement machine learning algorithms.
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* Set up a machine learning pipeline that compares the performance of each algorithm on the dataset using a set of carefully selected evaluation criteria.
* Another approach is to use the same algorithm on different subgroups of datasets.
* The best solution for this is to do it once or have a service running that does this in intervals when new data is added.

### 5-Optimize hyperparameters
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

- There are three options for optimizing hyperparameters-
    - Grid search
    - Random search and 
    - Bayesian optimization.
    
    
### Model Evoluation 
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

   - Train/test split
   - Cross-Validation

![Train_test_Flow](https://user-images.githubusercontent.com/12748752/141670977-ae3e17a8-0636-4fb6-a052-cf4de904f5a9.png)


#### Train/test split
* Split the dataset into two pieces, so that the model can be trained and tested on different data
* Testing accuracy is a better estimate than training accuracy of out-of-sample performance

#### Problem with train/test split
* It provides a high variance estimate since changing which observations happen to be in the testing set can significantly change testing accuracy.
* Testing accuracy can change a lot depending on a which observation happen to be in the testing set.



#### Cross-Validation		
* When only a limited amount of data is available, to achieve an unbiased estimate of the model performance we use k-fold cross-validation. 
* In k-fold cross-validation, we divide the data into k subsets of equal size. We build models k times, each time leaving out one of the subsets from training and use it as the test set. 
* If k equals the sample size, this is called "leave-one-out".

### 6-Fine-Tune the System
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

*  You will want to use as much data as possible for this step, especially as you move toward the end of fine-tuning.
* As always automate what you can.

* Fine-tune the hyperparameters using cross-validation.
    - Treat your data transformation choices as hyperparameters, especially when you are not sure about them (e.g., should I replace missing values with zero or with the median value? Or just drop the rows?).
    - Unless there are very few hyperparameter values to explore, prefer random search over grid search. If training is very long, you may prefer a Bayesian optimization approach (e.g., using Gaussian process priors, as described by Jasper Snoek, Hugo Larochelle, and Ryan Adams).
    
* Try Ensemble methods. Combining your best models will often perform better than running them individually.

* Once you are confident about your final model, measure its performance on the test set to estimate the generalization error.


## [Model Performance Evaluation - Classification](https://www.saedsayad.com/model_evaluation_c.htm)


#### 1) Confusion Matrix:
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

* A confusion matrix shows the number of correct and incorrect predictions made by the classification model compared to the actual outcomes (target value) in the data. 
* The matrix is NxN, where N is the number of target values (classes). 
* Performance of such models is commonly evaluated using the data in the matrix. The following table displays a 2x2 confusion matrix for two classes (Positive and Negative).



* **Classification reports**
    - Precision Score
    - Recall Score
    - F1 Score
    - Support
    - ConfusionMatrix



![confusion](https://user-images.githubusercontent.com/12748752/141670970-d95a8f0d-550e-4c66-9740-84e06ef269b7.jpg)
   
![accuracy](https://user-images.githubusercontent.com/12748752/141670978-009b1fe1-9dd3-496e-830f-7514383e3cd1.png)
 
 ---
![precision](https://user-images.githubusercontent.com/12748752/141670975-9bb6438b-f858-4b73-83d3-015d10f44027.png)

![recall](https://user-images.githubusercontent.com/12748752/141670976-318580ca-12e4-474b-95be-8d01862b50ca.png)

![f1](https://user-images.githubusercontent.com/12748752/141670972-4a4c0e56-8b64-4f35-a33c-89b31da196d1.png)



#### 2) Gain and Lift Charts
#### 3) Lift Chart
#### 4) K-S Chart 
#### 5) ROC Chart
#### 6) Area Under the Curve (AUC)
---

## Model Performance Evaluation - Regression


#### 1) Root Mean Squared Error		
RMSE is a popular formula to measure the error rate of a regression model. However, it can only be compared between models whose errors are measured in the same units.		




#### 2) Relative Squared Error

Unlike RMSE, the relative squared error (RSE) can be compared between models whose errors are measured in the different units.		


#### 3) Mean Absolute Error		
The mean absolute error (MAE) has the same unit as the original data, and it can only be compared between models whose errors are measured in the same units. It is usually similar in magnitude to RMSE, but slightly smaller.		


#### 4) Relative Absolute Error		
Like RSE , the relative absolute error (RAE) can be compared between models whose errors are measured in the different units.		


#### 5) Coefficient of Determination		
The coefficient of determination (R2) summarizes the explanatory power of the regression model and is computed from the sums-of-squares terms.		


R2 describes the proportion of variance of the dependent variable explained by the regression model. If the regression model is “perfect”, SSE is zero, and R2 is 1. If the regression model is a total failure, SSE is equal to SST, no variance is explained by regression, and R2 is zero.		
 		
#### 6) Standardized Residuals (Errors) Plot		
The standardized residual plot is a useful visualization tool in order to show the residual dispersion patterns on a standardized scale. There are no substantial differences between the pattern for a standardized residual plot and the pattern in the regular residual plot. The only difference is the standardized scale on the y-axis which allows us to easily detect potential outliers.		

---


