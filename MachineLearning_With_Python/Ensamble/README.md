## Ensemble Definition:


* Ensemble learning uses multiple machine learning models to try to make better predictions on a dataset.

* An ensemble model works by training different models on a dataset and having each model make predictions individually.

* The predictions of these models are then combined in the ensemble model to make a final prediction.

* The error emerging from any machine model can be broken down into three components mathematically. following are these component
    * **BIAS + VARIANCE + IRREDUCIBLE ERROR**
    
    
    
* **Bias** : BIAS IS A LEARNER’S TENDENCY TO CONSISTENTLY LEARN THE SAME WRONG THING. (Underfitting)

* Bias error is useful to quantify how much on an average are the predicted values different from the actual value. A high bias error means we have an under performing model which keeps on missing essential trends.



* **Varience** :  VARIANCE IS THE TENDENCY TO LEARN RANDOM THINGS IRRESPECTIVE OF THE REAL SIGNAL.(Overfitting) 

* Variance on the other side quantifies how are the prediction made on the same observation different from each other. a high variance model will over fit on your training population and perform poorly on any observation beyond training.




* Typically, as we increase the complexity of our model, you will see a reduction in error due to lower bias in the model. However, this only happens until a particular point. as we continue to make our model more complex, we end up over fitting our model, and hence our model will start suffering from the high variance.


* THERE ARE THREE MAIN TERMS DESCRIBING THE ENSEMBLE (COMBINATION) OF VARIOUS MODELS INTO ONE MORE EFFECTIVE MODEL:
    * **BAGGING** TO DECREASE THE MODEL’S VARIANCE.
    * **BOOSTING** TO DECREASING THE MODEL’S BIAS, AND.
    * **STACKING** TO INCREASING THE PREDICTIVE FORCE OF THE CLASSIFIER.



* The three most popular methods for combining the predictions from different models are:

    * BAGGING : Building multiple models (typically of the same type) from different subsamples of the training dataset.
    * BOOSTING : Building multiple models (typically of the same type) each of which learns to fix the prediction errors of a prior model in the chain.
    * VOTING : Building multiple models (typically of differing types) and simple statistics (like calculating the mean) are used to combine predictions.
    
    
* Bagging is one of the ensemble construction techniques which is also known as BOOTSTRAP AGGREGATION . Bootstrap establishes the foundation of bagging technique. 
* After the bootstrapped samples are formed, separate models are trained with the bootstrapped samples. 
* In real experiments, the bootstrapped samples are drawn from the training set, and the sub models are tested using the testing set. the final output prediction is combined across the projections of all the sub models.  





