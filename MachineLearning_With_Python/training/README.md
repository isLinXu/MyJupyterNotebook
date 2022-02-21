## Index
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
* [Gradient Descent](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/training/gradient_descent.md)
   * [Batch Gradient Descent](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/training/batch_gd.md)
   * [Stochastic Gradient Descent](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/training/stochastic_gradient_descent.md)
* [Terminologies](#terminologies)
   * [Convergence](#convergence)
   * [Theta _<a>&theta;</a>_](#theta-%CE%B8)
### Different types of optimizers and its performence   
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
|Algorithm| Large m|Out-of-core support|Large n| Hyperparams| Scaling required| Scikit-Learn|
|---|---|---|---|---|---|---|
|Normal Equation| Fast| No| Slow| 0| No| N/A|
SVD| Fast| No| Slow| 0| No| LinearRegression
Batch GD| Slow| No| Fast| 2| Yes| SGDRegressor
Stochastic GD| Fast| Yes| Fast| ≥2| Yes| SGDRegressor
Mini-batch GD| Fast| Yes| Fast| ≥2| Yes| SGDRegressor

  
   
   
   
## Terminologies
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
### _Convergence_
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* Once the agent, after many steps, realize the cost does not improve by a lot and it is stuck very near a particular point (minima), technically this is known as **_`convergence`_**. 
* The value of the parameters at that very last step is known as the ‘best’ set of parameters, and we have a trained model.

### _Theta <a>&theta;</a>_
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* By convention, the Greek letter **_<a>&theta;</a>_ (theta)** is frequently used to represent model parameters.
