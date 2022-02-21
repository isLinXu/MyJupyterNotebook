## Index
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

## The Idea
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* Gradient Descent is a _`generic optimization algorithm`_ capable of finding optimal solutions to a wide range of problems. 
* The general idea of _**Gradient Descent**_ is to tweak parameters iteratively in order to minimize a _cost function_. 

> ### Example 01
* _Suppose you are lost in the mountains in a dense fog, and you can only feel the slope of the ground below your feet_. 
* _A good strategy to get to the bottom of the valley quickly is to go downhill in the direction of the steepest slope_. 
* This is exactly what Gradient Descent does: it measures the local gradient of the error function with regard to the _`parameter vector`_ **_<a>&theta;</a>_**, and it goes in the direction of descending gradient. 
* Once the gradient is zero, you have reached a minimum!

> ### Steps
* **First**, start by filling **_<a>&theta;</a>_** with random values (this is called _`random initialization`_). 
* **Then** you improve it gradually, taking one baby step at a time, each step attempting to decrease the cost function (e.g., the MSE), until the algorithm converges to a minimum.

<img src="https://user-images.githubusercontent.com/12748752/146699335-d3b9e07e-1fe4-43d0-a805-14481610dc7e.png" width=40%/>

> In this depiction of Gradient Descent, the model parameters are initialized randomly and get tweaked repeatedly to minimize the cost function; the learning step size is proportional to the slope of the cost function, so the steps gradually get smaller as the parameters approach the minimum
### _`Learning rate`_
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
#### _Too Small_
<img src="https://user-images.githubusercontent.com/12748752/146699334-39c448ea-cc58-41fc-a85e-9e9cf56d72f2.png" width=40%/>

* An important parameter in Gradient Descent is the size of the steps, determined by the _`learning rate`_ hyperparameter. 
* If the learning rate is too small, then the algorithm will have to go through many iterations to converge, which will take a long time.
* On the other hand, if the learning rate is too high, you might jump across the valley and end up on the other side, possibly even higher up than you were before. 
* This might make the algorithm diverge, with larger and larger values, failing to find a good solution.
#### _Too Large_
<img src="https://user-images.githubusercontent.com/12748752/146699332-7793518a-29dd-4fab-94b5-746f0ff232c7.png" width=40% />

### The Problem
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
<img src="https://user-images.githubusercontent.com/12748752/146699336-bfff17ac-fcdf-48a5-9527-5c7ff8a89634.png" width=40%/>
* If the random initialization starts the algorithm on the left, then it will converge to a local minimum, which is not as good as the global minimum. 
* If it starts on the right, then it will take a very long time to cross the plateau. 
* And if you stop too early, you will never reach the global minimum.
### GD in Linear Regression
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* Fortunately, the MSE cost function for a Linear Regression model happens to be a convex function, which means that if you pick any two points on the curve, the line segment joining them never crosses the curve. 
* This implies that there are no local minima, just one global minimum.
* It is also a continuous function with a slope that never changes abruptly. 
* These two facts have a great consequence: Gradient Descent is guaranteed to approach arbitrarily close the global minimum (if you wait long enough and if the learning rate is not too high).

> ### When using _`Gradient Descent`_ , you should ensure that all features have a similar scale (e.g., using Scikit-Learn’s _StandardScaler_ class), or else it will take much longer to converge.
