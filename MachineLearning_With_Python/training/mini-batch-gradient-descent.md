## Index
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
## Mini-batch Gradient Descent
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
* At each step, instead of computing the gradients based on the full training set (as in Batch GD) or based on just one instance (as in Stochastic GD), 
* _**Mini-batch Gradient Descent**_ computes the gradients on small random sets of instances called mini-batches. 
* The main advantage of _Minibatch Gradient Descent_ over _Stochastic Gradient Descent_ is that you can get a performance boost from hardware optimization of matrix operations, especially when using GPUs.
* The algorithm’s progress in parameter space is less erratic than with Stochastic GD, especially with fairly large mini-batches.
* As a result, Minibatch GD will end up walking around a bit closer to the minimum than Stochastic GD—but it may be harder for it to escape from local minima (in the case of problems that suffer from local minima, unlike Linear Regression).
<img src="https://user-images.githubusercontent.com/12748752/147012339-32c5604d-c305-4ea0-b985-66486e1b046c.png" width = 45% />
