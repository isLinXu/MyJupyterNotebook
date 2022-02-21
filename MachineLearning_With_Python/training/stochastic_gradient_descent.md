## Index
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

## Stochastic Gradient Descent
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
* The main problem with **_Batch Gradient Descent_** is the fact that it uses the _whole training set_ to compute the gradients _at every step_, which makes it very slow when the training set is large. 
* At the opposite extreme, **_Stochastic Gradient Descent_** picks a _random instance_ in the training set at every step and computes the gradients based only on that _single instance_. 
* Obviously, working on a _single instance at a time_ makes the algorithm much faster because it has very little data to manipulate at every iteration. 
* It also makes it possible to train on huge training sets, since _only one instance needs to be in memory at each iteration_ (Stochastic GD can be implemented as an [outof- core algorithm](https://github.com/iAmKankan/MachineLearning_With_Python/blob/master/README.md#out-of-core-algorithm)).
* On the other hand, due to its _**stochastic**_ (i.e., random) nature, this algorithm is much less regular than **_Batch Gradient Descent_**: instead of gently decreasing until it reaches the minimum, the cost function will bounce up and down, decreasing only on average. 
* Over time it will end up very close to the minimum, but once it gets there it will continue to bounce around, never settling down. 
* So once the algorithm stops, the final parameter values are good, but not optimal.
<img src="https://user-images.githubusercontent.com/12748752/147009180-f4b49dc4-3e70-421e-bed2-4d6de3034532.png" width=30% />

* With _**Stochastic Gradient Descent**_, each training step is much faster but also much more _stochastic_ than when using _Batch Gradient Descent_
* When the cost function is very irregular, this can actually help the algorithm jump out of local minima, so **_Stochastic Gradient Descent_** has a better chance of finding the global minimum than **_Batch Gradient Descent_** does.

* One advantage is that the frequent updates allow us to have a pretty detailed rate of improvement. The frequent updates are more computationally expensive as the approach of Batch Gradient Descent.
* The frequency of those updates can also result in noisy gradients, which may cause the error rate to jump around, instead of slowly decreasing.
