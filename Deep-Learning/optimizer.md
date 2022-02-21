## Index
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)
* [Optimization](#optimization)
* [Goal of Optimization](#goal-of-optimization)
* [Training Error Optimization](#training-error-optimization)
* [Types of Optimizer](#types-of-optimizer)
   * [Gradient Descent](https://github.com/iAmKankan/Deep-Learning/blob/master/gradient-descent.md)
   * [Stochastic Gradient Descent](#)
   * [Minibatch Stochastic Gradient Descent](#)
   * [Momentum](#)
   * [Adagrad](#)
   * [RMSProp](#)
   * [Adadelta](#)
   * [Adam](#)
   * [Learning Rate Scheduling](#)
* [Optimization Challenges in Deep Learning](#challenges)
   * [Local Minima](#)
   * [Saddle Points](#)
   * [Vanishing Gradients](#)
## Optimization 
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)
* In Statistics, Machine Learning and other Data Science fields, we optimize a lot of stuff.
> #### Linear Regression we optimize
> *  _**Intercept**_ 
> *  _**Slope**_
<img src="https://user-images.githubusercontent.com/12748752/139344656-8e5f34a2-608d-45d5-90a9-0dc4676692e9.png" width=30%>

> #### When we use Logistic Regression we optimize 
> * _**A Squiggle**_
<img src="https://user-images.githubusercontent.com/12748752/139344662-2edb7ae2-2ee9-43d0-9bec-5d099e62bce5.png" width=30%>

> #### When we use t-SNE we optimize 
> * _**Clusters**_
### Goal of Optimization
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

> ### **The goal of Optimization is primarily concerned with minimizing an objective(loss function)**
> * The goal of optimization is to reduce the _**Training Error**_.


> ### **The goal of Deep Learning is finding a suitable model, given a finite amount of data.**
> * The goal of deep learning (or more broadly, statistical inference) is to reduce the _**Generalization Error**_. 
> * To accomplish so we need to pay attention to **overfitting** in addition to using the **optimization algorithm** to reduce the training error.

* **Empirical Risk:** The empirical risk is an average loss on the training dataset.
*  **Risk:**: The risk is the expected loss on the entire population of data. 

### Training Error Optimization
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)

* In Deep Learning optimizers are algorithms or methods used to change the attributes of the neural network such as **weights**, **Bias** and **learning rate** to reduce the losses. 
> <img src="https://latex.codecogs.com/svg.image?\\&space;W\&space;=\&space;W\&space;&plus;&space;\&space;\Delta&space;W&space;\\b\&space;=\&space;W\&space;&plus;&space;\&space;\Delta&space;b&space;\\&space;\Delta&space;W\&space;=\&space;-\eta&space;\nabla&space;c\&space;;\&space;\mathrm{[\eta=\&space;Learning\&space;rate,&space;\nabla&space;c=\&space;minimizing\&space;error&space;]}&space;\\&space;\\\Delta&space;W\&space;=\&space;-\eta&space;\frac{\partial&space;c}{\partial&space;w}&space;\&space;\mathrm{[Gredient\&space;Descent&space;]}" title="\\ W\ =\ W\ + \ \Delta W \\b\ =\ W\ + \ \Delta b \\ \Delta W\ =\ -\eta \nabla c\ ;\ \mathrm{[\eta=\ Learning\ rate, \nabla c=\ minimizing\ error ]} \\ \\\Delta W\ =\ -\eta \frac{\partial c}{\partial w} \ \mathrm{[Gredient\ Descent ]}" />

* Optimizers are used to solve optimization problems by minimizing the function.

* For a deep learning problem, we usually define a *loss function* first.
* Once we have the loss function, we can use an optimization algorithm in attempt to minimize the loss.
* In optimization, a loss function is often referred to as the *objective function* of the optimization problem. 
* In Deep Learning after calculationg weights  


## Optimization Challenges in Deep Learning
![dark](https://user-images.githubusercontent.com/12748752/136802585-2ef5b7ff-ddbc-417f-b963-ca233db3ded1.png)
* Here, we are going to focus specifically on the performance of optimization algorithms in minimizing the objective function, rather than a model’s generalization error.
> #### There are a clear difference between **analytical solutions** and **numerical solutions** in optimization problems. 
* In deep learning, most objective functions are complicated and do not have analytical solutions.
* The optimization algorithms all fall into **numerical optimization algorithms**.

> #### There are many challenges in deep learning optimization. Some of the most vexing ones are-
>> local minima
>>
>> saddle points
>> 
>> vanishing gradients

###  Local Minima
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* For any objective function  _f(x)_ , if the value of  _f(x)_  at  _x_  is smaller than the **values** of  *f(x)*  **at any other points in the vicinity of  x**. 
* Then  f(x)  could be a local minimum. 
* [A function _f(x)_ has a local minima at <img src="https://latex.codecogs.com/svg.image?\mathrm{x_0}" title="\mathrm{x_0}" />, if the value of <img src="https://latex.codecogs.com/svg.image?f(\mathrm{x_0})" title="f(\mathrm{x_0})" /> is smaller than the other values of *f*(x) of different points of x]

* If the value of  f(x)  at  x  is the minimum of the objective function over the entire domain, then  f(x)  is the global minimum.
![](https://d2l.ai/_images/output_optimization-intro_70d214_45_0.svg)

* The objective function of deep learning models usually has many local optima. 
* When the numerical solution of an optimization problem is near the local optimum, the numerical solution obtained by the final iteration may only minimize the objective function *locally*, rather than *globally*, as the gradient of the objective function's solutions approaches or becomes zero.
* Only some degree of noise might knock the parameter out of the local minimum.
* In fact, this is one of the beneficial properties of minibatch stochastic gradient descent where the natural variation of gradients over minibatches is able to dislodge the parameters from local minima.


### Saddle Points
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* Besides local minima, saddle points are another reason for gradients to vanish.
* A saddle point is any location where all gradients of a function vanish but which is neither a global nor a local minimum. 


### Vanishing Gradients
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* Probably the most insidious problem to encounter is the vanishing gradient.
* Recall our commonly-used activation functions and their derivatives in Section 4.1.2. For instance, assume that we want to minimize the function  f(x)=tanh(x)  and we happen to get started at  x=4 .
* As we can see, the gradient of  f  is close to nil. More specifically,  f′(x)=1−tanh2(x)  and thus  f′(4)=0.0013 .

### Summary
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* Minimizing the training error does *not* guarantee that we find the best set of parameters to minimize the generalization error.
* The optimization problems may have many local minima.
* The problem may have even more saddle points, as generally the problems are not convex.
* Vanishing gradients can cause optimization to stall. Often a reparameterization of the problem helps. Good initialization of the parameters can be beneficial, too.



