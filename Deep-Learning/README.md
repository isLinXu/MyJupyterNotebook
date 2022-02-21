## Index
![dark](https://user-images.githubusercontent.com/12748752/136656705-67e1f667-b192-4ce3-a95a-97dc1b982fd8.png)
* [Neural Network](#neural-network)
   * [The Perceptron](#the-perceptron)
     * [Perceptron learning Rule or Update Weights](#perceptron-learning-rule-or-update-weights)
   * [Bias](#bias)
* [Activation Function](https://github.com/iAmKankan/Deep-Learning/blob/master/activation.md)
   * [Sigmoid]()
* [Optimizer](https://github.com/iAmKankan/Deep-Learning/blob/master/optimizer.md)
   * [Gradient Descent](https://github.com/iAmKankan/Deep-Learning/blob/master/gradient-descent.md)
   * [Stochastic Gradient Descent](#)
   * [Minibatch Stochastic Gradient Descent](#)
   * [Momentum](#)
   * [Adagrad](#)
   * [RMSProp](#)
   * [Adadelta](#)
   * [Adam](#)
   * [Learning Rate Scheduling](#)

## Deep Learning
![dark](https://user-images.githubusercontent.com/12748752/136656705-67e1f667-b192-4ce3-a95a-97dc1b982fd8.png)
* Deep learning is a more approachable name for an artificial neural network. 
* The “deep” in deep learning refers to the depth of the network a.k.a Hidden layers. But an artificial neural network can also be very shallow.

### Neural Network
![light](https://user-images.githubusercontent.com/12748752/136656706-ad904776-3e69-4a32-bc28-edfc9fd41cf7.png)
* Neural networks are inspired by the structure of the cerebral cortex.
* At the basic level is the **Perceptron**, the mathematical representation of a biological neuron.
* Like in the cerebral cortex, there can be several layers of _interconnected perceptrons_.

### The Perceptron
![light](https://user-images.githubusercontent.com/12748752/136656706-ad904776-3e69-4a32-bc28-edfc9fd41cf7.png)

* Its the simplest ANN architecture. It was invented by Frank Rosenblatt in 1957 and published as `Rosenblatt, Frank (1958), The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain, Cornell Aeronautical Laboratory, Psychological Review, v65, No. 6, pp. 386–408. doi:10.1037/h0042519`
 
* Lets see the architecture shown below - 
    
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/ArtificialNeuronModel_english.png/1024px-ArtificialNeuronModel_english.png" width=40%> 
* Common activation functions used for Perceptrons are (with threshold at 0)- 
<img src="https://latex.codecogs.com/svg.image?step(z)\&space;or\&space;heaviside(z)&space;=\begin{cases}0&space;&&space;z<0\\&space;1&space;&&space;z\geq&space;0\end{cases}&space;" title="step(z)\ or\ heaviside(z) =\begin{cases}0 & z<0\\ 1 & z\geq 0\end{cases} " />

* Common activation functions used for Perceptrons are (with threshold at 0)- 

> <img src="https://latex.codecogs.com/svg.image?z\&space;=&space;\&space;X_1W_1&plus;X_2W_2&plus;X_3W_3&space;\&space;\&space;\textit{,&space;or&space;we&space;can&space;write&space;it&space;as&space;}\&space;\&space;\&space;z\&space;=&space;\&space;\sum_{i=1}^{n}&space;X_iW_i&space;&space;" title="z\ = \ X_1W_1+X_2W_2+X_3W_3 \ \ \textit{, or we can write it as }\ \ \ z\&space;=&space;\&space;\sum_{i=1}^{n} X_iW_i &space;" width=70% />

* If we want to multiply W and X we will end up with two matrices-
* For multiplication of 2 matrices we need to have 1st matrix column= 2nd matrix row thats why we take transpose of matrix W -> <img src="https://latex.codecogs.com/svg.image?W^T" title="W^T" />

> <img src="https://latex.codecogs.com/svg.image?z\&space;=&space;\&space;X_1W_1&plus;X_2W_2&plus;X_3W_3&space;---------(1)\\W^T=\begin{bmatrix}&space;W_1&W_2&space;&space;&&space;W_3&space;\\\end{bmatrix}_{(nXm)},&space;X=\begin{bmatrix}&space;X_1\\X_2\\&space;X_3\end{bmatrix}_{(mXn)}" title="z\ = \ X_1W_1+X_2W_2+X_3W_3 \\W^T=\begin{bmatrix} W_1&W_2 & W_3 \\\end{bmatrix}_{(nXm)}, X=\begin{bmatrix} X_1\\X_2\\ X_3\end{bmatrix}_{(mXn)}" />

> <img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}W_1&space;\\W_2\\W_3\\\end{bmatrix}^{T}\begin{bmatrix}X_1&space;\\X_2\\X_3\\\end{bmatrix}&space;\&space;\&space;=\&space;\&space;W^TX" title="\begin{bmatrix}W_1 \\W_2\\W_3\\\end{bmatrix}^{T}\begin{bmatrix}X_1 \\X_2\\X_3\\\end{bmatrix} \ \ =\ \ W^TX" />

### Derivation

* We are taking theta as the thrisold value-
> <img src="https://latex.codecogs.com/svg.image?\sigma(z)=\begin{cases}&plus;1\&space;\&space;\textit{if}\&space;\&space;\&space;z\geqslant&space;\theta&space;\\-1\&space;\&space;\textit{if}\&space;\&space;\&space;z<&space;\theta\end{cases}&space;" title="\sigma(z)=\begin{cases}+1\ \ \textit{if}\ \ \ z\geqslant \theta \\-1\ \ \textit{if}\ \ \ z< \theta\end{cases} " />

* Changing RHS to LHS

> <img src="https://latex.codecogs.com/svg.image?\sigma(z)=\begin{cases}&plus;1\&space;\&space;\textit{if}\&space;\&space;\&space;z-\theta\geqslant&space;0&space;\\-1\&space;\&space;\textit{if}\&space;\&space;\&space;z-&space;\theta<0\end{cases}" title="\sigma(z)=\begin{cases}+1\ \ \textit{if}\ \ \ z-\theta\geqslant 0 \\-1\ \ \textit{if}\ \ \ z- \theta<0\end{cases}" />

* We are taking theta as W_0X_0 and W_0X_0 which is 'y intercept' or 'c' in y=mX+c

> <img src="https://latex.codecogs.com/svg.image?\sigma(z)=\begin{cases}&plus;1\&space;\&space;\textit{if}\&space;\&space;\&space;W^TX&plus;bias\geqslant\&space;0\&space;\\-1\&space;\&space;\textit{if}\&space;\&space;\&space;W^TX&plus;bias<\&space;0\&space;\\\end{cases}" title="\sigma(z)=\begin{cases}+1\ \ \textit{if}\ \ \ W^TX+bias\geqslant\ 0\ \\-1\ \ \textit{if}\ \ \ W^TX+bias<\ 0\ \\\end{cases}" />

#### Bias
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* In y=mX+c ,
   * c or bias helps the **shifting** from +ve to -ve and vice versa so that the output is controlled.
   * m or the slope helps the **rotation**.

 <img src="https://user-images.githubusercontent.com/12748752/136802531-79edaea5-9b55-4ae2-b2c5-a2205e3fce31.png" width=50%>



* Bias effects the output as the following it change the output class +ve to -ve.

<img src="https://user-images.githubusercontent.com/12748752/136807286-303afa7c-d91e-4dae-94db-2ad88563fda7.png"  width=50%>

### Perceptron learning Rule or Update Weights and Errors
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
<img src="https://sebastianraschka.com/images/faq/classifier-history/perceptron-figure.png" width=40%>
* As we know error is **(_Predicted Value_ - _Acctual Value_)** .
* A Neural Network back propagate and updates **Weights** and **Bias**.


* Now, the question is What would be the update weight?

<img src="https://user-images.githubusercontent.com/12748752/138711107-3345c175-5d03-4b09-b8ad-0c0a517d74e6.png" width=40%/>

* The update value for each input is not same as each input weight has different contribution to the final error.
* So the rectification of weight for each input would be different.


<img src="https://user-images.githubusercontent.com/12748752/138711104-18be860b-4e6f-4baa-af73-c83f66853656.png" width=40%/>


> <img src="https://latex.codecogs.com/svg.image?w_{i,j}&space;\leftarrow&space;w_{i,j}&space;&plus;&space;\eta(y_j&space;-&space;\hat{y_j})x_i" title="w_{i,j} \leftarrow w_{i,j} + \eta(y_j - \hat{y_j})x_i" />


> #### Where
> 
>> <img src="https://latex.codecogs.com/svg.image?&space;w_{i,j}&space;\textrm{&space;:&space;connection&space;weight&space;between}&space;\&space;\&space;i^{th}&space;&space;\textrm{input&space;neuron&space;and&space;}&space;j^{th}&space;&space;\textrm{&space;output&space;neuron}" title=" w_{i,j} \textrm{ : connection weight between} \ \ i^{th} \textrm{input neuron and } j^{th} \textrm{ output neuron}" />.  
>>
>> <img src="https://latex.codecogs.com/svg.image?x_i&space;:&space;i^{th}\textrm{&space;input&space;value}" title="x_i : i^{th}\textrm{ input value}" />.
>>
>> <img src="https://latex.codecogs.com/svg.image?\hat{y_j}&space;:&space;\textrm{output&space;of}&space;\&space;j^{th}\&space;\textrm{&space;output&space;}" title="\hat{y_j} : \textrm{output of} \ j^{th}\ \textrm{ output }" />.
>>
>> <img src="https://latex.codecogs.com/svg.image?y_j&space;:&space;\textrm{target&space;output&space;of}\&space;\&space;j^{th}&space;\textrm{&space;output&space;neuron}" title="y_j : \textrm{target output of}\ \ j^{th} \textrm{ output neuron}" />.
>>
>> <img src="https://latex.codecogs.com/svg.image?\eta&space;:&space;\textrm{learning&space;rate}" title="\eta : \textrm{learning rate}" />.  

> #### It can also be written as for jth element of w vector 
> <img src="https://latex.codecogs.com/svg.image?w_j&space;=&space;w_j&space;&plus;&space;\triangle&space;w_j" title="w_j = w_j + \triangle w_j" />.
>
> <img src="https://latex.codecogs.com/svg.image?where,\&space;\triangle&space;w_j&space;=&space;&space;\eta(y^{(i)}&space;-&space;\hat{y_j}^{(i)})x_j^{(i)}" title="where,\ \triangle w_j = \eta(y^{(i)} - \hat{y_j}^{(i)})x_j^{(i)}" />.

#### Update Weights can be written as
><img src="https://latex.codecogs.com/svg.image?\\W=&space;W-\eta&space;\frac{\partial&space;e}{\partial&space;w}\\&space;\\\\\mathrm{Where,\&space;\&space;W\&space;=\&space;Weight,\&space;\&space;\eta\&space;=\&space;Learning&space;\&space;rate,\&space;\&space;\partial&space;e=Change&space;\&space;in\&space;error,&space;\&space;\&space;\partial&space;w=Change&space;\&space;in\&space;weight}&space;\\\mathrm{Here&space;\&space;\&space;-\eta&space;\frac{\partial&space;e}{\partial&space;w}\&space;\&space;=&space;\&space;\&space;\Delta&space;W,&space;\&space;\&space;\&space;From&space;\&space;the\&space;above&space;\&space;\&space;(W_j&plus;\Delta&space;W_j)}" title="\\W= W-\eta \frac{\partial e}{\partial w}\\ \\\\\mathrm{Where,\ \ W\ =\ Weight,\ \ \eta\ =\ Learning \ rate,\ \ \partial e=Change \ in\ error, \ \ \partial w=Change \ in\ weight} \\\mathrm{Here \ \ -\eta \frac{\partial e}{\partial w}\ \ = \ \ \Delta W, \ \ \ From \ the\ above \ \ (W_j+\Delta W_j)}" />


### Neural Network error Update Weights
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* Suppose we have a Neural Network with three input two hidden layers and we are using sigmoid as a activation function inside the hidden layer as well as in final weight calculation.
<img src="https://user-images.githubusercontent.com/12748752/138762029-20fc6d46-e47c-4131-b1d3-9ce33a3595af.png" width=50%/>

* **Input buffers**(that's why not having Bias, Input neuron would have Bias), **Hidden layers**, **Output Neuron** are like
><img src="https://latex.codecogs.com/svg.image?\\\mathrm{Input\&space;Buffer=\&space;}X_1,\&space;X_2,\&space;X_3&space;\&space;(No\&space;Bias,\&space;Input\&space;Neuron\&space;would\&space;have\&space;Bias)\\\mathrm{Weight&space;=\&space;W_{i&space;j}^{(z)}&space;\&space;(\&space;i=\&space;the&space;\&space;destination,\&space;j=\&space;the&space;\&space;source,\&space;z=\&space;location\&space;number)}&space;\\\mathrm{Bias=&space;\&space;b_i}&space;\\\mathrm{Weight&space;\&space;Summation=&space;\&space;Z_i^{(z)}(i=\&space;Hidden&space;\&space;or\&space;output\&space;neuron&space;\&space;number,\&space;z=\&space;location)}\\\mathrm{Activation&space;\&space;function=&space;\&space;a_i^{(z)}(i=\&space;Hidden&space;\&space;or\&space;output\&space;neuron&space;\&space;number,\&space;z=\&space;location)}&space;\\\widehat{Y}=\mathrm{\&space;Final\&space;output}&space;" title="\\\mathrm{Input\ Buffer=\ }X_1,\ X_2,\ X_3 \ (No\ Bias,\ Input\ Neuron\ would\ have\ Bias)\\\mathrm{Weight =\ W_{i j}^{(z)} \ (\ i=\ the \ destination,\ j=\ the \ source,\ z=\ location\ number)} \\\mathrm{Bias= \ b_i} \\\mathrm{Weight \ Summation= \ Z_i^{(z)}(i=\ Hidden \ or\ output\ neuron \ number,\ z=\ location)}\\\mathrm{Activation \ function= \ a_i^{(z)}(i=\ Hidden \ or\ output\ neuron \ number,\ z=\ location)} \\\widehat{Y}=\mathrm{\ Final\ output} " width=80% />

* Hidden Layer weight calculation

<img src="https://user-images.githubusercontent.com/12748752/138760860-1056cc68-b17d-4c8e-abd8-1d8e35ad72f7.png" width=50%/>

* Final layer weight calculation
 
<img src="https://user-images.githubusercontent.com/12748752/138760858-246fe6ec-c1f8-4807-821b-abeb18e08493.png" width=50%/>

* Weight and Bias update rule-
<img src="https://user-images.githubusercontent.com/12748752/138770072-79bdc601-ef95-4d6e-8bfb-03b6fa95f821.png" width=50%>

#### Matrix representation of above diagrams
![light](https://user-images.githubusercontent.com/12748752/136802581-e8e0607f-3472-44f7-a8b2-8ba82a0f8070.png)
* Calculationin **Hidden layers**
> <img src="https://latex.codecogs.com/svg.image?\\\mathrm{\&space;\begin{bmatrix}&space;W_{11}&W_{12}&space;&&space;W_{13}&space;\\&space;W_{21}&&space;W_{22}&space;&W_{23}&space;\\\end{bmatrix}_{(2X3)}*&space;\begin{bmatrix}&space;X_{1}&space;\\&space;X_{2}&space;\\X_{3}&space;\\\end{bmatrix}_{(3X1)}&space;&plus;&space;\begin{bmatrix}&space;b_{1}&space;\\&space;b_{2}&space;\\\end{bmatrix}&space;=&space;\\&space;&space;\begin{bmatrix}&space;Z_{1}&space;\\&space;Z_{2}&space;\\\end{bmatrix}\to&space;\begin{bmatrix}&space;activation(Z_{1})&space;\\&space;activation(Z_{2})&space;\\\end{bmatrix}\to&space;\begin{bmatrix}&space;a_{1}&space;\\&space;a_{2}&space;\\\end{bmatrix}&space;or\&space;\widehat{Y}&space;}" title="\\\mathrm{\ \begin{bmatrix} W_{11}&W_{12} & W_{13} \\ W_{21}& W_{22} &W_{23} \\\end{bmatrix}_{(2X3)}* \begin{bmatrix} X_{1} \\ X_{2} \\X_{3} \\\end{bmatrix}_{(3X1)} + \begin{bmatrix} b_{1} \\ b_{2} \\\end{bmatrix} = \\ \begin{bmatrix} Z_{1} \\ Z_{2} \\\end{bmatrix}\to \begin{bmatrix} activation(Z_{1}) \\ activation(Z_{2}) \\\end{bmatrix}\to \begin{bmatrix} a_{1} \\ a_{2} \\\end{bmatrix} or\ \widehat{Y} }" width=80%/>
* Calculation in **Output layer**
> <img src="https://latex.codecogs.com/svg.image?\\\mathrm{\&space;\begin{bmatrix}&space;W_{11}&W_{12}&space;\\\end{bmatrix}_{(1X2)}*&space;\begin{bmatrix}&space;a_{1}&space;\\&space;a_{2}&space;\end{bmatrix}_{(2X1)}&space;&plus;&space;\begin{bmatrix}&space;b_{1}&space;\\\end{bmatrix}&space;=&space;\\&space;&space;\begin{bmatrix}&space;Z_{1}&space;\\\end{bmatrix}\to&space;\begin{bmatrix}&space;activation(Z_{1})&space;&space;\\\end{bmatrix}\to&space;\begin{bmatrix}&space;a_{1}&space;\\\end{bmatrix}&space;or\&space;\widehat{Y}&space;}" title="\\\mathrm{\ \begin{bmatrix} W_{11}&W_{12} \\\end{bmatrix}_{(1X2)}* \begin{bmatrix} a_{1} \\ a_{2} \end{bmatrix}_{(2X1)} + \begin{bmatrix} b_{1} \\\end{bmatrix} = \\ \begin{bmatrix} Z_{1} \\\end{bmatrix}\to \begin{bmatrix} activation(Z_{1}) \\\end{bmatrix}\to \begin{bmatrix} a_{1} \\\end{bmatrix} or\ \widehat{Y} }" />

## What are Neural networks?
Neural networks are set of algorithms inspired by the functioning of human brian. Generally when you open your eyes, what you see is called data and is processed by the Nuerons(data processing cells) in your brain, and recognises what is around you. That’s how similar the Neural Networks works. They takes a large set of data, process the data(draws out the patterns from data), and outputs what it is.

## Neural Network Basics

* Neural networks were one of the first machine learning models.
* Deep learning implies the use of neural networks.  
* The **"deep"** in deep learning refers to a neural network with many hidden layers.  

* Neural networks accept input and produce output.  
    * The input to a neural network is called the feature vector.  
    * The size of this vector is always a fixed length.  
    * Changing the size of the feature vector means recreating the entire neural network.  
    * A vector implies a 1D array.  Historically the input to a neural network was always 1D.  
    * However, with modern neural networks you might see inputs, such as:-

* **1D Vector** - Classic input to a neural network, similar to rows in a spreadsheet.  Common in predictive modeling.
* **2D Matrix** - Grayscale image input to a convolutional neural network (CNN).
* **3D Matrix** - Color image input to a convolutional neural network (CNN).
* **nD Matrix** - Higher order input to a CNN.

Prior to CNN's, the image input was sent to a neural network simply by squashing the image matrix into a long array by placing the image's rows side-by-side.  CNNs are different, as the nD matrix literally passes through the neural network layers.


**Dimensions** The term dimension can be confusing in neural networks.  In the sense of a 1D input vector, dimension refers to how many elements are in that 1D array.  
* For example a neural network with 10 input neurons has 10 dimensions.  
* However, now that we have CNN's, the input has dimensions too.  
* The input to the neural network will *usually* have 1, 2 or 3 dimensions.  4 or more dimensions is unusual.  
----------
* You might have a 2D input to a neural network that has 64x64 pixels. 
* This would result in 4,096 input neurons.  
* This network is either** 2D or 4,096D, **depending on which set of dimensions you are talking about!

# Classification or Regression

Like many models, neural networks can function in classification or regression:

* **Regression** - You expect a number as your neural network's prediction.
* **Classification** - You expect a class/category as your neural network's prediction.(the number of Input = The number of Output)

The following shows a classification and regression neural network:

![Neural Network Classification and Regression](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_2_ann_class_reg.png "Neural Network Classification and Regression")


* Notice that **the output of the regression neural network is numeric** and **the output of the classification is a class.**
* **Regression, or two-class classification, networks always have a single output.**
* **Classification neural networks have an output neuron for each class.**


---

<img src='https://github.com/arijitBhadra/Deep-Learning/blob/master/Pictures/BasicNN2.jpeg?raw=true'>

The Calculation would be like-             
* **(I1W1+I2W2+B)** = Weighted sum ;and it would be go through the Activation function    
* Where **'I'** is the input and **'W'** is the weight.


The following diagram shows a typical neural network:

![Feedforward Neural Networks](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_2_ann.png "Feedforward Neural Networks")





There are usually four types of neurons in a neural network:

* **Input Neurons** - Each input neuron is mapped to one element in the feature vector.
* **Hidden Neurons** - Hidden neurons allow the neural network to abstract and process the input into the output.
* **Output Neurons** - Each output neuron calculates one part of the output.
* **Context Neurons** - Holds state between calls to the neural network to predict.
* **Bias Neurons** - Work similar to the y-intercept of a linear equation.  

These neurons are grouped into layers:

* **Input Layer** - The input layer accepts feature vectors from the dataset.  Input layers usually have a bias neuron.
* **Output Layer** - The output from the neural network.  The output layer does not have a bias neuron.
* **Hidden Layers** - Layers that occur between the input and output layers.  Each hidden layer will usually have a bias neuron.



# Deep-Learning
Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example

# Activation functions and what are it uses in a Neural Network Model?
Activation functions are really important for a Artificial Neural Network to learn and make sense of something really complicated and **Non-linear complex functional mappings between the inputs and response variable**.
* Their main purpose is to convert a input signal of a node in a A-NN to an output signal.
* A Neural Network without Activation function would simply be a Linear regression Model which has limited power and does not performs good most of the times.
*  Also without activation function our Neural network would not be able to learn and model other complicated kinds of data such as images, videos , audio , speech etc

## Why do we need Non-Linearities?
* Non-linear functions are those which have degree more than one and they have a curvature when we plot a Non-Linear function. Now we need a Neural Network Model to learn and represent almost anything and any arbitrary complex function which maps inputs to outputs.
* Also another important feature of a Activation function is that it should be differentiable. We need it to be this way so as to perform backpropogation optimization strategy while propogating backwards in the network to compute gradients of Error(loss) with respect to Weights and then accordingly optimize weights using Gradient descend or any other Optimization technique to reduce Error.

## Most popular types of Activation functions -
Sigmoid or Logistic
Tanh — Hyperbolic tangent
ReLu -Rectified linear units


## [Back-propagation](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) 
<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/Pictures/b1.png?raw=true">


### What is error?
* Error or loss is the difference between the actual value and the expected value.
* In deep Neural Net we adding up the Waights in every layer and at the end (or between-ReLu for hidden layer)we calculate all the waightes with a Activation function.
* The main perpose of back propagation is to go back in the Neural Network and modify the weights 
<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/Pictures/neural_network-9.png?raw=true">

* Backpropagation is a technique used to train certain classes of neural networks – it is essentially a principal that allows the machine learning program to adjust itself according to looking at its past function.
* Backpropagation is sometimes called the “backpropagation of errors.”
* Backpropagation as a technique uses gradient descent: It calculates the gradient of the loss function at output, and distributes it back through the layers of a deep neural network. The result is adjusted weights for neurons.


## Bibliography:
* https://medium.com/@purnasaigudikandula/recurrent-neural-networks-and-lstm-explained-7f51c7f6bbb9
* https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
* Andrew Ng- Coursera


