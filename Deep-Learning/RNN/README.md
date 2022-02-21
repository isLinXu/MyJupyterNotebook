# Recurrent Neural Network
* The idea behind RNNs is to make use of sequential information. 
* In a traditional neural network we assume that all inputs (and outputs) are independent of each other. 
* But for many tasks that’s a very bad idea. 
    * If you want to predict the next word in a sentence you better know which words came before it. 
* **RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations and they have a “memory” which captures information about what has been calculated so far.**

* Processing sequences and time series requires some sort of memory since dynamic temporal behaviour is also adding information to the whole picture. So by
introducing loopback connections between neurons such a Recurrent Neural Network can remember past events.


<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/RNN/rnn2.png?raw=true">

* Note that any Recurrent Neural Network can be unfolded through time into a Deep Feed Forward Neural Network. So again, this whole exercise is only there since
training can be improved by changing the neural network topology from a single hidden layer feed forward network to something else.

<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/RNN/rnn3.png?raw=true">
 
## Types of RNNS
<img src="https://github.com/iAmKankan/Deep-Learning/blob/master/RNN/types%20of%20rnn.png?raw=true">



## Backpropogate Through Time:
* Going back in Every time stamp to change/update the weights is called Backpropogate through time.
* We typically treat the full sequence (word) as one training example, so the total error is just the sum of the errors at each time step (character). The weights as we can see are the same at each time step.
* Note: Going back into every time stamp and updating its weights is really a slow process. It takes both the computational power and time.

* While Backpropogating you may get 2 types of issues.
    * Vanishing Gradient
    * Exploding Gradient
    
    
    
### Vanishing Gradient:
* where the contribution from the earlier steps becomes insignificant in the gradient descent step.

### Exploding Gradient:
We speak of Exploding Gradients when the algorithm assigns a stupidly high importance to the weights, without much reason. But fortunately, this problem can be easily solved if you truncate or squash the gradients.


### How can you overcome the Challenges of Vanishing and Exploding Gradience?
1. **Vanishing Gradience can be overcome with**
    * Relu activation function.
    * LSTM, GRU.
2. **Exploding Gradience can be overcome with**
    * Truncated BTT(instead starting backprop at the last time stamp, we can choose similar time stamp, which is just before it.)
    * Clip Gradience to threshold.
    * RMSprop to adjust learning rate



## Advantages of Recurrent Neural Network
* The main advantage of RNN over ANN is that RNN can model sequence of data (i.e. time series) so that each sample can be assumed to be dependent on previous ones
* Recurrent neural network are even used with convolutional layers to extend the effective pixel neighborhood.

## Disadvantages of Recurrent Neural Network
* Gradient vanishing and exploding problems.
* Training an RNN is a very difficult task.
* It cannot process very long sequences if using tanh or relu as an activation function.
