## Index
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

## Batch Gradient Descent
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
### Partial Derivative
* To implement Gradient Descent, you need to compute the gradient of the cost function with regard to each model parameter **_<a>&theta;</a><sub>j</sub>_** .
* In other words, you need to calculate how much the cost function will change if you change **_<a>&theta;</a>_** just a little bit. 
* This is called a partial derivative.
* Lets computes the partial derivative of the cost function with regard to parameter **_<a>&theta;</a>_** , noted **_MSE(<a>&theta;</a>)_**.
#### Partial derivatives of the cost function
<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;}{\partial&space;\theta_j}MSE(\theta)=\frac{2}{m}\sum_{i=1}^{m}\left&space;(\mathbf{\theta^\top&space;x^{(i)}&space;}&space;-y^{(i)}&space;\right&space;)\&space;x^{(i)}_j&space;&space;&space;" title="\frac{\partial }{\partial \theta_j}MSE(\theta)=\frac{2}{m}\sum_{i=1}^{m}\left (\mathbf{\theta^\top x^{(i)} } -y^{(i)} \right )\ x^{(i)}_j " width=45% />

* Instead of computing these partial derivatives individually, you can compute them all in one go. 
* The gradient vector, noted **<a>&nabla;</a> MSE(<a>&theta;</a>)** (_nabla Mean Squared Error of theta_), contains all the partial derivatives of the cost function (one for each model parameter).
#### Gradient vector of the cost function
<img src="https://latex.codecogs.com/svg.image?\nabla_{\theta}\textbf{MSE}(\theta)=&space;\begin{pmatrix}\frac{\partial&space;}{\partial&space;\theta_0}MSE(\theta)&space;\\\frac{\partial&space;}{\partial&space;\theta_1}MSE(\theta)\\\vdots&space;\\\frac{\partial&space;}{\partial&space;\theta_n}MSE(\theta)&space;\\\end{pmatrix}&space;=\frac{2}{m}\textbf{X}^\top&space;\left&space;(\mathbf{X&space;\theta&space;-&space;y&space;}\right&space;)&space;" title="\nabla_{\theta}\textbf{MSE}(\theta)= \begin{pmatrix}\frac{\partial }{\partial \theta_0}MSE(\theta) \\\frac{\partial }{\partial \theta_1}MSE(\theta)\\\vdots \\\frac{\partial }{\partial \theta_n}MSE(\theta) \\\end{pmatrix} =\frac{2}{m}\textbf{X}^\top \left (\mathbf{X \theta - y }\right ) " width=45% />

* Once you have the gradient vector, which points uphill, just go in the opposite direction to go downhill. 
* This means subtracting **<a>&nabla;</a> MSE(<a>&theta;</a>)** from **<a>&theta;</a>**. 
* This is where the **_learning rate  <a>&eta;</a> (eta)_** comes into play: multiply the gradient vector by **_<a>&eta;</a>_** to determine the size of the downhill step
### Gradient Descent step
<img src="https://latex.codecogs.com/svg.image?\theta^{(\mathrm{next\&space;step})}=\theta&space;-&space;\eta&space;\nabla_{\theta}\textbf{MSE}(\theta)" title="\theta^{(\mathrm{next\ step})}=\theta - \eta \nabla_{\theta}\textbf{MSE}(\theta)" width=30% />

