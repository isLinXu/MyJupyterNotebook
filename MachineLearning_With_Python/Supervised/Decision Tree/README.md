## Index
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)

## Decision tree
![grape](https://user-images.githubusercontent.com/12748752/126882595-d1f5449e-14bb-4ab3-809c-292caf0858a1.png)
* [Decission Tree Classification Notebook](https://nbviewer.org/github/iAmKankan/MachineLearning_With_Python/blob/master/Supervised/Decision%20Tree/Decision%20Trees1.ipynb)
* [Decission Tree Regression Notebook](https://nbviewer.org/github/iAmKankan/MachineLearning_With_Python/blob/master/Supervised/Decision%20Tree/DecissionTree2.ipynb)
### Definition
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* Decision Trees are versatile Machine Learning algorithms that can perform both classification and regression tasks, and even multi-output tasks. 
* They are powerful algorithms, capable of fitting complex datasets.
* Decision Trees are also the fundamental components of Random Forests
* One of the many qualities of Decision Trees is that they require very little data preparation. 
* In fact, they don’t require feature scaling or centering at all.

### Classification and Regression Tree (CART) algorithm
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

> #### Scikit-Learn uses the `Classification and Regression Tree` **(CART)** algorithm, which produces only binary trees: nonleaf nodes always have two children (i.e., questions only have yes/no answers). 
> #### However, other algorithms such as **ID3** can produce Decision Trees with nodes that have more than two children.

* _Scikit-Learn_ uses the _**`Classification and Regression Tree (CART)`**_ algorithm to train Decision Trees (also called “growing” trees). 
> #### How the algorithm works -
>> * **First** splitting the training set into _`two subsets`_ using a _`single feature`_ **k** and a threshold **t<sub>k</sub>** (e.g., “petal length ≤ 2.45 cm”). 
>> * **Then** after spliting the training set in two, it `recursively` splits the subsets using the same logic, then the sub-subsets, and so on. 
>> * **Last** It stops recursing once it reaches the maximum depth (defined by the `max_depth` hyperparameter), or if it cannot find a split that will reduce impurity. 

* A few other hyperparameters control additional stopping conditions (`min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf` and `max_leaf_nodes`).
> #### How does it choose **k** and **t<sub>k</sub>** ?
>>  It searches for the pair (**k**,**t<sub>k</sub>** ) that produces the purest subsets (weighted by their size). 


### CART cost function for classification
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)

<img src="https://latex.codecogs.com/svg.image?\\J(k,t_k)&space;=&space;\frac{m_{left}}{m}G_{left}\&space;&plus;&space;\frac{m_{right}}{m}G_{right}\\&space;\\&space;\\\mathrm{Where}\left\{\begin{matrix}G_{left&space;\setminus&space;&space;right}\textrm{\&space;=&space;\&space;measures&space;the&space;impurity&space;of&space;the&space;left/right&space;subset,}\\\&space;\&space;\&space;\&space;\&space;m_{left&space;\setminus&space;&space;right}\textrm{\&space;=&space;\&space;is&space;the&space;number&space;of&space;instances&space;in&space;the&space;left/right&space;subset.}\end{matrix}\right.&space;" title="\\J(k,t_k) = \frac{m_{left}}{m}G_{left}\ + \frac{m_{right}}{m}G_{right}\\ \\ \\\mathrm{Where}\left\{\begin{matrix}G_{left \setminus right}\textrm{\ = \ measures the impurity of the left/right subset,}\\\ \ \ \ \ m_{left \setminus right}\textrm{\ = \ is the number of instances in the left/right subset.}\end{matrix}\right. " /> 




> #### Note: 
> * As you can see, the CART algorithm is a greedy algorithm: it greedily searches for an optimum split at the top level, then repeats the process at each subsequent level. It does not check whether or not the split will lead to the lowest possible impurity several levels down. A greedy algorithm often produces a solution that’s reasonably good but not guaranteed to be optimal. 
> * Unfortunately, finding the optimal tree is known to be an NP-Complete problem: it requires O(exp(m)) time, making the problem intractable even for small training sets. This is why we must settle for a “reasonably good” solution.


### Computational Complexity
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)


### Decision Tree Splitting Methods
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* It’s tough to decide which variables to put at the **root** or at different levels of the tree as internal nodes when the dataset comprises N variables. 
* Choosing any node as the root at random will not solve the problem. 
* We can receive disappointing results with limited precision if we use a random technique. 
* Researchers recommended employing criteria such-

1. Gini Impurity
2. Information Gain
3. Chi-Square

### Gini impurity
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* The division is called **pure** if all elements are accurately separated into different classes (_**an ideal scenario**_). 
* The Gini impurity is used to predict the likelihood that a randomly selected example would be incorrectly classified by a specific node. 
* **It is called an “impurity” metric because it shows how the model differs from a pure division. **
* The degree of Gini impurity ranges from **_0 to 1_**, 
    * **0** indicating that all of the elements belong to a single class 
    * **1** indicates that only one class exists. 
* The Gini impurity of value 1 indicates that all of the items are randomly distributed over various classes, whereas a value of 0.5 indicates that the elements are uniformly distributed across some classes. 

> #### A node’s gini attribute measures its impurity:
> #### A node is “pure” (gini=0) if all training instances it applies to belong to the same class.

<img src="https://latex.codecogs.com/svg.image?\mathrm{G}_i&space;=&space;1-&space;\sum_{n}^{k=1}(p_{i,k})^2" title="\mathrm{G}_i = 1- \sum_{n}^{k=1}(p_{i,k})^2" width=25% />

>> * <img src="https://latex.codecogs.com/svg.image?p_{i,k}" title="p_{i,k}" /> is the ratio of class _**k**_ instances among the training instances in the _**i<sup>th</sup>**_ node. 

* Gini index is a metric to measure how often a randomly chosen element would be incorrectly identified. 
* It means an attribute with lower gini index should be preferred. 
* Sklearn supports “ Gini ” criteria for gini index and by default, it takes “ Gini ”.


### Entropy 
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* The concept of entropy originated in thermodynamics as a measure of molecular disorder: entropy approaches zero when molecules are still and well ordered. 
* In Machine Learning, entropy is frequently used as an _impurity measure_: **a set’s entropy is zero when it contains instances of only one class**.
 
<img src="https://latex.codecogs.com/svg.image?\mathrm{H}_i&space;=&space;-&space;\sum_{k=1,\&space;p_{i,k}\neq&space;0}^{n}p_{i,k}&space;\log_2&space;(p_{i,k})" title="\mathrm{H}_i = - \sum_{k=1,\ p_{i,k}\neq 0}^{n}p_{i,k} \log_2 (p_{i,k})" width=30% />

### Gini Impurity or Entrophy?
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
* The truth is, most of the time it does not make a big difference: they lead to similar trees. 
* _**Gini impurity**_ is slightly _faster to compute_, so it is a good default. 
* However, when they differ, 
    * **`Gini impurity`** **tends to isolate the most frequent class in its own branch of the tree**, 
    * While **`Entropy`** **tends to produce slightly more balanced trees**.

## Questions and Answers 
![plum](https://user-images.githubusercontent.com/12748752/126882596-b9ba4645-7001-435e-9a3c-d4416a2543c1.png)
1. What is the approximate depth of a Decision Tree trained (without restrictions) on a training set with one million instances?
   * log2(10⁶) ≈ 20 (actually a bit more since the tree will generally not be perfectly well balanced)
2. Is a node’s Gini impurity generally lower or greater than its parent’s? Is it generally lower/greater, or always lower/greater? 
   * A node’s Gini impurity is generally lower than its parent’s. 
   * This is ensured by the CART training algorithm’s cost function, which splits each node in a way that minimizes the weighted sum of its children’s Gini impurities. 
3. If a Decision Tree is overfitting the training set, is it a good idea to try decreasing `max_depth`? 
   * If a Decision Tree is overfitting the training set, it may be a good idea to decrease `max_depth`, since this will constrain the model, regularizing it.
4. If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features? 
   * Decision Trees don’t care whether or not the training data is scaled or centered; that’s one of the nice things about them. 
   * So if a Decision Tree underfits the training set, scaling the input features will just be a waste of time.
5. If it takes one hour to train a Decision Tree on a training set containing 1 million instances, roughly how much time will it take to train another Decision Tree on a training set containing 10 million instances? 
6. If your training set contains 100,000 instances, will setting presort=True speed up training? 
7. Train and fine-tune a Decision Tree for the moons dataset by following these steps:

