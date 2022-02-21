# K-Nearest Neighbor
## Definition
* It can be used for both classification and regression.
* Stores all the classes and classifies new classes based on a similarity measure.
* The 'K' is KNN algorithm is the nearest neighbours we wish to take a vote from.
* Also called an instance-based or memory-based learning
* Lazy Algorithm
    - K-NN is a lazy learner because Because it does not create a generalized model during the time of training, so the testing phase is very important where it does the actual job. 
    - Hence Testing is very costly - in terms of time & money.
    - For example, the logistic regression algorithm learns its model weights (parameters) during training time. In contrast, there is no training time in K-NN.
    - This property doesn’t come without a cost: The “prediction” step in K-NN is relatively expensive!
        
         - Each time we want to make a prediction, K-NN is searching for the nearest neighbor(s) in the entire training set! 
         - There are certain tricks such as **BallTrees** and **KDtrees** to speed this up a bit.)
    - To summarize: An eager learner has a model fitting or training step. A lazy learner does not have a training phase.
    
* Only we have to find the stable value of K
     - K to your chosen number of neighbors

<img src="knn.png?raw=true">

* Problemetic for large datasets
    - KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) with some mathematics.
    
* The result of current neughborhood every time.
* odd numbers of K (not getting the tie 
* K= the number of dots you need to consider.
* **Not working in High dimentional data.**

### Distance is measured by -

* For continuous variables.
    * Eucladian Distance 
    * Manhattan Distance
    * Minkowski Distance
* For categorical variables.
    * Hamming Distance
<img src="KNN_similarity.png?raw=true">

<img src="KNN_hamming.png?raw=true">

### Steps: 
**Step 1:** Determine the value of K.

**Step 2:** Calculate the distance between the query example(training data) and the current example(test data) from the data.

**Step 3:** Sort the ordered collection of distances and index from smallest to largest (in ascending order) by the distances.

**Step 4:** Pick the first K entries from the sorted collection.

**Step 5:** Get the labels of the selected K entries.

**Step 6:** 
    - If regression, return the mean of the K labels
    - If classification, return the mode of the K labels
    
### Advantages:
* It is extremely easy to implement 
* As said earlier, it is **Lazy Learning** algorithm and therefore requires no training prior to making real time predictions. this makes the k nn algorithm much faster than other algorithms that require trainin g e.g svm, linear regression, etc. 
* Since the algorithm requires no training before making predictio ns, new data can be added seamlessly. 
* There are only two parameters required to implement KNN i.e. the value of k and the distance function (e.g. euclidean or manhattan etc.
* It is extremely easy to implement 
* As said earlier, it is lazy learning algorithm and therefore req uires no training prior to making real time predictions. this makes the KNN algorithm much faster than other algorithms that require trainin g e.g SVM, Linear Regression, etc. 
* Since the algorithm requires no training before making predictio ns, new data can be added seamlessly. 
* There are only two parameters required to implement KNN i.e. the value of K and the distance function (e.g. Euclidean or Manhattan, Minkowski Distance etc.



### Disadvantages:
* The knn algorithm doesn't work well with high dimensional data because with large number of dimensions, it becomes difficult for the algorithm to calculate distance in each dimension. 
* The KNN algorithm has a high prediction cost for large datasets. this is because in large datasets the cost of calculating distance between new point and each existing point becomes higher. 
* Finally, the KNN algorithm doesn't work well with categorical features since it is difficult to find the distance between dime nsions with categorical features.


### Conclusion:

* KNN is a simple yet powerful classification algorithm. it requires no training for making predictions, which is typically one of the most difficult parts of a machine learningalgorithm.
* The KNN algorithm have been widely used to find document similarity and pattern recognition.
* It has also been employed for developing recommender systems and for dimensionality reduction and pre processing steps for computer vision, particularly face recognition tasks. 

