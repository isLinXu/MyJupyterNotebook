# K-Means Algorithm

## Definition

**K-means clustering** aims to partition n observations into _K_ clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.

The result of a cluster analysis shown below as the coloring of the squares into three clusters.

![Clustering](https://upload.wikimedia.org/wikipedia/commons/c/c8/Cluster-2.svg)

## How K-means forms cluster:

* K-means picks k number of points for each cluster known as centroids.
* Each data point forms a cluster with the closest centroids i.e. k clusters.
* Finds the centroid of each cluster based on existing cluster members. Here we have new centroids.
* As we have new centroids, repeat step 2 and 3. Find the closest distance for each data point from new centroids and get associated with new k-clusters. Repeat this process until convergence occurs i.e. centroids does not change.


## How to determine value of K:

In K-means, we have clusters and each cluster has its own centroid. Sum of square of difference between centroid and the data points within a cluster constitutes within sum of square value for that cluster. Also, when the sum of square values for all the clusters are added, it becomes total within sum of square value for the cluster solution.

We know that as the number of cluster increases, this value keeps on decreasing but if you plot the result you may see that the sum of squared distance decreases sharply up to some value of k, and then much more slowly after that. Here, we can find the optimum number of cluster.

## Description

Given a training set of observations:

![Training set](../images/k_means/training-set.svg)

![x-i](../images/k_means/x-i.svg)

Where each observation is a _d_-dimensional real vector, k-means clustering aims to partition the _m_ observations into _K_ (_≤ m_) clusters:

![Clusters](../images/k_means/clasters.svg)

... so as to minimize the within-cluster sum of squares (i.e. variance).

Below you may find an example of 4 random cluster centroids initialization and further clusters convergence:

![Clustering](http://shabal.in/visuals/kmeans/random.gif)

[Picture Source](http://shabal.in/visuals/kmeans/6.html)

Another illustration of k-means convergence:

![Clustering](https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif)

## Cost Function (Distortion)

![c-i](../images/k_means/c-i.svg) - index of cluster _(1, 2, ..., K)_ to which example _x<sup>(i)</sup>_ is currently assigned.

![mu-k](../images/k_means/mu-k.svg) - cluster centroid _k_ (![mu-k-2](../images/k_means/mu-k-2.svg)) and ![k](../images/k_means/k.svg).

![mu-c-i](../images/k_means/mu-c-i.svg) - cluster centroid of a cluster to which the example _x<sup>(i)</sup>_ has been assigned.

For example:

![Cluster example](../images/k_means/cluster-example.svg)

In this case optimization objective will look like the following:

![Cost Function](../images/k_means/cost-function.svg)

![Clustering](https://upload.wikimedia.org/wikipedia/commons/d/d1/KMeans-density-data.svg)

## The Algorithm

Randomly initialize _K_ cluster centroids (randomly pick _K_ training examples and set _K_ cluster centroids to that examples).

![Centroids](../images/k_means/centroids.svg)

![k-means-algorithm](../images/k_means/k-means-algorithm.svg)

## References

- [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning)
- [K-means on Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
