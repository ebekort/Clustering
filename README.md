# Image clustering with K-means

This repository contains Python scripts implementing K-Means clustering on MNIST handwritten digit data. The K-Means algorithm is applied to cluster the images into 10 groups corresponding to the 10 digit classes present in the MNIST dataset.

This github repository has python scripts that implement a K-means clustering algorithm to cluster handwritten images of the MNIST dataset. Since the dataset exists of handwritten digits from 0-9, 10 clusters have been used that each represent one digit

#### Dependencies
- `numpy`: For working with large arrays and computing the euclidean distance
- `scikit-learn`: to calculate the Adjusted Rand Index (ARI) 
- `matplotlib`: to visualize the data
- `scipy`: for certain functions in utils.py

## Files

### `main.py`
This program is the main program as the name suggests. It takes all of the other programs together and makes sure it functions well. Within this script the data is loaded, the K-means class is initialized and the evaluation of the algorithm is done, using the Adjusted Rand Index (ARI). It also has functionality to visualize the images or a confusion matrix of clusters made by the algorithm.



### `kmeans.py`

This module contains the implementation of the KMeansClusterer class, which performs K-Means clustering.
This program has the code for the KMeansClusterer class that has the wholy functionality of the K-means algorithm

#### Class: `KMeansClusterer`

- `__init__(self, n_clusters, n_feats)`: Initializes the class with the number of clusters and features and makes a numpy array for the number of means
- `initialize_clusters(self, data)`: Initializes the clusters based on the forgy method
- `assign(self, data)`: Assigns each datapoint to the closest cluster based on the euclidean distance and return a numpy array of the assignments
- `update(self, assignments, data)`: Updates the means based on the assignments of the assign function
- `fit_predict(self, data)`: Fits the algorithm with 100 iterations and returns the predictions of the algorithm

### `utils.py`

This program contains functions to plot the confusion matrix and to generate images from the data

## Usage

To run the program, execute `main.py`. You do need to install all the dependencies listed earlier.


