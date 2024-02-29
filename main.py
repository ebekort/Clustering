import numpy as np
from sklearn.metrics import adjusted_rand_score

from utils import plot_confusion_matrix, show_image_mnist
from kmeans import KMeansClusterer

# Loading in the data
data = np.load("data/data.npy")
labels = np.load("data/labels.npy")  # Note: given the labels,
# we will have 10 clusters.

# Show the first image
# show_image_mnist(data[0])


def main():
    np.random.seed(10)
    kclusterer = KMeansClusterer(10, 784)

    '''
    Uncomment this part to calculate the mean of the adjusted
    rand index over 5 iterations

    summedAri = 0
    for i in range(1,6):
        predictions = kclusterer.fit_predict(data)
        ari =  adjusted_rand_score(labels, predictions)
        summedAri += ari
        print(f"Adjusted Rand Index for iteration {i}: {ari}")
    print(f"Mean Adjusted Rand Index: {summedAri/5}")'''

    # comment this part when calculating the mean of the adjusted rand index
    predictions = kclusterer.fit_predict(data)
    print(adjusted_rand_score(labels, predictions))

    # uncomment this to print the confusion matrix
    # plot_confusion_matrix(labels, predictions)


if __name__ == "__main__":
    main()
