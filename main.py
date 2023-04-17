import math
import struct
import numpy as np
from array import array
import matplotlib.pyplot as plt
import random

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
     labels = []
     with open(labels_filepath, 'rb') as file:
         magic, size = struct.unpack(">II", file.read(8))
         if magic != 2049:
             raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
         labels = array("B", file.read())        
    
     with open(images_filepath, 'rb') as file:
         magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
         if magic != 2051:
             raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
         image_data = array("B", file.read())        
     images = []
     for i in range(size):
         images.append([0] * rows * cols)
     for i in range(size):
         img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
         img = img.reshape(28, 28)
         images[i][:] = img            
    
    # Convert the lists to NumPy arrays
     images = np.array(images)
     labels = np.array(labels)
    
     return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
		


#____________________________________main________________________________________

# Load data
loader = MnistDataloader('train-images.idx3-ubyte', 
                        'train-labels.idx1-ubyte',
                        't10k-images.idx3-ubyte',
                        't10k-labels.idx1-ubyte')
(x_train, y_train), (x_test, y_test) = loader.load_data()
x_train = x_train / 255.0 - 0.5
x_test = x_test / 255.0 - 0.5

#-------------------a+b--------------------
""" Write a program to reduce the dimension of the data (using PCA) into a dimension
 of size parameterized by p (later, we will set it to 40)."""

m=x_train.shape[0]
# Flatten the images into vectors

X = (x_train.reshape(x_train.shape[0], -1)).T
# Compute the covariance matrix
Theta = np.dot(X,X.T) / m

# Compute the eigendecomposition of the covariance matrix
U, S, Ut = np.linalg.svd(Theta)
S_squared=np.sqrt(S)

# Plot the singular values
plt.plot(S_squared)
plt.show()

# Select the first p columns of U as Up
p = 12
Up = U[:,:p]

# Compute the reduced-sized vectors for each image. now each coulm in w represent a picture with 40 elements instead of 784
w_train = np.dot(Up.T, X)

# Reconstruct an image from the reduced-sized vector
i = 9 # choose an image index
x_reconstructed = np.dot(Up, w_train[:, i])

# Reshape the reconstructed image back to a 2D array
x_reconstructed = x_reconstructed.reshape(28, 28)

# Plot the original and reconstructed images
plt.subplot(1, 2, 1)
plt.imshow(x_train[i], cmap='gray')
plt.title('Original image')
plt.subplot(1, 2, 2)
plt.imshow(x_reconstructed, cmap='gray')
plt.title('Reconstructed image')
plt.show()

#-------------------c--------------------
"""Write a program for applying the Kmeans clustering algorithm using k clusters.
Use random initialization (say, random values in [-0.5,0.5]). You should not use
built-in codes."""

def kmeans(X, k, centers_kmeans, max_iter=5):
    for i in range(max_iter):
        assigned_centers = assign_to_closest_center(X, centers_kmeans)
        centers_kmeans = recompute_centers(X, assigned_centers, k)
    
    plt.scatter(X[:, 0], X[:, 1], c=assigned_centers, cmap='viridis')
    plt.scatter(centers_kmeans[:, 0], centers_kmeans[:, 1], c='red', marker='x')
    plt.show()
    
    return assigned_centers, centers_kmeans

def assign_to_closest_center(X, centers_kmeans):
    assigned_centers = np.argmin(np.linalg.norm(X[:, np.newaxis, :] - centers_kmeans, axis=2), axis=1)
    return assigned_centers

def recompute_centers(X, assigned_centers, k):
    centers_kmeans = np.array([X[assigned_centers == i].mean(axis=0) if np.sum(assigned_centers == i) != 0 else np.random.uniform(low=-0.5, high=0.5, size=X.shape[1]) for i in range(k)])
    return centers_kmeans
#-------------------d--------------------
"""Using k = 10, classify the reduced images (p = 40) from the training data set."""
# Define the number of clusters
k = 10
# define each row  to be image, and each coulmn to be element
w_train=w_train.T
# Initialize the centroids randomly
centroids = np.random.uniform(low=-0.5, high=0.5, size=(k, w_train.shape[1])) 
assigned_centers,centers_assign_after_kmeans = kmeans(w_train, k, centroids)
assigned_centers=np.array(assigned_centers)

#---------------------------e------------------------------------
"""See which of the clusters you found corresponds to which digit. Assign a digit to
a cluster that you found using the most common label in that cluster - use the
train labels to determine that."""


def give_centers_lebles(assigned_centers, y):
    sums_of_images_per_center = np.zeros((k, k))
    centers_to_labels=np.zeros(k)
    for i in range (y.shape[0]):
        image_real_lable=y[i]
        image_assigned_center=assigned_centers[i]
        sums_of_images_per_center[image_assigned_center, image_real_lable] +=1

    for j in range (k):
        approximate_center=np.argmax(sums_of_images_per_center[j])
        centers_to_labels[j]=approximate_center
    return centers_to_labels


centers_to_labels=give_centers_lebles(assigned_centers, y_train)

#--------------------------------f------------------------------------
"""Test your success: for each of the images in the test data, estimate its label
using closest centroid (and its cluster’s label) and check the percentage of true
estimations using the test labels. Report your model’s results."""

#do the same procces on the x_test matrix
m=x_train.shape[0]
X = (x_test.reshape(x_test.shape[0], -1)).T

# Compute the reduced-sized vectors for each image. now each coulm in w represent a picture with 40 elements instead of 784
w_test = np.dot(Up.T, X)
w_test=w_test.T

def caculate_succses(centers_to_labels, assigned_photos_to_centers, y_test, w):
    succsus=0
    for i in range(w.shape[0]):
        kmeans_center_result=centers_to_labels[assigned_photos_to_centers[i]]
        if y_test[i]==kmeans_center_result:
            succsus+=1
    return (succsus / w.shape[0]) *100

assigned_photos_to_centers=assign_to_closest_center(w_test,centers_assign_after_kmeans)
succses_presents=caculate_succses(centers_to_labels, assigned_photos_to_centers, y_test,w_test)
print(succses_presents)
#-----------------------------g--------------------------------------
"""See if the process above is consistent or not with respect to the random initialization.
 Try this 3 times and report your results."""

for i in range(3):
    centroids = np.random.uniform(low=-0.5, high=0.5, size=(k, w_train.shape[1])) 
    assigned_W_train_photos_to_centers,centers_assign_after_kmeans = kmeans(w_train, k, centroids)
    assigned_W_train_photos_to_centers=np.array(assigned_W_train_photos_to_centers)
    centers_to_labels=give_centers_lebles(assigned_W_train_photos_to_centers, y_train)
    assigned_W_test_photos_to_centers=assign_to_closest_center(w_test,centers_assign_after_kmeans)
    succses_presents=caculate_succses(centers_to_labels, assigned_W_test_photos_to_centers, y_test,w_test)
    print(i, 'has a ',succses_presents, 'precent sucsses rate')

#-----------------------------i--------------------------------------
"""Try initializing each of the Kmeans centroids using the mean of 10 reduced images
that you pick from each label. Are the results better now (use p = 40 again)?"""

centroids=np.zeros((10,w_train.shape[1]))
for i in range(10):
        counter=0
        index=0
        while counter<10:
            if(y_train[index]==i):
                for p in range(w_train.shape[1]):
                    centroids[i,p] += w_train[index,p]
                counter+=1
            index+=1
        centroids[i]=centroids[i]/10

assigned_W_train_photos_to_centers,centers_assign_after_kmeans = kmeans(w_train, k, centroids)
assigned_W_train_photos_to_centers=np.array(assigned_W_train_photos_to_centers)
centers_to_labels=give_centers_lebles(assigned_W_train_photos_to_centers, y_train)
assigned_W_test_photos_to_centers=assign_to_closest_center(w_test,centers_assign_after_kmeans)
succses_presents=caculate_succses(centers_to_labels, assigned_W_test_photos_to_centers, y_test,w_test)


print('when computing the centers we get ',succses_presents, 'precent sucsses rate')