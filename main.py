import math
import struct
import numpy as np
from array import array
import matplotlib.pyplot as plt

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
def kmeans(X, k,centers_kmeans ,max_iter=5):
    for i in range (max_iter):
        assigned_centers=assign_to_closest_center(X,centers_kmeans)
        centers_kmeans=recompute_centers(X,assigned_centers,k, centers_kmeans)
    plt.scatter(X[:, 0], X[:, 1], c=assigned_centers, cmap='viridis')
    plt.scatter(centers_kmeans[:, 0], centers_kmeans[:, 1], c='red', marker='x')
    plt.show()
    return assigned_centers,centers_kmeans


def assign_to_closest_center(B, centers_assign):
  assigned_centers = []
  # Loop through each photo
  for d in range(B.shape[0]):
    # Calculate the distances between the photo and each center
    photo=B[d]
    distances = []
    for i in range(centers_assign.shape[0]):
      temp_center = centers_assign[i]
      distance = 0
      for j in range(B.shape[1]):
        distance += (photo[j] - temp_center[j]) ** 2
      distance = math.sqrt(distance)
      distances.append(distance)
    
    # Find the index of the center with the shortest distance
    min_index = distances.index(min(distances))
    
    # Assign the center with the shortest distance to the photo
    assigned_centers.append(min_index)
  return assigned_centers

def recompute_centers(B, assigned_centers,k,centers):
    numOfelements=B.shape[1]
    assigned_centers_array = np.array(assigned_centers)
    numOfImagesPerCenter = np.zeros(k)
    sumOfVectors = np.zeros((k, numOfelements))
    for i in range(len(assigned_centers_array)):
        center=assigned_centers_array[i]
        numOfImagesPerCenter[center] += 1
        for p in range(numOfelements):
            sumOfVectors[center,p] += B[i,p]
    for j in range (k):
        if numOfImagesPerCenter[j]!=0:
            sumOfVectors[j] = sumOfVectors[j] / numOfImagesPerCenter[j]
        else:
            sumOfVectors[j]=centers[j]
    return sumOfVectors
#-------------------d--------------------
# Define the number of clusters
k = 10
# define each row  to be image, and each coulmn to be element
w_train=w_train.T
# Initialize the centroids randomly
centroids = np.random.uniform(low=-0.5, high=0.5, size=(k, w_train.shape[1])) 
assigned_centers,centers_assign_after_kmeans = kmeans(w_train, k, centroids)
assigned_centers=np.array(assigned_centers)
#---------------------------e------------------------------------
def find_max_index(arr):
    max_index = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[max_index]:
            max_index = i
    return max_index

def give_centers_lebles(assigned_centers, y):
    sums_of_images_per_center = np.zeros((k, k))
    centers_to_labels=np.zeros(k)
    for i in range (y.shape[0]):
        image_real_lable=y[i]
        image_assigned_center=assigned_centers[i]
        sums_of_images_per_center[image_assigned_center, image_real_lable] +=1

    for j in range (k):
        approximate_center=find_max_index(sums_of_images_per_center[j])
        centers_to_labels[j]=approximate_center
    return centers_to_labels


centers_to_labels=give_centers_lebles(assigned_centers, y_train)

#--------------------------------f------------------------------------
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
for i in range(3):
    centroids = np.random.uniform(low=-0.5, high=0.5, size=(k, w_train.shape[1])) 
    assigned_W_train_photos_to_centers,centers_assign_after_kmeans = kmeans(w_train, k, centroids)
    assigned_W_train_photos_to_centers=np.array(assigned_W_train_photos_to_centers)
    centers_to_labels=give_centers_lebles(assigned_W_train_photos_to_centers, y_train)
    assigned_W_test_photos_to_centers=assign_to_closest_center(w_test,centers_assign_after_kmeans)
    succses_presents=caculate_succses(centers_to_labels, assigned_W_test_photos_to_centers, y_test,w_test)
    print(i, 'has a ',succses_presents, 'precent sucsses rate')

#-----------------------------i--------------------------------------
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