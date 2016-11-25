# http://cs231n.github.io/classification/

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.time_utils import time_function
import matplotlib.pyplot as plt


# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the raw CIFAR-10 data.
cifar10_dir = './cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]
num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]
# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print X_train.shape, X_test.shape

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
from cs231n.classifiers import KNearestNeighbor
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# Test your implementation:
# ((X_train[0] - X_test[0])**2).sum()**0.5
# np.linalg.norm(X_train[0, :] - X_test[0, :], axis=0)
# dists[i,j] = np.linalg.norm(self.X_train[j,:]-X[i,:], axis=0)
# no loops:
#   x2 = np.sum(X_train * X_train, axis=1)
#   y2 = np.sum(X_test * X_test, axis=1)[None].T
#   xy = np.dot(X_test, X_train.T)
dists2 = classifier.compute_distances_two_loops(X_test)
dists1 = classifier.compute_distances_one_loop(X_test)
dists0 = classifier.compute_distances_no_loops(X_test)
dists  = classifier.compute_distances_no_loops(X_test)
print dists.shape

# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

classifier.predict(X_test, k=1, num_loops=0)
classifier.pridict_currency(X_test, y_test, k=1, num_loops=0)

# cross validation

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
X_train_folds = []
y_train_folds = []
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
k_to_accuracies = {}
for k in k_choices:
  validation_accuracies = []
  for i in range(num_folds):
    current_x_test = X_train_folds[i]
    current_y_test = y_train_folds[i]
    current_x_train = []
    current_y_train = []
    for j in range(num_folds):
      if i != j:
        current_x_train.extend(X_train_folds[j])
        current_y_train.extend(y_train_folds[j])

    current_x_train = np.array(current_x_train)
    current_y_train = np.array(current_y_train)

    classifier.train(current_x_train, current_y_train)
    accuracy = classifier.pridict_currency(current_x_test, current_y_test, k)
    validation_accuracies.append(accuracy)

  k_to_accuracies[k] = validation_accuracies
  print 'k = %d, mean accuracy = %f' % (k, np.mean(k_to_accuracies[k]))

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)

# plot the raw observations
for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
