import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor
from datetime import datetime

# Initial loaders and checks
# %matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# %load_ext autoreload
# %autoreload 2

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

print 'Training data shape: ', X_train.shape # prints Training data shape:  (50000, 32, 32, 3)
print 'Training labels shape: ', y_train.shape # prints Training labels shape:  (50000,)
print 'Test data shape: ', X_test.shape # prints Test data shape:  (10000, 32, 32, 3)
print 'Test labels shape: ', y_test.shape # prints Test labels shape:  (10000,)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()

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
print X_train.shape, X_test.shape # prints (5000, 3072) (500, 3072)

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

start_timer = datetime.now()
dists = classifier.compute_distances_two_loops(X_test)
print dists.shape # prints (500, 5000)
print 'Time taken by two loops', str(datetime.now()-start_timer) # prints Time taken by two loops 0:02:13.502102

# plt.imshow(dists, interpolation='none')

y_test_pred = classifier.predict_labels(dists, k=1)
# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy) # prints Got 69 / 500 correct => accuracy: 0.138000

start_timer = datetime.now()
dists_one = classifier.compute_distances_one_loop(X_test)
print 'Time taken by one loop', datetime.now()-start_timer # prints Time taken by one loop 
difference = np.linalg.norm(dists - dists_one, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'

start_timer = datetime.now()
dists_two = classifier.compute_distances_no_loops(X_test)
print 'Time taken by no loop', datetime.now()-start_timer # prints Time taken by one loop 
# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'