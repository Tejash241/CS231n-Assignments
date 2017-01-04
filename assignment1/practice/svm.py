import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.linear_svm import svm_loss_naive, svm_loss_vectorized
from cs231n.gradient_check import grad_check_sparse
from cs231n.classifiers import LinearSVM
import matplotlib.pyplot as plt
from datetime import datetime

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)

num_training = 49000
num_validation = 1000
num_test = 1000

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# print 'Train data shape: ', X_train.shape # prints Train data shape:  (49000, 32, 32, 3)
# print 'Train labels shape: ', y_train.shape # prints Train labels shape:  (49000,)
# print 'Validation data shape: ', X_val.shape # prints Validation data shape:  (1000, 32, 32, 3)
# print 'Validation labels shape: ', y_val.shape # prints Validation labels shape:  (1000,)
# print 'Test data shape: ', X_test.shape # prints Test data shape:  (1000, 32, 32, 3)
# print 'Test labels shape: ', y_test.shape # prints Test labels shape:  (1000,)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print 'Training data shape: ', X_train.shape # prints Training data shape:  (49000, 3072)
print 'Validation data shape: ', X_val.shape # prints Validation data shape:  (1000, 3072)
print 'Test data shape: ', X_test.shape # prints Test data shape:  (1000, 3072)

mean_image = np.mean(X_train, axis=0)

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
# Also, lets transform both data matrices so that each image is a column.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T

print X_train.shape, X_val.shape, X_test.shape # prints (3073, 49000) (3073, 1000) (3073, 1000)

# generate a random SVM weight matrix of small numbers
W = np.random.randn(10, 3073) * 0.0001 
# start_time = datetime.now()
# loss, grad = svm_loss_naive(W, X_train, y_train, 0.00001)
# print 'loss naive: %f with time taken %s' % (loss, datetime.now()-start_time)

loss1, grad1 = svm_loss_vectorized(W, X_train, y_train, 0.00001)
# print 'loss vectorized: %f with time taken %s' % (loss1, datetime.now()-start_time)
# difference = np.linalg.norm(grad1 - grad, ord='fro')
# print 'difference: %f' % difference

# svm = LinearSVM()
# start_time = datetime.now()
# loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4, num_iters=15000, verbose=True)
# print 'Time taken', datetime.now()-start_time

# # A useful debugging strategy is to plot the loss as a function of iteration number:
# plt.plot(loss_hist)
# plt.xlabel('Iteration number')
# plt.ylabel('Loss value')
# plt.savefig('svm_loss.png')

# # Write the LinearSVM.predict function and evaluate the performance on both the
# # training and validation set
# y_train_pred = svm.predict(X_train)
# print 'training accuracy: %f' % (np.mean(y_train == y_train_pred), )
# y_val_pred = svm.predict(X_val)
# print 'validation accuracy: %f' % (np.mean(y_val == y_val_pred), )

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.4 on the validation set.
# learning_rates = [1e-7, 5e-5, 1e-10, 8e-9, 1, 10]
# regularization_strengths = [3e4, 9e-7, 1e-5, 1, 5e4]
learning_rates = [1e-7]
regularization_strengths = [3e4]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

for lr in learning_rates:
	for rs in regularization_strengths:
		print 'Training SVM for learning rate/step size %f and lambda %f' % (lr, rs)
		svm = LinearSVM()
		loss_hist = svm.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=1500, verbose=True)
		y_train_pred = svm.predict(X_train)
		train_acc = np.mean(y_train==y_train_pred)
		y_val_pred = svm.predict(X_val)
		val_acc = np.mean(y_val==y_val_pred)
		# print 'lr %f rs %f train_acc %f val_acc %f' % (lr, rs, train_acc, val_acc)
		if val_acc > best_val:
			best_val = val_acc
			best_svm = svm

		results[(lr, rs)] = (train_acc, val_acc)
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val # best_val obtained is approx 38.9%

# # Visualize the cross-validation results
# import math
# x_scatter = [math.log10(x[0]) for x in results]
# y_scatter = [math.log10(x[1]) for x in results]

# # plot training accuracy
# sz = [results[x][0]*1500 for x in results] # default size of markers is 20
# plt.subplot(1,2,1)
# plt.scatter(x_scatter, y_scatter, sz)
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('CIFAR-10 training accuracy')

# # plot validation accuracy
# sz = [results[x][1]*1500 for x in results] # default size of markers is 20
# plt.subplot(1,2,2)
# plt.scatter(x_scatter, y_scatter, sz)
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('CIFAR-10 validation accuracy')
# plt.savefig('validation_accuracy.png')

# Evaluate the best svm on test set
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print 'linear SVM on raw pixels final test set accuracy: %f' % test_accuracy

# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = best_svm.W[:,:-1] # strip out the bias
w = w.reshape(10, 32, 32, 3)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in xrange(10):
  plt.subplot(2, 5, i + 1)
    
  # Rescale the weights to be between 0 and 255
  wimg = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
  plt.imshow(wimg.astype('uint8'))
  plt.axis('off')
  plt.title(classes[i])
  plt.savefig('mean_images.png')