import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j, :] += X[:, i]
        dW[y[i], :] -= X[:, i]

  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W) # Reg(W) = sigmai(sigmaj(Wij(square))), how did divide by 2 come?
  dW += reg*W # LEARNING: dW(Loss+Regularization) = dW(loss) + dW(Regularization); dW(Reg) = 0.5*reg*2*W = reg*W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[1]
  num_classes = W.shape[0]
  test_scores = W.dot(X) # test_scores has the shape of (10, 49000)
  correct_scores = test_scores[y, np.arange(num_train)] # correct_scores has a shape of (49000,)
  margin = test_scores-correct_scores+1 # margin will have a shape of (10, 49000)
  margin[y, np.arange(num_train)] = 0
  loss = np.sum(margin[margin>0])

  # for i in xrange(num_train):
  #   test_score = W.dot(X[:, i]) # test_score has shape (10,)
  #   correct_score = test_score[y[i]] # correct_score is a single number
  #   margin = test_score-correct_score+1 # margin will have a shape of (10, )
  #   margin[y[i]] = 0
  #   loss += np.sum(margin[margin>0])

  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)

  bin_matrix = np.maximum(0, margin)
  bin_matrix[bin_matrix>0] = 1 # bin_matrix has the same shape as margin
  col_sum = np.sum(bin_matrix, axis=0)
  bin_matrix[y, range(num_train)] = -col_sum[range(num_train)]

  dW = bin_matrix.dot(X.T)

  dW /= num_train
  dW += reg*W

  return loss, dW