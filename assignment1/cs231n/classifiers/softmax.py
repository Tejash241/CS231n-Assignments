import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_classes = W.shape[0]
  for i in xrange(num_train):
    scores = W.dot(X[:, i]) # scores has the shape of (10, )
    scores -= np.max(scores) # numerical stability trick
    scores = np.exp(scores)
    score_sum = np.sum(scores)
    correct_score = scores[y[i]]
    S = correct_score/score_sum
    loss += -np.log(S)
    # LEARNING - softmax gradient formula derived - http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    for j in xrange(num_classes):
      dW[j, :] += [S-(j==y[i])] * X[:, i]

  loss /= num_train
  dW /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_classes = W.shape[0]
  scores = W.dot(X) # has shape (num_classes, num_train)
  scores -= np.max(scores)
  scores = np.exp(scores)
  correct_scores = scores[y, range(num_train)] # has shape (49000,)
  scores_sum = np.sum(scores, axis=0)
  S = correct_scores/scores_sum # same shape as correct_scores
  loss = np.sum(-np.log(S))
  bin_matrix = np.zeros_like(scores)
  bin_matrix[y, range(num_train)] = 1
  dW = np.dot(S-bin_matrix, X.T)
  loss /= num_train
  dW /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W

  return loss, dW
