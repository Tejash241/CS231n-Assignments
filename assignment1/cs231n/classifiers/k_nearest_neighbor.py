import numpy as np

class KNearestNeighbor:
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Input:
    X - A num_test x dimension array where each row is a test point.
    k - The number of nearest neighbors that vote for predicted label
    num_loops - Determines which method to use to compute distances
                between training points and test points.

    Output:
    y - A vector of length num_test, where y[i] is the predicted label for the
        test point X[i, :].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data. 

    TEJASH EDIT: Distance used will be Absolute Difference between self.X_train and X

    Input:
    X - An num_test x dimension array where each row is a test point.

    Output:
    dists - A num_test x num_train array where dists[i, j] is the distance
            between the ith test point and the jth training point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        dists[i][j] = np.sqrt(np.sum(np.square(self.X_train[j]-X[i])))

    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    TEJASH EDIT: Distance will be Absolute Distance between self.X_train and X

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      # print 'sdf', np.abs(np.sum(self.X_train-np.tile(X[i], (num_train, 1)), axis=1)).shape # Testing
      dists[i] = np.sqrt(np.sum(np.square(self.X_train-np.tile(X[i], (num_train, 1))), axis=1))
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    TEJASH EDIT: Distance will be Absolute Difference between self.X_train and X

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    Xtrain_sq = np.square(self.X_train).sum(axis=1)
    X_sq = np.square(X).sum(axis=1)
    mux = 2*(X.dot(self.X_train.T))
    dists = np.sqrt(Xtrain_sq+np.matrix(X_sq).T-mux)

    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Input:
    dists - A num_test x num_train array where dists[i, j] gives the distance
            between the ith test point and the jth training point.

    Output:
    y - A vector of length num_test where y[i] is the predicted label for the
        ith test point.
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      closest_y = [] # A list of length k storing the labels of the k nearest neighbors to the ith test point.
      nearest_k_matches = np.argsort(dists[i])[:k] # holds the indices of the minimum k values in dists
      closest_y = self.y_train[nearest_k_matches] # holds the labels of the nearest k matches
      y_pred[i] = np.bincount(closest_y).argmax() # holds the maximum occuring label value in closest_y
      """
      LEARNING:
      np.bincount (can be used only for 1d arrays with nonnegative ints) produces a ndarray 
      (which is a container for a numpy array of the same dimension) that contains the count of each integer occuring 
      in the array.
      np.argmax provides the first value of the maximum occurence in an array. 
      Applying it to the o/p of np.bincount will give the smaller label in case of ties
      """

    return y_pred

