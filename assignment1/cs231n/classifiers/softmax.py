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

  #############################################################################
  # Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_dims = W.shape[1]
  num_classes = W.shape[0]
  num_train = X.shape[1]

  for i in xrange(num_train):
    scores = W.dot(X[:,i])
    exp_scores = np.exp(scores)
    prob_scores = exp_scores/np.sum(exp_scores)
    loss += -np.log(prob_scores[y[i]])

    for d in xrange(num_dims):
      for k in xrange(num_classes):
        if k == y[i]:
          dW[k, d] += X[d, i] * (prob_scores[k]-1)
        else:
          dW[k, d] += X[d, i] * prob_scores[k]

  loss /= num_train
  dW /=num_train

  loss += 0.5 * reg * np.sum(W**2)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[1]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
