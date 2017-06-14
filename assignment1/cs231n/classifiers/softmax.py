import numpy as np
import math
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  N, D = X.shape
  C = W.shape[1]

  for i in xrange(N):
    scores = X[i].dot(W)
    y_i = y[i]
    s = np.sum(np.exp(scores))
    for j in xrange(C):
        o = math.exp(scores[j]) / s
        if j == y_i:
            dW[:,j] += (-1 + o) * X[i]
            loss += -math.log(o)
        else:
            dW[:,j] += o * X[i]

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW /= N
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
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  N = X.shape[0]
  C = W.shape[1]

  scores = np.exp(X.dot(W))
  sum_s = np.sum(scores, axis = 1).reshape(-1, 1)
  output = scores / sum_s
  loss = np.sum(-np.log(output[range(N), y]))
  output[range(N), y] -= 1
  dW = X.T.dot(output)
   

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

