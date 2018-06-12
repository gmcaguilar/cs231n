import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

    
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_classes_above_margin = 0
    
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        num_classes_above_margin += 1
        loss += margin
        
        # gradient for non-correct class weight
        dW[:,j] += X[i]
        
    # gradient for correct class weight 
    dW[:,y[i]] -= X[i] * num_classes_above_margin
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Average the gradient across the batch and add gradient of regularization term
  dW = dW /num_train + 2*reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  # compute scores matrix and get correct class scores
  scores = np.dot(X, W)
  correct_class_scores = np.choose(y, scores.T) 
    
  # remove correct class scores since we would not compute the difference for those
  mask = np.ones(scores.shape, dtype=bool)
  mask[range(scores.shape[0]),y] = False
  scores_ = scores[mask].reshape(scores.shape[0], scores.shape[1]-1)
  
  # compute margins
  margin = scores_ - correct_class_scores[..., np.newaxis] + 1

  # only add margin to our loss if it's greater than 0 (max function)
  # make margins less than 0 equal to 0 to prevent them from changing the loss
  margin[margin < 0] = 0
    
  # average data loss over size of the batch and add regularization term
  num_train = X.shape[0]
  loss = np.sum(margin) / num_train
  loss += reg * np.sum(W * W)
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  original_margin = scores - correct_class_scores[...,np.newaxis] + 1
    
  # create a mask that identifies where margin > 0
  margin_mask = (original_margin > 0).astype(float)
  
  # count how many values > 0 for each image but don't count correct class hence -1
  num_classes_above_zero = margin_mask.sum(1) - 1

  # make correct class margin be negative total of how many > 0 
  margin_mask[range(margin_mask.shape[0]), y] = -num_classes_above_zero
  
  # calculate gradient
  dW = np.dot(X.T, margin_mask)
  
  # average over batch and add regularization derivative
  dW /= num_train + 2 * reg * W
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
