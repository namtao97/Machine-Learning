import numpy as np

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    y_predict = [sum([_w * _x for _w, _x in zip(x, W)]) for x in X]
    for yi, yi_pred in zip(y, y_predict):
        loss += (yi - yi_pred) ** 2
    loss /= 2 * len(y_predict)

    dW = sum([yi_pred - yi] * _x for _x, yi, yi_pred in zip(X, y, y_predict)) / len(y_predict)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    y_predict = X.dot(W)
    loss_list = y_predict - y
    loss = sum(loss_list * loss_list) / (2 * len(y_predict))
    tmp = np.array(y_predict - y)
    dW = (X.transpose().dot(tmp)) / len(y_predict)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW