"""Loss functions."""

import tensorflow as tf
import semver

# from ipdb import set_trace as debug

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    assert max_grad > 0.

    x = y_true - y_pred

    return tf.where(
      tf.abs(x) <= max_grad, 
      .5*tf.square(x),
      max_grad*tf.abs(x)-.5*max_grad*max_grad
    )

def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    loss = huber_loss(y_true,y_pred,max_grad)
    return tf.reduce_mean(loss, 0)
