import numpy as np
from numpy import ndarray


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def mae(pred: ndarray, label: ndarray) -> ndarray:
    """Returns the mean absolute error between the predicted values and the
    actual values.

    Args:
        pred (ndarray): the array of predicted values.
        label (ndarray): the array of ground truths.
    Returns:
        (ndarray): mean absolute errors.
    """
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(label, pred)))


def sse(pred: ndarray, label: ndarray) -> ndarray:
    """Returns the residual sum of squared errors between the predicted
    values and the actual values.

    Args:
        pred (ndarray): the array of predicted values.
        label (ndarray): the array of ground truths.
    Returns:
        (ndarray): residual sum of squared errors.
    """
    return np.sum((label-pred)**2)


def mse(pred: ndarray, label: ndarray) -> ndarray:
    """Returns the mean squared errors between the predicted
    values and the actual values.

    Args:
        pred (ndarray): the array of predicted values.
        label (ndarray): the array of ground truths.
    Returns:
        (ndarray): mean squared errors.
    """
    """ Mean Squared Error """
    return np.mean(np.square(_error(label, pred)))


def rmse(pred: ndarray, label: ndarray) -> ndarray:
    """Returns the root mean squared error between the predicted values
    and the actual values.

    Args:
        pred (ndarray): the array of predicted values.
        label (ndarray): the array of ground truths.
    Returns:
        (ndarray): root mean squared errors.
    """
    """ Root Mean Squared Error """
    return np.sqrt(mse(label, pred))