import numpy as np


def tied_rank(x):
    """
    Computes the tied rank of elements in x.
    This function computes the tied rank of elements in x.
    Parameters
    ----------
    x : list of numbers, numpy array
    Returns
    -------
    score : list of numbers
            The tied rank f each element in x
    """
    sorted_x = sorted(zip(x, range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank + 1 + i) / 2.0
            last_rank = i
        if i == len(sorted_x) - 1:
            for j in range(last_rank, i + 1):
                r[sorted_x[j][1]] = float(last_rank + i + 2) / 2.0
    return r


def auc(y_true, y_score):
    """
    copy from: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py

    Computes the area under the receiver-operater characteristic (AUC)
    This function computes the AUC error metric for binary classification.
    Parameters
    ----------
    y_true : list of binary numbers, numpy array
             The ground truth value
    y_score : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.
    Returns
    -------
    score : double
            The mean squared error between actual and posterior
    """
    r = tied_rank(y_score)
    num_positive = len([0 for x in y_true if x == 1])
    num_negative = len(y_true) - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if y_true[i] == 1])
    auc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) /
           (num_negative * num_positive))
    return auc


def logloss(y_true, y_pred, normalize=True):
    loss_array = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    if normalize:
        return np.mean(loss_array)
    else:
        return np.sum(loss_array)
