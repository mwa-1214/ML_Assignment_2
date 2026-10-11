import numpy as np


def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)

    if total == 0:
        return 0.0

    return correct / total


def confusion_matrix(y_true, y_pred):
    # Class 0 = no fault, Class 1 = fault
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([
        [tn, fp],
        [fn, tp]
    ])


def missed_faults(y_true, y_pred):
    # Actual fault but predicted as no fault
    return int(np.sum((y_true == 1) & (y_pred == 0)))