import numpy as np

def zero_one_loss(y_true, y_pred):
    """Compute zero-one loss (fraction of incorrect predictions)."""
    return np.mean(y_true != y_pred)

def accuracy_score(y_true, y_pred):
    """Compute accuracy score."""
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix for binary classification."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])