import numpy as np

def gini(y):
    """Compute Gini impurity of a label array y."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return 1.0 - np.sum(prob ** 2)

def entropy(y):
    """Compute entropy of a label array y."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    prob = prob[prob > 0]  # Avoid log2(0)
    return -np.sum(prob * np.log2(prob))

def misclassification_error(y):
    """Compute misclassification error rate."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return 1.0 - np.max(prob)

def info_gain_ratio(parent_y, left_y, right_y):
    """Compute Information Gain Ratio for a split."""
    n = len(parent_y)
    n_left = len(left_y)
    n_right = len(right_y)
    
    if n == 0 or n_left == 0 or n_right == 0:
        return 0.0
    
    #Information gain using entropy
    impurity_parent = entropy(parent_y)
    weighted_impurity = (n_left / n) * entropy(left_y) + (n_right / n) * entropy(right_y)
    info_gain = impurity_parent - weighted_impurity
    
    #Split info (intrinsic information)
    split_proportions = np.array([n_left / n, n_right / n])
    split_proportions = split_proportions[split_proportions > 0]
    split_info = -np.sum(split_proportions * np.log2(split_proportions))
    
    if split_info == 0:
        return 0.0
    
    return info_gain / split_info