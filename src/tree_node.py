class TreeNode:
    def __init__(self, is_leaf=False, prediction=None, feature_index=None, 
                 test_func=None, left=None, right=None, threshold=None):
        """
        Initialize a TreeNode.
        
        Parameters:
        - is_leaf (bool): Whether this node is a leaf node
        - prediction: Class prediction if it's a leaf node
        - feature_index (int): Index of feature used for splitting
        - test_func (function): Function that takes a feature value and returns True/False
        - left (TreeNode): Left child node
        - right (TreeNode): Right child node
        - threshold: The threshold/value used for splitting (for interpretability)
        """
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_index = feature_index
        self.test_func = test_func
        self.left = left
        self.right = right
        self.threshold = threshold  
    
    def evaluate(self, x):
        """
        evaluate a single data point to decide which child to follow.
        
        Parameters:
        - x (list or np.array): Feature vector
        
        Returns:
        - TreeNode: either self.left or self.right
        """
        if self.is_leaf:
            return self
        
        if self.test_func(x[self.feature_index]):
            return self.left
        else:
            return self.right