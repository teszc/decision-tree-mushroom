import numpy as np
from tree_node import TreeNode
from splitting_criteria import gini, entropy, misclassification_error, info_gain_ratio

class DecisionTreeClassifier:
    def __init__(self, 
                 max_depth=None, 
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_impurity_decrease=0.0,
                 splitting_criterion='gini',
                 categorical_threshold=10):
        """
        Decision Tree Classifier with multiple stopping criteria.
        
        Parameters:
        - max_depth: Maximum depth of the tree
        - min_samples_split: Minimum samples required to split a node
        - min_samples_leaf: Minimum samples required in a leaf node
        - min_impurity_decrease: Minimum impurity decrease required for split
        - splitting_criterion: 'gini', 'entropy', 'misclassification', or 'info_gain_ratio'
        - categorical_threshold: Features with <= this many unique values treated as categorical
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.splitting_criterion = splitting_criterion
        self.categorical_threshold = categorical_threshold
        self.root = None
        self.feature_types = None  
        
        #Map criterion names to functions
        self.criterion_funcs = {
            'gini': gini,
            'entropy': entropy,
            'misclassification': misclassification_error
        }
        
        if splitting_criterion not in self.criterion_funcs and splitting_criterion != 'info_gain_ratio':
            raise ValueError(f"Invalid splitting criterion: {splitting_criterion}")
    
    def fit(self, X, y):
        """Fit the decision tree to training data."""
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        
        #Determine feature types based on unique values
        self.feature_types = []
        for i in range(self.n_features):
            unique_vals = len(np.unique(X[:, i]))
            self.feature_types.append(unique_vals <= self.categorical_threshold)
        
        self.root = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        num_samples = len(y)
        
        #Check stopping conditions
        if self._should_stop(X, y, depth):
            leaf_value = self._most_common_label(y)
            return TreeNode(is_leaf=True, prediction=leaf_value)
        
        #Find best split
        best_split = self._find_best_split(X, y)
        
        if best_split is None or best_split['gain'] < self.min_impurity_decrease:
            leaf_value = self._most_common_label(y)
            return TreeNode(is_leaf=True, prediction=leaf_value)
        
        #Create split masks
        left_mask = self._create_split_mask(X[:, best_split['feature_idx']], 
                                          best_split['threshold'], 
                                          best_split['is_categorical'])
        right_mask = ~left_mask
        
        #Check minimum samples in leaf constraint
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            leaf_value = self._most_common_label(y)
            return TreeNode(is_leaf=True, prediction=leaf_value)
        
        #Recursively build subtrees
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        #Create test function
        feature_idx = best_split['feature_idx']
        threshold = best_split['threshold']
        is_categorical = best_split['is_categorical']
        
        if is_categorical:
            test_func = lambda val: val in threshold
        else:
            test_func = lambda val: val <= threshold
        
        return TreeNode(
            is_leaf=False,
            feature_index=feature_idx,
            test_func=test_func,
            threshold=threshold,
            left=left_node,
            right=right_node
        )
    
    def _should_stop(self, X, y, depth):
        """Check if we should stop splitting."""
        num_samples = len(y)
        num_labels = len(np.unique(y))
        
        #Pure node
        if num_labels == 1:
            return True
        
        #Depth limit
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        
        #Minimum samples to split
        if num_samples < self.min_samples_split:
            return True
        
        return False
    
    def _find_best_split(self, X, y):
        """Find the best split for the current node."""
        best_gain = 0
        best_split = None
        
        for feature_idx in range(self.n_features):
            feature_values = X[:, feature_idx]
            is_categorical = self.feature_types[feature_idx]
            
            if is_categorical:
                #Categorical feature: try each unique value
                unique_values = np.unique(feature_values)
                for val in unique_values:
                    threshold = [val] 
                    gain = self._calculate_split_gain(X, y, feature_idx, threshold, True)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {
                            'feature_idx': feature_idx,
                            'threshold': threshold,
                            'is_categorical': True,
                            'gain': gain
                        }
            else:
                #Numerical feature: try midpoints between sorted unique values
                unique_values = np.unique(feature_values)
                if len(unique_values) <= 1:
                    continue
                
                sorted_vals = np.sort(unique_values)
                thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2
                
                for threshold in thresholds:
                    gain = self._calculate_split_gain(X, y, feature_idx, threshold, False)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {
                            'feature_idx': feature_idx,
                            'threshold': threshold,
                            'is_categorical': False,
                            'gain': gain
                        }
        
        return best_split
    
    def _calculate_split_gain(self, X, y, feature_idx, threshold, is_categorical):
        """Calculate the gain from a potential split."""
        feature_values = X[:, feature_idx]
        left_mask = self._create_split_mask(feature_values, threshold, is_categorical)
        right_mask = ~left_mask
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        #Skip if split results in empty partition
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        if self.splitting_criterion == 'info_gain_ratio':
            return info_gain_ratio(y, y_left, y_right)
        else:
            criterion_func = self.criterion_funcs[self.splitting_criterion]
            impurity_parent = criterion_func(y)
            n = len(y)
            n_left = len(y_left)
            n_right = len(y_right)
            
            weighted_impurity = (n_left / n) * criterion_func(y_left) + \
                               (n_right / n) * criterion_func(y_right)
            
            return impurity_parent - weighted_impurity
    
    def _create_split_mask(self, feature_values, threshold, is_categorical):
        """Create boolean mask for splitting."""
        if is_categorical:
            return np.array([val in threshold for val in feature_values])
        else:
            return feature_values <= threshold
    
    def _most_common_label(self, y):
        """Return the most common label in y."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def predict_one(self, x):
        """Predict class for a single sample."""
        node = self.root
        while not node.is_leaf:
            node = node.evaluate(x)
        return node.prediction
    
    def predict(self, X):
        """Predict classes for multiple samples."""
        return np.array([self.predict_one(x) for x in X])