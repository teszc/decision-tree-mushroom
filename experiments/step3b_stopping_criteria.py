# step3_stopping_criteria.py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from decision_tree import DecisionTreeClassifier
from decision_functions import zero_one_loss

def evaluate_stopping_criteria():
    """Evaluate multiple stopping criteria."""
    #Load data
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    
    #Test1: Max depth effect
    print("=== Testing Max Depth Effect ===")
    max_depths = range(1, 21)
    depth_results = {'train_errors': [], 'val_errors': []}
    
    for depth in max_depths:
        clf = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=2,
            splitting_criterion='gini'
        )
        clf.fit(X_train, y_train)
        
        train_err = zero_one_loss(y_train, clf.predict(X_train))
        val_err = zero_one_loss(y_val, clf.predict(X_val))
        
        depth_results['train_errors'].append(train_err)
        depth_results['val_errors'].append(val_err)
        
        if depth % 5 == 0:
            print(f"Depth {depth}: Train Error = {train_err:.4f}, Val Error = {val_err:.4f}")
    
    #Test2: Min samples split effect
    print("\n=== Testing Min Samples Split Effect ===")
    min_samples_splits = [2, 4, 8, 16, 32, 64]
    split_results = {'train_errors': [], 'val_errors': []}
    
    for min_samples in min_samples_splits:
        clf = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=min_samples, 
            splitting_criterion='gini'
        )
        clf.fit(X_train, y_train)
        
        train_err = zero_one_loss(y_train, clf.predict(X_train))
        val_err = zero_one_loss(y_val, clf.predict(X_val))
        
        split_results['train_errors'].append(train_err)
        split_results['val_errors'].append(val_err)
        
        print(f"Min samples {min_samples}: Train Error = {train_err:.4f}, Val Error = {val_err:.4f}")
    
    #Test3: Min samples leaf effect
    print("\n=== Testing Min Samples Leaf Effect ===")
    min_samples_leafs = [1, 2, 4, 8, 16]
    leaf_results = {'train_errors': [], 'val_errors': []}
    
    for min_leaf in min_samples_leafs:
        clf = DecisionTreeClassifier(
            max_depth=15,
            min_samples_leaf=min_leaf,  
            splitting_criterion='gini'
        )
        clf.fit(X_train, y_train)
        
        train_err = zero_one_loss(y_train, clf.predict(X_train))
        val_err = zero_one_loss(y_val, clf.predict(X_val))
        
        leaf_results['train_errors'].append(train_err)
        leaf_results['val_errors'].append(val_err)
        
        print(f"Min leaf {min_leaf}: Train Error = {train_err:.4f}, Val Error = {val_err:.4f}")
    
    #Test4: Min impurity decrease effect
    print("\n=== Testing Min Impurity Decrease Effect ===")
    min_impurity_decreases = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    impurity_results = {'train_errors': [], 'val_errors': []}
    
    for min_impurity in min_impurity_decreases:
        clf = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=2,
            min_impurity_decrease=min_impurity, 
            splitting_criterion='gini'
        )
        clf.fit(X_train, y_train)
        
        train_err = zero_one_loss(y_train, clf.predict(X_train))
        val_err = zero_one_loss(y_val, clf.predict(X_val))
        
        impurity_results['train_errors'].append(train_err)
        impurity_results['val_errors'].append(val_err)
        
        print(f"Min impurity {min_impurity}: Train Error = {train_err:.4f}, Val Error = {val_err:.4f}")
    
    #plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    #Plot1: Max depth
    axes[0, 0].plot(max_depths, depth_results['train_errors'], 'b-', marker='o', label='Training Error')
    axes[0, 0].plot(max_depths, depth_results['val_errors'], 'r-', marker='s', label='Validation Error')
    axes[0, 0].set_xlabel('Max Depth')
    axes[0, 0].set_ylabel('0-1 Loss')
    axes[0, 0].set_title('Effect of Max Depth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    #Plot2: Min samples split
    axes[0, 1].plot(min_samples_splits, split_results['train_errors'], 'b-', marker='o', label='Training Error')
    axes[0, 1].plot(min_samples_splits, split_results['val_errors'], 'r-', marker='s', label='Validation Error')
    axes[0, 1].set_xlabel('Min Samples Split')
    axes[0, 1].set_ylabel('0-1 Loss')
    axes[0, 1].set_title('Effect of Min Samples Split')
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    #Plot3: Min samples leaf
    axes[1, 0].plot(min_samples_leafs, leaf_results['train_errors'], 'b-', marker='o', label='Training Error')
    axes[1, 0].plot(min_samples_leafs, leaf_results['val_errors'], 'r-', marker='s', label='Validation Error')
    axes[1, 0].set_xlabel('Min Samples Leaf')
    axes[1, 0].set_ylabel('0-1 Loss')
    axes[1, 0].set_title('Effect of Min Samples Leaf')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    #Plot4: Min impurity decrease
    axes[1, 1].plot(min_impurity_decreases, impurity_results['train_errors'], 'b-', marker='o', label='Training Error')
    axes[1, 1].plot(min_impurity_decreases, impurity_results['val_errors'], 'r-', marker='s', label='Validation Error')
    axes[1, 1].set_xlabel('Min Impurity Decrease')
    axes[1, 1].set_ylabel('0-1 Loss')
    axes[1, 1].set_title('Effect of Min Impurity Decrease')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    #Save plot
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/stopping_criteria_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    #Summary analysis
    print("\n" + "="*60)
    print("STOPPING CRITERIA ANALYSIS SUMMARY")
    print("="*60)
    
    #Find optimal values based on validation error
    best_depth_idx = np.argmin(depth_results['val_errors'])
    best_depth = list(max_depths)[best_depth_idx]
    
    best_split_idx = np.argmin(split_results['val_errors'])
    best_split = min_samples_splits[best_split_idx]
    
    best_leaf_idx = np.argmin(leaf_results['val_errors'])
    best_leaf = min_samples_leafs[best_leaf_idx]
    
    best_impurity_idx = np.argmin(impurity_results['val_errors'])
    best_impurity = min_impurity_decreases[best_impurity_idx]
    
    print(f"Best Max Depth: {best_depth} (Val Error: {depth_results['val_errors'][best_depth_idx]:.4f})")
    print(f"Best Min Samples Split: {best_split} (Val Error: {split_results['val_errors'][best_split_idx]:.4f})")
    print(f"Best Min Samples Leaf: {best_leaf} (Val Error: {leaf_results['val_errors'][best_leaf_idx]:.4f})")
    print(f"Best Min Impurity Decrease: {best_impurity} (Val Error: {impurity_results['val_errors'][best_impurity_idx]:.4f})")
    
    # Test combined best parameters
    print(f"\n=== Testing Combined Best Parameters ===")
    best_clf = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_split=best_split,
        min_samples_leaf=best_leaf,
        min_impurity_decrease=best_impurity,
        splitting_criterion='gini'
    )
    best_clf.fit(X_train, y_train)
    
    best_train_err = zero_one_loss(y_train, best_clf.predict(X_train))
    best_val_err = zero_one_loss(y_val, best_clf.predict(X_val))
    
    print(f"Combined Best Model - Train Error: {best_train_err:.4f}, Val Error: {best_val_err:.4f}")
    
    return (depth_results, split_results, leaf_results, impurity_results, 
            {'max_depth': best_depth, 'min_samples_split': best_split, 
             'min_samples_leaf': best_leaf, 'min_impurity_decrease': best_impurity})

if __name__ == "__main__":
    evaluate_stopping_criteria()