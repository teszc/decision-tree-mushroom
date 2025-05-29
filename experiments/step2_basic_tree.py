import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from decision_tree import DecisionTreeClassifier
from decision_functions import zero_one_loss, accuracy_score

def evaluate_splitting_criteria():
    """Evaluating 3 different splitting criteria."""
    #Load processed data
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    
    #Test different splitting criteria
    criteria = ['gini', 'entropy', 'misclassification', 'info_gain_ratio']
    
    results = []
    
    for criterion in criteria:
        print(f"\n=== Testing {criterion.upper()} criterion ===")
        
        clf = DecisionTreeClassifier(
            splitting_criterion=criterion,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2
        )
        
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        
        # Evaluation
        train_error = zero_one_loss(y_train, y_pred_train)
        val_error = zero_one_loss(y_val, y_pred_val)
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        
        results.append({
            'criterion': criterion,
            'train_error': train_error,
            'val_error': val_error,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        })
        
        print(f"Training accuracy: {train_acc:.4f} (error: {train_error:.4f})")
        print(f"Validation accuracy: {val_acc:.4f} (error: {val_error:.4f})")
    
    #Summary table
    print("\n" + "="*60)
    print("SUMMARY: Splitting Criteria Comparison")
    print("="*60)
    print(f"{'Criterion':<15} {'Train Acc':<10} {'Val Acc':<10} {'Train Err':<10} {'Val Err':<10}")
    print("-"*60)
    
    for result in results:
        print(f"{result['criterion']:<15} {result['train_accuracy']:<10.4f} "
              f"{result['val_accuracy']:<10.4f} {result['train_error']:<10.4f} "
              f"{result['val_error']:<10.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_splitting_criteria()