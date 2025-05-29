import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from decision_tree import DecisionTreeClassifier
from decision_functions import zero_one_loss, accuracy_score, confusion_matrix

def final_comprehensive_evaluation():
    """
    evaluation of multiple tree configurations.
    """
    #load data
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    #Defining different tree configurations
    configurations = [
        {
            'name': 'Shallow Gini Tree',
            'params': {
                'splitting_criterion': 'gini',
                'max_depth': 5,
                'min_samples_split': 8,
                'min_samples_leaf': 4
            }
        },
        {
            'name': 'Deep Entropy Tree',
            'params': {
                'splitting_criterion': 'entropy',
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        },
        {
            'name': 'Balanced Info Gain Ratio Tree',
            'params': {
                'splitting_criterion': 'info_gain_ratio',
                'max_depth': 10,
                'min_samples_split': 4,
                'min_samples_leaf': 2
            }
        },
        {
            'name': 'Conservative Misclassification Tree',
            'params': {
                'splitting_criterion': 'misclassification',
                'max_depth': 8,
                'min_samples_split': 16,
                'min_samples_leaf': 8
            }
        }
    ]
    
    print("="*80)
    print("COMPREHENSIVE DECISION TREE EVALUATION")
    print("="*80)
    
    results = []
    
    for config in configurations:
        print(f"\n--- {config['name']} ---")
        
        #Train model
        clf = DecisionTreeClassifier(**config['params'])
        clf.fit(X_train, y_train)
        
        #Predictions
        y_train_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)
        y_test_pred = clf.predict(X_test)
        
        #Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        train_err = zero_one_loss(y_train, y_train_pred)
        val_err = zero_one_loss(y_val, y_val_pred)
        test_err = zero_one_loss(y_test, y_test_pred)
        
        #Confusion matrix on test set
        cm = confusion_matrix(y_test, y_test_pred)
        
        result = {
            'name': config['name'],
            'params': config['params'],
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'train_error': train_err,
            'val_error': val_err,
            'test_error': test_err,
            'confusion_matrix': cm
        }
        
        results.append(result)
        
        print(f"Parameters: {config['params']}")
        print(f"Training Accuracy:   {train_acc:.4f} (Error: {train_err:.4f})")
        print(f"Validation Accuracy: {val_acc:.4f} (Error: {val_err:.4f})")
        print(f"Test Accuracy:       {test_acc:.4f} (Error: {test_err:.4f})")
        print(f"Confusion Matrix (Test):")
        print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
        print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    #Summary table
    print(f"\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Model':<30} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Train Err':<10} {'Val Err':<10} {'Test Err':<10}")
    print("-"*100)
    
    for result in results:
        print(f"{result['name']:<30} {result['train_accuracy']:<10.4f} "
              f"{result['val_accuracy']:<10.4f} {result['test_accuracy']:<10.4f} "
              f"{result['train_error']:<10.4f} {result['val_error']:<10.4f} "
              f"{result['test_error']:<10.4f}")
    
    #find best model based on validation performance
    best_model = min(results, key=lambda x: x['val_error'])
    print(f"\nBest model based on validation error: {best_model['name']}")
    print(f"Test accuracy of best model: {best_model['test_accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    final_comprehensive_evaluation()