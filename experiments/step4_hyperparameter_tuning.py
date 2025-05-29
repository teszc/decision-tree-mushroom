import numpy as np
import os
import sys
import pickle
from sklearn.model_selection import ParameterSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from decision_tree import DecisionTreeClassifier
from decision_functions import zero_one_loss

def hyperparameter_tuning():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    param_grid = {
        'max_depth': [3, 5, 8, 12, 15, 20, None],
        'min_samples_split': [2, 4, 8, 16, 32],
        'min_samples_leaf': [1, 2, 4, 8],
        'splitting_criterion': ['gini', 'entropy', 'info_gain_ratio'],
        'min_impurity_decrease': [0.0, 0.001, 0.01, 0.05]
    }

    n_iter = 50
    sampled_params = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

    print(f"Randomized hyperparameter tuning with {n_iter} sampled configurations...")

    best_params = None
    best_val_error = float('inf')
    results = []
    failed_configs = 0

    for i, params in enumerate(sampled_params):
        print(f" Testing {i+1}/{n_iter}...")
        try:
            clf = DecisionTreeClassifier(
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                min_impurity_decrease=params['min_impurity_decrease'],
                splitting_criterion=params['splitting_criterion'],
                categorical_threshold=10
            )
            clf.fit(X_train, y_train)
            val_error = zero_one_loss(y_val, clf.predict(X_val))
            results.append({'params': params.copy(), 'val_error': val_error})

            if val_error < best_val_error:
                best_val_error = val_error
                best_params = params.copy()
        except Exception as e:
            failed_configs += 1
            print(f" Failed config #{failed_configs}: {e}")

    print(f"\n Tuning complete. {len(results)} successful, {failed_configs} failed.")
    print(f" Best validation error: {best_val_error:.6f}")
    print("🔧 Best parameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    #Final evaluation
    final_clf = DecisionTreeClassifier(**best_params)
    final_clf.fit(X_train, y_train)
    y_train_pred = final_clf.predict(X_train)
    y_val_pred = final_clf.predict(X_val)
    y_test_pred = final_clf.predict(X_test)

    train_error = zero_one_loss(y_train, y_train_pred)
    val_error = zero_one_loss(y_val, y_val_pred)
    test_error = zero_one_loss(y_test, y_test_pred)

    print(f"\n Final Errors — Train: {train_error:.4f}, Val: {val_error:.4f}, Test: {test_error:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/hyperparameter_tuning_results.pkl", "wb") as f:
        pickle.dump({
            'best_params': best_params,
            'best_val_error': best_val_error,
            'all_results': results,
            'final_errors': {
                'train': train_error,
                'val': val_error,
                'test': test_error
            }
        }, f)
    print(" Results saved to results/hyperparameter_tuning_results.pkl")

    return best_params, final_clf, results

def analyze_performance_tradeoffs():
    print(f"\n{'='*80}")
    print("PERFORMANCE TRADE-OFF ANALYSIS")
    print(f"{'='*80}")

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))

    trade_off_configs = [
        {'name': 'Very Shallow', 'max_depth': 2, 'min_samples_split': 32},
        {'name': 'Shallow', 'max_depth': 5, 'min_samples_split': 16},
        {'name': 'Moderate', 'max_depth': 10, 'min_samples_split': 8},
        {'name': 'Deep', 'max_depth': 20, 'min_samples_split': 2},
        {'name': 'Very Deep', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
    ]

    print(f"{'Configuration':<20} {'Train Error':<12} {'Val Error':<12} {'Gap':<12} {'Interpretation'}")
    print("-" * 80)

    for config in trade_off_configs:
        name = config.pop('name')
        clf = DecisionTreeClassifier(splitting_criterion='gini', min_samples_leaf=2, **config)
        clf.fit(X_train, y_train)
        train_err = zero_one_loss(y_train, clf.predict(X_train))
        val_err = zero_one_loss(y_val, clf.predict(X_val))
        gap = val_err - train_err

        interpretation = (
            "Overfitting" if gap > 0.05 else
            "Underfitting" if train_err > 0.1 else
            "Good fit"
        )
        print(f"{name:<20} {train_err:<12.6f} {val_err:<12.6f} {gap:<12.6f} {interpretation}")

if __name__ == "__main__":
    best_params, final_clf, results = hyperparameter_tuning()
    analyze_performance_tradeoffs()
