data/
Contains the dataset and all preprocessed files.
secondary_data.csv: Raw mushroom dataset (semicolon-delimited).
processed/:
X_train.npy, X_test.npy, X_val.npy: NumPy arrays for train/test/validation features.
y_train.npy, y_test.npy, y_val.npy: NumPy arrays for train/test/validation labels.
encoders.pkl: Pickle file storing label encoders used for categorical preprocessing.

src/
Source code for the custom decision tree implementation.
tree_node.py: Defines the TreeNode class used to build the tree structure.
decision_tree.py: Implements the DecisionTreeClassifier class, including training and prediction logic.
decision_functions.py: Contains utilities for decision-making at nodes (e.g., thresholds, set membership).
splitting_criteria.py: Implements splitting heuristics (e.g., Gini index, entropy, gain ratio).
__init__.py: Marks this folder as a Python package.
__pycache__/: Automatically generated Python bytecode (can be ignored or excluded from Git).

experiments/
Scripts for different stages of the project.
step1_data_exploration.py: Loads, cleans, encodes, and splits the dataset. Also generates class distribution plots.
step2_basic_tree.py: Trains a basic decision tree with fixed depth and reports performance.
step3b_stopping_criteria.py: Compares tree performance under various stopping criteria.
step4_hyperparameter_tuning.py: Performs hyperparameter tuning for depth, samples per node, etc.
step5_final_evaluation.py: Final evaluation using tuned parameters.
__init__.py: Marks this folder as a Python module.

results/
Generated outputs from experiments.
stopping_criteria_analysis.png: Visualization of tree performance across stopping rules.
hyperparameter_tuning_results.pkl: Pickle file storing results of tuning runs.

📄 Root Files
main.py: Optional entry-point script to run a specific tree training pipeline.
requirements.txt: Lists all required Python packages.
.vscode/settings.json: Editor-specific settings for Visual Studio Code.
.DS_Store: Mac-specific system file (safe to ignore).
inspect_npy.py: Debugging utility to view contents of .npy files.
view_pickle.py: Debugging utility to inspect contents of .pkl files.

Note
All experiments are fully reproducible using the provided scripts and preprocessed data.
No information leakage from test/validation data during preprocessing or training.
Tree predictors use single-feature binary tests (thresholding for numeric and set-membership for categorical).

