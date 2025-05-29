import numpy as np
import os

data_dir = '/Users/tesizace/Desktop/DSE/2nd trimester 1st year/Machine Learning/decision-tree-mushroom/data/processed'


files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy", "X_val.npy", "y_val.npy"]

for file in files:
    filepath = os.path.join(data_dir, file)
    data = np.load(filepath)
    
    print(f"--- {file} ---")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    if len(data.shape) > 1:
        preview = data[:7]
    else:
        preview = data[:10]
    print("Preview:")
    print(preview)
    print()
