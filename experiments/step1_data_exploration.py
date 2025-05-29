import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.join("data", "secondary_data.csv")

def load_data(path):
    """Load the dataset"""
    df = pd.read_csv(path, sep=';')
    return df

def preprocess_data(df):
    """
    Preprocess the mushroom dataset, no test set
    """
    #Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    #Handle missing or empty values
    df.replace('', 'missing', inplace=True)
    df.fillna('missing', inplace=True)
    
    #Split target and features
    y = df['class']
    X = df.drop(columns=['class'])
    
    return X, y

def encode_features(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Encode features using only training set information to avoid data leakage.
    """
    #Identify column types from training set only
    numerical_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    
    encoders = {}
    
    #Encode categorical columns using training set only
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        encoders[col] = le
        
        #Apply same encoding to validation and test sets
        #Handle unseen categories by assigning them to a default value
        def safe_transform(values, encoder):
            result = np.zeros(len(values), dtype=int)
            for i, val in enumerate(values.astype(str)):
                if val in encoder.classes_:
                    result[i] = encoder.transform([val])[0]
                else:
                    #Assign unseen categories to the most frequent class
                    result[i] = 0  #or could use mode from training set
            return result
        
        X_val[col] = safe_transform(X_val[col], le)
        X_test[col] = safe_transform(X_test[col], le)
    
    #Encode target using training set
    y_encoder = LabelEncoder()
    y_train_encoded = y_encoder.fit_transform(y_train)
    y_val_encoded = y_encoder.transform(y_val)
    y_test_encoded = y_encoder.transform(y_test)
    
    return (X_train, X_val, X_test, 
            y_train_encoded, y_val_encoded, y_test_encoded, 
            encoders, y_encoder)

def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print("Dataset loaded with shape:", df.shape)
    
    print("Preprocessing dataset...")
    X, y = preprocess_data(df)
    print("Preprocessing complete. Feature matrix shape:", X.shape)
    
    #split1: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    #split2: training (64%) and validation (16%) from remaining 80%
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples")
    
    # Encode features properly
    X_train, X_val, X_test, y_train, y_val, y_test, encoders, y_encoder = encode_features(
        X_train.copy(), X_val.copy(), X_test.copy(), y_train, y_val, y_test
    )
    
    print("Class label mapping:", dict(zip(y_encoder.classes_, range(len(y_encoder.classes_)))))
    
    #Save processed datasets
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train.npy", X_train.to_numpy())
    np.save("data/processed/X_val.npy", X_val.to_numpy())
    np.save("data/processed/X_test.npy", X_test.to_numpy())
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/y_test.npy", y_test)
    
    #Save encoders for later use
    import pickle
    with open("data/processed/encoders.pkl", "wb") as f:
        pickle.dump((encoders, y_encoder), f)

if __name__ == "__main__":
    main()