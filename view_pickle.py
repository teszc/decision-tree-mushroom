import pickle

with open("results/hyperparameter_tuning_results.pkl", "rb") as f:
    data = pickle.load(f)

print(data)
