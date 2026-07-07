"""Save/load data dict as pickle."""
import os
import pickle


def save_data(data, filepath):
    dir_path = os.path.dirname(filepath)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data saved to: {filepath}")


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from: {filepath}")
    return data
