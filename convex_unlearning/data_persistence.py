"""
Data persistence module for saving and loading datasets.
"""
import pickle
import os
import numpy as np


def save_data(data, filepath):
    """
    Save data dictionary to pickle file.
    
    Parameters:
    -----------
    data : dict
        Data dictionary to save
    filepath : str
        Path to save the pickle file
    """
    # Create directory if needed (only if path includes directories)
    dir_path = os.path.dirname(filepath)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Data saved to: {filepath}")


def load_data(filepath):
    """
    Load data dictionary from pickle file.
    
    Parameters:
    -----------
    filepath : str
        Path to the pickle file
        
    Returns:
    --------
    data : dict
        Loaded data dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data loaded from: {filepath}")
    return data


def check_data_exists(filepath):
    """
    Check if data file exists.
    
    Parameters:
    -----------
    filepath : str
        Path to check
        
    Returns:
    --------
    bool
        True if file exists, False otherwise
    """
    return os.path.exists(filepath)


def print_data_summary(data):
    """
    Print summary of loaded data.
    
    Parameters:
    -----------
    data : dict
        Data dictionary
    """
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    # Training data
    if 'X_train' in data:
        print(f"Training set: {data['X_train'].shape}")
    if 'X_retain' in data:
        print(f"  Retain: {data['X_retain'].shape}")
    if 'X_forget' in data:
        print(f"  Forget: {data['X_forget'].shape}")
    
    # Validation and test
    if 'X_val' in data:
        print(f"Validation set: {data['X_val'].shape}")
    if 'X_test' in data:
        print(f"Test set: {data['X_test'].shape}")
    
    # Parameters
    if 'theta_star' in data:
        print(f"\nTrue parameter θ*: {data['theta_star'].shape}")
        print(f"  ||θ*|| = {np.linalg.norm(data['theta_star']):.4f}")
    
    if 'theta_f1' in data:
        print(f"Forget parameter θ^{{f1}}: {data['theta_f1'].shape}")
        print(f"  ||θ^{{f1}}|| = {np.linalg.norm(data['theta_f1']):.4f}")
    
    if 'theta_f2' in data:
        print(f"Forget parameter θ^{{f2}}: {data['theta_f2'].shape}")
        print(f"  ||θ^{{f2}}|| = {np.linalg.norm(data['theta_f2']):.4f}")
    
    # Forget split info
    if 'forget_split' in data:
        n_f1, n_f2 = data['forget_split']
        print(f"\nForget set split: {n_f1} (first half) + {n_f2} (second half)")
    
    print("="*70)


if __name__ == "__main__":
    # Test save/load
    test_data = {
        'X_train': np.random.randn(100, 10),
        'y_train': np.random.randint(0, 2, 100),
        'theta_star': np.random.randn(10)
    }
    
    filepath = '/tmp/test_data.pkl'
    save_data(test_data, filepath)
    loaded_data = load_data(filepath)
    print_data_summary(loaded_data)
    
    # Verify
    assert np.allclose(test_data['X_train'], loaded_data['X_train'])
    print("\n✓ Save/load test passed!")
