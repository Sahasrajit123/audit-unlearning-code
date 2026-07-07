import torch
import math
import os
import sys
import torchvision
from torch.utils.data import random_split
from torchvision import datasets, transforms
from model import MLP, AllCNN, ResNet18


def load_dataset(dataset):

    if not os.path.exists('./data'):
        os.mkdir('./data')
        
    if dataset == 'mnist':
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'cifar10':
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'svhn':
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        raise ValueError('Undefined Dataset.')

    return train_set, test_set


def load_data(dataset, batch_size, seed=42):

    train_set, test_set = load_dataset(dataset)

    torch.manual_seed(seed)
    
    val_size = int(len(train_set) * 0.2)
    train_set, val_set = random_split(train_set, [len(train_set) - val_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader


def load_unlearn_data(dataset, num_unlearn, seed=0):
    train_set, _ = load_dataset(dataset)

    torch.manual_seed(seed)
    res_set, unl_set = random_split(train_set, [len(train_set) - num_unlearn, num_unlearn])

    # unlearnloader = torch.utils.data.DataLoader(unl_set, batch_size=len(unl_set), shuffle=False, num_workers=2)
    # residualloader = torch.utils.data.DataLoader(res_set, batch_size=len(res_set), shuffle=False, num_workers=2)

    return unl_set, res_set


def load_train_data(dataset, batch_size, seed=42):
    torch.manual_seed(seed)
    
    val_size = int(len(dataset) * 0.2)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, valloader


def load_subsampled_data_from_folders(dataset_name, data_dir='./data', subsample_prob=0.5, seed=None):
    """
    Load data from organized folders and subsample forget set.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mnist')
        data_dir: Root directory containing train/val/test/retain/forget folders
        subsample_prob: Probability of keeping each sample from forget set (default: 0.5)
        seed: Random seed for subsampling (if None, uses random subsampling)
    
    Returns:
        train_data: Subsampled forget set (tilde_forget) as TensorDataset
        retain_data: Full retain set as TensorDataset (unchanged)
        val_data: Validation set as TensorDataset
        test_data: Test set as TensorDataset
        forget_data: Full forget set as TensorDataset (original, before subsampling)
    """
    import os
    from torch.utils.data import TensorDataset
    import numpy as np
    
    # Load forget set
    forget_path = os.path.join(data_dir, 'forget', f'{dataset_name}_forget.pt')
    forget_dict = torch.load(forget_path)
    forget_full_data = forget_dict['data']
    forget_full_labels = forget_dict['labels']
    
    # Subsample forget set with probability subsample_prob
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    num_forget = len(forget_full_data)
    # Create mask: keep with probability subsample_prob
    mask = torch.rand(num_forget) < subsample_prob
    
    tilde_forget_data = forget_full_data[mask]
    tilde_forget_labels = forget_full_labels[mask]
    
    print(f"Subsampled forget set: {len(tilde_forget_data)}/{num_forget} samples (p={subsample_prob})")
    
    # Load retain set (unchanged)
    retain_path = os.path.join(data_dir, 'retain', f'{dataset_name}_retain.pt')
    retain_dict = torch.load(retain_path)
    retain_data_tensor = retain_dict['data']
    retain_labels_tensor = retain_dict['labels']
    
    # Load validation set
    val_path = os.path.join(data_dir, 'val', f'{dataset_name}_val.pt')
    val_dict = torch.load(val_path)
    val_data_tensor = val_dict['data']
    val_labels_tensor = val_dict['labels']
    
    # Load test set
    test_path = os.path.join(data_dir, 'test', f'{dataset_name}_test.pt')
    test_dict = torch.load(test_path)
    test_data_tensor = test_dict['data']
    test_labels_tensor = test_dict['labels']
    
    # Create TensorDatasets
    train_data = TensorDataset(tilde_forget_data, tilde_forget_labels)
    retain_data = TensorDataset(retain_data_tensor, retain_labels_tensor)
    val_data = TensorDataset(val_data_tensor, val_labels_tensor)
    test_data = TensorDataset(test_data_tensor, test_labels_tensor)
    forget_data = TensorDataset(forget_full_data, forget_full_labels)
    
    print(f"Loaded datasets:")
    print(f"  Train (tilde_forget): {len(train_data)} samples")
    print(f"  Retain: {len(retain_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Forget (full): {len(forget_data)} samples")
    
    return train_data, retain_data, val_data, test_data, forget_data


def params_to_vec(parameters, grad=False):
    vec = []
    for param in parameters:
        if grad:
            vec.append(param.grad.view(1, -1))
        else:
            vec.append(param.data.view(1, -1))
    return torch.cat(vec, dim=1).squeeze()


def vec_to_params(vec, parameters):
    param = []
    for p in parameters:
        size = p.view(1, -1).size(1)
        param.append(vec[:size].view(p.size()))
        vec = vec[size:]
    return param


def batch_grads_to_vec(parameters):
    vec = []
    for param in parameters:
        # vec.append(param.view(1, -1))
        vec.append(param.reshape(1, -1))
    return torch.cat(vec, dim=1).squeeze()


def batch_vec_to_grads(vec, parameters):
    grads = []
    for param in parameters:
        size = param.view(1, -1).size(1)
        grads.append(vec[:size].view(param.size()))
        vec = vec[size:]
    return grads


class TensorDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for tensors."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_data_from_folders(dataset_name, forget_prob=0.5, seed=42, data_dir='./data', independent_sampling=False):
    """
    Load data from organized folders and sub-sample from forget set.
    
    This function implements the new unlearning style where:
    - Retain set stays the same (no randomness)
    - Forget set is sub-sampled with probability forget_prob to get ~forget
    - Training set = retain + ~forget (combined)
    - Unlearning will be performed on the ~forget samples
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mnist')
        forget_prob: Probability of keeping each sample from forget set (default: 0.5)
        seed: Random seed for reproducibility
        data_dir: Root directory containing data folders
        independent_sampling: If True, sample each point independently with probability forget_prob.
                             If False (default), sample forget_prob * total_points uniformly at random.
    
    Returns:
        train_set: retain + ~forget (combined training set)
        retain_set: Full retain set (unchanged)
        forget_set: Sub-sampled forget set (~forget) - samples to unlearn
        val_set: Validation set
        test_set: Test set
        forget_indices: Indices of selected forget samples (numpy array)
    """
    import numpy as np
    from torch.utils.data import ConcatDataset
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load retain set (stays constant)
    retain_path = os.path.join(data_dir, 'retain', f'{dataset_name}_retain.pt')
    retain_data = torch.load(retain_path)
    retain_set = TensorDataset(retain_data['data'], retain_data['labels'])
    
    # Load full forget set
    forget_path = os.path.join(data_dir, 'forget', f'{dataset_name}_forget.pt')
    forget_data = torch.load(forget_path)
    full_forget_size = len(forget_data['data'])
    
    # Sub-sample forget set
    if independent_sampling:
        # Sample each point independently with probability forget_prob
        keep_mask = np.random.rand(full_forget_size) < forget_prob
        keep_indices = np.where(keep_mask)[0]
    else:
        # Sample forget_prob * total_points uniformly at random (without replacement)
        num_samples = int(forget_prob * full_forget_size)
        keep_indices = np.random.choice(full_forget_size, size=num_samples, replace=False)
        keep_indices = np.sort(keep_indices)  # Sort for consistency
    
    # Create ~forget set (subsampled forget)
    subsampled_forget_data = forget_data['data'][keep_indices]
    subsampled_forget_labels = forget_data['labels'][keep_indices]
    forget_set = TensorDataset(subsampled_forget_data, subsampled_forget_labels)
    
    # Create combined training set: retain + ~forget
    train_set = ConcatDataset([retain_set, forget_set])
    
    # Load validation set
    val_path = os.path.join(data_dir, 'val', f'{dataset_name}_val.pt')
    val_data = torch.load(val_path)
    val_set = TensorDataset(val_data['data'], val_data['labels'])
    
    # Load test set
    test_path = os.path.join(data_dir, 'test', f'{dataset_name}_test.pt')
    test_data = torch.load(test_path)
    test_set = TensorDataset(test_data['data'], test_data['labels'])
    
    print(f"Loaded data from {data_dir}:")
    print(f"  Retain set: {len(retain_set)} samples (constant)")
    print(f"  Full forget set: {full_forget_size} samples")
    print(f"  ~forget (subsampled): {len(forget_set)} samples ({len(forget_set)/full_forget_size*100:.1f}% of full forget)")
    print(f"  Selected forget indices: {keep_indices.tolist()[:10]}{'...' if len(keep_indices) > 10 else ''}")
    print(f"  Total selected: {len(keep_indices)} indices")
    print(f"  Train set (retain + ~forget): {len(train_set)} samples")
    print(f"  Validation set: {len(val_set)} samples")
    print(f"  Test set: {len(test_set)} samples")
    
    return train_set, retain_set, forget_set, val_set, test_set, keep_indices


def load_data_from_folders_universal(dataset_name, forget_prob=0.5, seed=42, data_dir='./data', independent_sampling=False, batch_level_sampling=False):
    """
    Universal data loader that supports both .pt files and .pkl batch files.
    Automatically detects the format and loads accordingly.
    
    This function maintains backward compatibility with .pt files while adding
    support for batch pickle files (like cifar_1 format).
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mnist', 'cifar10')
        forget_prob: Probability of keeping each sample/batch from forget set (default: 0.5)
        seed: Random seed for reproducibility
        data_dir: Root directory containing data folders
        independent_sampling: If True, sample each point independently with probability forget_prob.
                             If False (default), sample forget_prob * total_points uniformly at random.
        batch_level_sampling: If True, sample at batch level (keep entire batches). Only applies to .pkl format.
                             If False (default), sample at sample level (individual samples).
                             When True, returns batch numbers (1-based) instead of sample indices.
    
    Returns:
        train_set: retain + ~forget (combined training set)
        retain_set: Full retain set (unchanged)
        forget_set: Sub-sampled forget set (~forget) - samples to unlearn
        val_set: Validation set
        test_set: Test set
        forget_indices: Indices of selected forget samples/batches (numpy array)
                       - Sample indices (0-based) if batch_level_sampling=False
                       - Batch numbers (1-based) if batch_level_sampling=True
    """
    import numpy as np
    import pickle
    import glob
    from torch.utils.data import ConcatDataset
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Check data format by looking for .pt or .pkl files
    retain_dir = os.path.join(data_dir, 'retain')
    pt_file = os.path.join(retain_dir, f'{dataset_name}_retain.pt')
    pkl_files = glob.glob(os.path.join(retain_dir, '*.pkl'))
    
    use_pkl_format = len(pkl_files) > 0 and not os.path.exists(pt_file)
    
    if use_pkl_format:
        print(f"Detected .pkl batch format in {data_dir}")
        return _load_from_pkl_batches(dataset_name, forget_prob, seed, data_dir, independent_sampling, batch_level_sampling)
    else:
        print(f"Detected .pt format in {data_dir}")
        # batch_level_sampling only applies to .pkl format
        return load_data_from_folders(dataset_name, forget_prob, seed, data_dir, independent_sampling)


def _load_from_pkl_batches(dataset_name, forget_prob, seed, data_dir, independent_sampling=False, batch_level_sampling=False):
    """
    Load data from pickle batch files (e.g., cifar_1 format).
    
    Each folder contains multiple batch_XXXXX.pkl files.
    
    Args:
        dataset_name: Name of the dataset
        forget_prob: Probability of keeping each sample/batch from forget set
        seed: Random seed for reproducibility
        data_dir: Root directory containing data folders
        independent_sampling: If True, sample each point independently with probability forget_prob.
                             If False (default), sample forget_prob * total_points uniformly at random.
        batch_level_sampling: If True, sample at batch level (keep entire batches). 
                             If False (default), sample at sample level (individual samples).
                             When True, returns batch numbers (1-based) instead of sample indices.
    """
    import numpy as np
    import pickle
    import glob
    import re
    from torch.utils.data import ConcatDataset
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    def load_pkl_batches(folder_path, selected_batch_files=None):
        """Load pickle batch files from a folder.
        
        Args:
            folder_path: Path to folder containing batch files
            selected_batch_files: If provided, only load these specific batch files.
                                If None, load all batch files.
        """
        if selected_batch_files is None:
            pkl_files = sorted(glob.glob(os.path.join(folder_path, '*.pkl')))
        else:
            pkl_files = sorted(selected_batch_files)
        
        if len(pkl_files) == 0:
            raise ValueError(f"No .pkl files found in {folder_path}")
        
        all_data = []
        all_labels = []
        
        print(f"  Loading {len(pkl_files)} batch files from {os.path.basename(folder_path)}...")
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                batch = pickle.load(f)
                
                # Handle different pickle formats
                if isinstance(batch, dict):
                    # Format: {'data': array, 'labels': array}
                    data = batch.get('data', batch.get('images', None))
                    labels = batch.get('labels', batch.get('targets', None))
                elif isinstance(batch, (tuple, list)):
                    # Format: (data, labels) or (data, labels, ...) - take first two elements
                    if len(batch) < 2:
                        raise ValueError(f"Pickle batch must contain at least 2 elements (data, labels), got {len(batch)} in {pkl_file}")
                    data, labels = batch[0], batch[1]
                else:
                    raise ValueError(f"Unknown pickle format in {pkl_file}")
                
                # Convert to torch tensors if needed
                if not isinstance(data, torch.Tensor):
                    data = torch.from_numpy(data) if hasattr(data, 'shape') else torch.tensor(data)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.from_numpy(labels) if hasattr(labels, 'shape') else torch.tensor(labels)
                
                all_data.append(data)
                all_labels.append(labels)
        
        # Concatenate all batches
        full_data = torch.cat(all_data, dim=0) if len(all_data) > 1 else all_data[0]
        full_labels = torch.cat(all_labels, dim=0) if len(all_labels) > 1 else all_labels[0]
        
        # Ensure correct dtype
        full_data = full_data.float()
        if full_labels.dtype == torch.int32 or full_labels.dtype == torch.int64:
            full_labels = full_labels.long()
        
        # Fix shape: PyTorch CNNs need [N, C, H, W] but data might be [N, H, W, C]
        if len(full_data.shape) == 4 and full_data.shape[-1] in [1, 3]:
            # Last dimension is channels (likely from numpy), transpose to [N, C, H, W]
            full_data = full_data.permute(0, 3, 1, 2)
            print(f"    Transposed data from [N, H, W, C] to [N, C, H, W]: {full_data.shape}")
        elif len(full_data.shape) == 3:
            # Grayscale without channel dimension, add it: [N, H, W] -> [N, 1, H, W]
            full_data = full_data.unsqueeze(1)
            print(f"    Added channel dimension: {full_data.shape}")
        
        return full_data, full_labels
    
    def extract_batch_number(filename):
        """Extract batch number from filename like batch_00000.pkl -> 0, batch_00001.pkl -> 1"""
        # Extract number from filename (preserves 0-based or 1-based from filename)
        match = re.search(r'batch[_\s]*(\d+)', os.path.basename(filename), re.IGNORECASE)
        if match:
            return int(match.group(1))  # Returns the number as-is from filename (0-based for batch_00000.pkl)
        # If extraction fails, raise an error (don't use array index as fallback)
        raise ValueError(f"Could not extract batch number from filename: {filename}. Expected format like 'batch_00000.pkl' or 'batch_00001.pkl'")
    
    print(f"\nLoading data from {data_dir} (pickle batch format):")
    
    # Load retain set (stays constant)
    retain_dir = os.path.join(data_dir, 'retain')
    retain_data, retain_labels = load_pkl_batches(retain_dir)
    retain_set = TensorDataset(retain_data, retain_labels)
    
    # Handle forget set with batch-level or sample-level sampling
    forget_dir = os.path.join(data_dir, 'forget')
    all_forget_batch_files = sorted(glob.glob(os.path.join(forget_dir, '*.pkl')))
    
    if batch_level_sampling:
        # Batch-level sampling: select which batches to keep
        num_batches = len(all_forget_batch_files)
        
        if independent_sampling:
            # Sample each batch independently with probability forget_prob
            print(f"\n  Batch-level sampling: selecting batches independently with probability {forget_prob}...")
            keep_mask = np.random.rand(num_batches) < forget_prob
            selected_batch_indices = np.where(keep_mask)[0]
        else:
            # Sample forget_prob * num_batches batches uniformly at random
            num_selected_batches = int(forget_prob * num_batches)
            print(f"\n  Batch-level sampling: selecting {num_selected_batches} batches uniformly at random (forget_prob={forget_prob})...")
            selected_batch_indices = np.random.choice(num_batches, size=num_selected_batches, replace=False)
            selected_batch_indices = np.sort(selected_batch_indices)  # Sort for consistency
        
        # Get the selected batch files
        selected_batch_files = [all_forget_batch_files[i] for i in selected_batch_indices]
        
        # Extract batch numbers from filenames (preserves 0-based or 1-based from filename)
        batch_numbers = []
        for idx in selected_batch_indices:
            batch_file = all_forget_batch_files[idx]
            batch_num = extract_batch_number(batch_file)  # Will raise error if extraction fails
            batch_numbers.append(batch_num)
        batch_numbers = np.array(sorted(batch_numbers))  # Sort batch numbers
        
        # Load only selected batches
        forget_data, forget_labels = load_pkl_batches(forget_dir, selected_batch_files)
        full_forget_size = len(forget_data)  # Total samples in selected batches
        
        # Create forget set from selected batches
        forget_set = TensorDataset(forget_data, forget_labels)
        
        # Return batch numbers instead of sample indices
        keep_indices = batch_numbers
        
        print(f"  Selected {len(selected_batch_indices)} batches out of {num_batches} total batches")
        print(f"  Batch numbers: {batch_numbers[:10].tolist()}{'...' if len(batch_numbers) > 10 else ''}")
        
    else:
        # Sample-level sampling (original behavior): load all batches, then sample samples
        forget_data, forget_labels = load_pkl_batches(forget_dir)
        full_forget_size = len(forget_data)
        
        # Sub-sample forget set at sample level
        if independent_sampling:
            # Sample each point independently with probability forget_prob
            print(f"\n  Sample-level sampling: selecting samples independently with probability {forget_prob}...")
            keep_mask = np.random.rand(full_forget_size) < forget_prob
            keep_indices = np.where(keep_mask)[0]
        else:
            # Sample forget_prob * total_points uniformly at random (without replacement)
            num_samples = int(forget_prob * full_forget_size)
            print(f"\n  Sample-level sampling: selecting {num_samples} samples uniformly at random (forget_prob={forget_prob})...")
            keep_indices = np.random.choice(full_forget_size, size=num_samples, replace=False)
            keep_indices = np.sort(keep_indices)  # Sort for consistency
        
        # Create ~forget set (subsampled forget)
        subsampled_forget_data = forget_data[keep_indices]
        subsampled_forget_labels = forget_labels[keep_indices]
        forget_set = TensorDataset(subsampled_forget_data, subsampled_forget_labels)
    
    # Create combined training set: retain + ~forget
    train_set = ConcatDataset([retain_set, forget_set])
    
    # Load validation set
    val_dir = os.path.join(data_dir, 'val')
    val_data, val_labels = load_pkl_batches(val_dir)
    val_set = TensorDataset(val_data, val_labels)
    
    # Load test set
    test_dir = os.path.join(data_dir, 'test')
    test_data, test_labels = load_pkl_batches(test_dir)
    test_set = TensorDataset(test_data, test_labels)
    
    # Create combined training set: retain + ~forget
    train_set = ConcatDataset([retain_set, forget_set])
    
    # Load validation set
    val_dir = os.path.join(data_dir, 'val')
    val_data, val_labels = load_pkl_batches(val_dir)
    val_set = TensorDataset(val_data, val_labels)
    
    # Load test set
    test_dir = os.path.join(data_dir, 'test')
    test_data, test_labels = load_pkl_batches(test_dir)
    test_set = TensorDataset(test_data, test_labels)
    
    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"{'='*60}")
    print(f"  Retain set: {len(retain_set):,} samples (constant)")
    if batch_level_sampling:
        print(f"  Full forget set: {len(all_forget_batch_files)} batches")
        print(f"  ~forget (selected batches): {len(forget_set):,} samples from {len(keep_indices)} batches")
        print(f"  Selected {len(keep_indices)} batch numbers")
        print(f"  Batch numbers: {keep_indices[:10].tolist()}{'...' if len(keep_indices) > 10 else ''}")
    else:
        print(f"  Full forget set: {full_forget_size:,} samples")
        print(f"  ~forget (subsampled): {len(forget_set):,} samples ({len(forget_set)/full_forget_size*100:.1f}% of full forget)")
        print(f"  Selected {len(keep_indices):,} sample indices")
        print(f"  First 10 indices: {keep_indices[:10].tolist()}{'...' if len(keep_indices) > 10 else ''}")
    print(f"  Train set (retain + ~forget): {len(train_set):,} samples")
    print(f"  Validation set: {len(val_set):,} samples")
    print(f"  Test set: {len(test_set):,} samples")
    print(f"  Data shape: {retain_data[0].shape}")
    print(f"  Data type: {retain_data.dtype}")
    print(f"{'='*60}\n")
    
    return train_set, retain_set, forget_set, val_set, test_set, keep_indices
