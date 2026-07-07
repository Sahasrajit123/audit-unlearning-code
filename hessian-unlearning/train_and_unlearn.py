"""
New style of unlearning: Train + Unlearn in a single run

This script:
1. Sub-samples from forget set with probability 0.5 to get ~forget
2. Trains a model on retain + ~forget (combined)
3. Unlearns the ~forget samples (using retain as residual)
4. Saves trained model, unlearned model, and forget indices
5. Retain set stays constant (no randomness)
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader, Subset
from utils import load_data_from_folders, load_data_from_folders_universal
from model import MLP, AllCNN, ResNet18, TinyNet, TinyNetCIFAR100
from train import train
from unlearn import unlearn
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
import sys

class UnbufferedFile:
    """File wrapper that flushes on every write for real-time log updates."""
    def __init__(self, file):
        self.file = file
    
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        # Force OS-level flush for real-time updates
        try:
            os.fsync(self.file.fileno())
        except (OSError, AttributeError):
            # If fsync is not available or fails, just continue
            pass
    
    def flush(self):
        self.file.flush()
        # Force OS-level flush for real-time updates
        try:
            os.fsync(self.file.fileno())
        except (OSError, AttributeError):
            # If fsync is not available or fails, just continue
            pass
    
    def __getattr__(self, name):
        return getattr(self.file, name)

# Set CuBLAS workspace configuration for deterministic behavior (required for CUDA >= 10.2)
# This must be set before any CUDA operations
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def evaluate_accuracy(model, dataset, batch_size, device, dataset_name="Dataset"):
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: PyTorch model
        dataset: PyTorch dataset
        batch_size: Batch size for evaluation
        device: Device to evaluate on
        dataset_name: Name of dataset for printing
    
    Returns:
        Accuracy as percentage
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def run_single_experiment(run_idx, args_dict, model_dir, data_dir, assigned_gpu_id=None):
    """
    Execute a single train+unlearn run.
    This function is designed to be called in parallel.
    
    Args:
        run_idx: Index of the run
        args_dict: Dictionary of arguments
        model_dir: Directory for saving models
        data_dir: Directory containing data
        assigned_gpu_id: GPU ID assigned to this run (None for CPU or auto-assignment)
    """
    # Reconstruct args object from dict (needed for multiprocessing)
    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    args = Args(args_dict)
    
    # Set device (for CUDA, use assigned GPU or auto-assign)
    # Note: GPU assignment message will be printed to terminal by caller
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        if assigned_gpu_id is not None:
            # Use pre-assigned GPU
            device = torch.device(f"cuda:{assigned_gpu_id}")
        else:
            # Fallback: auto-assign based on run_idx (for backward compatibility)
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                gpu_id = run_idx % num_gpus
                device = torch.device(f"cuda:{gpu_id}")
            else:
                device = torch.device("cpu")
                use_cuda = False
    else:
        device = torch.device("cpu")
    
    # Use same seed for everything (training, model init, etc.) to ensure identical models
    # BUT use different seed for forget set sampling so each run gets different random subset
    seed = args.seed
    forget_seed = args.seed + run_idx  # Different seed for forget set sampling per run
    
    # Set seeds and deterministic flags FIRST (before any operations)
    # This ensures identical model initialization, batch shuffling, training, and unlearn operations
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic CUDA operations for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Try to enable deterministic algorithms (if available in PyTorch version)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # Older PyTorch versions don't have this
    
    # Create subfolder for this run
    run_subdir = os.path.join(model_dir, f'run_{run_idx}')
    if not os.path.exists(run_subdir):
        os.makedirs(run_subdir)
    
    # Define log file path
    log_file_path = os.path.join(run_subdir, f'run_{run_idx}.log')
    
    # Define paths for this run
    scheduler_str = f"_sched_{args.scheduler_type}" if args.scheduler_type else ""
    base_name = f"clip_{str(args.C).replace('.','_')}_{args.model}_{args.dataset}_opt_{args.optimizer}{scheduler_str}_lr_{str(args.lr).replace('.','_')}_bs_{args.batch_size}_wd_{str(args.weight_decay).replace('.','_')}_epoch_{args.epochs}_seed_{args.seed}_forgetprob_{str(args.forget_prob).replace('.','_')}_run_{run_idx}"
    PATH_trained = os.path.join(run_subdir, f'trained_{base_name}.pth')
    PATH_unlearned = os.path.join(run_subdir, f'unlearned_{base_name}_unlearnbs_{args.unlearn_bs}_s1_{args.s1}_s2_{args.s2}_std_{str(args.std).replace(".","_")}_gamma_{str(args.gamma).replace(".","_")}_scale_{str(args.scale).replace(".","_")}.pth')
    PATH_indices = os.path.join(run_subdir, f'forget_indices_{base_name}.npy')
    
    # Redirect all output to log file
    exit_code = 0
    try:
        # Open log file with line buffering (buffering=1) for real-time updates
        with open(log_file_path, 'w', buffering=1) as log_file:
            # Wrap file with unbuffered wrapper for real-time updates
            unbuffered_log = UnbufferedFile(log_file)
            # Redirect both stdout and stderr to log file
            with redirect_stdout(unbuffered_log), redirect_stderr(unbuffered_log):
                # Load data with different forget set sampling per run
                # Note: This will temporarily set the seed in the data loading function
                independent_sampling = args.independent_sampling if hasattr(args, 'independent_sampling') else False
                batch_level_sampling = args.batch_level_sampling if hasattr(args, 'batch_level_sampling') else False
                train_set, retain_set, forget_set, val_set, test_set, forget_indices = load_data_from_folders_universal(
                    args.dataset, 
                    forget_prob=args.forget_prob,  # Use the specified forget probability
                    seed=forget_seed,  # Different seed per run for forget set sampling
                    data_dir=data_dir,
                    independent_sampling=independent_sampling,
                    batch_level_sampling=batch_level_sampling
                )
                
                # RESET seeds again after data loading to ensure consistent state for model initialization
                # (Data loading function may have consumed some random state)
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                
                # Create generator for DataLoader to ensure reproducible shuffling (same across all runs)
                generator = torch.Generator()
                generator.manual_seed(seed)
                
                print(f"\n{'='*70}")
                print(f"RUN {run_idx} (Process {os.getpid()})")
                print(f"  Main seed: {seed} (same across all runs - for training, model init)")
                print(f"  Forget set seed: {forget_seed} (different per run - for batch selection and unlearning)")
                print(f"{'='*70}\n")
                
                # Determine input size for MLP based on dataset
                # Map dataset names to their input sizes (C * H * W)
                dataset_input_sizes = {
                    'mnist': 1 * 28 * 28,      # 784
                    'cifar10': 3 * 32 * 32,    # 3072
                    'cifar': 3 * 32 * 32,      # 3072 (alias)
                    'svhn': 3 * 32 * 32,       # 3072
                }
                dataset_lower = args.dataset.lower()
                input_size = dataset_input_sizes.get(dataset_lower, None)
                if input_size is None:
                    # Fallback: get from dataset sample if dataset not in mapping
                    sample_data, _ = train_set[0]
                    if len(sample_data.shape) == 3:  # [C, H, W]
                        input_size = sample_data.shape[0] * sample_data.shape[1] * sample_data.shape[2]
                    elif len(sample_data.shape) == 1:  # Already flattened
                        input_size = sample_data.shape[0]
                    else:
                        raise ValueError(f"Unexpected data shape: {sample_data.shape} for dataset {args.dataset}")
                    print(f"  Calculated input_size from dataset sample: {input_size} (dataset: {args.dataset})")
                else:
                    print(f"  Using input_size from mapping: {input_size} (dataset: {args.dataset})")
                
                # Optional one-time sample-level shuffle: permute dataset once before batching.
                shuffle_train_samples = args.shuffle_train_samples if hasattr(args, 'shuffle_train_samples') else False
                if shuffle_train_samples:
                    sample_perm_generator = torch.Generator()
                    sample_perm_generator.manual_seed(seed)
                    sample_permutation = torch.randperm(len(train_set), generator=sample_perm_generator).tolist()
                    train_set = Subset(train_set, sample_permutation)

                # Per-epoch shuffle controls whether DataLoader reshuffles sample indices every epoch.
                shuffle_train_batches_each_epoch = (
                    args.shuffle_train_batches_each_epoch
                    if hasattr(args, 'shuffle_train_batches_each_epoch')
                    else True
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    shuffle=shuffle_train_batches_each_epoch,
                    num_workers=0,
                    generator=generator if shuffle_train_batches_each_epoch else None,
                )  # num_workers=0 for multiprocessing
                val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
                
                # Create model (seed is already set, so initialization will be identical)
                if args.model == 'mlp':
                    print(f"  Creating MLP with input_size={input_size}, num_classes={args.num_classes}")
                    model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters, input_size=input_size).to(device)
                elif args.model in ['cnn', 'allcnn']:
                    model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(device)
                elif args.model == 'resnet18':
                    model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(device)
                elif args.model == 'tinynet':
                    model = TinyNet(num_classes=args.num_classes, filters_percentage=args.filters).to(device)
                elif args.model in ['tinynetcifar100', 'tinynet_cifar100']:
                    model = TinyNetCIFAR100(num_classes=args.num_classes, filters_percentage=args.filters).to(device)
                else:
                    raise ValueError(f"Unknown model: {args.model}. Choose from: mlp, cnn, allcnn, resnet18, tinynet, tinynetcifar100")
                
                # Step 1: Train on retain + ~forget
                print(f"\n[STEP 1] Training model on retain + ~forget set... (Run {run_idx})")
                print(
                    f"  Optimizer: {args.optimizer}, Scheduler: {args.scheduler_type or 'None'}, "
                    f"shuffle_train_samples={shuffle_train_samples}, "
                    f"shuffle_train_batches_each_epoch={shuffle_train_batches_each_epoch}"
                )
                train(train_loader, val_loader, model, args.lr, args.weight_decay, args.epochs, args.C, PATH_trained, device,
                      optimizer_type=args.optimizer, momentum=args.momentum, 
                      scheduler_type=args.scheduler_type, scheduler_params=args.scheduler_params,
                      steps_per_epoch=len(train_loader))
                
                # Reload trained model
                model.load_state_dict(torch.load(PATH_trained))
                print(f"Trained model saved to: {PATH_trained}")
                
                # Evaluate trained model
                print(f"\n[EVALUATION - AFTER TRAINING]")
                train_acc = evaluate_accuracy(model, train_set, args.batch_size, device, "Train")
                val_acc = evaluate_accuracy(model, val_set, args.batch_size, device, "Val")
                retain_acc = evaluate_accuracy(model, retain_set, args.batch_size, device, "Retain")
                test_acc = evaluate_accuracy(model, test_set, args.batch_size, device, "Test")
                print(f"  Train Accuracy: {train_acc:.2f}%")
                print(f"  Val Accuracy: {val_acc:.2f}%")
                print(f"  Retain Accuracy: {retain_acc:.2f}%")
                print(f"  Test Accuracy: {test_acc:.2f}%")
                
                # Step 2: Unlearn the ~forget samples
                # Use forget_seed for unlearning to ensure different randomness per run (same as batch selection)
                # This ensures each run has different unlearning randomness while remaining reproducible
                # Set seeds for unlearning with forget_seed (different per run)
                torch.manual_seed(forget_seed)
                np.random.seed(forget_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(forget_seed)
                    torch.cuda.manual_seed_all(forget_seed)
                
                # Create a fresh generator for unlearn with forget_seed (different for each run)
                # (Training consumed the generator state, so we need a fresh one with forget_seed)
                unlearn_generator = torch.Generator()
                unlearn_generator.manual_seed(forget_seed)
                
                print(f"\n[STEP 2] Unlearning ~forget samples... (Run {run_idx})")
                print(f"  Using forget_seed={forget_seed} for unlearning (different per run)")
                unlearn(model, forget_set, retain_set, args.batch_size, args.unlearn_bs, args.weight_decay,
                        args.gamma, args.s1, args.s2, args.scale, args.std, device, generator=unlearn_generator)
                
                # Save the unlearned model
                torch.save(model.state_dict(), PATH_unlearned)
                print(f"Unlearned model saved to: {PATH_unlearned}")
                
                # Evaluate unlearned model
                print(f"\n[EVALUATION - AFTER UNLEARNING]")
                train_acc_unl = evaluate_accuracy(model, train_set, args.batch_size, device, "Train")
                val_acc_unl = evaluate_accuracy(model, val_set, args.batch_size, device, "Val")
                retain_acc_unl = evaluate_accuracy(model, retain_set, args.batch_size, device, "Retain")
                forget_acc_unl = evaluate_accuracy(model, forget_set, args.batch_size, device, "Forget")
                test_acc_unl = evaluate_accuracy(model, test_set, args.batch_size, device, "Test")
                print(f"  Train Accuracy: {train_acc_unl:.2f}%")
                print(f"  Val Accuracy: {val_acc_unl:.2f}%")
                print(f"  Retain Accuracy: {retain_acc_unl:.2f}%")
                print(f"  Forget Accuracy: {forget_acc_unl:.2f}%")
                print(f"  Test Accuracy: {test_acc_unl:.2f}%")
                
                # Step 3: Save forget indices
                print(f"\n[STEP 3] Saving forget indices... (Run {run_idx})")
                np.save(PATH_indices, forget_indices)
                print(f"Forget indices saved to: {PATH_indices}")
                
                print(f"\n{'='*70}")
                print(f"RUN {run_idx} COMPLETE")
                print(f"{'='*70}")
                print(f"Trained model:   {PATH_trained}")
                print(f"Unlearned model: {PATH_unlearned}")
                print(f"Forget indices:  {PATH_indices}")
                print(f"Log file:       {log_file_path}")
                print(f"{'='*70}\n")
    except Exception as e:
        exit_code = 1
        # Write exception to log file (append mode in case file was created)
        try:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"\n{'='*70}\n")
                log_file.write(f"ERROR in RUN {run_idx}\n")
                log_file.write(f"{'='*70}\n")
                import traceback
                log_file.write(traceback.format_exc())
                log_file.write(f"\n{'='*70}\n")
        except:
            # If we can't write to log file, create it
            with open(log_file_path, 'w') as log_file:
                log_file.write(f"ERROR in RUN {run_idx}\n")
                import traceback
                log_file.write(traceback.format_exc())
        # Re-raise to be caught by the caller
        raise
    
    return run_idx, exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--model', default='tinynet')
    parser.add_argument('--mlp-layer', type=int, default=3, metavar='N',
                        help='number of layers of MLP (default: 3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--unlearn-bs', type=int, default=10, metavar='N',
                        help='input batch size for unlearning (default: 10)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--C', type=float, default=20,
                        help='Norm constraint of parameters')
    parser.add_argument('--s1', type=int, default=10, help='Number of samples in Hessian approximation')
    parser.add_argument('--s2', type=int, default=1000, help='The order number of Taylor expansion in Hessian approximation')
    parser.add_argument('--std', type=float, default=.001, help='The standard deviation of Gaussian noise')
    parser.add_argument('--gamma', type=float, default=.01, help='The convex approximation coefficient')
    parser.add_argument('--scale', type=float, default=1000., help='The scale of Hessian')
    parser.add_argument('--forget-prob', type=float, default=0.5, help='Probability of keeping samples from forget set (default: 0.5)')
    parser.add_argument('--independent-sampling', action='store_true', default=False, help='If set, sample each point independently with probability forget_prob. If not set (default), sample forget_prob * total_points uniformly at random.')
    parser.add_argument('--batch-level-sampling', action='store_true', default=True, help='If set, sample at batch level (keep entire batches). Only applies to .pkl batch format. Default is True (batch-level sampling). When True, forget_indices will contain batch numbers (from filenames) instead of sample indices.')
    parser.add_argument('--run-dir', type=str, required=True, help='Directory for saving models (REQUIRED)')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing data (REQUIRED)')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of train+unlearn runs to perform (default: 1)')
    parser.add_argument('--start-run-idx', type=int, default=0, help='Starting run index (default: 0). Runs will be performed from start_run_idx to start_run_idx + num_runs.')
    parser.add_argument('--overwrite-existing-runs', dest='overwrite_existing_runs', action='store_true', default=True,
                        help='Overwrite outputs in existing run_X folders (default: enabled).')
    parser.add_argument('--skip-existing-runs', dest='overwrite_existing_runs', action='store_false',
                        help='Skip runs whose run_X folders already exist.')
    parser.add_argument('--max-workers-per-gpu', type=int, default=2, help='Maximum workers per GPU when using CUDA (default: 2). Prevents GPU memory issues.')
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated list of GPU IDs to use (e.g., "0,1,2"). If not specified, uses all available GPUs.')
    parser.add_argument('--max-runs-per-gpu', type=int, default=2, help='Maximum total number of runs to assign to each GPU (default: 2).')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer type: adam or sgd (default: adam)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer (default: 0.9)')
    
    # Data loading arguments
    parser.add_argument('--shuffle-train-samples', type=str, default=None, choices=['true', 'false', 'True', 'False'],
                        help='If true, shuffle training samples once before creating batches. If omitted, defaults to false.')
    parser.add_argument('--shuffle-train-batches-each-epoch', type=str, default=None, choices=['true', 'false', 'True', 'False'],
                        help='If true, reshuffle training sample order each epoch. If omitted, defaults to true.')
    
    # Scheduler arguments
    parser.add_argument('--scheduler-type', type=str, default=None, 
                        choices=['step', 'cosine', 'exponential', 'plateau', 'onecycle', 'none', None],
                        help='Learning rate scheduler type (default: None)')
    parser.add_argument('--scheduler-step-size', type=int, default=30,
                        help='Step size for StepLR scheduler (default: 30)')
    parser.add_argument('--scheduler-gamma', type=float, default=0.1,
                        help='Gamma (decay factor) for StepLR/ExponentialLR scheduler (default: 0.1)')
    parser.add_argument('--scheduler-t-max', type=int, default=100,
                        help='T_max for CosineAnnealingLR scheduler (default: 100)')
    parser.add_argument('--scheduler-eta-min', type=float, default=0.0,
                        help='eta_min for CosineAnnealingLR scheduler (default: 0.0)')
    parser.add_argument('--scheduler-patience', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler (default: 10)')
    parser.add_argument('--scheduler-factor', type=float, default=0.1,
                        help='Factor for ReduceLROnPlateau scheduler (default: 0.1)')
    parser.add_argument('--scheduler-mode', type=str, default='min', choices=['min', 'max'],
                        help='Mode for ReduceLROnPlateau scheduler (default: min)')
    parser.add_argument('--scheduler-pct-start', type=float, default=0.3,
                        help='pct_start for OneCycleLR scheduler (default: 0.3)')
    parser.add_argument('--scheduler-anneal-strategy', type=str, default='cos', choices=['cos', 'linear'],
                        help='anneal_strategy for OneCycleLR scheduler (default: cos)')
    parser.add_argument('--model-dir-folders', type=str, default='.', help='Directory for saving models (default: models_no_shuffle)')
    
    # Config file support
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (overrides command-line arguments)')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Override args with config values
        for key, value in config.items():
            # Skip comment keys (keys starting with underscore)
            if key.startswith('_'):
                continue
            # Convert config keys to match argument names (replace - with _)
            arg_key = key.replace('-', '_')
            if hasattr(args, arg_key):
                setattr(args, arg_key, value)
            else:
                print(f"Warning: Unknown config key '{key}', ignoring...")
    
    # Normalize scheduler_type (handle 'none' string)
    if args.scheduler_type and args.scheduler_type.lower() == 'none':
        args.scheduler_type = None
    
    # Normalize optional shuffle controls.
    if isinstance(args.shuffle_train_samples, str):
        args.shuffle_train_samples = args.shuffle_train_samples.lower() == 'true'
    elif args.shuffle_train_samples is not None and not isinstance(args.shuffle_train_samples, bool):
        args.shuffle_train_samples = bool(args.shuffle_train_samples)

    if isinstance(args.shuffle_train_batches_each_epoch, str):
        args.shuffle_train_batches_each_epoch = args.shuffle_train_batches_each_epoch.lower() == 'true'
    elif args.shuffle_train_batches_each_epoch is not None and not isinstance(args.shuffle_train_batches_each_epoch, bool):
        args.shuffle_train_batches_each_epoch = bool(args.shuffle_train_batches_each_epoch)

    # Defaults when new flags are not provided.
    if args.shuffle_train_samples is None:
        args.shuffle_train_samples = False
    if args.shuffle_train_batches_each_epoch is None:
        args.shuffle_train_batches_each_epoch = False
    
    # Build scheduler_params dictionary
    scheduler_params = {}
    if args.scheduler_type:
        if args.scheduler_type.lower() == 'step':
            scheduler_params = {
                'step_size': args.scheduler_step_size,
                'gamma': args.scheduler_gamma
            }
        elif args.scheduler_type.lower() == 'cosine':
            scheduler_params = {
                'T_max': args.scheduler_t_max,
                'eta_min': args.scheduler_eta_min
            }
        elif args.scheduler_type.lower() == 'exponential':
            scheduler_params = {
                'gamma': args.scheduler_gamma
            }
        elif args.scheduler_type.lower() == 'plateau':
            scheduler_params = {
                'mode': args.scheduler_mode,
                'factor': args.scheduler_factor,
                'patience': args.scheduler_patience
            }
        elif args.scheduler_type.lower() == 'onecycle':
            scheduler_params = {
                'max_lr': args.lr,  # Will use lr as max_lr for onecycle
                'pct_start': args.scheduler_pct_start,
                'anneal_strategy': args.scheduler_anneal_strategy
            }
    args.scheduler_params = scheduler_params if scheduler_params else None
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    # Note: We don't set seeds here in the main process because each worker process
    # will set its own seed based on run_idx. This ensures each run has different randomness.
    # The main process seed setting is removed to avoid confusion.

    # Create run directory structure (now required)
    if args.run_dir is None:
        raise ValueError("--run-dir is required. Please specify the directory for saving models using --run-dir.")
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
    
    # Set data directory (now required)
    if args.data_dir is None:
        raise ValueError("--data-dir is required. Please specify the directory containing data using --data-dir.")
    data_dir = args.data_dir
    model_dir = os.path.join(args.run_dir, args.model_dir_folders)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save configuration to top-level runs folder
    config_dict = {
        'config_file': args.config if args.config else None,
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'lr': args.lr,
        'filters': args.filters,
        'model': args.model,
        'mlp_layer': args.mlp_layer,
        'no_cuda': args.no_cuda,
        'num_classes': args.num_classes,
        'seed': args.seed,
        'unlearn_bs': args.unlearn_bs,
        'weight_decay': args.weight_decay,
        'C': args.C,
        's1': args.s1,
        's2': args.s2,
        'std': args.std,
        'gamma': args.gamma,
        'scale': args.scale,
        'forget_prob': args.forget_prob,
        'independent_sampling': args.independent_sampling if hasattr(args, 'independent_sampling') else False,
        'batch_level_sampling': args.batch_level_sampling if hasattr(args, 'batch_level_sampling') else False,
        'num_runs': args.num_runs,
        'start_run_idx': args.start_run_idx,
        'optimizer': args.optimizer,
        'momentum': args.momentum,
        'scheduler_type': args.scheduler_type,
        'scheduler_params': args.scheduler_params,
        'shuffle_train_samples': args.shuffle_train_samples if hasattr(args, 'shuffle_train_samples') else False,
        'shuffle_train_batches_each_epoch': args.shuffle_train_batches_each_epoch if hasattr(args, 'shuffle_train_batches_each_epoch') else True,
        'run_dir': args.run_dir,
        'data_dir': data_dir,
        'model_dir_folders': args.model_dir_folders,
        'overwrite_existing_runs': args.overwrite_existing_runs,
        'gpus': args.gpus,
        'max_runs_per_gpu': args.max_runs_per_gpu,
        'max_workers_per_gpu': args.max_workers_per_gpu,
    }
    
    # Add scheduler-specific parameters
    if args.scheduler_type:
        if args.scheduler_type.lower() == 'step':
            config_dict['scheduler_step_size'] = args.scheduler_step_size
            config_dict['scheduler_gamma'] = args.scheduler_gamma
        elif args.scheduler_type.lower() == 'cosine':
            config_dict['scheduler_t_max'] = args.scheduler_t_max
            config_dict['scheduler_eta_min'] = args.scheduler_eta_min
        elif args.scheduler_type.lower() == 'exponential':
            config_dict['scheduler_gamma'] = args.scheduler_gamma
        elif args.scheduler_type.lower() == 'plateau':
            config_dict['scheduler_mode'] = args.scheduler_mode
            config_dict['scheduler_factor'] = args.scheduler_factor
            config_dict['scheduler_patience'] = args.scheduler_patience
        elif args.scheduler_type.lower() == 'onecycle':
            config_dict['scheduler_pct_start'] = args.scheduler_pct_start
            config_dict['scheduler_anneal_strategy'] = args.scheduler_anneal_strategy
    
    print(f"Using directories:")
    print(f"  Data directory: {data_dir}")
    print(f"  Model directory: {model_dir}")
    print(f"  Run directory: {args.run_dir}")

    dataset_num_classes = {
        'mnist': 10,
        'cifar10': 10,
        'cifar': 10,
        'svhn': 10,
        'cifar100': 100,
    }
    if args.num_classes is None:
        args.num_classes = dataset_num_classes.get(args.dataset.lower(), 10)
    else:
        args.num_classes = int(args.num_classes)
    
    # Update config_dict with num_classes (set after model_dir creation)
    config_dict['num_classes'] = args.num_classes
    
    # Save config to JSON file in model_dir (after all processing)
    config_save_path = os.path.join(model_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to: {config_save_path}")
    
    # Parse GPU selection
    available_gpus = None
    if use_cuda:
        total_gpus = torch.cuda.device_count()
        if args.gpus is not None:
            # Parse comma-separated GPU list
            try:
                gpu_list = [int(x.strip()) for x in args.gpus.split(',')]
                # Validate GPU IDs
                available_gpus = [gpu_id for gpu_id in gpu_list if 0 <= gpu_id < total_gpus]
                if len(available_gpus) == 0:
                    print(f"Warning: No valid GPUs found in --gpus '{args.gpus}'. Available GPUs: 0-{total_gpus-1}")
                    print("  Falling back to all available GPUs.")
                    available_gpus = list(range(total_gpus))
                elif len(available_gpus) < len(gpu_list):
                    invalid = [gpu_id for gpu_id in gpu_list if gpu_id not in available_gpus]
                    print(f"Warning: Invalid GPU IDs {invalid} ignored. Using GPUs: {available_gpus}")
            except ValueError:
                print(f"Warning: Invalid --gpus format '{args.gpus}'. Expected comma-separated integers (e.g., '0,1,2').")
                print("  Falling back to all available GPUs.")
                available_gpus = list(range(total_gpus))
        else:
            # Use all available GPUs
            available_gpus = list(range(total_gpus))
        print(f"Using GPUs: {available_gpus} (out of {total_gpus} available)")
    
    # Determine number of workers - always auto-calculate based on GPUs and max_runs_per_gpu
    if use_cuda:
        # For CUDA: use max_runs_per_gpu to determine workers (allows 2 runs per GPU by default)
        num_available_gpus = len(available_gpus) if available_gpus else 0
        num_workers = min(num_available_gpus * args.max_runs_per_gpu, mp.cpu_count())
        print(f"CUDA detected: {num_available_gpus} GPU(s) selected, using {num_workers} worker(s) ({args.max_runs_per_gpu} runs per GPU)")
    else:
        # For CPU: use conservative default (half of cores, max 4)
        cpu_count = mp.cpu_count()
        num_workers = min(4, max(1, cpu_count // 2))
        print(f"CPU mode: {cpu_count} cores available, using {num_workers} worker(s) to avoid resource hogging")
    
    # Convert args to dict for multiprocessing (args objects aren't picklable)
    args_dict = {
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'lr': args.lr,
        'filters': args.filters,
        'model': args.model,
        'mlp_layer': args.mlp_layer,
        'no_cuda': args.no_cuda,
        'num_classes': args.num_classes,
        'seed': args.seed,
        'unlearn_bs': args.unlearn_bs,
        'weight_decay': args.weight_decay,
        'C': args.C,
        's1': args.s1,
        's2': args.s2,
        'std': args.std,
        'gamma': args.gamma,
        'scale': args.scale,
        'forget_prob': args.forget_prob,
        'independent_sampling': args.independent_sampling if hasattr(args, 'independent_sampling') else False,
        'batch_level_sampling': args.batch_level_sampling if hasattr(args, 'batch_level_sampling') else False,
        'num_runs': args.num_runs,
        'optimizer': args.optimizer,
        'momentum': args.momentum,
        'scheduler_type': args.scheduler_type,
        'scheduler_params': args.scheduler_params,
        'shuffle_train_samples': args.shuffle_train_samples if hasattr(args, 'shuffle_train_samples') else False,
        'shuffle_train_batches_each_epoch': args.shuffle_train_batches_each_epoch if hasattr(args, 'shuffle_train_batches_each_epoch') else True,
        'overwrite_existing_runs': args.overwrite_existing_runs,
    }
    
    # Determine starting run index
    start_run_idx = args.start_run_idx
    end_run_idx = start_run_idx + args.num_runs
    
    print(f"\n{'='*70}")
    print(f"Starting {args.num_runs} runs with {num_workers} parallel worker(s)")
    print(f"Run indices: {start_run_idx} to {end_run_idx - 1} (inclusive)")
    if use_cuda and available_gpus:
        print(f"GPU assignment: Dynamic (max_runs_per_gpu={args.max_runs_per_gpu})")
    print(f"{'='*70}\n")
    
    # Helper function to check if run folder already exists
    def run_folder_exists(run_idx):
        run_subdir = os.path.join(model_dir, f'run_{run_idx}')
        return os.path.exists(run_subdir)

    def should_skip_run(run_idx):
        return run_folder_exists(run_idx) and not args.overwrite_existing_runs
    
    # Perform multiple train+unlearn runs in parallel
    if num_workers == 1:
        # Sequential execution (useful for debugging)
        for run_idx in range(start_run_idx, end_run_idx):
            # Check if run folder already exists
            if should_skip_run(run_idx):
                print(f"⊘ Run {run_idx} skipped (folder already exists)")
                continue
            
            # For sequential execution, use round-robin GPU assignment
            assigned_gpu = None
            if use_cuda and available_gpus:
                assigned_gpu = available_gpus[run_idx % len(available_gpus)]
                print(f"Run {run_idx} assigned to GPU {assigned_gpu}")
            elif use_cuda:
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if num_gpus > 0:
                    auto_gpu = run_idx % num_gpus
                    print(f"Run {run_idx} using GPU {auto_gpu} (auto-assigned)")
                else:
                    print(f"Run {run_idx} using CPU")
            else:
                print(f"Run {run_idx} using CPU")
            
            # Execute run
            try:
                _, exit_code = run_single_experiment(run_idx, args_dict, model_dir, data_dir, assigned_gpu_id=assigned_gpu)
                print(f"✓ Run {run_idx} completed (exit code: {exit_code})")
            except Exception as exc:
                print(f"✗ Run {run_idx} failed with exception (exit code: 1)")
                print(f"  Exception: {exc}")
                import traceback
                traceback.print_exc()
    else:
        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        # This is required because CUDA cannot be re-initialized in forked processes
        if use_cuda:
            # Use spawn context for CUDA compatibility
            ctx = mp.get_context('spawn')
            executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
        else:
            # Use default (fork) for CPU-only, which is more efficient
            executor = ProcessPoolExecutor(max_workers=num_workers)
        
        # Parallel execution with dynamic GPU assignment
        runs_to_process = args.num_runs
        
        # Filter out runs that already have folders
        all_runs = list(range(start_run_idx, end_run_idx))
        pending_runs = [run_idx for run_idx in all_runs if not should_skip_run(run_idx)]
        skipped_runs = [run_idx for run_idx in all_runs if should_skip_run(run_idx)]
        
        if skipped_runs:
            print(f"Skipping {len(skipped_runs)} runs that already have folders: {skipped_runs[:10]}{'...' if len(skipped_runs) > 10 else ''}")
            runs_to_process = len(pending_runs)
        
        # Track GPU usage: how many runs are currently running on each GPU
        gpu_current_runs = {gpu_id: 0 for gpu_id in available_gpus} if (use_cuda and available_gpus) else {}
        # Track which GPU each running task is using
        future_to_gpu = {}
        
        with executor:
            futures = {}
            completed = 0
            
            # Helper function to find available GPU
            def find_available_gpu():
                if not (use_cuda and available_gpus):
                    return None
                # Find GPU with fewest current runs that hasn't hit the limit
                available_gpus_list = [
                    gpu_id for gpu_id in available_gpus 
                    if gpu_current_runs[gpu_id] < args.max_runs_per_gpu
                ]
                if len(available_gpus_list) == 0:
                    return None
                # Return GPU with fewest current runs
                return min(available_gpus_list, key=lambda gpu: gpu_current_runs[gpu])
            
            # Submit initial batch of tasks (up to available GPU slots)
            # Allow up to max_runs_per_gpu runs per GPU
            max_concurrent = len(available_gpus) * args.max_runs_per_gpu if (use_cuda and available_gpus) else num_workers
            while pending_runs and len(futures) < max_concurrent:
                assigned_gpu = find_available_gpu()
                if assigned_gpu is None and use_cuda and available_gpus:
                    # No GPU available, wait for one to free up
                    break
                
                run_idx = pending_runs.pop(0)
                
                # Check if folder was created between check and now (race condition protection)
                if should_skip_run(run_idx):
                    print(f"⊘ Run {run_idx} skipped (folder already exists)")
                    completed += 1
                    # Try to get next run if available
                    if not pending_runs:
                        break
                    continue
                
                # Print assignment message
                if use_cuda and assigned_gpu is not None:
                    print(f"Run {run_idx} assigned to GPU {assigned_gpu}")
                    gpu_current_runs[assigned_gpu] += 1
                elif use_cuda:
                    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    if num_gpus > 0:
                        auto_gpu = run_idx % num_gpus
                        print(f"Run {run_idx} using GPU {auto_gpu} (auto-assigned)")
                    else:
                        print(f"Run {run_idx} using CPU")
                else:
                    print(f"Run {run_idx} using CPU")
                
                future = executor.submit(run_single_experiment, run_idx, args_dict, model_dir, data_dir, assigned_gpu_id=assigned_gpu)
                futures[future] = run_idx
                if assigned_gpu is not None:
                    future_to_gpu[future] = assigned_gpu
            
            # Process completed tasks and submit new ones
            while completed < runs_to_process:
                # Wait for at least one task to complete
                for future in as_completed(futures):
                    completed_idx = futures.pop(future)
                    assigned_gpu = future_to_gpu.pop(future, None)
                    
                    # Free up the GPU slot
                    if assigned_gpu is not None:
                        gpu_current_runs[assigned_gpu] -= 1
                    
                    # Print completion message with exit code
                    try:
                        _, exit_code = future.result()
                        completed += 1
                        status = "✓" if exit_code == 0 else "✗"
                        print(f"{status} Run {completed_idx} completed (exit code: {exit_code}) ({completed}/{args.num_runs})")
                    except Exception as exc:
                        completed += 1
                        print(f"✗ Run {completed_idx} failed with exception (exit code: 1) ({completed}/{args.num_runs})")
                        print(f"  Exception: {exc}")
                    
                    # Submit next task if available and GPU slot is free
                    if pending_runs:
                        assigned_gpu = find_available_gpu()
                        if assigned_gpu is not None or not (use_cuda and available_gpus):
                            run_idx = pending_runs.pop(0)
                            
                            # Check if folder was created between check and now (race condition protection)
                            if should_skip_run(run_idx):
                                print(f"⊘ Run {run_idx} skipped (folder already exists)")
                                completed += 1
                                # Don't submit this run, continue to next iteration
                                if not pending_runs:
                                    break
                                continue
                            
                            # Print assignment message
                            if use_cuda and assigned_gpu is not None:
                                print(f"Run {run_idx} assigned to GPU {assigned_gpu}")
                                gpu_current_runs[assigned_gpu] += 1
                            elif use_cuda:
                                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                                if num_gpus > 0:
                                    auto_gpu = run_idx % num_gpus
                                    print(f"Run {run_idx} using GPU {auto_gpu} (auto-assigned)")
                                else:
                                    print(f"Run {run_idx} using CPU")
                            else:
                                print(f"Run {run_idx} using CPU")
                            
                            new_future = executor.submit(run_single_experiment, run_idx, args_dict, model_dir, data_dir, assigned_gpu_id=assigned_gpu)
                            futures[new_future] = run_idx
                            if assigned_gpu is not None:
                                future_to_gpu[new_future] = assigned_gpu
                    
                    break  # Break to check if we're done
    
    print(f"\n{'='*70}")
    print(f"ALL {args.num_runs} RUNS COMPLETE!")
    print(f"{'='*70}\n")

