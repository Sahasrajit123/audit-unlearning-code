#!/usr/bin/env python3
import argparse
import os
import time
import copy
import random
import pickle

import numpy as np

import torch
import torch.optim as optim

from tqdm import tqdm
import models
import datasets
from utils import *

def get_Lipschitz(model1, grad_vector1, model2, grad_vector2):
    '''
    Estimates Lipschitz constant based on model1, model2
    '''
    params1 = []
    params2 = []

    for param in model1.parameters():
        params1.append(param.view(-1))
            
    for param in model2.parameters():
        params2.append(param.view(-1))

    param1_vector = torch.cat(params1)
    param2_vector = torch.cat(params2)

 
    numer = torch.norm(grad_vector1 - grad_vector2, p=2).item()   
    denom = torch.norm(param1_vector - param2_vector, p=2).item()
    L = numer/denom
    return L 

def compute_l2_norm(model): #computes l2 norm of gradients
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:  # Check if gradient exists
            param_norm = param.grad.data.norm(2)  # L2 norm of gradients
            total_norm += param_norm.item() ** 2  # Sum of squared norms

    total_norm = total_norm ** 0.5  # Square root of sum of squared norms (Euclidean norm)
    return(total_norm)

def copy_model_with_grads(model):
    new_model = copy.deepcopy(model)

    for (new_param, param) in zip(new_model.parameters(), model.parameters()):
        if param.grad is not None:
            new_param.grad = param.grad.clone()

    return new_model


def add_gaussian_noise_to_weights(args, model, sigma):
    # Iterate over all model parameters
    for param in model.parameters():
        # Check if the parameter has a gradient (usually it will if it's a learnable weight)
        if param.requires_grad:
            # Create Gaussian noise with mean 0 and standard deviation sigma
            noise = torch.normal(mean=0.0, std=sigma, size=param.size()).to(args.device)
            # Add the noise to the parameter
            param.data += noise


def compute_full_gradient(args, model, data_loader, criterion):
    model.eval()
    #model.train()
    model.zero_grad()
    
    total_gradient = None

    for batch_idx, (data, target, identity) in enumerate(data_loader):

        model.zero_grad(set_to_none=True)
        data, target = data.to(args.device), target.to(args.device)
                
        output = model(data)
        loss = criterion(output, target) 
        loss.backward()

        gradients = []
        params = []

        for param in model.parameters():
            if param.grad is not None:  # Check if gradient exists
                gradients.append(param.grad.view(-1))  # Flatten the gradient and add to list

        # Concatenate all gradients into a single vector
        grad_vector = torch.cat(gradients) 
        
        # Use actual batch size from the data (data_loader uses micro_batch_size)
        actual_batch_size = data.size(0)

        if total_gradient is None:
            total_gradient = grad_vector * actual_batch_size
        else:
            total_gradient += grad_vector * actual_batch_size

    total_gradient = total_gradient/len(data_loader.dataset)

    #print(f"gradnorm: {torch.linalg.vector_norm(total_gradient)}")
    #print(f"one batch gradnorm: {compute_l2_norm(model)}")

    return(total_gradient)

def estimate_Lipschitz(model, Nsamples = 100):
    L_list = []
    model1 = copy.deepcopy(model)
        
    grad_vector = compute_full_gradient(args, model, train_loader, criterion)

    Nsamples = 100 #number of samples for lipschitz estimate
    for i in tqdm(range(Nsamples)):
        model1 = copy.deepcopy(model)
        add_gaussian_noise_to_weights(args, model1, 0.01)
        grad_vector1 = compute_full_gradient(args, model1, train_loader, criterion)
            
        lip = get_Lipschitz(model1, grad_vector1, model, grad_vector)
        L_list.append(lip)

    del model1
    return(max(L_list))



    
def run_epoch_simple(model, data_loader, device, criterion=torch.nn.CrossEntropyLoss(), optimizer=None):
    model.train()

    with torch.set_grad_enabled(True):

        for batch_idx, (data, target, identity) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

                
            output = model(data)
            loss = criterion(output, target) 

            optimizer.zero_grad() 
            loss.backward()

            optimizer.step()
           
    
def run_epoch(args, model, data_loader, criterion=torch.nn.CrossEntropyLoss(), optimizer=None, epoch=0, mode='train'):
    
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    
    metrics = AverageMeter() #reset after each epoch
    
    # Gradient accumulation setup (only for training mode)
    accumulation_steps = args.accumulation_steps if mode == 'train' else 1
    accumulation_counter = 0
    actual_samples_in_batch = 0  # Track actual number of samples in current effective batch

    with torch.set_grad_enabled(mode == 'train'):

        for batch_idx, (data, target, identity) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            actual_batch_size = data.size(0)  # Actual size of this micro-batch (may be smaller than micro_batch_size)

                
            output = model(data)
            loss = criterion(output, target) 

            if not args.quiet:
                metrics.update(n=actual_batch_size, loss=loss.item(), error=get_error(output, target))
            
            if mode == 'train':
                # Scale loss by accumulation_steps to average gradients
                # Note: This assumes all micro-batches are full size, but we'll correct for incomplete batches later
                loss = loss / accumulation_steps
                
                # Zero gradients only at the start of each effective batch
                if accumulation_counter == 0:
                    optimizer.zero_grad() 
                    model.zero_grad(set_to_none=True) #double assurance?
                    actual_samples_in_batch = 0
                
                loss.backward()
                accumulation_counter += 1
                actual_samples_in_batch += actual_batch_size
                
                # Update optimizer and compute gradient norm after accumulating all micro-batches
                if accumulation_counter == accumulation_steps:
                    # Check if we have fewer samples than expected (due to incomplete micro-batches)
                    # Since loss is averaged per sample and we divided by accumulation_steps,
                    # if actual_samples < effective_batch_size, we need to scale gradients
                    # to make them equivalent to processing the full effective_batch_size
                    if actual_samples_in_batch < args.batch_size:
                        scale_factor = args.batch_size / actual_samples_in_batch
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad *= scale_factor
                    
                    if not args.no_gradient_estimation:
                        grad_norm = compute_l2_norm(model)
                        metrics.update(n=actual_samples_in_batch, grad_norm=grad_norm)
                    
                    optimizer.step()
                    accumulation_counter = 0
                    actual_samples_in_batch = 0
                

    # Handle remaining accumulated gradients if epoch ends mid-accumulation
    if mode == 'train' and accumulation_counter > 0:
        # Scale gradients to make them equivalent to processing the full effective_batch_size
        # This handles both incomplete effective batches and smaller last micro-batches
        scale_factor = args.batch_size / actual_samples_in_batch
        for param in model.parameters():
            if param.grad is not None:
                param.grad *= scale_factor
        
        if not args.no_gradient_estimation:
            grad_norm = compute_l2_norm(model)
            metrics.update(n=actual_samples_in_batch, grad_norm=grad_norm)
        
        optimizer.step()

    log_metrics(mode, metrics, epoch)
    
    if mode == 'train':
        print('Learning Rate : {}'.format(optimizer.param_groups[0]['lr']))
    return metrics


    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None)
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='something about suppressing logs')

    parser.add_argument('--batch-size', type=str, default='256', metavar='N',
                        help='Effective batch size for training (default: 256). Can be an integer or "full" for full-batch gradient descent. Gradients are accumulated across micro-batches to reach this size.')
    parser.add_argument('--micro-batch-size', type=int, default=None, metavar='N',
                        help='Micro batch size for GPU loading (default: same as batch-size). If smaller than batch-size, gradients are accumulated. Must be <= batch-size and batch-size must be divisible by micro-batch-size.')
    parser.add_argument('--dataset', default='lacuna100binary128')
    parser.add_argument('--dataroot', type=str, default='data/lacuna100binary128')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--num-ids-forget', type=int, default=None,
                        help='Number of IDs to forget')
    parser.add_argument('--forget-ids-file', type=str, default=None,
                        help='Optional path to a file containing forget IDs (json or npy). If provided, these IDs are used.')
    parser.add_argument('--lossfn', type=str, default='ce',
                        help='Cross Entropy: ce or mse')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--scheduler', type=float, default=1,
                        help='exponential scheduler')
    parser.add_argument('--model', default='resnetsmooth')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume')
    parser.add_argument('--device', type=int, default=0, metavar='S',
                        help='GPU device number')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='shuffle training/test loaders (default: True)')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                        help='disable shuffling for all loaders')
    parser.add_argument('--save-checkpoints', action='store_true', default=False,
                        help='save checkpoints')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot training and validation loss')
    parser.add_argument('--model-selection', action='store_true', default=False,
                        help='store and log model with best validation error')
    
    parser.add_argument('--no-gradient-estimation', action='store_true', default=False, help='disables gradient norm estimation')

    parser.add_argument('--compute-lipschitz', action='store_true', default=True, help='estimate Lipschitz constant')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base output directory for checkpoints, logs, and plots. If None, uses current directory.')

    args = parser.parse_args()
    
    # Handle "full" batch size - will be set after data is loaded
    use_full_batch = False
    if args.batch_size.lower() == 'full':
        use_full_batch = True
        args.batch_size = None  # Will be set to dataset size later
    else:
        try:
            args.batch_size = int(args.batch_size)
        except ValueError:
            raise ValueError(f"batch_size must be an integer or 'full', got: {args.batch_size}")
    
    # For data loading, use micro_batch_size if provided, otherwise use a default
    # (we'll set up proper accumulation after we know the dataset size)
    if args.micro_batch_size is None:
        if args.batch_size is not None:
            args.micro_batch_size = args.batch_size
        else:
            args.micro_batch_size = 256  # Default for "full" case
   
    manual_seed(args.seed)
    
    # Set up output directories
    if args.output_dir is not None:
        output_base = args.output_dir
        checkpoint_dir = os.path.join(output_base, 'checkpoints')
        log_dir = os.path.join(output_base, 'logs')
        plot_dir = os.path.join(output_base, 'plots')
    else:
        checkpoint_dir = 'checkpoints'
        log_dir = 'logs'
        plot_dir = 'plots'
    
    os.makedirs(log_dir, exist_ok=True)

    #DEVICE MANAGEMENT
    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")
    assert use_cuda

    os.makedirs(checkpoint_dir, exist_ok=True)

    ood = True
    if args.dataset == 'eicu':
        ood=False

    #LOAD DATA!
    # Load optional forget ids from file if provided
    # Note: forget_ids_file may contain either identity IDs or batch indices
    # If it contains batch indices, we'll convert them to identity IDs
    forget_ids = None
    if args.forget_ids_file is not None:
        if os.path.exists(args.forget_ids_file):
            try:
                if args.forget_ids_file.endswith(".npy"):
                    # For .npy files, assume it's a legacy format (list of identity IDs)
                    loaded_data = np.load(args.forget_ids_file).tolist()
                    forget_ids = loaded_data
                else:
                    import json
                    with open(args.forget_ids_file, "r") as fh:
                        loaded_data = json.load(fh)
                    
                    # Check if it's a structured format with type indicator
                    if isinstance(loaded_data, dict) and "type" in loaded_data and "values" in loaded_data:
                        data_type = loaded_data["type"]
                        values = loaded_data["values"]
                        
                        if data_type == "batch_indices":
                            # Convert batch indices to identity IDs
                            # For CIFAR10, when identities are not in batch files,
                            # CIFARPickle generates them sequentially (0, 1, 2, ...)
                            # Since each batch has 1 sample, batch_index == identity
                            # So batch indices can be used directly as identity IDs
                            forget_ids = values
                            print(f"Using {len(forget_ids)} batch indices as identity IDs")
                        elif data_type == "identity_ids":
                            # Already identity IDs
                            forget_ids = values
                        else:
                            raise ValueError(f"Unknown forget_ids type: {data_type}")
                    elif isinstance(loaded_data, list):
                        # Legacy format: assume it's a list of identity IDs
                        forget_ids = loaded_data
                    else:
                        # Unknown format, try to use as-is
                        forget_ids = loaded_data
            except Exception as e:
                print(f"Failed to load forget ids from {args.forget_ids_file}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Forget ids file not found: {args.forget_ids_file}")

    loaders = datasets.get_loaders_large(
        args.dataset,
        num_ids_forget=args.num_ids_forget,
        forget_ids=forget_ids,
        batch_size=args.micro_batch_size,  # Use micro_batch_size for actual data loading
        seed=args.seed,
        root=args.dataroot,
        augment=False,
        shuffle=args.shuffle,
        ood=ood,
        test=False,
    )

    # Persist the actual forget IDs used (if available)
    if args.forget_ids_file is not None and 'forget_ids' in loaders:
        try:
            os.makedirs(os.path.dirname(args.forget_ids_file), exist_ok=True)
            with open(args.forget_ids_file, "w") as fh:
                import json
                json.dump(list(loaders['forget_ids']), fh)
            # Also store npy alongside for convenience
            np.save(args.forget_ids_file.replace(".json", ".npy"), np.array(loaders['forget_ids']))
        except Exception as e:
            print(f"Warning: could not save forget ids to {args.forget_ids_file}: {e}")
    train_loader = loaders['train_loader']
    valid_loader = loaders['valid_loader']

    # When resuming, use the actual forget set size from loaded data instead of the argument
    if args.resume is not None and 'num_forget_ids' in loaders:
        actual_forget_size = loaders['num_forget_ids']
        if args.num_ids_forget != actual_forget_size:
            print(f"Updating num_ids_forget from {args.num_ids_forget} to actual loaded forget set size: {actual_forget_size}")
            args.num_ids_forget = actual_forget_size

    # Check if we'll be using retain_loader for training
    will_use_retain = args.resume is not None and 'retain_loader' in loaders

    # Handle "full" batch size
    # If using retain_loader, use retain set size; otherwise use training set size
    if use_full_batch:
        if will_use_retain:
            retain_loader = loaders['retain_loader']
            args.batch_size = len(retain_loader.dataset)
            print(f"Full batch size enabled: using retain dataset size = {args.batch_size} (retain_loader will be used for training)")
        else:
            args.batch_size = len(train_loader.dataset)
            print(f"Full batch size enabled: using training dataset size = {args.batch_size}")

    # If resuming and a retain split is available, train on retain set directly
    if will_use_retain:
        train_loader = loaders['retain_loader']
        print("Using retain_loader for training during resume.")
    
    # Now set up micro-batch-size and gradient accumulation
    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size
    
    # Validation
    if args.micro_batch_size > args.batch_size:
        raise ValueError(f"micro_batch_size ({args.micro_batch_size}) must be <= batch_size ({args.batch_size})")
    
    # For "full" batch size, allow non-divisibility (we handle incomplete batches in training loop)
    # For fixed batch sizes, require divisibility for cleaner implementation
    if not use_full_batch and args.batch_size % args.micro_batch_size != 0:
        raise ValueError(f"batch_size ({args.batch_size}) must be divisible by micro_batch_size ({args.micro_batch_size})")
    
    # Calculate accumulation steps
    # Use ceiling division to ensure we accumulate enough micro-batches to reach (or exceed) batch_size
    # This is especially important for "full" batch size where dataset size may not be divisible by micro_batch_size
    import math
    args.accumulation_steps = math.ceil(args.batch_size / args.micro_batch_size)
    
    if use_full_batch and args.batch_size % args.micro_batch_size != 0:
        print(f"Note: dataset size ({args.batch_size}) is not divisible by micro_batch_size ({args.micro_batch_size}). "
              f"Using accumulation_steps={args.accumulation_steps} (will accumulate {args.accumulation_steps * args.micro_batch_size} samples per effective batch, "
              f"which may exceed dataset size for the last effective batch).")
    
    if args.accumulation_steps > 1:
        print(f"Gradient accumulation enabled: effective batch size={args.batch_size}, micro batch size={args.micro_batch_size}, accumulation steps={args.accumulation_steps}")
    else:
        print(f"No gradient accumulation: batch size={args.batch_size}")
    
    # Generate name after batch_size and accumulation_steps are set (so "full" shows actual number)
    if args.name is None:
        args.name = f"{args.dataset}_{args.model}_{str(args.filters).replace('.','_')}"
        # Use actual forget set size from loaders if available, otherwise use args.num_ids_forget
        if 'num_forget_ids' in loaders:
            forget_size = loaders['num_forget_ids']
        elif args.num_ids_forget is not None:
            forget_size = args.num_ids_forget
        else:
            forget_size = 0
        if forget_size > 0:
            args.name += f"_forget_{forget_size}"
        args.name+=f"_lr_{str(args.lr).replace('.','_')}"
        if args.accumulation_steps > 1:
            args.name+=f"_bs_{str(args.batch_size)}_mbs_{str(args.micro_batch_size)}"
        else:
            args.name+=f"_bs_{str(args.batch_size)}"
        args.name+=f"_ls_{args.lossfn}"
        args.name+=f"_seed_{str(args.seed)}"
    if args.scheduler is not None:
        args.name+=f"_scheduler_{str(args.scheduler).replace('.','_')}"

    print(f'Checkpoint name: {args.name}')
    
    num_classes = max(train_loader.dataset.targets) + 1 if args.num_classes is None else args.num_classes
    args.num_classes = num_classes
    print(f"Number of Classes: {num_classes}")

    #GET MODEL
    model = models.get_model(args.model, num_classes=num_classes, filters_percentage=args.filters).to(args.device)
    
    if args.resume is not None:
        state = torch.load(args.resume,weights_only=True)
        model.load_state_dict(state)
        print(f"Loading state from: {args.resume}")
        args.name += f"_loadedfrom{args.resume.split('.')[0].split('_')[-1]}"
    
    torch.save(model.state_dict(), f"{checkpoint_dir}/{args.name}_init.pt")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=0)
    
    if args.scheduler is not None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler)

    criterion = torch.nn.CrossEntropyLoss().to(args.device) if args.lossfn=='ce' else torch.nn.MSELoss().to(args.device)

    train_time = 0

    plot_loss = [] #training loss
    plot_valid_loss = []

    plot_train_error = []
    plot_valid_error = []
    
    plot_grad_norm = []
    plot_lr = []

    args_dict = vars(args)
    logs = dict()
    for key, value in args_dict.items():
        logs[key] = value

    valid_loss = 1000 #for model selection 

    for epoch in range(args.epochs):

        model1 = copy.deepcopy(model)
        t1 = time.time()

        if args.scheduler is not None:
            plot_lr.append(scheduler.get_last_lr())

        metrics = run_epoch(args, model, train_loader, criterion, optimizer, epoch, mode='train')
        valid_metrics = run_epoch(args, model, valid_loader, criterion, optimizer, epoch, mode='valid')

        if args.scheduler is not None:
            scheduler.step()

        if args.model_selection:
            if valid_metrics.avg['error'] < valid_loss:
                valid_epoch = epoch
                valid_name = f"{checkpoint_dir}/{args.name}_selected.pt"
                torch.save(model.state_dict(), valid_name)
                valid_loss = valid_metrics.avg['error'] 
        
        plot_loss.append(metrics.avg['loss'])
        plot_train_error.append(metrics.avg['error'])
        plot_valid_loss.append(valid_metrics.avg['loss'])
        plot_valid_error.append(valid_metrics.avg['error'])
        if not args.no_gradient_estimation:
            plot_grad_norm.append(metrics.max['grad_norm'])

        t2 = time.time()
        train_time += np.round(t2-t1,2)
        
        if epoch % 5 == 0 and args.save_checkpoints:
            torch.save(model.state_dict(), f"{checkpoint_dir}/{args.name}_{epoch}.pt")
        

        print(f'Epoch Time: {np.round(time.time()-t1,2)} sec')


    print (f'Pure training time: {train_time} sec')


    torch.save(model.state_dict(), f"{checkpoint_dir}/{args.name}_{epoch}_final.pt") 


    final_it = -1

    if args.model_selection:
        state = torch.load(valid_name, weights_only=True)
        model.load_state_dict(state)
        print(f"Model selection: Epoch {valid_epoch} with valid error {valid_loss}")
        final_it = valid_epoch
        logs['selected epoch'] = final_it


    #log information for plotting
    logs['train loss over epochs'] = plot_loss
    logs['train error over epochs'] = plot_train_error
    logs['final train loss'] = plot_loss[final_it]
    logs['final train error'] = plot_train_error[final_it]

    logs['valid loss over epochs'] = plot_valid_loss
    logs['valid error over epochs'] = plot_valid_error
    logs['final valid loss'] = plot_valid_loss[final_it]
    logs['final valid error'] = plot_valid_error[final_it]

    # Explicit end-of-phase evaluation metrics for quick visibility.
    final_eval_epoch = args.epochs - 1
    final_valid_metrics = run_epoch(args, model, valid_loader, criterion, optimizer, final_eval_epoch, mode='valid_final')
    logs['final eval valid loss'] = final_valid_metrics.avg['loss']
    logs['final eval valid error'] = final_valid_metrics.avg['error']

    final_test_metrics = None
    if 'test_loader' in loaders:
        test_loader = loaders['test_loader']
        final_test_metrics = run_epoch(args, model, test_loader, criterion, optimizer, final_eval_epoch, mode='test')
        logs['final eval test loss'] = final_test_metrics.avg['loss']
        logs['final eval test error'] = final_test_metrics.avg['error']

    phase_name = 'unlearn' if args.resume is not None else 'train'
    final_val_acc = 1.0 - final_valid_metrics.avg['error']
    print(f"[{phase_name}] final val accuracy: {final_val_acc:.4f} ({100.0 * final_val_acc:.2f}%)")
    if final_test_metrics is not None:
        final_test_acc = 1.0 - final_test_metrics.avg['error']
        print(f"[{phase_name}] final test accuracy: {final_test_acc:.4f} ({100.0 * final_test_acc:.2f}%)")
    else:
        print(f"[{phase_name}] test loader not available, skipping final test accuracy")

    if not args.no_gradient_estimation:
        logs['grad norm over epochs'] = plot_grad_norm
    
    logs['lr over epochs'] = plot_lr

    
    if args.dataset != "eicu":
        if 'ood_loader' in loaders:
            print("Testing on OOD data")
            ood_loader = loaders['ood_loader']
            ood_metrics = run_epoch(args, model, ood_loader, criterion, optimizer, epoch, mode='ood')
            logs['final ood loss'] = ood_metrics.avg['loss']
            logs['final ood error'] = ood_metrics.avg['error']
        else:
            print("OOD loader not available, skipping OOD testing")

    # Log forget-set metrics when available
    if 'train_forget_loader' in loaders:
        train_forget_loader = loaders['train_forget_loader']
        train_forget_metrics = run_epoch(args, model, train_forget_loader, criterion, optimizer, epoch, mode='train forget')
        logs['final train forget loss'] = train_forget_metrics.avg['loss']
        logs['final train forget error'] = train_forget_metrics.avg['error']
    elif 'forget_loader' in loaders:
        # At minimum, record forget evaluation if only forget_loader is present (pre-split path)
        forget_loader = loaders['forget_loader']
        forget_metrics = run_epoch(args, model, forget_loader, criterion, optimizer, epoch, mode='forget')
        logs['final forget loss'] = forget_metrics.avg['loss']
        logs['final forget error'] = forget_metrics.avg['error']


    if args.compute_lipschitz:
        print("Computing Lipschitz constant...")
        L_list = []

        model1 = copy.deepcopy(model)
        
        grad_vector = compute_full_gradient(args, model, train_loader, criterion)

        Nsamples = 100 #number of samples for lipschitz estimate
        for i in tqdm(range(Nsamples)):
            model1 = copy.deepcopy(model)
            add_gaussian_noise_to_weights(args, model1, 0.01)
            grad_vector1 = compute_full_gradient(args, model1, train_loader, criterion)
            
            lip = get_Lipschitz(model1, grad_vector1, model, grad_vector)
            L_list.append(lip)

        del model1
        print(f"Lipschitz constant: {max(L_list)}")
        logs['Lipschitz'] = max(L_list)

    with open(f"{log_dir}/{args.name}.pkl", 'wb') as f:
        pickle.dump(logs, f)


    #plotting! 

    if args.plot:
        os.makedirs(plot_dir, exist_ok=True)
        import matplotlib.pyplot as plt

        # plotting loss over time 
        plt.figure()
        plt.plot(plot_loss, label="Loss")
        plt.plot(plot_valid_loss, label="Validation Loss")
        plt.plot(plot_train_error, label="Training Error")
        plt.plot(plot_valid_error, label="Validation Error")
        plt.legend()
        plt.xlabel("Epoch")
        plt.title(args.name)
        plt.savefig(f"{plot_dir}/{args.name}.png")

        #plotting gradnorm over time
        if not args.no_gradient_estimation:
            plt.figure()
            
            plt.plot(plot_grad_norm, label="Gradient Norm")
            plt.legend()
            plt.xlabel("Epoch")
            plt.title(args.name)
            plt.savefig(f"{plot_dir}/{args.name}gradientnorm.png")

        print("plots saved successfully!")
