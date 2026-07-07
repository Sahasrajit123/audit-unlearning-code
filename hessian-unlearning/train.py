import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau, OneCycleLR
from sklearn.metrics import f1_score
from utils import load_data, load_dataset
from model import MLP, AllCNN, ResNet18, ResNet50
import os


def create_optimizer(model, optimizer_type, lr, wd, momentum=0.9):
    """
    Create optimizer based on type.
    
    Args:
        model: The model to optimize
        optimizer_type: 'adam' or 'sgd'
        lr: Learning rate
        wd: Weight decay
        momentum: Momentum for SGD (default: 0.9)
    
    Returns:
        Optimizer instance
    """
    optimizer_type = optimizer_type.lower()
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose from: 'adam', 'sgd'")
    return optimizer


def create_scheduler(optimizer, scheduler_type=None, scheduler_params=None):
    """
    Create learning rate scheduler based on type.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_type: 'step', 'cosine', 'exponential', 'plateau', or None
        scheduler_params: Dictionary of scheduler parameters
    
    Returns:
        Scheduler instance or None
    """
    if scheduler_type is None or scheduler_type.lower() == 'none':
        return None
    
    scheduler_type = scheduler_type.lower()
    scheduler_params = scheduler_params or {}
    
    if scheduler_type == 'step':
        step_size = scheduler_params.get('step_size', 30)
        gamma = scheduler_params.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = scheduler_params.get('T_max', 100)
        eta_min = scheduler_params.get('eta_min', 0)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == 'exponential':
        gamma = scheduler_params.get('gamma', 0.95)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == 'plateau':
        mode = scheduler_params.get('mode', 'min')
        factor = scheduler_params.get('factor', 0.1)
        patience = scheduler_params.get('patience', 10)
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    elif scheduler_type == 'onecycle':
        max_lr = scheduler_params.get('max_lr', 0.1)
        total_steps = scheduler_params.get('total_steps', None)
        pct_start = scheduler_params.get('pct_start', 0.3)
        anneal_strategy = scheduler_params.get('anneal_strategy', 'cos')
        if total_steps is None:
            raise ValueError("OneCycleLR requires 'total_steps' in scheduler_params")
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, 
                              pct_start=pct_start, anneal_strategy=anneal_strategy)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose from: 'step', 'cosine', 'exponential', 'plateau', 'onecycle', 'none'")
    
    return scheduler


def train(train_loader, val_loader, model, lr, wd, num_epochs, max_norm, path, device, 
          optimizer_type='adam', momentum=0.9, scheduler_type=None, scheduler_params=None, steps_per_epoch=None):
    """
    Train the model with specified optimizer and scheduler.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: The model to train
        lr: Learning rate
        wd: Weight decay
        num_epochs: Number of training epochs
        max_norm: Maximum norm for gradient clipping
        path: Path to save the best model
        device: Device to train on
        optimizer_type: 'adam' or 'sgd' (default: 'adam')
        momentum: Momentum for SGD (default: 0.9)
        scheduler_type: Type of scheduler ('step', 'cosine', 'exponential', 'plateau', or None)
        scheduler_params: Dictionary of scheduler parameters
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, optimizer_type, lr, wd, momentum)
    
    # For OneCycleLR, we need total_steps. If not provided, calculate from steps_per_epoch
    if scheduler_type and scheduler_type.lower() == 'onecycle':
        if scheduler_params is None:
            scheduler_params = {}
        if 'total_steps' not in scheduler_params:
            if steps_per_epoch is None:
                steps_per_epoch = len(train_loader)
            scheduler_params['total_steps'] = steps_per_epoch * num_epochs
        if 'max_lr' not in scheduler_params:
            scheduler_params['max_lr'] = lr
    
    scheduler = create_scheduler(optimizer, scheduler_type, scheduler_params)
    
    min_val_loss = 1e10
    for epoch in range(num_epochs):
        # Training phase
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(data)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Perform backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Step OneCycleLR scheduler per batch (if using)
            if scheduler is not None and scheduler_type == 'onecycle':
                scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)

            param_norm = nn.utils.parameters_to_vector(model.parameters()).norm()
            if param_norm > max_norm:
                scale_factor = max_norm / param_norm
                for param in model.parameters():
                    param.data *= scale_factor

        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()  # Set the model in evaluation mode
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
            
        val_loss /= len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), path)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            elif scheduler_type == 'onecycle':
                # OneCycleLR should be stepped per batch, not per epoch
                # But we're calling it per epoch for simplicity
                # This is a limitation - ideally OneCycleLR should be stepped in the training loop
                pass  # OneCycleLR is typically stepped per batch, handled below
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = lr
        
        # Print the validation loss and accuracy for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, LR: {current_lr:.6f}")

    print('Finished Training')


def test(testloader, model, device):
    criterion = nn.CrossEntropyLoss()
    loss = 0
    correct = 0
    total = 0
    pred_test = []
    label_test = []
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_test.append(predicted)
            label_test.append(labels)
    pred_test = torch.cat(pred_test, 0)
    label_test = torch.cat(label_test, 0)
    f1 = f1_score(label_test.detach().cpu().numpy(), pred_test.detach().cpu().numpy(), average='micro')
    print(f"Test Loss: {loss / len(testloader):.4f}, Test Accuracy: {100.0 * correct / total:.2f}%, Test Micro F1: {100.0 * f1:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--model', default='mlp')
    parser.add_argument('--mlp-layer', type=int, default=3, metavar='N',
                        help='number of layers of MLP (default: 3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--C', type=float, default=20,
                        help='Norm constraint of parameters')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # Set the random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists('./model'):
        os.mkdir('./model')

    PATH = './model/'+f"clip_{str(args.C).replace('.','_')}_"+args.model+'_'+args.dataset+f"_lr_{str(args.lr).replace('.','_')}"+f"_bs_{str(args.batch_size)}"+f"_wd_{str(args.weight_decay).replace('.','_')}"+f"_epoch_{str(args.epochs)}"+f"_seed_{str(args.seed)}"+'.pth'

    trainloader, valloader, testloader = load_data(args.dataset, args.batch_size, args.seed)
    train_set, _ = load_dataset(args.dataset)
    num_classes = 10
    args.num_classes = num_classes

    if args.model == 'mlp':
        model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'cnn':
        model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet50':
        model = ResNet50(num_classes=args.num_classes).to(args.device)

    train(trainloader, valloader, model, args.lr, args.weight_decay, args.epochs, args.C, PATH, args.device)
    model.load_state_dict(torch.load(PATH))
    test(testloader, model, args.device)
    