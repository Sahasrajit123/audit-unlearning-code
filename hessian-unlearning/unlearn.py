from __future__ import print_function
import argparse
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd import grad
from torchvision import datasets, transforms
import argparse
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from utils import load_data, load_dataset, load_unlearn_data, batch_grads_to_vec, vec_to_params
from model import MLP, AllCNN, ResNet18


def grad_batch(batch_loader, lam, model, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    params = [p for p in model.parameters() if p.requires_grad]
    grad_batch = [torch.zeros_like(p).cpu() for p in params]
    num = 0
    for batch_idx, (data, targets) in enumerate(batch_loader):
        num += targets.shape[0]
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)

        grad_mini = list(grad(criterion(outputs, targets), params))
        for i in range(len(grad_batch)):
            grad_batch[i] += grad_mini[i].cpu().detach()

    for i in range(len(grad_batch)):
        grad_batch[i] /= num

    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)
    grad_reg = list(grad(lam * l2_reg, params))
    for i in range(len(grad_batch)):
        grad_batch[i] += grad_reg[i].cpu().detach()
    return [p.to(device) for p in grad_batch]


def grad_batch_approx(batch_loader, lam, model, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0
    for batch_idx, (data, targets) in enumerate(batch_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss += criterion(outputs, targets)

    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)

    params = [p for p in model.parameters() if p.requires_grad]
    return list(grad(loss + lam * l2_reg, params))


def hvp(y, w, v):
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def inverse_hvp(y, w, v):

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    vec_first_grads = batch_grads_to_vec(first_grads)
    
    hessian_list = []
    for i in range(vec_first_grads.shape[0]):
        sec_grads = grad(vec_first_grads[i], w, retain_graph=True)
        hessian_list.append(batch_grads_to_vec(sec_grads).unsqueeze(0))
    
    hessian_mat = torch.cat(hessian_list, 0)
    return torch.linalg.solve(hessian_mat, v.view(-1, 1))


def newton_update(g, batch_size, res_set, lam, gamma, model, s1, s2, scale, device, generator=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    H_res = [torch.zeros_like(p) for p in g]
    for i in tqdm(range(s1)):
        H = [p.clone() for p in g]
        sampler = RandomSampler(res_set, replacement=True, num_samples=batch_size * s2, generator=generator)
        # Create a data loader with the sampler
        res_loader = DataLoader(res_set, batch_size=batch_size, sampler=sampler)
        res_iter = iter(res_loader)
        for j in range(s2):
            data, target = next(res_iter)
            data, target = data.to(device), target.to(device)
            z = model(data)
            loss = criterion(z, target)
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            # Add L2 regularization to the loss
            loss += (lam + gamma) * l2_reg
            H_s = hvp(loss, params, H)
            with torch.no_grad():
                for k in range(len(params)):
                    H[k] = H[k] + g[k] - H_s[k] / scale
                if j % int(s2 / 10) == 0:
                    h_norm = torch.norm(batch_grads_to_vec(H)).item()
                    print(f'Epoch: {j}, ||H|| = {h_norm:.6f}')
        h_norm_final = torch.norm(batch_grads_to_vec(H)).item()
        print(f'Sample {i}: final ||H|| = {h_norm_final:.6f}')
        for k in range(len(params)):
            H_res[k] = H_res[k] + H[k] / scale

    return [p / s1 for p in H_res]


def unlearn(model, unl_set, res_set, batch_size, unlearn_bs, weight_decay, gamma, s1, s2, scale, std, device, generator=None):
    """
    Perform certified unlearning on the model.
    
    Args:
        model: The trained model to unlearn from (modified in-place)
        unl_set: The unlearning dataset (samples to forget)
        res_set: The residual dataset (samples to retain)
        batch_size: Batch size for gradient computation
        unlearn_bs: Batch size for unlearning Newton update
        weight_decay: L2 regularization parameter (lambda)
        gamma: Convex approximation coefficient
        s1: Number of samples in Hessian approximation
        s2: Order of Taylor expansion in Hessian approximation
        scale: Scale of Hessian
        std: Standard deviation of Gaussian noise
        device: Device to run on (cuda/cpu)
        generator: Optional torch.Generator for reproducible randomness
    
    Returns:
        None (model is modified in-place)
    """
    print(f"Starting unlearning process...")
    print(f"Unlearning set size: {len(unl_set)}, Residual set size: {len(res_set)}")
    
    # Compute gradients on residual set
    res_loader = torch.utils.data.DataLoader(res_set, batch_size=batch_size, shuffle=False, num_workers=0)
    g = grad_batch(res_loader, weight_decay, model, device)
    
    # Perform Newton update
    print("Computing Newton update...")
    delta = newton_update(g, unlearn_bs, res_set, weight_decay, gamma, model, s1, s2, scale, device, generator=generator)
    
    # Update model parameters with noise
    print("Updating model parameters...")
    for i, param in enumerate(model.parameters()):
        if generator is not None:
            # Generate noise on CPU with generator, then move to device
            # (Generators work on CPU, so we generate there and transfer)
            noise = torch.randn(param.data.size(), generator=generator) * std
            noise = noise.to(device)
        else:
            noise = torch.randn(param.data.size(), device=device) * std
        param.data.add_(-delta[i] + noise)
    
    print("Unlearning complete!")


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
    parser.add_argument('--unlearn-bs', type=int, default=10, metavar='N',
                        help='input batch size for unlearning (default: 10)')
    parser.add_argument('--num-unlearn', default=1000, type=int, help='number of unlearned samples')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--C', type=float, default=20,
                        help='Norm constraint of parameters')
    parser.add_argument('--s1', type=int, default=10, help='Number of samples in Hessian approximation')
    parser.add_argument('--s2', type=int, default=1000, help='The order number of Taylor expansion in Hessian approximation')
    parser.add_argument('--std', type=float, default=.001, help='The standard deviation of Gaussian noise')
    parser.add_argument('--gamma', type=float, default=.01, help='The convex approximation coefficient')
    parser.add_argument('--scale', type=float, default=1000., help='The scale of Hessian')
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
    
    PATH_load = './model/'+f"clip_{str(args.C).replace('.','_')}_"+args.model+'_'+args.dataset+f"_lr_{str(args.lr).replace('.','_')}"+f"_bs_{str(args.batch_size)}"+f"_wd_{str(args.weight_decay).replace('.','_')}"+f"_epoch_{str(args.epochs)}"+f"_seed_{str(args.seed)}"+'.pth'
    PATH_save = './model/unlearn_'+f"clip_{str(args.C).replace('.','_')}_"+args.model+'_'+args.dataset+f"_lr_{str(args.lr).replace('.','_')}"+f"_bs_{str(args.batch_size)}"+f"_wd_{str(args.weight_decay).replace('.','_')}"+f"_epoch_{str(args.epochs)}"+f"_seed_{str(args.seed)}"+f"_num_{str(args.num_unlearn)}"+f"_unlearnbs_{str(args.unlearn_bs)}"+f"_s1_{str(args.s1)}"+f"_s2_{str(args.s2)}"+f"_std_{str(args.std).replace('.','_')}"+f"_gamma_{str(args.gamma).replace('.','_')}"+f"_scale_{str(args.scale).replace('.','_')}"+'.pth'

    unl_set, res_set = load_unlearn_data(args.dataset, args.num_unlearn)
    train_set, _ = load_dataset(args.dataset)
    num_classes = 10
    args.num_classes = num_classes

    if args.model == 'mlp':
        model = MLP(num_layer=args.mlp_layer, num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'cnn':
        model = AllCNN(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_classes, filters_percentage=args.filters).to(args.device)
    
    model.load_state_dict(torch.load(PATH_load))
    
    unlearn(model, unl_set, res_set, args.batch_size, args.unlearn_bs, args.weight_decay, 
            args.gamma, args.s1, args.s2, args.scale, args.std, args.device)
    
    # Save the unlearned model
    torch.save(model.state_dict(), PATH_save)
    print(f"Unlearned model saved to: {PATH_save}")
    
    # Exact Computation
    # model.eval()
    # loss = 0
    # criterion = nn.CrossEntropyLoss()
    # params = [p for p in model.parameters() if p.requires_grad]
    # data_loader = DataLoader(res_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # for data, labels in data_loader:
    #     data, labels = data.to(args.device), labels.to(args.device)
    #     outputs = model(data)
    #     loss += criterion(outputs, labels)
    # l2_reg = 0
    # for param in model.parameters():
    #     l2_reg += torch.norm(param, p=2)
    # # Add L2 regularization to the loss
    # loss += (args.weight_decay + args.gamma) * l2_reg
    # delta = inverse_hvp(loss, params, batch_grads_to_vec(g))
    # delta = vec_to_params(delta, params)
    # for i, param in enumerate(model.parameters()):
    #     param.data.add_(-delta[i])
    #     # param.data.add_(delta[i] + args.std * torch.randn(param.data.size()).to(args.device))

    # origin_model.load_state_dict(torch.load(PATH_load))
    # retrain_model.load_state_dict(torch.load(PATH_retrain))

    # norm_dist = torch.linalg.vector_norm(nn.utils.parameters_to_vector(origin_model.parameters()) - nn.utils.parameters_to_vector(model.parameters()))
    # norm_dist_rt = torch.linalg.vector_norm(nn.utils.parameters_to_vector(retrain_model.parameters()) - nn.utils.parameters_to_vector(model.parameters()))
    # norm_dist_before = torch.linalg.vector_norm(nn.utils.parameters_to_vector(retrain_model.parameters()) - nn.utils.parameters_to_vector(origin_model.parameters()))
    # print(f"original distance: {norm_dist}, retrain distance: {norm_dist_rt}, origin-retrain distance: {norm_dist_before}")
    # print(f'Time: {time.time() - start}')

    # PATH_exact = './model/exact_unlearn_clip_20_'+args.model+'_'+args.dataset+f"_lr_{str(args.lr).replace('.','_')}"+f"_bs_{str(args.batch_size)}"+f"_wd_{str(args.weight_decay).replace('.','_')}"+f"_epoch_{str(args.epochs)}"+f"_seed_{str(args.seed)}"+f"_num_{str(args.num_unlearn)}"+f"_unlearnbs_{str(args.unlearn_bs)}"+f"_s1_{str(args.s1)}"+f"_s2_{str(args.s2)}"+f"_std_{str(args.std).replace('.','_')}"+f"_gamma_{str(args.gamma).replace('.','_')}"+f"_scale_{str(args.scale).replace('.','_')}"+'.pth'
    # torch.save(model.state_dict(), PATH_exact)
    
