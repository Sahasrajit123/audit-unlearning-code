#!/usr/bin/env python3
import argparse
import json
import copy
import random
from collections import defaultdict

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import os

import models

def manual_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)
        self.max = defaultdict(float)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]

            if val[k] > self.max[k]:
                self.max[k] = val[k]



def log_metrics(split, metrics, epoch, **kwargs):
    print(f'[{epoch}] {split} metrics:' + json.dumps(metrics.avg))
    #print(f'[{epoch}] {split} max metrics:' + json.dumps(metrics.max))

def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return 1. - pred.eq(target.view_as(pred)).float().mean().item()
    else:
        pred = output.clone()
        pred[pred>0]=1
        pred[pred<=0]=-1
        return 1 - pred.eq(target.view_as(pred)).float().mean().item()

def set_batchnorm_mode(model, train=True):
    if isinstance(model, torch.nn.BatchNorm1d) or isinstance(model, torch.nn.BatchNorm2d):
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_batchnorm_mode(l, train=train)
        

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def balance_classes(dataset, desired_balance): #I think this only works if desired_balance < 0.5 
    labels = dataset.targets[dataset.indices]

    new_indices = []

    large_number_indices = []

    n = len(labels)
    num_ones = sum(labels)
    current_balance = num_ones/n

    print(current_balance)

    if current_balance < desired_balance:
        m = current_balance*n/desired_balance
        for i in range(len(dataset.indices)):
            if labels[i] == 1:
                new_indices.append(dataset.indices[i])
            elif labels[i] == 0:
                large_number_indices.append(dataset.indices[i])
        subset_zeros = random.sample(large_number_indices, int((1 - desired_balance)*m))
        final_indices = new_indices + list(subset_zeros)
    else:
        m = (1 - current_balance)*n/(1 - desired_balance)
        for i in range(len(dataset.indices)):
            if labels[i] == 1:
                large_number_indices.append(dataset.indices[i])
            elif labels[i] == 0:
                new_indices.append(dataset.indices[i])
        subset_ones = random.sample(large_number_indices, int(desired_balance*m))
        final_indices = new_indices + list(subset_ones)


    
    dataset.indices = np.array(final_indices)       

    print(sum(dataset.targets[dataset.indices])/len(dataset)) 

