from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
import random
import torch
import torch.nn as nn
import numpy as np
import random

from sklearn.metrics import accuracy_score, roc_auc_score

import copy

def membership_inference_attack(model, t_loader, f_loader, device, n_splits = 5, seed=1, metric=None, normalize=None, n_repeats = 100, balance=True, scoring='roc_auc', n_samples=1):
  #adapted from SCRUB code to include both logits and loss in logistic regression model
  #Honestly, the variable names here are a bit problematic. I think there are some names that don't really reflect what is happening
    '''
    t_loader: data that has never been seen by the model (test/ood set)
    f_loader: data learned and then unlearned (forget set)
    model: unlearned model
    device: device
    n_samples: number of undersamples (for making a balanced dataset)
    setting metric = 'loss' only considers the loss of the model
    
    '''
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    cr = nn.CrossEntropyLoss(reduction='none') #reduction=none means we compute loss on each sample 
    test_losses = []
    forget_losses = []
    test_outputs = []
    forget_outputs = []
    
    all_scores = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(t_loader):
            data, target = data.to(device), target.to(device)            
            output = model(data)
            loss = cr(output, target)
            output = torch.cat([output, loss.unsqueeze(1)], dim=1) #logits and loss
            if metric == 'loss':
                test_outputs.append(loss)
            else:
                test_outputs.append(output)

        test_losses = torch.cat(test_outputs, dim=0)

        for batch_idx, (data, target, _) in enumerate(f_loader):
            data, target = data.to(device), target.to(device)            
            output = model(data)
            loss = cr(output, target)

            output = torch.cat([output, loss.unsqueeze(1)], dim=1)
            #forget_losses = forget_losses + list(loss.cpu().detach().numpy())
            if metric == 'loss':
                forget_outputs.append(loss)
            else:
                forget_outputs.append(output)

        forget_losses = torch.cat(forget_outputs, dim=0)
        
        og_forget_losses = copy.deepcopy(forget_losses)
        og_test_losses = copy.deepcopy(test_losses)

        for k in range(n_samples):
            if balance:
                if len(og_forget_losses) > len(og_test_losses):
                  #forget_losses = list(random.sample(forget_losses, len(test_losses)))
                    indices = torch.randperm(og_forget_losses.size(0))[:len(og_test_losses)]
                  # Select the corresponding data points
                    forget_losses = og_forget_losses[indices]

                elif len(og_test_losses) > len(og_forget_losses):
                      #test_losses = list(random.sample(test_losses, len(forget_losses)))
                    indices = torch.randperm(og_test_losses.size(0))[:len(og_forget_losses)]
                      # Select the corresponding data points
                    test_losses = og_test_losses[indices]


            features = torch.cat([test_losses, forget_losses], dim=0).cpu().numpy()

            #normalize
            if normalize=='zscore':
                mean_vals = features.mean(axis=0, keepdims=True)
                std_vals = features.std(axis=0, keepdims=True)

                features = (features - mean_vals) / std_vals
            elif normalize =='minmax':
            # Min-Max normalization
                min_vals = features.min(axis=0, keepdims=True)
                max_vals = features.max(axis=0, keepdims=True)
                features = (features - min_vals) / (max_vals - min_vals)

            elif normalize is None:
                features = features

            if metric == 'loss':
                features = features.reshape(-1, 1)
            #features = np.array(test_losses + forget_losses).reshape(-1,1)

            test_labels = [0]*len(test_losses)
            forget_labels = [1]*len(forget_losses)

            labels = np.array(test_labels + forget_labels).reshape(-1)
            #features = np.clip(features, -100, 100)

            permutation = np.random.permutation(len(labels))

            labels=labels[permutation]
            features = features[permutation]

            
            attack_model = LogisticRegression(class_weight='balanced')
            #attack_model = LogisticRegression()
            cv = RepeatedStratifiedKFold(n_splits=n_splits, random_state=seed + k, n_repeats=n_repeats) #automatically shuffles
            score = cross_val_score(attack_model, features, labels, cv=cv, scoring=scoring)

            all_scores += list(score)
        return all_scores, features, labels


