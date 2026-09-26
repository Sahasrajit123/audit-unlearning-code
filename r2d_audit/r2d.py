import torch
import math
import numpy as np
from math import exp, sqrt, log
from scipy.special import erf


def h_function(K, eta, L, m, n, T):
    """
    Compute h(K) from Theorem 3.1 using log-space computation for numerical stability.
    
    Formula: h(K) = ((1 + eta * L * n/(n-m))**(T-K) - 1) * (1 + eta * L)**K
    
    For large K or T-K, we compute in log space to avoid overflow.
    """
    # Compute bases
    base1 = 1 + eta * L * n / (n - m)
    base2 = 1 + eta * L
    
    # Compute exponents in log space
    exp1 = (T - K) * log(base1)
    exp2 = K * log(base2)
    
    # Check for potential overflow
    # If exp1 or exp2 are too large, the result will be inf or very large
    # We'll compute carefully
    if exp1 > 700:  # exp(700) is near the limit of float64
        # If exp1 is huge, then base1**(T-K) dominates, so we approximate
        # h(K) ≈ base1**(T-K) * base2**K
        log_h = exp1 + exp2
        # Return a large but finite value, or handle as needed
        if log_h > 700:
            # Return a very large number (but not inf)
            return float('inf')
        return exp(log_h)
    
    # Compute term1 = base1**(T-K) - 1
    term1_power = exp(exp1)
    term1 = term1_power - 1
    
    # Compute term2 = base2**K
    if exp2 > 700:
        return float('inf')
    term2 = exp(exp2)
    
    # Return h(K) = term1 * term2
    result = term1 * term2
    
    # Check for inf or nan
    if not np.isfinite(result):
        # If result is inf or nan, try to return a large finite value
        # or handle the overflow case
        return float('inf')
    
    return result


def add_gaussian_noise_to_weights(model, sigma, device, seed=None):

    if seed is not None:
        torch.manual_seed(seed)
    # Iterate over all model parameters
    for param in model.parameters():
        # Check if the parameter has a gradient (usually it will if it's a learnable weight)
        if param.requires_grad:
            # Create Gaussian noise with mean 0 and standard deviation sigma
            noise = torch.normal(mean=0.0, std=sigma, size=param.size())
            # Add the noise to the parameter
            param.data += noise.to(device)


def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return (1. - pred.eq(target.view_as(pred)).float().mean().item(), len(pred))


def compute_test_error(model, loader, device=None):
    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        
    model.eval()
    with torch.no_grad():
        
        error_list = []

        for batch_idx, (data, target, identity) in enumerate(loader):
                    #data, target = data.to(args.device), target.to(args.device)   
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            error, length = get_error(output, target)

            error_list.append(error * length)

        return(sum(error_list)/len(loader.dataset))


    

from scipy.special import erf, log_ndtr

def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol=1e-12):
    """
    Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism
    [Balle and Wang, ICML'18] with numerical stability improvements for large epsilon.

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)

    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5 * (1.0 + erf(t / sqrt(2.0)))

    def caseA(epsilon, s):
        a = sqrt(epsilon * s)
        b = sqrt(epsilon * (s + 2.0))
        log_term = epsilon + log_ndtr(-b)
        return Phi(a) - np.exp(log_term)

    def caseB(epsilon, s):
        a = sqrt(epsilon * s)
        b = sqrt(epsilon * (s + 2.0))
        log_term = epsilon + log_ndtr(-b)
        return Phi(-a) - np.exp(log_term)

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while not predicate_stop(s_sup):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while not predicate_stop(s_mid):
            if predicate_left(s_mid):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    # Compute delta threshold
    delta_thr = caseA(epsilon, 0.0)

    if delta == delta_thr:
        alpha = 1.0
    else:
        if delta > delta_thr:
            predicate_stop_DT = lambda s: caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s: caseA(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)
        else:
            predicate_stop_DT = lambda s: caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s: caseB(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

        predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha * GS / sqrt(2.0 * epsilon)
    return sigma

from sklearn.metrics import roc_auc_score

def compute_auroc(model, dataloader, device=None):
    """
    Compute AUROC for a model over a dataset.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device or None): Device to use (cpu or cuda). If None, use model's device.

    Returns:
        float: AUROC score.
    """
    model.eval()
    all_labels = []
    all_probs = []

    # Infer the device if not given
    if device is None:
        device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # If outputs are logits, apply sigmoid for binary classification
            if outputs.shape[-1] == 1 or len(outputs.shape) == 1:
                probs = torch.sigmoid(outputs).squeeze()
            else:
                # For multi-class, take softmax probabilities (choose the positive class)
                probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return roc_auc_score(all_labels, all_probs)