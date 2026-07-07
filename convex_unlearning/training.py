"""
Training module: logistic regression or Ridge (MSE) regression.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, mean_squared_error, r2_score
from lipschitz_constants import compute_lipschitz_constants_from_data


def train(X_train, y_train, X_val, y_val, X_test, y_test,
          per_sample_reg=1.0, loss='logistic', max_iter=1000, random_state=42, verbose=True, loss_params=None):
    """
    Train model with L2 per-sample regularization.
    
    loss='logistic': binary classification, per-sample f = (λ_i/2)||w||^2 + log loss.
    loss='mse': regression, per-sample f = (1/2)(y_i - w^T x_i)^2 + (λ_i/2)||w||^2.
    loss='cubic': f(w,z) = (λ/2)||w||² + (M/6)(a,w)³ - (z,w); loss_params = config 'cubic' section (lambda, M, B, a).
    
    Parameters:
    -----------
    X_train, y_train, X_val, y_val, X_test, y_test : data
    per_sample_reg : scalar or array (n_train,); L2 strength λ_i
    loss : 'logistic', 'mse', or 'cubic'
    max_iter : int
    random_state : int
    verbose : bool
    loss_params : dict, optional; for loss='cubic' must contain lambda, M, B, and optionally a (default 'e1')
        
    Returns:
    --------
    dict with model, weights, hessian, metrics, C, Lambda, per_sample_reg, L, M, loss.
    """
    n_train = len(X_train)
    per_sample_reg = np.atleast_1d(np.asarray(per_sample_reg, dtype=float))
    if per_sample_reg.ndim == 0 or per_sample_reg.size == 1:
        lam = float(per_sample_reg.flat[0])
        Lambda = n_train * lam
        per_sample_reg = np.full(n_train, lam)
    else:
        if per_sample_reg.shape[0] != n_train:
            raise ValueError("per_sample_reg length must match X_train size")
        Lambda = float(np.sum(per_sample_reg))
    C = 1.0 / Lambda
    X_train_with_intercept = np.column_stack([X_train, np.ones(len(X_train))])

    if loss == 'logistic':
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state,
                                   solver='lbfgs', penalty='l2', fit_intercept=True)
        model.fit(X_train, y_train)
        w_coef = model.coef_.flatten()
        w_intercept = model.intercept_[0]
        w_full = np.concatenate([w_coef, [w_intercept]])
        logits = X_train_with_intercept @ w_full
        probs = 1 / (1 + np.exp(-logits))
        weights_diag = probs * (1 - probs)
        weighted_X = X_train_with_intercept * weights_diag[:, np.newaxis]
        hessian = X_train_with_intercept.T @ weighted_X
        reg_matrix = np.eye(len(w_full))
        reg_matrix[-1, -1] = 0
        hessian += Lambda * reg_matrix
        lipschitz_constants = compute_lipschitz_constants_from_data(
            X_train, per_sample_reg=per_sample_reg, verbose=verbose
        )
        L, M = lipschitz_constants['L'], lipschitz_constants['M']
        metrics = {}
        for name, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            metrics[f'{name}_accuracy'] = accuracy_score(y, y_pred)
            metrics[f'{name}_loss'] = log_loss(y, y_proba)
            metrics[f'{name}_auc'] = roc_auc_score(y, y_proba)
        if verbose:
            print("\n" + "="*60)
            print("TRAINING RESULTS (logistic)")
            print("="*60)
            print(f"Train - Acc: {metrics['train_accuracy']:.4f}, Loss: {metrics['train_loss']:.4f}, AUC: {metrics['train_auc']:.4f}")
            print(f"Val   - Acc: {metrics['val_accuracy']:.4f}, Loss: {metrics['val_loss']:.4f}, AUC: {metrics['val_auc']:.4f}")
            print(f"Test  - Acc: {metrics['test_accuracy']:.4f}, Loss: {metrics['test_loss']:.4f}, AUC: {metrics['test_auc']:.4f}")
            print("="*60)

    elif loss == 'mse':
        model = Ridge(alpha=Lambda, fit_intercept=True, random_state=random_state)
        model.fit(X_train, y_train)
        w_coef = model.coef_
        w_intercept = model.intercept_
        w_full = np.concatenate([w_coef, [w_intercept]])
        # Hessian for Ridge: ∇²f = X'^T X' + Λ I (constant, no reg on intercept)
        hessian = X_train_with_intercept.T @ X_train_with_intercept
        reg_matrix = np.eye(len(w_full))
        reg_matrix[-1, -1] = 0
        hessian += Lambda * reg_matrix
        lipschitz_constants = compute_lipschitz_constants_from_data(
            X_train, per_sample_reg=per_sample_reg, loss='mse', verbose=verbose
        )
        L, M = lipschitz_constants['L'], lipschitz_constants['M']
        metrics = {}
        for name, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
            X_aug = np.column_stack([X, np.ones(len(X))])
            y_pred = (X_aug @ w_full)
            metrics[f'{name}_mse'] = mean_squared_error(y, y_pred)
            metrics[f'{name}_r2'] = r2_score(y, y_pred)
        if verbose:
            print("\n" + "="*60)
            print("TRAINING RESULTS (MSE)")
            print("="*60)
            print(f"Train - MSE: {metrics['train_mse']:.4f}, R2: {metrics['train_r2']:.4f}")
            print(f"Val   - MSE: {metrics['val_mse']:.4f}, R2: {metrics['val_r2']:.4f}")
            print(f"Test  - MSE: {metrics['test_mse']:.4f}, R2: {metrics['test_r2']:.4f}")
            print("="*60)

    elif loss == 'cubic':
        from cubic_loss import train_cubic
        cubic_params = (loss_params or {}).copy()
        # λ is taken from per_sample_reg only; cubic.lambda is not used
        return train_cubic(
            X_train, X_val, X_test,
            per_sample_reg=per_sample_reg,
            cubic_params=cubic_params,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose
        )

    else:
        raise ValueError(f"loss must be 'logistic', 'mse', or 'cubic', got {loss!r}")

    return {
        'model': model,
        'weights': w_full,
        'hessian': hessian,
        'metrics': metrics,
        'C': C,
        'Lambda': Lambda,
        'per_sample_reg': per_sample_reg,
        'L': L,
        'M': M,
        'loss': loss,
        'mu': Lambda / n_train,  # effective strong convexity: λ for logistic/MSE
    }
