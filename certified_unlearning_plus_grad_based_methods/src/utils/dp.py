import numpy as np
import scipy.special as scsp
from jax import numpy as jnp


def theta_epsilon(epsilon: np.float64, r: np.float64) -> np.float64:
    """
    Calculating theta_epsilon function from the notes.
    Handles large epsilon values with numerical stability improvements using log-space computations.
    """

    def Q(x):
        return (1 - scsp.erf(x / np.sqrt(2))) / 2

    try:
        arg1 = epsilon / r - r / 2
        arg2 = epsilon / r + r / 2
        
        # Compute Q(arg1) and Q(arg2) directly
        # For very negative arg1, Q(arg1) ≈ 1.0
        # For very positive arg2, Q(arg2) ≈ 0.0
        q1 = Q(arg1)
        q2 = Q(arg2)
        
        # Result = Q(arg1) - exp(epsilon) * Q(arg2)
        # For large epsilon, we need to handle exp(epsilon) * Q(arg2) carefully
        
        if epsilon > 700:
            # exp(epsilon) would overflow, work in log space
            # If Q(arg2) is essentially 0, then exp(epsilon) * Q(arg2) ≈ 0
            # Check if Q(arg2) is negligible
            if q2 < 1e-100:
                # Q(arg2) is essentially 0, so exp(epsilon) * Q(arg2) ≈ 0
                # Result ≈ Q(arg1)
                return max(0.0, min(1.0, q1))
            else:
                # Q(arg2) is not negligible, compute in log space
                # log(exp(epsilon) * Q(arg2)) = epsilon + log(Q(arg2))
                log_q2 = np.log(q2)
                log_exp_q2 = epsilon + log_q2
                
                # Compare Q(arg1) and exp(epsilon) * Q(arg2) in log space
                log_q1 = np.log(max(q1, 1e-100))
                diff_log = log_q1 - log_exp_q2
                
                max_exp = np.log(np.finfo(np.float64).max) - 1
                min_exp = np.log(np.finfo(np.float64).tiny) + 1
                
                if diff_log > 50:
                    # Q(arg1) >> exp(epsilon) * Q(arg2), result ≈ Q(arg1)
                    return max(0.0, min(1.0, q1))
                elif diff_log < -50:
                    # exp(epsilon) * Q(arg2) >> Q(arg1), result is negative, clamp to 0
                    return 0.0
                else:
                    # Both terms comparable, compute difference
                    # Use: Q(arg1) - exp(epsilon) * Q(arg2)
                    # = exp(log_q1) - exp(log_exp_q2)
                    if log_exp_q2 < max_exp:
                        exp_q2_term = np.exp(log_exp_q2)
                        result = q1 - exp_q2_term
                    else:
                        # exp(epsilon) * Q(arg2) would overflow, it dominates
                        return 0.0
                    
                    return max(0.0, min(1.0, result))
        else:
            # Standard calculation for smaller epsilon
            exp_epsilon = np.exp(epsilon)
            result = q1 - exp_epsilon * q2
            return max(0.0, min(1.0, result))
            
    except (OverflowError, ValueError, ZeroDivisionError):
        # Fallback: return 0 for numerical issues
        raise ValueError(f"Numerical issue with epsilon={epsilon}, r={r}")


def clamp_matrix(matrix, min_val, max_val):
    return jnp.clip(matrix, min_val, max_val)
