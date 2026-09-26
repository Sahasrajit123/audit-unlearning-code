import flax.linen as nn
import jax.numpy as jnp


class LogisticRegression(nn.Module):
    """Single linear layer for logistic regression."""
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True, mutable=None):
        # Flatten input to 2D: (batch_size, features)
        x = x.reshape((x.shape[0], -1))
        # Single linear layer - Flax will automatically infer input features
        x = nn.Dense(features=self.num_classes)(x)
        return x
