import jax.numpy as jnp
from jax import jit
import jax


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


@jit
def update(params, step_size, grads):
    return jax.tree_map(lambda param, g: param - g * step_size, params, grads)
