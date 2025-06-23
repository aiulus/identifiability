import jax
import jax.numpy as jnp
from typing import Tuple


def sparse_matrix(shape: Tuple[int, int], sparsity: float, key: jax.random.PRNGKey) -> jnp.ndarray:
    mask = jax.random.bernoulli(key, 1.0 - sparsity, shape)
    values = jax.random.normal(key, shape)
    return mask * values


def sparse_rand_mc(shape, sparsity, key):
    mask = jnp.where(jax.random.uniform(key, shape) < (1 - sparsity), 1, 0)
    vals = jax.random.normal(key, shape) * mask
    return vals
