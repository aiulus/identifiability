import jax.numpy as jnp
from jax import Array
from typing import List
import numpy as np


def H_wd(w: np.ndarray, T, num_rows: int) -> np.ndarray:
    T, q = w.shape
    cols = T - num_rows + 1
    blocks = [ w[i:i+cols] for i in range(num_rows) ]
    return np.vstack(blocks)


def build_LTI_hankel(A: Array, B: Array, n: int) -> Array:
    """
    Constructs the block-Hankel matrix for a controlled LTI system.

    Assuming full observability (p = n), the matrix is:
    H_n,n = [[B, AB, ..., A^(n-1)B],
             [AB, A^2B, ..., A^n B],
             ...,
             [A^(n-1)B, ..., A^(2n-2)B]]

     Args:
        A: The system matrix (n, n)
        B: The input matrix (n, m)
        n: The dimension of the state space (used for block sizes)

    Returns:
        The block-Hankel matrix H_n,n of shape (n*n, n*m)
    """
    powers: List[Array] = [B]
    for i in range(1, 2*n - 1):
        powers.append(A @ powers[-1])

    rows: List[Array] = []
    for i in range(n):
        row = jnp.hstack(powers[i: i + n])
        rows.append(row)

    hankel_matrix = jnp.vstack(rows)

    return hankel_matrix