
import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable

# A VectorField is a function mapping a state `x` to a vector of the same dimension.
VectorField = Callable[[Array], Array]
# A ScalarField maps a state `x` to a scalar value.
ScalarField = Callable[[Array], Array]

def lie_derivative(f: VectorField, h: ScalarField) -> ScalarField:
    """
    Computes the Lie derivative of a scalar field `h` along a vector field `f`.
    L_f h(x) = ∇h(x) · f(x)

    Args:
        f: A vector field, a function from R^n -> R^n.
        h: A scalar field (e.g., an output function), a function from R^n -> R.

    Returns:
        A new scalar field representing the Lie derivative.
    """
    grad_h = jax.grad(h)
    
    def Lf_h(x: Array) -> Array:
        return jnp.dot(grad_h(x), f(x))
    
    return Lf_h

def lie_bracket(f: VectorField, g: VectorField) -> VectorField:
    """
    Computes the Lie bracket of two vector fields, `f` and `g`.
    [f, g](x) = (∇g)f(x) - (∇f)g(x)

    Args:
        f: The first vector field, R^n -> R^n.
        g: The second vector field, R^n -> R^n.

    Returns:
        A new vector field representing the Lie bracket.
    """
    jac_f = jax.jacobian(f)
    jac_g = jax.jacobian(g)

    def bracket_fg(x: Array) -> Array:
        return jac_g(x) @ f(x) - jac_f(x) @ g(x)
        
    return bracket_fg

