"""
Safe kernel initializers that avoid GPU cusolver QR calls.

The orthogonal initializer uses QR decomposition (jax.numpy.linalg.qr) which can
trigger cuSolver problems (incompatible driver/lib versions). As a defensive
measure, we expose a `orthogonal` function that uses the Flax orthogonal
initializer on CPU, but falls back to a safe variance scaling initializer on
GPU. This provides resilience when running on systems with faulty cuSolver.
"""
import jax
from jax.nn.initializers import variance_scaling, lecun_normal
from flax.linen.initializers import orthogonal as flax_orthogonal


def orthogonal(scale):
    """Return an initializer: use QR orthogonalization on CPU, variance scaling on GPU.

    Args:
        scale: gain for orthogonal initializer; passed as-is to orthogonal or used
               as the scale parameter for variance scaling.
    Returns:
        A callable initializer function compatible with Flax `kernel_init`.
    """
    platform = jax.devices()[0].platform if jax.devices() else 'cpu'
    if platform == 'gpu':
        # Use a variance scaling initializer on GPU to avoid QR decomposition
        # which may rely on cuSolver. This approximate replacement is robust
        # across platforms and avoids runtime cuSolver issues.
        return variance_scaling(scale, 'fan_in', 'truncated_normal')
    else:
        # Use the Flax orthogonal initializer on CPU for the usual benefits
        return flax_orthogonal(scale)
