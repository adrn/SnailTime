from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=['K'])
def ln_gmm_density(x, amps, locs, scales, K):
    """
    Evaluate the log-density for a Gaussian Mixture Model (GMM).

    The resulting distribution is not necessarily a probability distribution, as the
    amplitudes can be anything.

    Parameters
    ----------
    x : `jax.numpy.DeviceArray`
        The values/locations to evaluate the GMM density at.
    amps : `jax.numpy.DeviceArray`
        The amplitudes of the mixture components.
    locs : `jax.numpy.DeviceArray`
        The means of the mixture components.
    scales : `jax.numpy.DeviceArray`
        The standard deviations of the mixture components.
    K : int
        The number of mixture components.

    Returns
    -------
    ln_density : `jax.numpy.DeviceArray`
        The log-density, with the same shape as ``x``.
    """

    ln_vals = []
    for k in range(K):
        ln_dens = jax.scipy.stats.norm.logpdf(x, loc=locs[k], scale=scales[k])
        ln_vals.append(jnp.log(amps[k]) + ln_dens)
    return jax.scipy.special.logsumexp(jnp.array(ln_vals), axis=0)


@jax.jit
def simpson(y, x):
    """
    Evaluate the definite integral of a function using Simpson's rule

    Note: x must be a regularly-spaced grid of points!
    """

    dx = jnp.diff(x)[0]
    num_points = len(x)
    if num_points % 2 == 0:
        raise ValueError("Because of laziness, the input size must be odd")

    weights_first = jnp.asarray([1.0])
    weights_mid = jnp.tile(jnp.asarray([4.0, 2.0]), [(num_points - 3) // 2])
    weights_last = jnp.asarray([4.0, 1.0])
    weights = jnp.concatenate([weights_first, weights_mid, weights_last], axis=0)

    return dx / 3 * jnp.sum(y * weights, axis=-1)
