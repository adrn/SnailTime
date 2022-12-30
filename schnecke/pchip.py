"""
.. author:: Michael Taylor <mtaylor@atlanticsciences.com>
.. author:: Mathieu Virbel <mat@meltingrocks.com>

Copyright (c) 2016 Michael Taylor and Mathieu Virbel
Copyright (c) 2022 Adrian Price-Whelan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import jax.numpy as jnp

__all__ = ["pchip_interpolate_uniform"]


# @partial(jax.jit, static_argnames=['xi', 'mode'])
def pchip_interpolate_uniform(xi, yi, x):
    """
    Interpolation using piecewise cubic Hermite polynomial, enforcing that the resulting
    interpolated values are monotonic if the input is.

    This function requires a uniform grid in ``xi``. For support for nonuniform grids,
    use `pchip_interpolate_nonuniform()`.
    """

    xi = jnp.atleast_1d(xi)
    yi = jnp.atleast_1d(yi)
    x = jnp.atleast_1d(x)

    # ensure that input arrays have increasing x
    idx = jnp.argsort(xi)
    xi = xi[idx]
    yi = yi[idx]

    if len(xi) != len(yi):
        raise ValueError("Input arrays must be the same length.")
    if xi.ndim != 1 or yi.ndim != 1:
        raise ValueError("Input arrays must be 1D.")

    # uniform input grid
    xi_step = (xi[-1] - xi[0]) / (len(xi) - 1)
    x_index = jnp.clip(jnp.floor((x - xi[0]) / xi_step).astype(int), 0, len(xi) - 2)

    # Calculate gradients d
    h = (xi[-1] - xi[0]) / (len(xi) - 1)

    # monotonic: Fritsch-Carlson algorithm from fortran numerical recipe
    delta = jnp.diff(yi) / h

    # Needed for rev-mode autodiff. See:
    # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    d1 = jnp.where(delta[:-1] != 0.0, delta[0:-1], 1.)
    d2 = jnp.where(delta[1:] != 0.0, delta[1:], 1.)
    d = jnp.where(
        (delta[:-1] != 0.0) & (delta[1:] != 0.0),
        2 / (1 / d1 + 1 / d2),
        0.0,
    )
    d = jnp.concatenate((delta[0:1], d, delta[-1:]))

    zero_mask1 = jnp.concatenate(
        (
            jnp.array([False]),
            jnp.logical_xor(delta[0:-1] > 0, delta[1:] > 0),
            jnp.array([False]),
        )
    )
    zero_mask2 = jnp.logical_or(
        jnp.concatenate((jnp.array([False]), delta == 0)),
        jnp.concatenate((delta == 0, jnp.array([False]))),
    )
    d = jnp.where(zero_mask1 | zero_mask2, 0.0, d)

    # Calculate output values y
    dxxi = x - xi[x_index]
    dxxid = x - xi[x_index + 1]
    y = 2 / h**3 * (
        yi[x_index] * dxxid**2 * (dxxi + h / 2)
        - yi[1 + x_index] * dxxi**2 * (dxxid - h / 2)
    ) + 1 / h**2 * (
        d[x_index] * dxxid**2 * dxxi + d[1 + x_index] * dxxi**2 * dxxid
    )

    return y


# @partial(jax.jit, static_argnames=['xi', 'mode'])
def pchip_interpolate_nonuniform(xi, yi, x):
    """
    Interpolation using piecewise cubic Hermite polynomial, enforcing that the resulting
    interpolated values are monotonic if the input is.

    This function does not require a uniform grid in ``xi``. For faster support for
    uniform grids, use `pchip_interpolate_uniform()`.
    """

    raise NotImplementedError("Not yet jaxified!")

    xi = jnp.atleast_1d(xi)
    yi = jnp.atleast_1d(yi)
    x = jnp.atleast_1d(x)

    # ensure that input arrays have increasing x
    idx = jnp.argsort(xi)
    xi = xi[idx]
    yi = yi[idx]

    if len(xi) != len(yi):
        raise ValueError("Input arrays must be the same length.")
    if xi.ndim != 1 or yi.ndim != 1:
        raise ValueError("Input arrays must be 1D.")

    # non-uniform input/output grids, output grid not monotonic
    x_index = jnp.zeros(len(x), dtype=int)
    for j in range(len(x)):
        loc = jnp.where(x[j] < xi)[0]
        if loc.size == 0:
            x_index[j] = len(xi) - 2
        elif loc[0] == 0:
            x_index[j] = 0
        else:
            x_index[j] = loc[0] - 1

    # Calculate gradients d
    h = np.diff(xi)
    d = np.zeros(len(xi), dtype="double")
    delta = np.diff(yi) / h
    if mode == "quad":
        # quadratic polynomial fit
        d[[0, -1]] = delta[[0, -1]]
        d[1:-1] = (delta[1:] * h[0:-1] + delta[0:-1] * h[1:]) / (h[0:-1] + h[1:])
    else:
        # mode=='mono', Fritsch-Carlson algorithm from fortran numerical
        # recipe
        d = np.concatenate(
            (
                delta[0:1],
                3
                * (h[0:-1] + h[1:])
                / (
                    (h[0:-1] + 2 * h[1:]) / delta[0:-1]
                    + (2 * h[0:-1] + h[1:]) / delta[1:]
                ),
                delta[-1:],
            )
        )
        d[
            np.concatenate(
                (
                    np.array([False]),
                    np.logical_xor(delta[0:-1] > 0, delta[1:] > 0),
                    np.array([False]),
                )
            )
        ] = 0
        d[
            np.logical_or(
                np.concatenate((np.array([False]), delta == 0)),
                np.concatenate((delta == 0, np.array([False]))),
            )
        ] = 0
    dxxi = x - xi[x_index]
    dxxid = x - xi[1 + x_index]
    dxxi2 = pow(dxxi, 2)
    dxxid2 = pow(dxxid, 2)
    y = 2 / pow(h[x_index], 3) * (
        yi[x_index] * dxxid2 * (dxxi + h[x_index] / 2)
        - yi[1 + x_index] * dxxi2 * (dxxid - h[x_index] / 2)
    ) + 1 / pow(h[x_index], 2) * (
        d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid
    )

    return y
