from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import jaxopt

from .data import get_data_im
from .jax_helpers import ln_gmm_density


@jax.jit
def distort_rz(rz, th, e2, e4):
    return 1 + e2 * rz * jnp.cos(2 * th) + e4 * rz * jnp.cos(4 * th)


class VerticalOrbitModel:

    def __init__(self, e2_knots, e4_knots):
        e2_knots = jnp.array(e2_knots)
        e4_knots = jnp.array(e4_knots)

    @partial(jax.jit, static_argnames=['self'])
    def get_distorted_rz(self, init_rz, init_th, e2_vals, ln_e4_vals):
        # TODO: figure out also how to get out the corrected thetas in here?
        e2_interp = InterpolatedUnivariateSpline(
            self.e2_knots,
            e2_vals,
            k=3
        )
        ln_e4_interp = InterpolatedUnivariateSpline(
            self.e4_knots,
            ln_e4_vals,
            k=3
        )
        rz = init_rz * distort_rz(
            e2=e2_interp(init_rz),
            e4=jnp.exp(ln_e4_interp(init_rz)),
            rz=init_rz,
            th=init_th
        )
        return rz

    @partial(jax.jit, static_argnames=['self'])
    def get_rz_th(self, z, vz, Omega, e2_vals, ln_e4_vals):
        x = vz / jnp.sqrt(Omega)
        y = z * jnp.sqrt(Omega)

        init_rz = jnp.sqrt(x**2 + y**2)
        init_th = jnp.arctan2(y, x)
        rz = self.get_distorted_rz(init_rz, init_th, e2_vals, ln_e4_vals)

        return rz, init_rz, init_th

    @partial(jax.jit, static_argnames=['self'])
    def ln_density(self, params, z, vz):
        rz, _, th = self.get_rz_th(
            z - params['z0'], vz - params['vz0'],
            Omega=jnp.exp(params['ln_Omega']),
            e2_vals=params['e2_vals'],
            ln_e4_vals=params['ln_e4_vals']
        )

        amps = jnp.exp(params['ln_amps'])
        scales = jnp.exp(params['ln_scales'])
        locs = jnp.zeros_like(scales)
        return ln_gmm_density(rz, amps=amps, locs=locs, scales=scales, K=len(amps))

    @partial(jax.jit, static_argnames=['self'])
    def ln_poisson_likelihood(self, params, z, vz, H):
        # Expected number:
        ln_Lambda = self.ln_density(params, z, vz)

        # gammaln(x+1) = log(factorial(x))
        return (H * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(H + 1)).sum()

    @partial(jax.jit, static_argnames=['self'])
    def objective(self, params, z, vz, H):
        return -(self.ln_poisson_likelihood(params, z, vz, H)) / H.size

    @partial(jax.jit, static_argnames=['self'])
    def optimize(self, params0, z, vz, bins, bounds=None):
        data = get_data_im(z=z, vz=vz, bins=bins)

        if bounds is not None:
            optimizer = jaxopt.ScipyBoundedMinimize(
                fun=self.objective,
                method='L-BFGS-B',
                maxiter=16384,
                options=dict(disp=False)
            )
            res = optimizer.run(
                init_params=params0,
                bounds=bounds,
                z=data['z'],
                vz=data['vz'],
                H=data['H'].T,
            )

        return data, res
