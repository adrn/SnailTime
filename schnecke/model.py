from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import jaxopt

# from .pchip import pchip_interpolate_uniform


@jax.jit
def distort_rz(rz, th, e2, e4):
    return 1 + e2 * rz * jnp.cos(2 * th) + e4 * rz * jnp.cos(4 * th)


class VerticalOrbitModel:

    def __init__(self, dens_knots, e2_knots, e4_knots):
        self.dens_knots = jnp.array(dens_knots)
        self.e2_knots = jnp.array(e2_knots)
        self.e4_knots = jnp.array(e4_knots)

    @partial(jax.jit, static_argnames=['self'])
    def get_e2_e4(self, init_rz, e2_vals, e4_vals):
        # e2s = pchip_interpolate_uniform(
        #     self.e2_knots,
        #     e2_vals,
        #     init_rz
        # )
        # e4s = pchip_interpolate_uniform(
        #     self.e4_knots,
        #     e4_vals,
        #     init_rz
        # )
        e2s = InterpolatedUnivariateSpline(
            self.e2_knots,
            jnp.cumsum(e2_vals),
            k=1
        )(init_rz)
        e4s = InterpolatedUnivariateSpline(
            self.e4_knots,
            jnp.cumsum(e4_vals),
            k=1
        )(init_rz)
        return e2s, e4s

    @partial(jax.jit, static_argnames=['self'])
    def get_distorted_rz(self, init_rz, init_th, e2_vals, e4_vals):
        e2s, e4s = self.get_e2_e4(init_rz, e2_vals, e4_vals)
        rz = init_rz * distort_rz(
            e2=e2s,
            e4=e4s,
            rz=init_rz,
            th=init_th
        )
        return rz

    @partial(jax.jit, static_argnames=['self'])
    def get_rz_th(self, z, vz, Omega, e2_vals, e4_vals):
        x = vz / jnp.sqrt(Omega)
        y = z * jnp.sqrt(Omega)

        init_rz = jnp.sqrt(x**2 + y**2)
        init_th = jnp.arctan2(y, x)
        rz = self.get_distorted_rz(init_rz, init_th, e2_vals, e4_vals)

        return rz, init_rz, init_th

    @partial(jax.jit, static_argnames=['self'])
    def get_ln_dens(self, rz, ln_dens_vals):
        spl = InterpolatedUnivariateSpline(
            self.dens_knots,
            ln_dens_vals,
            k=3
        )
        return spl(rz)

    @partial(jax.jit, static_argnames=['self'])
    def ln_density(self, params, z, vz):
        rz, _, th = self.get_rz_th(
            z - params['z0'], vz - params['vz0'],
            Omega=jnp.exp(params['ln_Omega']),
            e2_vals=params['e2_vals'],
            e4_vals=params['e4_vals']
        )
        return self.get_ln_dens(rz, params['ln_dens_vals'])

    @partial(jax.jit, static_argnames=['self'])
    def ln_poisson_likelihood(self, params, z, vz, H):
        # Expected number:
        ln_Lambda = self.ln_density(params, z, vz)

        # gammaln(x+1) = log(factorial(x))
        return (H * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(H + 1)).sum()

    @partial(jax.jit, static_argnames=['self'])
    def objective(self, params, z, vz, H):
        return -(self.ln_poisson_likelihood(params, z, vz, H)) / H.size

    def optimize(self, params0, z, vz, H, bounds=None, jaxopt_kwargs=None):
        if jaxopt_kwargs is None:
            jaxopt_kwargs = dict()
        jaxopt_kwargs.setdefault('maxiter', 16384)

        if bounds is not None:
            jaxopt_kwargs.setdefault('method', 'L-BFGS-B')
            optimizer = jaxopt.ScipyBoundedMinimize(
                fun=self.objective,
                **jaxopt_kwargs,
            )
            res = optimizer.run(
                init_params=params0,
                bounds=bounds,
                z=z,
                vz=vz,
                H=H,
            )

        else:
            jaxopt_kwargs.setdefault('method', 'BFGS')
            raise NotImplementedError("TODO")

        return res
