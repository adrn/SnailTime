from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import jaxopt

from .jax_helpers import simpson


class VerticalOrbitModel:

    def __init__(self, dens_knots, e_knots):
        r"""
        Notation:
        - :math:`\nu_0` or ``nu_0``: A scale frequency used to compute the elliptical
          radius ``rz_prime``.
        - :math:`r_z'` or ``rz_prime``: The "raw" elliptical radius :math:`\sqrt{z^2\,
          \nu_0 + v_z^2 \, \nu_0^{-1}}`.
        - :math:`\theta'` or ``theta_prime``: The "raw" z angle defined as :math:`\tan
          {\theta'} = \frac{z}{v_z}\,\nu_0`.
        - :math:`r_z` or ``rz``: The distorted elliptical radius :math:`r_z = r_z' \,
          f(r_z', \theta_z')` where :math:`f(\cdot)` is the distortion function.
        - :math:`\theta` or ``theta``: The true vertical angle.
        - :math:`f(r_z', \theta_z')`: The distortion function is a Fourier expansion,
          defined as: :math:`f(r_z', \theta_z') = 1+\sum_m e_m(r_z')\,\cos(m\,\theta')`

        Parameters
        ----------
        dens_knots : array_like
            The knot locations for the spline that controls the density function. These
            are locations in :math:`r_z`.
        e_knots : dict
            Keys should be the (integer) "m" order of the distortion term (for the
            distortion function), and values should be the knot locations for
            interpolating the values of the distortion coefficients :math:`e_m(r_z')`.
            Currently, this functionality has been partially disabled and the functions
            are required to be linear, so you must pass in two knots.

        """
        self.dens_knots = jnp.array(dens_knots)
        self.e_knots = {int(k): jnp.array(knots) for k, knots in e_knots.items()}

        for m, knots in self.e_knots.items():
            if len(knots) != 2:
                raise NotImplementedError(
                    "The current implementation of the model requires a purely linear "
                    "function for the e_m coefficients, which is equivalent to having "
                    f"just two knots in the (linear) spline. You passed {len(knots)} "
                    f"knots for the m={m} expansion term."
                )

    @partial(jax.jit, static_argnames=["self"])
    def get_rz(self, rz_prime, theta_prime, e_vals):
        """
        Compute the distorted radius :math:`r_z`
        """
        es = self.get_es(rz_prime, e_vals)
        return rz_prime * (
            1
            + jnp.sum(
                jnp.array([e * jnp.cos(n * theta_prime) for n, e in es.items()]), axis=0
            )
        )

    @partial(jax.jit, static_argnames=["self"])
    def get_rz_prime(self, rz, theta_prime, e_vals):
        thp = theta_prime

        # convert e_vals and e_knots to slope and intercept
        e_as = {
            k: (e_vals[k][1] - e_vals[k][0]) / (self.e_knots[k][1] - self.e_knots[k][0])
            for k in e_vals
        }
        e_bs = {k: -e_as[k] * self.e_knots[k][0] + e_vals[k][0] for k in e_vals}

        terms1 = jnp.sum(jnp.array([e_bs[k] * jnp.cos(k * thp) for k in e_bs]), axis=0)
        terms2 = jnp.sum(jnp.array([e_as[k] * jnp.cos(k * thp) for k in e_bs]), axis=0)
        return (2 * rz) / (1 + terms1 + jnp.sqrt((1 + terms1) ** 2 + 4 * rz * terms2))

    @partial(jax.jit, static_argnames=["self"])
    def get_z(self, rz, theta_prime, e_vals, Omega):
        rzp = self.get_rz_prime(rz, theta_prime, e_vals)
        return rzp * jnp.sin(theta_prime) / jnp.sqrt(Omega)

    @partial(jax.jit, static_argnames=["self"])
    def get_vz(self, rz, theta_prime, e_vals, Omega):
        rzp = self.get_rz_prime(rz, theta_prime, e_vals)
        return rzp * jnp.cos(theta_prime) * jnp.sqrt(Omega)

    @partial(jax.jit, static_argnames=["self"])
    def get_Tz_Jz_thz(self, z, vz, e_vals, Omega, N_grid=101):
        rz, _, thp_ = self.get_rz_th(z, vz, Omega, e_vals)

        dz_dthp_func = jax.vmap(
            jax.grad(self.get_z, argnums=1),
            in_axes=[None, 0, None, None]
        )

        # Grid of theta_prime to do the integral over:
        thp = jnp.linspace(0, jnp.pi / 2, N_grid)
        vz_th = self.get_vz(rz, thp, e_vals, Omega)
        dz_dthp = dz_dthp_func(rz, thp, e_vals, Omega)

        Tz = 4 * simpson(dz_dthp / vz_th, thp)
        Jz = 4 / (2*jnp.pi) * simpson(dz_dthp * vz_th, thp)

        thp_partial = jnp.linspace(0, thp_, N_grid)
        vz_th_partial = self.get_vz(rz, thp_partial, e_vals, Omega)
        dz_dthp_partial = dz_dthp_func(rz, thp_partial, e_vals, Omega)
        dt = simpson(dz_dthp_partial / vz_th_partial, thp_partial)
        thz = 2*jnp.pi * dt / Tz

        return Tz, Jz, thz

    @partial(jax.jit, static_argnames=["self"])
    def get_es(self, rz_prime, e_vals):
        es = {}
        for k, vals in e_vals.items():
            es[k] = InterpolatedUnivariateSpline(self.e_knots[k], vals, k=1)(rz_prime)
        return es

    @partial(jax.jit, static_argnames=["self"])
    def get_rz_th(self, z, vz, Omega, e_vals):
        x = vz / jnp.sqrt(Omega)
        y = z * jnp.sqrt(Omega)

        rz_prime = jnp.sqrt(x**2 + y**2)
        th_prime = jnp.arctan2(y, x)
        rz = self.get_rz(rz_prime=rz_prime, theta_prime=th_prime, e_vals=e_vals)

        return rz, rz_prime, th_prime

    @partial(jax.jit, static_argnames=["self"])
    def get_ln_dens(self, rz, ln_dens_vals):
        spl = InterpolatedUnivariateSpline(self.dens_knots, ln_dens_vals, k=3)
        return spl(rz)

    @partial(jax.jit, static_argnames=["self"])
    def ln_density(self, params, z, vz):
        rz, *_ = self.get_rz_th(
            z - params["z0"],
            vz - params["vz0"],
            Omega=jnp.exp(params["ln_Omega"]),
            e_vals=params["e_vals"],
        )
        return self.get_ln_dens(rz, params["ln_dens_vals"])

    @partial(jax.jit, static_argnames=["self"])
    def ln_poisson_likelihood(self, params, z, vz, H):
        # Expected number:
        ln_Lambda = self.ln_density(params, z, vz)

        # gammaln(x+1) = log(factorial(x))
        return (H * ln_Lambda - jnp.exp(ln_Lambda) - gammaln(H + 1)).sum()

    @partial(jax.jit, static_argnames=["self"])
    def objective(self, params, z, vz, H):
        return -(self.ln_poisson_likelihood(params, z, vz, H)) / H.size

    def optimize(self, params0, z, vz, H, bounds=None, jaxopt_kwargs=None):
        if jaxopt_kwargs is None:
            jaxopt_kwargs = dict()
        jaxopt_kwargs.setdefault("maxiter", 16384)

        if bounds is not None:
            jaxopt_kwargs.setdefault("method", "L-BFGS-B")
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
            jaxopt_kwargs.setdefault("method", "BFGS")
            raise NotImplementedError("TODO")

        return res
