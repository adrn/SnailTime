from functools import partial
import pickle

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import median_absolute_deviation as MAD
import astropy.table as at
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

from scipy.special import factorial, hermitenorm

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import jaxopt

from schnecke.model import VerticalOrbitModel
from schnecke.data import get_data_im
from schnecke.jax_helpers import simpson

from gala.units import galactic


# TODO: load data

init_model = VerticalOrbitModel(
    dens_knots=jnp.linspace(0, np.sqrt(1.5), 8)**2,
    e_knots={
        2: jnp.array([0., 1.]),
        4: jnp.array([0., 1.]),
    }
)