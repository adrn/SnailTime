import astropy.units as u
import numpy as np
import jax.numpy as jnp

from .config import usys

def get_data_im(z, vz, bins):
    """
    Convert the raw data -- stellar positions and velocities z, vz -- into a binned 2D
    histogram / image of number counts.
    """
    data_H, xe, ye = np.histogram2d(
        vz.decompose(usys).value,
        z.decompose(usys).value,
        bins=(bins['vz'], bins['z'])
    )
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    xc, yc = np.meshgrid(xc, yc)

    return {'z': jnp.array(yc), 'vz': jnp.array(xc), 'H': jnp.array(data_H.T)}
