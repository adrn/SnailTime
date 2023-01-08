import os
import pathlib

import astropy.coordinates as coord
import astropy.units as u
from gala.units import galactic

# Same values from Hunt+2022
galcen_frame = coord.Galactocentric(
    galcen_distance=8.275 * u.kpc,
    galcen_v_sun=[8.4, 251.8, 8.4] * u.km/u.s,
    z_sun=0*u.pc
)

usys = galactic


def login_gea():
    from astroquery.gaia import Gaia

    user = os.environ.get("GAIA_USER", None)
    password = os.environ.get("GAIA_PASSWORD", None)
    credentials_path = pathlib.Path('~/.gaia/archive.login').expanduser()

    if user is not None:
        Gaia.login(user=user, password=password)

    elif credentials_path.exists():
        Gaia.login(credentials_file=credentials_path)

    else:
        raise RuntimeError("Unable to log in to Gaia archive.")

    return Gaia