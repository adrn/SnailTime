import sys

import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np
import scipy.interpolate as sci
import gala.dynamics as gd
from pyia import GaiaData

from schnecke.config import galcen_frame
import sywpaths

data_file = sywpaths.data / 'parent-sample-cube.fits'
if data_file.exists():
    print(f"Data file at '{str(data_file)}' already exists")
    sys.exit(0)

initial_tbl = at.QTable.read(sywpaths.data / "dr3-rv-gspphot-parent.fits")
print("Initial data table loaded...")
fidelity_tbl = at.QTable.read(sywpaths.data / "dr3-rv-sample-fidelity.fits")
print("Fidelity data table loaded...")
tbl = at.join(initial_tbl, fidelity_tbl, keys="source_id")
tbl = tbl[
    (tbl["fidelity_v2"] > 0.5) &
    (tbl["ruwe"] < 1.4) &
    (tbl["rv_template_teff"] < 14_500 * u.K)
]
print("Tables joined and filtered...")

# Weirdness to make invalid values nan's and not 1e20
for col in tbl.colnames:
    if np.issubdtype(tbl[col].dtype, np.floating):
        if tbl[col].unit is None:
            nanval = np.nan
        else:
            try:
                nanval = np.nan * tbl[col].unit
            except ValueError:
                nanval = np.nan
        tbl[col][tbl[col].value > 1e15] = nanval

# -------------------------------------------------------------------------------------
# Parallax zeropoint
#
# TODO: something about parallax zero point
# plx_zpt =
plx = tbl["parallax"]
new_dist = coord.Distance(parallax=plx)

# -------------------------------------------------------------------------------------
# Correct radial velocities:
#
new_rv = tbl["radial_velocity"].copy()

# Katz et al. 2022, Eqn. XX
mask1 = (tbl["grvs_mag"] > 11.0 * u.mag) & (tbl["rv_template_teff"] < 8500 * u.K)
rv = tbl["radial_velocity"][mask1]
Grvs = tbl["grvs_mag"][mask1].value
new_rv[mask1] = rv + (-0.02755 * Grvs**2 + 0.55863 * Grvs - 2.81129) * u.km / u.s

# Blomme et al. 2022, Section 5
mask2 = (
    (tbl["rv_template_teff"] > 8500 * u.K)
    & (tbl["rv_template_teff"] < 14_500 * u.K)
    & (tbl["grvs_mag"] > 6.0 * u.mag)
    & (tbl["grvs_mag"] < 12.0 * u.mag)
)
rv = tbl["radial_velocity"][mask2]
Grvs = tbl["grvs_mag"][mask2].value
new_rv[mask2] = rv + (-7.98 + 1.135 * Grvs) * u.km / u.s

# -------------------------------------------------------------------------------------
# TODO - compute other shit
#

g = GaiaData(tbl)
c = g.get_skycoord(distance=new_dist, radial_velocity=new_rv)
galcen = c.transform_to(galcen_frame)
w = gd.PhaseSpacePosition(galcen.data)
L = w.angular_momentum()

xsun = [-1., 0, 0] * galcen_frame.galcen_distance

cyl = w.cylindrical
R = cyl.rho

# Use the Eilers et al. 2019 circular velocity curve to compute R_G without a potential
eilers_tbl = at.Table.read(
    sywpaths.data / "Eilers2019-circ-velocity.txt", format="ascii.basic"
)
eilers_vcirc = sci.InterpolatedUnivariateSpline(eilers_tbl["R"], eilers_tbl["v_c"], k=3)
interp_vcirc = eilers_vcirc(R.to_value(u.kpc))

Rg = np.abs((L[2] / interp_vcirc).to(u.kpc))

tbl['dist_fixed'] = new_dist
tbl['rv_fixed'] = new_rv

tbl['xyz'] = w.xyz.T
tbl['v_xyz'] = w.v_xyz.T
tbl['z'] = w.z
tbl['vz'] = w.v_z

tbl['R'] = R
tbl['phi'] = cyl.phi

tbl['L'] = L.T
tbl['Rg'] = Rg

cube_mask = np.all(np.abs(tbl['xyz'] - xsun) < (4 * u.kpc / np.sqrt(2)), axis=1)

# Fix the logg column units:
for col in tbl.colnames:
    if col.startswith('logg'):
        tbl[col] = tbl[col].value

print("Final table prepared -- writing...")
tbl[cube_mask].write(data_file)
