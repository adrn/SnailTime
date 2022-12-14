# +
import os
import sys
import pathlib

from astroquery.gaia import Gaia

import sywpaths

# +
data_file = sywpaths.data / 'dr3-rv-plx0.2-gspphot.fits'
if data_file.exists():
    sys.exit()

# +
user = os.environ.get("GAIA_USER", None)
password = os.environ.get("GAIA_PASSWORD", None)
credentials_path = pathlib.Path('~/.gaia/archive.login').expanduser()

if user is not None:
    Gaia.login(user=user, password=password)

elif credentials_path.exists():
    Gaia.login(credentials_file=credentials_path)

else:
    raise RuntimeError("Unable to log in to Gaia archive.")
# -

query = """
SELECT
    source_id, ra, dec, parallax, parallax_error,
    pmra, pmra_error, pmdec, pmdec_error,
    ruwe,
    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
    phot_g_mean_flux_over_error,
    phot_bp_mean_flux_over_error, phot_rp_mean_flux_over_error,
    radial_velocity, radial_velocity_error,
    teff_gspphot, teff_gspphot_lower, teff_gspphot_upper,
    logg_gspphot, logg_gspphot_lower, logg_gspphot_upper,
    mh_gspphot, ag_gspphot, abp_gspphot, arp_gspphot
FROM
    gaiadr3.gaia_source
WHERE
    parallax > 0.2 AND
    radial_velocity BETWEEN -1000 AND 1000
"""


job = Gaia.launch_job_async(query)

tbl = job.get_results()

tbl.write(data_file)
