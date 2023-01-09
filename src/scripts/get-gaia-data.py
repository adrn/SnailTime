import sys
from schnecke.config import login_gea


def main(data_file):
    if data_file.exists():
        sys.exit(0)

    Gaia = login_gea()

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
        mh_gspphot, mh_gspphot_lower, mh_gspphot_upper,
        ag_gspphot, ebpminrp_gspphot,
        grvs_mag, rv_template_teff,
        spur.fidelity_v2
    FROM
        gaiadr3.gaia_source AS gaia
    LEFT JOIN external.gaiaedr3_spurious AS spur USING (source_id)
    WHERE
        gaia.parallax > 0.25 AND
        gaia.radial_velocity IS NOT NULL
    """

    job = Gaia.launch_job_async(query)

    tbl = job.get_results()
    tbl.write(data_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Output filename", dest="out")
    args = parser.parse_args()

    main(args.out)
