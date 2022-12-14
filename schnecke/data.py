gaiadr3_query = """
SELECT
    source_id, ra, dec,
    parallax, parallax_error,
    pmra, pmra_error,
    pmdec, pmdec_error,
    phot_g_mean_mag, phot_g_mean_flux_over_error,
    phot_bp_mean_mag, phot_bp_mean_flux_over_error,
    phot_rp_mean_mag, phot_rp_mean_flux_over_error,
    radial_velocity, radial_velocity_error,
    teff_gspphot, logg_gspphot, mh_gspphot,
    distance_gspphot, ag_gspphot, ebpminrp_gspphot
FROM
    gaiadr3.gaia_source_lite
WHERE
    parallax > 0.1 AND
    radial_velocity BETWEEN -1000 AND 1000
"""