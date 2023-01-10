import pathlib
from schnecke.config import login_gea


def main(data_file):
    if data_file.exists():
        return

    Gaia = login_gea()

    query = """
    SELECT spur.source_id, spur.fidelity_v2
    FROM external.gaiaedr3_spurious as spur
    JOIN gaiadr3.gaia_source as gaia USING (source_id)
    WHERE
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

    main(pathlib.Path(args.out))
