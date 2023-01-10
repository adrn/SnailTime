import yaml

rule plots:
    input:
        "src/plots/fit_by_guiding_radius/"

# NOTE: temporarily disabled these rules because I ran the scripts manually and they
# involve some big queries / downloads to the Gaia archive
# rule get_gaia_data:
#     input:
#         "src/scripts/get-gaia-data.py"
#     output:
#         "src/data/dr3-rv-gspphot-parent.fits"
#     log:
#         "logs/get_dr3_data.log"
#     conda:
#         "environment.yml"
#     shell:
#         "python {input[0]} --out={output[0]}"

# rule get_fidelity_data:
#     input:
#         "src/scripts/get-fidelity-data.py"
#     output:
#         "src/data/dr3-rv-sample-fidelity.fits"
#     log:
#         "logs/get_fidelity_data.log"
#     conda:
#         "environment.yml"
#     shell:
#         "python {input[0]} --out={output[0]}"

# rule clean_data:
#     input:
#         "src/scripts/clean-update-data.py",
#         "src/data/dr3-rv-gspphot-parent.fits",
#         "src/data/dr3-rv-sample-fidelity.fits"
#     output:
#         "src/data/parent-sample-cube.fits"
#     log:
#         "logs/clean_data.log"
#     conda:
#         "environment.yml"
#     shell:
#         "python {input[0]}"

# -------------------------------------------------------------------------------------

with open("src/config/fit-by-guiding-radius.yml", "r") as f:
    fbgr_config = yaml.load(f.read(), Loader=yaml.FullLoader)

# rule run_fit_by_guiding_radius:
#     input:
#         "src/scripts/fit-by-guiding-radius.py",
#         "src/data/parent-sample-cube.fits"
#     output:
#         "src/cache/fit_by_guiding_radius/Rg-model-0.pkl"
#     params:
#         config="src/config/fit-by-guiding-radius.yml",
#     log:
#         "logs/fit_by_guiding_radius/Rg-model-0.log"
#     conda:
#         "environment.yml"
#     shell:
#         """
#         python {input[0]} \\
#             --config={params.config} \\
#             --data={input[1]} \\
#             --bin-idx=0 \\
#             --output={output} \\
#             &> {log}
#         """

rule run_fit_by_guiding_radius:
    input:
        "src/data/parent-sample-cube.fits"
    output:
        "src/cache/fit_by_guiding_radius/Rg-model-{n}.pkl"
    params:
        config="src/config/fit-by-guiding-radius.yml",
    log:
        "logs/fit_by_guiding_radius/Rg-model-{n}.log"
    conda:
        "environment.yml"
    shell:
        """
        python src/scripts/fit-by-guiding-radius.py \\
            --config={params.config} \\
            --input={input} \\
            --bin-idx={wildcards.n} \\
            --output={output} \\
            &> {log}
        """

rule plot_fit_by_guiding_radius:
    input:
        expand(
            "src/cache/fit_by_guiding_radius/Rg-model-{n}.pkl",
            n=range(
                int(
                    (fbgr_config["Rg_bin_cen_max"] - fbgr_config["Rg_bin_cen_min"])
                    / fbgr_config["Rg_bin_cen_step"] + 1
                )
            ),
            dataset="{dataset}",
        )
    output:
        "src/plots/fit_by_guiding_radius/"
    log:
        "logs/fit_by_guiding_radius/plot-Rg-model.log"
    conda:
        "environment.yml"
    shell:
        """
        python src/scripts/plot-fit-by-guiding-radius.py \\
            --output-path={output} \\
            {input} \\
            &> {log}
        """
