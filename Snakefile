rule get_gaia_data:
    input:
        "src/scripts/get-gaia-data.py"
    output:
        "src/data/dr3-rv-gspphot-parent.fits"
    log:
        "logs/get_dr3_data.log"
    conda:
        "environment.yml"
    shell:
        "python {input[0]} --out={output[0]}"

rule get_fidelity_data:
    input:
        "src/scripts/get-fidelity-data.py"
    output:
        "src/data/dr3-rv-sample-fidelity.fits"
    log:
        "logs/get_fidelity_data.log"
    conda:
        "environment.yml"
    shell:
        "python {input[0]} --out={output[0]}"

rule clean_data:
    input:
        "src/scripts/clean-update-data.py",
        "src/data/dr3-rv-gspphot-parent.fits",
        "src/data/dr3-rv-sample-fidelity.fits"
    output:
        "src/data/parent-sample-cube.fits"
    log:
        "logs/clean_data.log"
    conda:
        "environment.yml"
    shell:
        "python {input[0]}"

rule run_fit_by_guiding_radius:
    input:
        "src/scripts/TODO.py",
        "src/data/parent-sample-cube.fits"
    output:
        "src/data/fit_by_guiding_radius/TODO"
    log:
        "logs/TODO.log"
    conda:
        "environment.yml"
    shell:
        "mpiexec python {input[0]} --gaia-data={input[1]}"
