rule sgrA_star:
    input:
        "src/scripts/get-gaia-data.py"
    output:
        "src/data/gaiadr3-rv-plx0.1.fits"
    conda:
        "environment.yml"
    shell:
        "python {input[0]}"