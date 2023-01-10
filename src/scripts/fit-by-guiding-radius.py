import pickle

import astropy.table as at
import numpy as np
from gala.units import galactic

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from schnecke.model import VerticalOrbitModel
from schnecke.data import get_data_im
from schnecke.config import usys


def run_fit_orbit_model(init_model, data):
    # From the initial model, estimate initial parameter values and make copies for both
    # keeping track of the model at parameter initialization and optimized model
    model0 = init_model.get_initial_params(
        data["z"].decompose(galactic).value, data["vz"].decompose(galactic).value
    )
    model = model0.copy()

    # Set up binning scheme to make phase-space density maps
    # TODO: number of bins and limits are hard-set!
    im_bins = {"z": np.linspace(-2.5, 2.5, 151)}
    im_bins["vz"] = im_bins["z"] * model0.state["Omega"]
    data_H = get_data_im(data["z"], data["vz"], im_bins)

    # Bounds for the optimizer:
    bounds_l = {
        "vz0": -0.1,
        "z0": -0.5,
        "ln_dens_vals": np.full_like(model0.state["ln_dens_vals"], -5.0),
        "ln_Omega": -5.0,
        "e_vals": {},
    }

    bounds_r = {
        "vz0": 0.1,
        "z0": 0.5,
        "ln_dens_vals": np.full_like(model0.state["ln_dens_vals"], 15.0),
        "ln_Omega": 0.0,
        "e_vals": {},
    }

    for m in model.e_knots:
        bounds_l["e_vals"][m] = np.full_like(model0.state["e_vals"][m], -0.3)
        bounds_r["e_vals"][m] = np.full_like(model0.state["e_vals"][m], 0.3)

    # Run the optimizer to fit
    res = model.optimize(
        **data_H,
        bounds=(bounds_l, bounds_r),
        jaxopt_kwargs=dict(options=dict(maxls=1000, disp=False))
    )
    print("Optimizer state:", res.state)

    return model0, model, data_H


def main(data, metadata, cache_file):
    # Initial model that specifies the knot locations for spline components of the model
    init_model = VerticalOrbitModel(
        dens_knots=jnp.linspace(0, np.sqrt(1.5), 8) ** 2,
        e_knots={
            2: jnp.array([0.0, 1.0]),
            4: jnp.array([0.0, 1.0]),
            6: jnp.array([0.0, 1.0]),
        },
    )

    model0, model, data_H = run_fit_orbit_model(init_model, data)

    # cache the initial and fitted model with the metadata
    cache = {
        "data": data_H,
        "model0": model0,
        "model-fitted": model,
        "metadata": metadata
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-idx", required=True, type=int, dest="bin_idx")
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    Rg_bin_cs = np.arange(
        config["Rg_bin_cen_min"],
        config["Rg_bin_cen_max"] + config["Rg_bin_cen_step"],
        config["Rg_bin_cen_step"],
    )
    half_width = config["Rg_bin_width"] / 2.0

    Rg_bin_l = Rg_bin_cs[args.bin_idx] - half_width
    Rg_bin_r = Rg_bin_cs[args.bin_idx] + half_width

    metadata = {}
    metadata["Rg_bin_idx"] = args.bin_idx
    metadata["Rg_bin"] = (Rg_bin_l, Rg_bin_r)

    # Load the parent sample dataset
    tbl = at.QTable.read(args.input)

    Rg = tbl["Rg"].decompose(usys).value
    Rg_mask = (Rg > Rg_bin_l) & (Rg <= Rg_bin_r)

    main(tbl[Rg_mask], metadata=metadata, cache_file=args.output)
