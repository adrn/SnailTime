import pathlib
import pickle

import jax
jax.config.update("jax_enable_x64", True)

from schnecke.plot import plot_data_models_residual


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", required=True, type=str)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()

    output_path = pathlib.Path(args.output_path)
    # output_path.mkdir(exist_ok=True)

    models = []
    for model_file in args.models:
        with open(model_file, 'rb') as f:
            models.append(pickle.load(f))
    models = sorted(models, key=lambda x: x['metadata']['Rg_bin'][0])

    for n, cache in enumerate(models):
        fig, axes = plot_data_models_residual(
            cache['data'], cache['model0'], cache['model-fitted'],
            smooth_residual=None, vlim_residual=0.3
        )
        fig.savefig(output_path / f'Rg-model-{n}-plot.png', dpi=250)
