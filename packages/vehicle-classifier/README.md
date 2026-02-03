# Acies Vehicle Classifiers

Acoustic- and seismic-based vehicle classifiers.

If you use this repository in your research, please cite our accompanying paper
to acknowledge the work. You can do so with the following BibTeX entry:

```bibtex
@inproceedings{li2024aciesos,
  title={Acies-OS: A Content-Centric Platform for Edge AI Twinning and Orchestration},
  author={Li, Jinyang and Chen, Yizhuo and Kimura, Tomoyoshi and et al.},
  booktitle={2024 33rd International Conference on Computer Communications and Networks (ICCCN)},
  pages={1--1},
  year={2024},
  organization={IEEE}
}
```

## Setup

### Python Environment Management

This project uses [`uv`](https://docs.astral.sh/uv) (or its predecessor [`rye`](https://rye.astral.sh)) to manage the Python environment.

To check if `rye` is already installed, run:

```bash
which rye
```

- If the command prints a path, `rye` is available and you can skip this step.
- If not, we recommend installing `uv` for new setups. Follow [uv’s official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

For backward compatibility, you may also install [rye](https://rye.astral.sh/guide/installation/).

### Clone and install dependencies

```shell
$ git clone git@github.com:acies-os/vehicle-classifier.git
$ cd vehicle-classifier
vehicle-classifier$ uv sync

# or, if using rye
vehicle-classifier$ rye sync
```

### Install `just`

Install `just` use [your package manager](https://just.systems/man/en/packages.html) or [pre-built binary](https://just.systems/man/en/pre-built-binaries.html).

## Download Model Weights

Place the weight files under `models/`. You can either download them from the
[GitHub release](https://github.com/acies-os/vehicle-classifier/releases/tag/weight-v1.0.0)
or pull them directly with `wget`.

```bash
# in repo's root folder
vehicle-classifier$ cd models/

# you can also use curl, just replace `wget` with `curl -LO`
vehicle-classifier/models$ wget https://github.com/acies-os/vehicle-classifier/releases/download/weight-v1.0.0/gcq202410_mae.pt
vehicle-classifier/models$ wget https://github.com/acies-os/vehicle-classifier/releases/download/weight-v1.0.0/Parkland_TransformerV4_vehicle_classification_finetune_gcq202410_1.0_multiclasslatest.pt
```

## Usage

There are 2 classifiers, to run them:

```shell
# VibroFM
just vfm

# FreqMAE
just mae
```

## Documentation

The documentation is managed using **Sphinx**, which fetch docstring comments from code and compile them into html pages.

Sphinx sources live in the `docs/source/` folder:

- `conf.py` — Sphinx configuration
- `*.rst` files — reStructuredText source documents that define docs content

To build Sphinx documentation, run this command:

```shell
just build-doc
```

Now the built documentation should live in `docs/_build/` folder.

To view the documentation in browser, run this command:

```shell
just view-doc
```
