# Installation

FMMAX requires Python 3.10 or newer. The commands below install this repository directly from Git; replace `<reviewed-tag-or-full-commit>` with the revision you intend to use.

## CPU quick start

Create an isolated environment and install FMMAX with pip:

```sh
python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "fmmax @ git+https://github.com/matt8s/fmmax.git@<reviewed-tag-or-full-commit>"
python -m pip check
```

Check the devices available to JAX:

```sh
python -c "import jax; print(jax.devices())"
```

> **Distribution note:** `pip install fmmax` and the currently hosted documentation refer to the original upstream resources. Install this fork from [`matt8s/fmmax`](https://github.com/matt8s/fmmax), preferably at a reviewed tag or full commit so that the environment is reproducible.

The package metadata accepts JAX and jaxlib versions from 0.4.38 through 0.11.2.

## Developer installation with pip

For an editable installation from a source checkout:

```sh
git clone https://github.com/matt8s/fmmax.git
cd fmmax
python -m pip install -e ".[dev]"
```

The optional extras are:

- `tests`: test runners and comparison dependencies;
- `examples`: plotting, image-processing, and scientific-computing dependencies;
- `docs`: documentation generators;
- `dev`: the combined development toolchain;
- `jeig`: the optional `jeig` eigensolver backend;
- `cuda12`: JAX's CUDA 12 packages;
- `cuda13`: JAX's CUDA 13 packages.

Extras can be combined, for example:

```sh
python -m pip install -e ".[dev,jeig]"
```

## uv

uv reads the same `pyproject.toml`; FMMAX does not require a separate uv project file.

```sh
uv venv --python 3.14
source .venv/bin/activate
# On Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install "fmmax @ git+https://github.com/matt8s/fmmax.git@<reviewed-tag-or-full-commit>"
uv pip check
```

For development from a source checkout:

```sh
uv pip install -e ".[dev]"
```

Runtime dependencies are not locked to a single environment. The tested compatibility endpoints are pinned explicitly when they are exercised.

## Conda environments

Conda can manage the Python environment while pip installs FMMAX and its Python dependencies:

```sh
conda create --name fmmax python=3.14 pip
conda activate fmmax
python -m pip install "fmmax @ git+https://github.com/matt8s/fmmax.git@<reviewed-tag-or-full-commit>"
python -m pip check
```

A source checkout also provides `environment.yml` for an editable development environment:

```sh
conda env create --file environment.yml
conda activate fmmax-dev
python -m pip check
```

FMMAX is installed with pip inside these environments because a separate Conda package is not currently published.

## NVIDIA GPU installation

Install one CUDA extra, not both. CUDA 13 requires JAX 0.10.2 or newer and sufficiently recent NVIDIA hardware and drivers. CUDA 12 retains the JAX 0.4.38 compatibility floor and supports a broader range of hardware.

From a source checkout:

```sh
python -m pip install -e ".[cuda13]"
# Or: python -m pip install -e ".[cuda12]"
```

Add other extras when needed, for example:

```sh
python -m pip install -e ".[dev,cuda13]"
```

With uv, replace `python -m pip install` with `uv pip install`.

Confirm that JAX actually selected the GPU rather than assuming that a successful import did so:

```sh
python -c "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu'"
```

### Native cuSOLVER eigensolver

The optional device-native nonsymmetric eigensolver is selected explicitly with `EigBackend.CUSOLVER`:

```python
from fmmax import utils

eig_backend = utils.EigBackend.CUSOLVER
```

Pass this value as the `eig_backend` argument to the relevant eigensolve function. It requires a compatible recent JAX release and an NVIDIA GPU. Unsupported configurations raise an error rather than silently selecting this backend. The default eigensolver remains compatible with the JAX 0.4.38 floor.

## Tested configurations

The following configurations have been exercised on Linux:

| Python | JAX/jaxlib | Backend | Coverage |
| --- | --- | --- | --- |
| 3.10 | 0.4.38 | CPU | Complete `tests/fmmax` suite and example/comparator checks |
| 3.12 | 0.11.2 | CPU | Conda editable install, dependency check, basis tests, and static typing |
| 3.14 | 0.11.2 | CPU | Complete inherited CPU suite and focused optional-backend checks |
| 3.14 | 0.11.2 | NVIDIA CUDA 13 | Solver, gradient, field, flux, S4, and native cuSOLVER checks |

Versions within the declared JAX range have not all received the same depth of testing. The GPU results were obtained on NVIDIA A100 hardware and do not establish compatibility with every accelerator, driver, or operating system.
