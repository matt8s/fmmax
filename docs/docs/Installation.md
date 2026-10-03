# Installation

## Distribution status

The package currently published as `fmmax` on PyPI is the original upstream
distribution. Development from this community-maintenance fork should be installed
from a reviewed Git tag or full commit hash until ownership and publishing access
are coordinated with the original maintainers. Pinning a revision makes an install
reproducible; installing a moving branch such as `main` does not.

FMMAX requires Python 3.10 or newer. The declared JAX and jaxlib range is
0.4.38–0.11.2. The compatibility endpoints have been exercised on Linux as follows:

| Python | JAX/jaxlib | Backend | Coverage |
| --- | --- | --- | --- |
| 3.10 | 0.4.38 | CPU | Complete `tests/fmmax` suite and example/comparator checks |
| 3.12 | 0.11.2 | CPU | Conda editable install, dependency check, basis tests, and static typing |
| 3.14 | 0.11.2 | CPU | Complete inherited CPU suite and focused optional-backend checks |
| 3.14 | 0.11.2 | NVIDIA CUDA 13 | Solver, gradient, field, flux, S4, and native cuSOLVER checks |

Versions inside the declared range are supported by the metadata but have not all
received the same exhaustive test coverage. GPU results were obtained on NVIDIA
A100 hardware. They do not establish compatibility with every accelerator, driver,
or operating system.

In the commands below, replace `<reviewed-tag-or-full-commit>` with the revision
you intend to use.

## pip

For an isolated CPU installation directly from Git:

```sh
python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "fmmax @ git+https://github.com/matt8s/fmmax.git@<reviewed-tag-or-full-commit>"
python -m pip check
```

For an editable developer installation from a checkout:

```sh
git clone https://github.com/matt8s/fmmax.git
cd fmmax
python -m pip install -e ".[dev]"
```

The optional extras are `tests`, `examples`, `docs`, `dev`, `jeig`, `cuda12`, and
`cuda13`. Multiple extras can be combined, for example `.[dev,jeig]`.

## uv

uv reads the same standards-based `pyproject.toml`; a separate uv-specific project
file is not required for installing this library:

```sh
uv venv --python 3.14
source .venv/bin/activate
# On Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install "fmmax @ git+https://github.com/matt8s/fmmax.git@<reviewed-tag-or-full-commit>"
uv pip check
```

From a source checkout, use `uv pip install -e ".[dev]"` for development. The
project intentionally does not lock runtime dependencies to one environment;
compatibility endpoints are pinned explicitly in CI instead.

## Conda environments

There is not yet an official Conda package. Conda can manage the Python environment
while pip installs FMMAX and its Python dependencies:

```sh
conda create --name fmmax python=3.14 pip
conda activate fmmax
python -m pip install "fmmax @ git+https://github.com/matt8s/fmmax.git@<reviewed-tag-or-full-commit>"
python -m pip check
```

A source checkout also includes `environment.yml` for a development environment:

```sh
conda env create --file environment.yml
conda activate fmmax-dev
python -m pip check
```

## NVIDIA GPU installation

Use one CUDA extra, not both. CUDA 13 requires JAX 0.10.2 or newer and sufficiently
recent NVIDIA hardware and drivers. CUDA 12 retains the JAX 0.4.38 compatibility
floor and broader hardware support.

From a checkout:

```sh
python -m pip install -e ".[cuda13]"
# Or: python -m pip install -e ".[cuda12]"
```

For uv, replace `python -m pip install` with `uv pip install`. Confirm actual device
selection rather than assuming that a successful import uses the accelerator:

```sh
python -c "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu'"
```

The optional native nonsymmetric eigensolver is selected explicitly through
`EigBackend.CUSOLVER`; it requires a compatible recent JAX release and NVIDIA GPU.
The default backend remains compatible with the JAX 0.4.38 floor.
