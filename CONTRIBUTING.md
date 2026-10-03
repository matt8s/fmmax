# Contributing to fmmax

Thank you for contributing to fmmax. Bug reports, documentation improvements, tests, and code changes are all welcome.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Add or update tests for changed behavior.
3. Update the documentation when changing an API or user-facing behavior.
4. Run the relevant development checks.
5. Submit the pull request to `matt8s/fmmax`.
6. Link any relevant upstream issue, pull request, or source contribution.

For numerical changes, include coverage for batching, complex-valued inputs, gradients, flux conservation, field and phase conventions, and physical regression or convergence cases where applicable.

## Development checks

Install the development dependencies with:

```sh
python -m pip install -e ".[dev]"
```

Run the relevant checks before submitting:

```sh
black --check src examples tests
isort --profile black --check-only src examples tests
mypy src examples
darglint src examples --strictness=short --ignore-raise=ValueError
pytest tests/fmmax tests/grcwa tests/examples
```

CI includes a compatibility-floor lane using Python 3.10 / JAX 0.4.38 and a current-dependency lane using Python 3.14 / JAX 0.11.2. It also exercises the optional `jeig` backend. See `.github/workflows/build-ci.yml` for the complete job definitions.

Documentation contributors can find the site build and generation workflow in [`docs/README.md`](docs/README.md).

## Issues and security reports

Use [GitHub issues](https://github.com/matt8s/fmmax/issues) for public bug reports and feature discussions. Include a clear description and enough information to reproduce the behavior.

Report suspected vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## License and attribution

Contributions to this community fork are accepted under the existing MIT license in [LICENSE](LICENSE). No Meta contributor agreement is required for contributions to this fork.

Preserve existing copyright notices and original credit. When adapting work from upstream, another fork, or another project, identify its source and retain the attribution and license information it requires.
