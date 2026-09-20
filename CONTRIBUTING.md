# Contributing to fmmax
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Submit the pull request to `matt8s/fmmax` and link relevant upstream issues.

This community fork accepts contributions under the existing MIT license; it
does not require Meta's contributor agreement. Preserve copyright notices and
credit when adapting upstream or other fork contributions.

## Development checks

Current CI uses Python 3.10. Install with `python -m pip install -e ".[dev]"`.
Run the relevant checks before submitting:

```sh
black --check src examples tests
isort --profile black --check-only src examples tests
mypy src examples
darglint src examples --strictness=short --ignore-raise=ValueError
pytest tests/fmmax tests/grcwa tests/examples
```

The CI also tests the optional `jeig` backend. Numerical changes should cover
gradients, batching, complex-valued inputs, and physical regression cases where
applicable. See `.github/workflows/build-ci.yml` for the complete job definitions.

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Report bugs at <https://github.com/matt8s/fmmax/issues>. See [SECURITY.md](SECURITY.md)
for confidential vulnerability reporting. This fork is independently maintained.

## License
By contributing to fmmax, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
