"""Configure numerical precision before test modules import solver constants."""

import jax

# Enabling x64 inside individual test modules is too late if an earlier import
# has already initialized shared lattice vectors or module-level solve fixtures.
jax.config.update("jax_enable_x64", True)
