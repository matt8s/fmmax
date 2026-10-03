---
slug: /
---

# FMMAX documentation

FMMAX implements the Fourier modal method (FMM), also known as rigorous coupled-wave analysis (RCWA), in JAX. It models electromagnetic scattering in periodic layered media and supports automatic differentiation, batching, CPU and GPU execution, Brillouin-zone integration, vector formulations, magnetic media, and transverse anisotropy with a decoupled z axis.

## Start here

1. Follow the [installation guide](Installation.md) to create a CPU or NVIDIA GPU environment.
2. Work through a tutorial that matches your problem:
   - [Periodic dipole](Tutorials/dipoles.md) introduces sources, scattering matrices, field reconstruction, and extraction efficiency.
   - [Metal, dipole, and PML](Tutorials/metal_dipole.md) uses anisotropic materials to model absorbing boundaries around a dipole above metal.
   - [Brillouin-zone integration on a photonic crystal](Tutorials/crystal_bz.md) treats localized dipole and Gaussian-beam sources.
   - [Metal grating](Metal_grating.md) builds a layered grating simulation and compares convergence across expansion choices.
3. Consult the [API reference](API/basis.LatticeVectors.md) for functions, classes, and argument definitions.

## Numerical methods and validation

- [Analytic lamellar material coefficients](Analytic_materials.md) explains raster-free Fourier coefficients and direct convolution-matrix eigensolves for binary stripes.
- [Exact-modal characteristic evaluation](Exact_modal.md) documents the binary-lamellar characteristic function and how to interpret its outputs.
- [Cross-solver numerical validation](Numerical_validation.md) describes comparisons with analytic results, S4, and torcwa, including field, phase, flux, and accelerator conventions.
- [References](References.md) expands the abbreviated literature labels used throughout the documentation and API pages.

## Conventions

- The speed of light, vacuum permittivity, and vacuum permeability are all 1.
- Fields evolve in time as $\exp(-i\omega t)$.
- For primitive lattice vectors $\mathbf{u}$ and $\mathbf{v}$, the unit cell is the parallelogram with vertices $\mathbf{0}$, $\mathbf{u}$, $\mathbf{u}+\mathbf{v}$, and $\mathbf{v}$.
- Grid index `(0, 0)` corresponds to the physical location $\mathbf{0}$.
- The scattering-matrix block $\mathbf{S}_{11}$ maps incident forward-going amplitudes to transmitted forward-going amplitudes. The other blocks follow the corresponding FMMAX convention.

## Batching

Use batched calculations where possible instead of Python loops. Most quantities use leading batch axes; wave amplitudes and electromagnetic fields use a trailing batch axis. This layout permits calculations such as transmission for several polarizations to use a matrix-matrix product:

```python
transmitted_amplitudes = S11 @ incident_amplitudes
```
