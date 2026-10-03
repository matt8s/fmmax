# FMMAX: Fourier Modal Method with JAX

<a href="https://facebookresearch.github.io/fmmax/"><img src="https://img.shields.io/badge/Docs-blue.svg"/></a>
![Continuous integration](https://github.com/matt8s/fmmax/actions/workflows/build-ci.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/fmmax)

FMMAX implements the Fourier modal method (FMM), also known as rigorous coupled-wave analysis (RCWA), in [JAX](https://github.com/google/jax). It solves Maxwell's equations in periodic layered media by expanding the in-plane directions in a truncated Fourier basis and propagating through the layer stack with scattering matrices [1999 Whittaker, 2012 Liu, 2020 Jin].

The JAX implementation supports automatic differentiation, batched calculations, and execution on CPUs and GPUs. FMMAX also provides Brillouin-zone integration, vector FMM formulations, and support for magnetic media and transverse anisotropy with a decoupled z axis.

## Quick start

FMMAX requires Python 3.10 or newer. To install this repository in an editable development environment, choose a reviewed tag or full commit and run:

```sh
git clone https://github.com/matt8s/fmmax.git
cd fmmax
git checkout <reviewed-tag-or-full-commit>
python -m pip install -e ".[dev]"
python -m pip check
```

Confirm that JAX can see the expected devices:

```sh
python -c "import jax; print(jax.devices())"
```

> **Package and documentation note:** `pip install fmmax` and the [hosted documentation](https://facebookresearch.github.io/fmmax/) currently refer to the original upstream resources. Install this fork from [`matt8s/fmmax`](https://github.com/matt8s/fmmax), preferably at a reviewed tag or full commit. See [MAINTENANCE.md](MAINTENANCE.md) for maintenance and provenance details.

For CPU, NVIDIA GPU, pip, uv, and Conda instructions, see the [installation guide](docs/docs/Installation.md).

## Where to start

Start with a workflow that matches your problem:

- [Periodic dipole](notebooks/dipoles.ipynb): define a source, assemble scattering matrices, reconstruct fields, and calculate extraction efficiency.
- [Metal, dipole, and PML](notebooks/metal_dipole.ipynb): model a dipole above a metal plane using anisotropic perfectly matched layers.
- [Brillouin-zone integration](notebooks/crystal_bz.ipynb): simulate localized dipole and Gaussian-beam sources in a photonic-crystal slab.
- [Metal grating](docs/docs/Metal_grating.md): build a layered grating simulation and compare convergence across expansion choices.

Additional numerical-method documentation covers:

- [analytic lamellar material coefficients](docs/docs/Analytic_materials.md);
- [exact-modal characteristic evaluation](docs/docs/Exact_modal.md);
- [cross-solver numerical validation](docs/docs/Numerical_validation.md).

## Capabilities

### Brillouin-zone integration

Brillouin-zone integration [2022 Lopez-Fraguas] represents localized sources in periodic structures as a batch of Bloch-periodic calculations. The `crystal` example applies this method to a Gaussian beam incident on a photonic-crystal slab and to a localized dipole in a finite supercell within the slab.

![Gaussian beam incident on photonic crystal](img/crystal_beam.gif)

### Vector FMM formulations

Vector formulations introduce local coordinate systems normal and tangent to material interfaces, allowing the corresponding field components to be treated differently to improve convergence. FMMAX implements the _Pol_, _Normal_, and _Jones_ methods described by [2012 Liu], as well as a _Jones direct_ formulation. Vector fields can be generated automatically by functional minimization, including for anisotropic and magnetic materials. See the `vector_fields` example for a comparison.

![Comparison of automatically generated vector fields](img/vector_fields.png)

### Anisotropic and magnetic materials

FMMAX accepts permittivity and permeability tensors with `xx`, `xy`, `yx`, `yy`, and `zz` components. Among other applications, this enables uniaxial perfectly matched layers. The `metal_dipole` example uses these materials to simulate a dipole in vacuum above a metal substrate.

![Dipole suspended above metal substrate with PML](img/metal_dipole.png)

## FMM conventions

- The speed of light, vacuum permittivity, and vacuum permeability are all 1.
- Fields evolve in time as $\exp(-i\omega t)$.
- For primitive lattice vectors $\mathbf{u}$ and $\mathbf{v}$, the unit cell is the parallelogram with vertices $\mathbf{0}$, $\mathbf{u}$, $\mathbf{u}+\mathbf{v}$, and $\mathbf{v}$.
- For a gridded quantity such as patterned-layer permittivity, grid index `(0, 0)` corresponds to the physical location $\mathbf{0}$.
- The scattering-matrix block $\mathbf{S}_{11}$ maps incident forward-going amplitudes to transmitted forward-going amplitudes. The other blocks follow the corresponding FMMAX convention, which differs from conventions commonly used for photonic integrated circuits.

## Batching

FMMAX supports batched calculations and generally favors batching over Python loops.

- Most arrays use leading batch axes.
- Wave amplitudes and electromagnetic fields use a trailing batch axis.

For example, transmission for several incident polarizations can be evaluated as a matrix-matrix operation:

```python
transmitted_amplitudes = S11 @ incident_amplitudes
```

## Citing FMMAX

If you use FMMAX, please consider citing [the FMMAX paper](https://doi.org/10.1364/OE.503481):

```bibtex
@article{schubert2023fourier,
      author={Martin F. Schubert and Alec M. Hammond},
      title={Fourier modal method for inverse design of metasurface-enhanced micro-LEDs},
      journal={Optics Express},
      volume={31},
      number={26},
      pages={42945--42960},
      year={2023},
      doi={10.1364/OE.503481}
}
```

## License

FMMAX is licensed under the [MIT license](LICENSE).

## References

- [2012 Liu] V. Liu and S. Fan, [S4: A free electromagnetic solver for layered periodic structures](https://www.sciencedirect.com/science/article/pii/S0010465512001658), _Comput. Phys. Commun._ **183**, 2233–2244 (2012).

- [1999 Whittaker] D. M. Whittaker and I. S. Culshaw, [Scattering-matrix treatment of patterned multilayer photonic structures](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.60.2610), _Phys. Rev. B_ **60**, 2610 (1999).

- [2020 Jin] W. Jin, W. Li, M. Orenstein, and S. Fan, [Inverse design of lightweight broadband reflector for relativistic lightsail propulsion](https://pubs.acs.org/doi/10.1021/acsphotonics.0c00768), _ACS Photonics_ **7**, 9, 2350–2355 (2020).

- [2022 Lopez-Fraguas] E. López-Fraguas, F. Binkowski, S. Burger, S. Hagedorn, B. García-Cámara, R. Vergaz, C. Becker, and P. Manley, [Tripling the light extraction efficiency of a deep ultraviolet LED using a nanostructured p-contact](https://www.nature.com/articles/s41598-022-15499-7), _Scientific Reports_ **12**, 11480 (2022).
