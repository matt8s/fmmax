# Metal grating

This example computes the complex TE- and TM-polarized reflection amplitudes of a one-dimensional metallic stripe grating at normal incidence. It also sweeps the Fourier expansion size, truncation rule, and FMM formulation so that convergence can be assessed rather than assumed.

The complete source is available in [`examples/metal_grating.py`](https://github.com/matt8s/fmmax/blob/main/examples/metal_grating.py).

## Geometry and units

The structure is periodic along $x$, uniform along $y$, and layered along the propagation direction. From the ambient toward the substrate, the stack is:

| Layer | Relative permittivity | Thickness |
| --- | --- | ---: |
| Ambient | $1$ | Exterior |
| Planarization | $2.25$ | 20 nm |
| Patterned grating | Metal and planarization | 80 nm |
| Metal substrate | $-7.632 + 0.731i$ | Exterior |

The grating has a 180 nm pitch and a nominal 60 nm metal-line width. The metal line uses the same permittivity as the substrate, while the surrounding material uses the planarization permittivity. All lengths—including wavelength, lattice vectors, layer thicknesses, and rasterization resolution—are expressed consistently in nanometers.

The square computational cell has lattice vectors $(180,0)$ nm and $(0,180)$ nm. Although a two-dimensional sampled cell is supplied to FMMAX, the density depends only on $x$, so the structure is uniform along $y$.

## Complete example

The following is a complete runnable version of the example. Running it directly performs the default convergence sweep.

```python
import itertools
from typing import Tuple

import jax.numpy as jnp

from fmmax import basis, fmm, scattering, utils


NUM_TERMS_SWEEP = (
    9,
    25,
    49,
    81,
    121,
    169,
    225,
    289,
    361,
    441,
    529,
    625,
    729,
    841,
)


def simulate_grating(
    permittivity_ambient: complex = 1.0 + 0.0j,
    permittivity_planarization: complex = 2.25 + 0.0j,
    permittivity_substrate: complex = -7.632 + 0.731j,
    wavelength_nm: float = 500.0,
    pitch_nm: float = 180.0,
    grating_width_nm: float = 60.0,
    grating_thickness_nm: float = 80.0,
    planarization_thickness_nm: float = 20.0,
    resolution_nm: float = 1.0,
    approximate_num_terms: int = 20,
    truncation: basis.Truncation = basis.Truncation.CIRCULAR,
    formulation: fmm.Formulation = fmm.Formulation.FFT,
) -> Tuple[int, complex, complex]:
    """Computes the TE- and TM-polarized reflection from a 1D stripe grating."""
    x_nm, _ = jnp.meshgrid(
        jnp.arange(-pitch_nm / 2, pitch_nm / 2, resolution_nm),
        jnp.arange(-pitch_nm / 2, pitch_nm / 2, resolution_nm),
        indexing="ij",
    )
    density = (jnp.abs(x_nm) <= grating_width_nm / 2).astype(float)

    permittivities = [
        jnp.asarray([[permittivity_ambient]]),
        jnp.asarray([[permittivity_planarization]]),
        utils.interpolate_permittivity(
            permittivity_solid=jnp.asarray(permittivity_substrate),
            permittivity_void=jnp.asarray(permittivity_planarization),
            density=density,
        ),
        jnp.asarray([[permittivity_substrate]]),
    ]
    thicknesses = [0, planarization_thickness_nm, grating_thickness_nm, 0]

    in_plane_wavevector = jnp.asarray([0.0, 0.0])
    primitive_lattice_vectors = basis.LatticeVectors(
        u=jnp.asarray([pitch_nm, 0.0]),
        v=jnp.asarray([0.0, pitch_nm]),
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=truncation,
    )

    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength_nm),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=formulation,
        )
        for p in permittivities
    ]

    s_matrix = scattering.stack_s_matrix(
        layer_solve_results=layer_solve_results,
        layer_thicknesses=[jnp.asarray(t) for t in thicknesses],
    )

    r_te = s_matrix.s21[0, 0]
    r_tm = s_matrix.s21[expansion.num_terms, expansion.num_terms]
    return expansion.num_terms, complex(r_te), complex(r_tm)


def convergence_study(
    approximate_num_terms: Tuple[int, ...] = NUM_TERMS_SWEEP,
    truncations: Tuple[basis.Truncation, ...] = (
        basis.Truncation.CIRCULAR,
        basis.Truncation.PARALLELOGRAMIC,
    ),
    fmm_formulations: Tuple[fmm.Formulation, ...] = (
        fmm.Formulation.FFT,
        fmm.Formulation.JONES_DIRECT,
        fmm.Formulation.JONES,
        fmm.Formulation.NORMAL,
        fmm.Formulation.POL,
    ),
) -> Tuple[
    Tuple[fmm.Formulation, basis.Truncation, int, complex, complex], ...
]:
    """Sweeps over expansion sizes, truncations, and FMM formulations."""
    results = []
    for formulation, truncation, n in itertools.product(
        fmm_formulations,
        truncations,
        approximate_num_terms,
    ):
        num_terms, r_te, r_tm = simulate_grating(
            approximate_num_terms=n,
            truncation=truncation,
            formulation=formulation,
        )
        results.append((formulation, truncation, num_terms, r_te, r_tm))
        print(
            f"{formulation.value}/{truncation.value}/n={num_terms}: "
            f"r_te={complex(r_te):.3f}, r_tm={complex(r_tm):.3f}"
        )
    return tuple(results)


if __name__ == "__main__":
    convergence_study()
```

## From sampled geometry to layer permittivity

The patterned layer is represented by a sampled density array:

```python
density = (jnp.abs(x_nm) <= grating_width_nm / 2).astype(float)
```

A value of one denotes metal, and a value of zero denotes planarization material. Because the expression uses only `x_nm`, every row along $y$ is identical, producing a stripe that extends continuously in that direction. With the default 1 nm grid and inclusive `<=` boundary test, the sampled stripe occupies 61 grid columns; 60 nm is the nominal continuum width. `resolution_nm` controls this material rasterization and is independent of the number of Fourier terms used for the fields.

`utils.interpolate_permittivity` maps the density to the two material permittivities. For this binary density, it selects the metal or planarization endpoint. For fractional densities, the utility interpolates the real and imaginary parts of refractive index before squaring, avoiding problematic permittivity zero crossings that can occur with metallic or lossy materials.

Uniform layers are supplied as arrays with shape `(1, 1)`, while the grating layer is supplied as the sampled two-dimensional array. This lets `eigensolve_isotropic_media` distinguish uniform layers from the patterned layer.

The first and last thicknesses are zero because the ambient and substrate are exterior port media, not finite films. They remain in the layer list to define the incident and outgoing modes. A zero thickness places the scattering-matrix reference planes at their interfaces without adding an arbitrary propagation distance through either exterior medium.

## Expansion and layer eigensolves

The zero in-plane wavevector specifies normal incidence. `basis.generate_expansion` then selects the reciprocal-lattice orders retained in the Fourier expansion. `approximate_num_terms` is a target: preserving a symmetric set of orders may cause `expansion.num_terms` to differ from the requested value. The function therefore returns and reports the actual number of terms.

`CIRCULAR` and `PARALLELOGRAMIC` select different shapes for truncating the reciprocal lattice. Each layer receives its own eigensolve using the same wavelength, lattice, in-plane wavevector, expansion, and formulation. The resulting layer modes are passed in physical stack order to `scattering.stack_s_matrix`, which combines interface matching and propagation through the finite layers.

## Why `s21` is reflection

FMMAX defines the scattering matrix by

$$
a_{\mathrm{end}} = s_{11}a_{\mathrm{start}} + s_{12}b_{\mathrm{end}},
$$

$$
b_{\mathrm{start}} = s_{21}a_{\mathrm{start}} + s_{22}b_{\mathrm{end}},
$$

where $a$ denotes forward-going amplitudes and $b$ denotes backward-going amplitudes. With illumination only from the ambient, $b_{\mathrm{end}}=0$, so $b_{\mathrm{start}}=s_{21}a_{\mathrm{start}}$. Under the FMMAX convention, `s21` therefore contains reflection amplitudes and `s11` contains transmission amplitudes.

For normal incidence in the uniform ambient, the two polarization blocks each contain `expansion.num_terms` modes. The zero-order TE mode is first in the first block, while the zero-order TM mode is first in the second block. At normal incidence, TE has its electric field along $y$, and TM along $x$. These values are complex modal amplitudes, not reflected powers; power calculations require the appropriate flux normalization.

## Checking convergence

`convergence_study` evaluates the Cartesian product of expansion-size requests, truncation rules, and formulations. Each returned record contains

```text
(formulation, truncation, actual_num_terms, r_te, r_tm)
```

Use `actual_num_terms` when comparing runs because it can differ from `approximate_num_terms`.

Convergence should be judged by whether the complex TE and TM amplitudes stabilize as the expansion grows. Compare multiple formulations and truncations: discontinuous metallic structures can converge differently depending on polarization and factorization, and no formulation is universally best. Agreement among increasingly large expansions and different formulations provides stronger evidence than a single high-order calculation; analytic results or a separate solver are needed for an independent reference.

The example's regression test uses a one-point version of this sweep to check selected formulations against stored reference amplitudes. That protects against unintended numerical changes, but it is not a substitute for a convergence study. When changing the geometry, material values, or wavelength, repeat the expansion sweep; when boundary sampling may matter, also refine `resolution_nm` separately.
