# Analytic lamellar material coefficients

## When to use analytic coefficients

Use `fmmax.fft.binary_lamellar_fourier_coefficients` and `binary_lamellar_convolution_matrix` for a periodic binary stripe with straight, parallel interfaces. The stripe is invariant along the second fractional lattice coordinate; its width and center are specified as fractions of the first lattice period.

These functions evaluate the material Fourier coefficients analytically rather than obtaining them from a sampled spatial grid. This removes interface rasterization error. The electromagnetic fields are still represented by a finite Fourier expansion, so Fourier truncation error remains and should be checked by increasing the retained orders.

## Fourier coefficients and matrix construction

For inside value $a$, outside value $b$, fill fraction $f$ in $[0,1]$, and center $c$, the Fourier convention and coefficients are

$$
\widehat g_{m,n}=\int_0^1\int_0^1 g(u,v)e^{-2\pi i(mu+nv)}\,du\,dv
=\left[b\delta_{m0}+(a-b)f\operatorname{sinc}(mf)e^{-2\pi imc}\right]\delta_{n0}.
$$

Here sinc is normalized:

```text
sinc(x) = sin(pi*x)/(pi*x)
```

The convolution-matrix entry for a row and column uses the **difference** of their reciprocal indices. Those differences can lie outside the reciprocal orders retained for the field expansion, so constructing the matrix requires coefficients for the full set of pairwise differences rather than only the field-order set itself.

The formula follows by integrating the binary profile in Lifeng Li, chapter 13, Eq. (13.1), of [*Gratings: Theory and Numeric Applications*, second revisited edition (2014)](https://www.fresnel.fr/files/gratings/Second-Edition/Chapter13.pdf).

Material values, fill fraction, and center support broadcasting. The resulting calculations are compatible with JIT compilation and differentiation.

## Usage

The convolution matrix can be constructed directly from an `fmmax.basis.Expansion`:

```python
import jax.numpy as jnp
from fmmax import basis, fft

expansion = basis.Expansion(jnp.array([[0, 0], [1, 0], [-1, 0]]))
epsilon = fft.binary_lamellar_convolution_matrix(
    value=jnp.asarray(4.0 + 0.1j),
    background=jnp.asarray(1.0),
    fill_fraction=jnp.asarray(0.4),
    center=jnp.asarray(0.0),
    expansion=expansion,
)
```

For inverse Fourier factorization, construct the reciprocal-material coefficients separately. In general,

```text
C(1/epsilon) != inv(C(epsilon))
```

as discussed in Eqs. (13.15)–(13.18). The two matrices can then be passed to `fmmax.fmm.eigensolve_isotropic_media_from_convolution_matrices`:

```python
from fmmax import fmm

inverse_epsilon = fft.binary_lamellar_convolution_matrix(
    value=jnp.asarray(1 / (4.0 + 0.1j)),
    background=jnp.asarray(1.0),
    fill_fraction=jnp.asarray(0.4),
    center=jnp.asarray(0.0),
    expansion=expansion,
)
layer = fmm.eigensolve_isotropic_media_from_convolution_matrices(
    wavelength=jnp.asarray(0.73),
    in_plane_wavevector=jnp.asarray([0.0, 0.0]),
    primitive_lattice_vectors=basis.LatticeVectors(
        u=jnp.asarray([1.0, 0.0]), v=jnp.asarray([0.0, 1.0])
    ),
    permittivity_matrix=epsilon,
    inverse_permittivity_matrix=inverse_epsilon,
    expansion=expansion,
    # Interfaces are parallel to y, so their Cartesian tangent is (0, 1).
    tangent_vector=jnp.asarray([0.0, 1.0]),
)
```

## Interpreting the factorization choice

Omitting `tangent_vector` reproduces the direct FFT factorization. Supplying a constant tangent applies the inverse rule to the electric-field component normal to the interfaces. This form is appropriate for straight, parallel interfaces; spatially varying interface directions require a sampled vector formulation.

The tangent is a Cartesian direction. On a skew lattice, the direction of a stripe expressed in fractional coordinates is not automatically the Cartesian interface direction, so it must be converted explicitly.

## Validation and convergence

As an independent regression, the analytic inverse-rule path reproduces the chapter-10 §10.6 TM grating result. For period and wavelength 1, rod permittivity 12.96, fill 0.28, thickness `1 / (2*sqrt(2))`, and 45-degree incidence, the zeroth reflected-order efficiency converges from 0.94651 with 41 Fourier terms to 0.94838 with 81 terms, toward the four-digit published value 0.9487. Total propagating flux is conserved.

The TE Fourier sequence is substantially slower for this discontinuous profile and is not asserted against the published exact-modal value. This distinction illustrates the role of the analytic material representation: it removes interface rasterization error, but it does not turn the Fourier-modal calculation into the exact modal method of chapter 10 or eliminate modal truncation error.
