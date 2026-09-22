# Analytic lamellar material coefficients

`fmmax.fft.binary_lamellar_fourier_coefficients` and
`binary_lamellar_convolution_matrix` describe a periodic binary stripe without
rasterizing its interfaces. The stripe is invariant along the second fractional
lattice coordinate; width and center are fractions of the first lattice period.

For inside value `a`, outside value `b`, fill fraction `f` in `[0, 1]`, and center
`c`, the Fourier convention and coefficients are

$$
\widehat g_{m,n}=\int_0^1\int_0^1 g(u,v)e^{-2\pi i(mu+nv)}\,du\,dv
=\left[b\delta_{m0}+(a-b)f\operatorname{sinc}(mf)e^{-2\pi imc}\right]\delta_{n0}.
$$

Here sinc is normalized: `sinc(x) = sin(pi*x)/(pi*x)`. A convolution-matrix entry
uses the **difference** of the row and column reciprocal indices, including
differences outside the retained field-order set. Material values, fill and
center support broadcasting, JIT compilation and differentiation.

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

The formula follows by integrating the binary profile in Lifeng Li, chapter 13,
Eq. (13.1), of [*Gratings: Theory and Numeric Applications*, second revisited
edition (2014)](https://www.fresnel.fr/files/gratings/Second-Edition/Chapter13.pdf).
For inverse Fourier factorization, compute the coefficients of the reciprocal
material separately: `C(1/epsilon)` is generally not `inv(C(epsilon))`; see
Eqs. (13.15)–(13.18).

The matrices can be passed directly to
`fmmax.fmm.eigensolve_isotropic_media_from_convolution_matrices`. The reciprocal
material matrix must be supplied separately:

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

Omitting `tangent_vector` reproduces the direct FFT factorization. A constant
tangent applies the inverse rule to the normal electric-field component and is
restricted to straight parallel interfaces. Use a sampled vector formulation
for spatially varying interface directions. On a skew lattice, the fractional
stripe direction is not automatically a Cartesian interface direction.

Analytic coefficients eliminate rasterization error, but do not provide the
exact modal method of chapter 10 or eliminate Fourier truncation error.

As an independent regression, the analytic inverse-rule path reproduces the
chapter-10 §10.6 TM grating result. For period and wavelength 1, rod permittivity
12.96, fill 0.28, thickness `1 / (2*sqrt(2))`, and 45-degree incidence, the
zeroth reflected-order efficiency converges from 0.94651 with 41 Fourier terms
to 0.94838 with 81 terms, toward the four-digit published value 0.9487. Total
propagating flux is conserved. The TE Fourier sequence is substantially slower
for this discontinuous profile and is not asserted against the published exact-
modal value; analytic material coefficients do not remove modal truncation error.
