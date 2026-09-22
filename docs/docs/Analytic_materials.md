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

These are material-matrix utilities. The existing public layer eigensolvers
still accept sampled material arrays; direct analytic-matrix solver input is
a subsequent extension. Analytic coefficients eliminate rasterization error,
but do not provide the exact modal method of chapter 10 or eliminate Fourier
truncation error.
