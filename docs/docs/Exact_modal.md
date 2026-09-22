# Exact-modal building blocks

`fmmax.exact_modal.binary_lamellar_characteristic` evaluates the transverse
characteristic equation for a positive-real binary lamellar medium at
nonconical incidence. It implements the trace condition in Boris Gralak,
chapter 10, Eqs. (10.85)–(10.91), of [*Gratings: Theory and Numeric Applications*,
second revisited edition (2014)](https://www.fresnel.fr/files/gratings/Second-Edition/Chapter10.pdf).

The spectral parameter is the squared longitudinal wavevector. TE uses the
permeability continuity parameter and TM the permittivity parameter. The
implementation uses entire sine/cosine combinations at transverse cutoff and
exponential scaling for strongly evanescent segment solutions.

```python
import jax.numpy as jnp
from fmmax import exact_modal

residual, scaled_derivative, log_scale = (
    exact_modal.binary_lamellar_characteristic(
        eigenvalue_squared=jnp.asarray(20.0),
        wavelength=jnp.asarray(1.0),
        period=jnp.asarray(1.0),
        bloch_wavevector=jnp.asarray(jnp.pi / 4),
        permittivities=(jnp.asarray(12.96), jnp.asarray(1.0)),
        fill_fraction=jnp.asarray(0.28),
        polarization=exact_modal.Polarization.TM,
    )
)
```

The residual and derivative have the same positive scale. Their ratio therefore
gives the unscaled Newton correction, but `scaled_derivative` is deliberately
not described as the derivative of `residual` where the scale varies.

This is a bounded exact-modal building block, not a complete solver. It does not
claim to find all roots, reconstruct exact modes, compute overlaps, or produce
scattering efficiencies. A future root API must provide finite search intervals,
multiplicity-aware isolation, unresolved-cluster reporting, and an independently
validated root count before claiming completeness. Exact-mode scattering also
requires visually verified overlap and interface equations from the original
chapter pages.

The chapter's §10.6 scattering benchmark is presently exercised through the
independent analytic Fourier-matrix solver for TM polarization. It validates the
material factorization and scattering stack, not this characteristic evaluator
or any future exact-root completeness claim.
