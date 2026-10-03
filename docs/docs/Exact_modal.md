# Exact-modal characteristic evaluation

## What the evaluator does

`fmmax.exact_modal.binary_lamellar_characteristic` evaluates the characteristic equation for a positive-real binary lamellar medium at nonconical incidence. It is useful for evaluating a candidate squared longitudinal wavevector and as the characteristic function in a root-search method supplied by the user.

The implementation follows the trace condition in Boris Gralak, chapter 10, Eqs. (10.85)–(10.91), of [*Gratings: Theory and Numeric Applications*, second revisited edition (2014)](https://www.fresnel.fr/files/gratings/Second-Edition/Chapter10.pdf).

## Spectral and polarization parameters

The spectral parameter `eigenvalue_squared` is the squared longitudinal wavevector. For TE polarization, the segment matching condition uses the permeability continuity parameter. For TM polarization, it uses the permittivity parameter.

At transverse cutoff, the evaluator uses entire sine/cosine combinations to avoid removable singularities. Strongly evanescent segment solutions are evaluated with exponential scaling to avoid overflow.

## Usage

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

A root-search procedure can evaluate this function at candidate values of `eigenvalue_squared` and use the returned residual and derivative information to refine those candidates.

## Interpreting the returned values

The function returns the scaled characteristic residual, the correspondingly scaled spectral derivative, and the logarithm of the positive scale factor.

`residual` and `scaled_derivative` have the same positive scale. Their ratio therefore gives the unscaled Newton correction. However, where the scale varies with the spectral parameter, `scaled_derivative` is not the derivative of the scaled `residual`; it is the scale times the derivative of the unscaled characteristic equation.

## Scope and validation

This function is a characteristic evaluator rather than a complete exact-modal solver. Users supply root isolation and decide the finite search interval, multiplicity treatment, and handling of unresolved root clusters. The function does not reconstruct exact modes, compute modal overlaps, or produce scattering efficiencies.

Scattering calculations continue to use the analytic Fourier-matrix path. The chapter's §10.6 scattering benchmark exercises that path for TM polarization and validates its material factorization and scattering stack. It does not by itself validate root completeness for the characteristic evaluator.
