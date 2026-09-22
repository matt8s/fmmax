"""Exact-modal building blocks for binary lamellar media."""

import enum
from typing import Tuple

import jax
import jax.numpy as jnp


@enum.unique
class Polarization(enum.Enum):
    """Polarization for a nonconical one-dimensional exact-modal problem."""

    TE = "te"
    TM = "tm"


def binary_lamellar_characteristic(
    eigenvalue_squared: jnp.ndarray,
    wavelength: jnp.ndarray,
    period: jnp.ndarray,
    bloch_wavevector: jnp.ndarray,
    permittivities: Tuple[jnp.ndarray, jnp.ndarray],
    fill_fraction: jnp.ndarray,
    polarization: Polarization,
    permeabilities: Tuple[jnp.ndarray, jnp.ndarray] = (
        jnp.asarray(1.0),
        jnp.asarray(1.0),
    ),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluates the binary-lamellar exact-mode characteristic equation.

    This implements the trace condition of chapter 10, equations (10.85)--
    (10.91), in Popov (ed.), *Gratings: Theory and Numeric Applications*, second
    revisited edition (2014). Materials must be positive real scalars and the
    incidence nonconical. The spectral variable is the squared longitudinal
    wavevector.

    The returned residual and spectral derivative share a positive exponential
    scale that prevents overflow for strongly evanescent segment solutions. The
    second result is the scale times the derivative of the *unscaled*
    characteristic, not the derivative of the scaled residual. Their ratio is
    therefore suitable for a Newton update. Root completeness and exact-mode
    scattering are outside this evaluator's scope.

    Args:
        eigenvalue_squared: Spectral parameter, the squared longitudinal wavevector.
        wavelength: Free-space wavelength.
        period: Grating period.
        bloch_wavevector: Bloch wavevector along the periodic direction.
        permittivities: Relative permittivity in the two lamellar segments.
        fill_fraction: Width of the first segment divided by `period`.
        polarization: TE or TM polarization.
        permeabilities: Relative permeability in the two segments.

    Returns:
        The scaled characteristic residual, scaled spectral derivative, and
        logarithm of the positive scale factor.
    """
    if not isinstance(polarization, Polarization):
        raise ValueError(f"Unsupported polarization {polarization!r}.")
    epsilon_1, epsilon_2 = permittivities
    mu_1, mu_2 = permeabilities
    nu_1, nu_2 = (
        (mu_1, mu_2) if polarization is Polarization.TE else (epsilon_1, epsilon_2)
    )
    width_1 = fill_fraction * period
    width_2 = (1 - fill_fraction) * period
    angular_frequency = 2 * jnp.pi / wavelength
    t_1 = angular_frequency**2 * epsilon_1 * mu_1 - eigenvalue_squared
    t_2 = angular_frequency**2 * epsilon_2 * mu_2 - eigenvalue_squared
    p_1, dp_1, log_scale_1 = _scaled_segment_matrix(t_1, width_1, nu_1)
    p_2, dp_2, log_scale_2 = _scaled_segment_matrix(t_2, width_2, nu_2)
    scale = jnp.exp(log_scale_1 + log_scale_2)
    target_trace = 2 * scale * jnp.cos(bloch_wavevector * period)
    residual = jnp.trace(p_2 @ p_1, axis1=-2, axis2=-1) - target_trace
    derivative = jnp.trace(
        dp_2 @ p_1 + p_2 @ dp_1,
        axis1=-2,
        axis2=-1,
    )
    return residual, derivative, log_scale_1 + log_scale_2


def _scaled_segment_matrix(
    t: jnp.ndarray,
    width: jnp.ndarray,
    nu: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns scaled segment propagation and its scaled spectral derivative."""
    t, width, nu = jnp.broadcast_arrays(
        jnp.asarray(t), jnp.asarray(width), jnp.asarray(nu)
    )
    dtype = jnp.result_type(t.dtype, width.dtype, nu.dtype, jnp.float32)
    t, width, nu = t.astype(dtype), width.astype(dtype), nu.astype(dtype)
    near_zero = jnp.abs(t) * width**2 < 0.1
    negative = t < 0
    positive_regular = (t > 0) & ~near_zero
    b = jnp.where(negative, jnp.sqrt(jnp.where(negative, -t, 1)), 0)
    log_scale = -width * b
    scale = jnp.exp(log_scale)

    positive_beta = jnp.where(
        positive_regular,
        jnp.sqrt(jnp.where(positive_regular, t, 1)),
        0,
    )
    c_positive = jnp.cos(width * positive_beta)
    s_positive = width * jnp.sinc(width * positive_beta / jnp.pi)
    a = width * b
    c_negative = -jnp.expm1(-2 * a) / 2 + jnp.exp(-2 * a)
    b_safe = jnp.where(b == 0, 1, b)
    s_negative = -jnp.expm1(-2 * a) / (2 * b_safe)

    c = jnp.where(t < 0, c_negative, c_positive)
    s = jnp.where(t < 0, s_negative, s_positive)
    # Near cutoff, use entire power series before applying the scale. This avoids
    # evaluating the removable division in the derivative formula.
    c_series = (
        1
        - t * width**2 / 2
        + t**2 * width**4 / 24
        - t**3 * width**6 / 720
        + t**4 * width**8 / 40320
    )
    s_series = (
        width
        - t * width**3 / 6
        + t**2 * width**5 / 120
        - t**3 * width**7 / 5040
        + t**4 * width**9 / 362880
    )
    c = jnp.where(near_zero, scale * c_series, c)
    s = jnp.where(near_zero, scale * s_series, s)

    t_safe = jnp.where(near_zero, 1, t)
    dc = width * s / 2
    ds = (s - width * c) / (2 * t_safe)
    dc_series = scale * (
        width**2 / 2
        - t * width**4 / 12
        + t**2 * width**6 / 240
        - t**3 * width**8 / 10080
        + t**4 * width**10 / 725760
    )
    ds_series = scale * (
        width**3 / 6
        - t * width**5 / 60
        + t**2 * width**7 / 1680
        - t**3 * width**9 / 90720
        + t**4 * width**11 / 7983360
    )
    dc = jnp.where(near_zero, dc_series, dc)
    ds = jnp.where(near_zero, ds_series, ds)

    matrix = jnp.stack(
        (jnp.stack((c, nu * s), axis=-1), jnp.stack((-t * s / nu, c), axis=-1)),
        axis=-2,
    )
    derivative = jnp.stack(
        (
            jnp.stack((dc, nu * ds), axis=-1),
            jnp.stack(((s - t * ds) / nu, dc), axis=-1),
        ),
        axis=-2,
    )
    return matrix, derivative, log_scale


jax.tree_util.register_pytree_node(
    Polarization,
    lambda x: ((), x.value),
    lambda value, _: Polarization(value),
)
