"""Tests for exact-modal binary-lamellar building blocks."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from fmmax import exact_modal

jax.config.update("jax_enable_x64", True)


def test_segment_matrix_matches_independent_matrix_exponential():
    for t in (3.2, 0.0, -4.7):
        width, nu = 0.37, 2.3
        matrix, _, log_scale = exact_modal._scaled_segment_matrix(
            jnp.asarray(t), jnp.asarray(width), jnp.asarray(nu)
        )
        generator = np.array([[0.0, nu], [-t / nu, 0.0]])
        expected = np.exp(log_scale) * expm(width * generator)
        np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-12)


def test_zero_beta_matrix_and_derivative_limits():
    width, nu = 0.37, 2.3
    matrix, derivative, log_scale = exact_modal._scaled_segment_matrix(
        jnp.asarray(0.0), jnp.asarray(width), jnp.asarray(nu)
    )
    np.testing.assert_allclose(
        matrix, [[1.0, nu * width], [0.0, 1.0]], rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        derivative,
        [[width**2 / 2, nu * width**3 / 6], [width / nu, width**2 / 2]],
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_array_equal(log_scale, 0.0)


def test_uniform_medium_has_analytic_roots():
    wavelength, period, epsilon, alpha = 0.83, 1.2, 2.7, 0.31
    k0 = 2 * np.pi / wavelength
    for polarization in exact_modal.Polarization:
        for order in range(-3, 4):
            eigenvalue_squared = (
                k0**2 * epsilon - (alpha + 2 * np.pi * order / period) ** 2
            )
            residual, _, _ = exact_modal.binary_lamellar_characteristic(
                eigenvalue_squared=jnp.asarray(eigenvalue_squared),
                wavelength=jnp.asarray(wavelength),
                period=jnp.asarray(period),
                bloch_wavevector=jnp.asarray(alpha),
                permittivities=(jnp.asarray(epsilon), jnp.asarray(epsilon)),
                fill_fraction=jnp.asarray(0.37),
                polarization=polarization,
            )
            np.testing.assert_allclose(residual, 0.0, atol=5e-13)


def test_scaled_derivative_matches_finite_difference_without_scaling():
    kwargs = dict(
        wavelength=jnp.asarray(0.83),
        period=jnp.asarray(1.2),
        bloch_wavevector=jnp.asarray(0.31),
        permittivities=(jnp.asarray(2.7), jnp.asarray(1.3)),
        fill_fraction=jnp.asarray(0.37),
        polarization=exact_modal.Polarization.TM,
    )
    eigenvalue_squared = jnp.asarray(20.0)  # Both segment solutions oscillatory.
    residual, derivative, log_scale = exact_modal.binary_lamellar_characteristic(
        eigenvalue_squared=eigenvalue_squared, **kwargs
    )
    np.testing.assert_array_equal(log_scale, 0.0)
    for step in (1e-3, 3e-4, 1e-4):
        upper = exact_modal.binary_lamellar_characteristic(
            eigenvalue_squared=eigenvalue_squared + step, **kwargs
        )[0]
        lower = exact_modal.binary_lamellar_characteristic(
            eigenvalue_squared=eigenvalue_squared - step, **kwargs
        )[0]
        np.testing.assert_allclose(derivative, (upper - lower) / (2 * step), rtol=2e-7)
    assert np.isfinite(residual)


def test_strongly_evanescent_evaluation_is_finite_and_symmetric():
    kwargs = dict(
        eigenvalue_squared=jnp.asarray(1e8),
        wavelength=jnp.asarray(0.83),
        period=jnp.asarray(1.2),
        permittivities=(jnp.asarray(12.96), jnp.asarray(1.0)),
        fill_fraction=jnp.asarray(0.28),
        polarization=exact_modal.Polarization.TM,
    )
    result_positive = exact_modal.binary_lamellar_characteristic(
        bloch_wavevector=jnp.asarray(0.31), **kwargs
    )
    result_negative = exact_modal.binary_lamellar_characteristic(
        bloch_wavevector=jnp.asarray(-0.31), **kwargs
    )
    assert all(np.all(np.isfinite(x)) for x in result_positive)
    for positive, negative in zip(result_positive, result_negative):
        np.testing.assert_allclose(positive, negative)


def test_scaled_derivative_is_scale_times_unscaled_derivative():
    wavelength, period, alpha, fill = 0.83, 1.2, 0.31, 0.37
    epsilon = (2.7, 1.3)
    eigenvalue_squared = 500.0

    def unscaled(value):
        matrices = []
        for material, width in zip(epsilon, (fill * period, (1 - fill) * period)):
            t = (2 * np.pi / wavelength) ** 2 * material - value
            beta = np.sqrt(t + 0j)
            c = np.cos(width * beta)
            s = np.sin(width * beta) / beta
            matrices.append(np.array([[c, material * s], [-t * s / material, c]]))
        return np.trace(matrices[1] @ matrices[0]) - 2 * np.cos(alpha * period)

    _, derivative, log_scale = exact_modal.binary_lamellar_characteristic(
        eigenvalue_squared=jnp.asarray(eigenvalue_squared),
        wavelength=jnp.asarray(wavelength),
        period=jnp.asarray(period),
        bloch_wavevector=jnp.asarray(alpha),
        permittivities=tuple(map(jnp.asarray, epsilon)),
        fill_fraction=jnp.asarray(fill),
        polarization=exact_modal.Polarization.TM,
    )
    step = 1e-4
    expected = (
        np.exp(log_scale)
        * (unscaled(eigenvalue_squared + step) - unscaled(eigenvalue_squared - step))
        / (2 * step)
    )
    np.testing.assert_allclose(derivative, expected.real, rtol=2e-8, atol=1e-12)


def test_float32_cutoff_derivative_and_reverse_mode_are_finite():
    width, nu = 0.7, 2.3
    for t in (-0.21, -1e-4, 1e-4, 0.21):
        _, derivative32, _ = exact_modal._scaled_segment_matrix(
            jnp.asarray(t, dtype=jnp.float32),
            jnp.asarray(width, dtype=jnp.float32),
            jnp.asarray(nu, dtype=jnp.float32),
        )
        _, derivative64, _ = exact_modal._scaled_segment_matrix(
            jnp.asarray(t, dtype=jnp.float64),
            jnp.asarray(width, dtype=jnp.float64),
            jnp.asarray(nu, dtype=jnp.float64),
        )
        np.testing.assert_allclose(derivative32, derivative64, rtol=2e-5, atol=2e-6)

        gradient = jax.grad(
            lambda value: jnp.sum(
                exact_modal._scaled_segment_matrix(value, width, nu)[0]
            )
        )(jnp.asarray(t))
        assert np.isfinite(gradient)


def test_jit_and_batching():
    fn = jax.jit(
        lambda value: exact_modal.binary_lamellar_characteristic(
            eigenvalue_squared=value,
            wavelength=jnp.asarray(0.83),
            period=jnp.asarray(1.2),
            bloch_wavevector=jnp.asarray(0.31),
            permittivities=(jnp.asarray(2.7), jnp.asarray(1.3)),
            fill_fraction=jnp.asarray(0.37),
            polarization=exact_modal.Polarization.TE,
        )
    )
    result = fn(jnp.asarray([10.0, 20.0, 1e8]))
    assert all(x.shape == (3,) for x in result)
    assert all(np.all(np.isfinite(x)) for x in result)
