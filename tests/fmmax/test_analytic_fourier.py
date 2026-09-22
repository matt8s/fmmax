"""Independent checks of analytic binary-lamellar material coefficients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmmax import basis, fft

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.complex64, jnp.float64, jnp.complex128]
)
def test_dtypes_and_parameter_broadcasting(dtype):
    real_dtype = jnp.asarray(0, dtype=dtype).real.dtype
    value = jnp.asarray(4, dtype=dtype)
    fills = jnp.asarray([[0.2], [0.7]], dtype=real_dtype)
    centers = jnp.asarray([0.0, 0.1, 0.2], dtype=real_dtype)
    orders = jnp.array([[0, 0], [1, 0]])
    fn = jax.jit(fft.binary_lamellar_fourier_coefficients)
    actual = fn(value, jnp.ones_like(value), fills, centers, orders)
    assert actual.shape == (2, 3, 2)
    assert actual.dtype == jnp.promote_types(dtype, jnp.complex64)
    np.testing.assert_allclose(
        actual[..., 0], np.broadcast_to(1 + 3 * fills, (2, 3)), rtol=1e-6
    )


@pytest.mark.parametrize("orders", [jnp.zeros((3,)), jnp.zeros((2, 2))])
def test_invalid_orders(orders):
    with pytest.raises(ValueError, match="orders"):
        fft.binary_lamellar_fourier_coefficients(4.0, 1.0, 0.4, 0.0, orders)


def test_integral_and_translation():
    orders = np.array([[0, 0], [1, 0], [-3, 0], [2, 1]])
    value, background, fill, center = 4 + 0.2j, 1.0, 0.37, 0.23
    actual = fft.binary_lamellar_fourier_coefficients(
        value, background, fill, center, orders
    )
    # Independent midpoint quadrature over the stripe, including a full-cell
    # background integral. This checks Fourier phase and normalization.
    u = center - fill / 2 + (np.arange(20000) + 0.5) * fill / 20000
    expected = []
    for m, n in orders:
        integral = fill * np.mean(np.exp(-2j * np.pi * m * u))
        expected.append(
            ((value - background) * integral + background * (m == 0)) * (n == 0)
        )
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    translated = fft.binary_lamellar_fourier_coefficients(
        value, background, fill, center + 0.17, orders
    )
    np.testing.assert_allclose(
        translated, actual * np.exp(-2j * np.pi * orders[:, 0] * 0.17)
    )


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_uniform_limits_and_indexing(fill):
    expansion = basis.Expansion(np.array([[2, 0], [0, 1], [-1, 0], [0, 0]]))
    actual = fft.binary_lamellar_convolution_matrix(4.0, 1.0, fill, 0.21, expansion)
    np.testing.assert_allclose(actual, np.eye(4) * (1 + 3 * fill), atol=1e-14)


def test_hermiticity_batching_and_inverse_material():
    expansion = basis.Expansion(np.array([[2, 0], [0, 1], [-1, 0], [0, 0]]))
    fn = jax.jit(
        lambda v: fft.binary_lamellar_convolution_matrix(v, 1.0, 0.4, 0.13, expansion)
    )
    matrices = fn(jnp.array([2.0, 4.0]))
    assert matrices.shape == (2, 4, 4)
    np.testing.assert_allclose(matrices, np.swapaxes(matrices.conj(), -1, -2))
    np.testing.assert_allclose(matrices[:, 1, [0, 2, 3]], 0.0, atol=0)
    # Row (2,0) minus column (-1,0) is (3,0). Check the phase independently:
    # Hermiticity alone would not detect reversing the Toeplitz difference.
    expected = 3 * np.sin(3 * np.pi * 0.4) / (3 * np.pi) * np.exp(-6j * np.pi * 0.13)
    np.testing.assert_allclose(matrices[1, 0, 2], expected, atol=1e-12)
    reciprocal = fft.binary_lamellar_convolution_matrix(0.25, 1.0, 0.4, 0.13, expansion)
    assert not np.allclose(reciprocal, np.linalg.inv(matrices[1]))


def test_parameter_gradients_and_periodicity():
    orders = jnp.array([[0, 0], [1, 0], [-3, 0]])
    fn = lambda f, c: fft.binary_lamellar_fourier_coefficients(4.0, 1.0, f, c, orders)
    fill, center = 0.37, 0.23
    derivative_fill = jax.jacfwd(fn, 0)(fill, center)
    derivative_center = jax.jacfwd(fn, 1)(fill, center)
    m = np.asarray(orders[:, 0])
    np.testing.assert_allclose(
        derivative_fill,
        3 * np.cos(np.pi * m * fill) * np.exp(-2j * np.pi * m * center),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        derivative_center, -2j * np.pi * m * (fn(fill, center) - (m == 0)), atol=1e-12
    )
    np.testing.assert_allclose(fn(fill, center), fn(fill, center + 1), atol=1e-12)
