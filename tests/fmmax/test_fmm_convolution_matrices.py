"""Tests for eigensolves from explicit material convolution matrices."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fmmax import basis, fft, fields, fmm, fmm_matrices, scattering, utils

jax.config.update("jax_enable_x64", True)

EXPANSION = basis.Expansion(
    basis_coefficients=np.column_stack((np.arange(-2, 3), np.zeros(5, int)))
)
LATTICE = basis.LatticeVectors(u=jnp.array([1.0, 0.0]), v=jnp.array([0.0, 1.0]))


def _sort(values):
    return jnp.take_along_axis(values, jnp.argsort(jnp.abs(values), axis=-1), axis=-1)


def _matrices(permittivity):
    return (
        fft.fourier_convolution_matrix(permittivity, EXPANSION),
        fft.fourier_convolution_matrix(1 / permittivity, EXPANSION),
    )


def test_matches_sampled_fft_path_and_preserves_inverse_matrix():
    x = np.arange(128) / 128
    permittivity = jnp.asarray(
        np.where(((x - 0.17) % 1) < 0.37, 3.4 + 0.2j, 1.2 + 0.05j)[:, None]
    )
    eps_matrix, eta_matrix = _matrices(permittivity)
    expected = fmm.eigensolve_isotropic_media(
        wavelength=jnp.asarray(0.83),
        in_plane_wavevector=jnp.asarray([0.21, 0.07]),
        primitive_lattice_vectors=LATTICE,
        permittivity=permittivity,
        expansion=EXPANSION,
        formulation=fmm.Formulation.FFT,
    )
    actual = fmm.eigensolve_isotropic_media_from_convolution_matrices(
        wavelength=jnp.asarray(0.83),
        in_plane_wavevector=jnp.asarray([0.21, 0.07]),
        primitive_lattice_vectors=LATTICE,
        permittivity_matrix=eps_matrix,
        inverse_permittivity_matrix=eta_matrix,
        expansion=EXPANSION,
    )
    np.testing.assert_allclose(
        _sort(actual.eigenvalues**2), _sort(expected.eigenvalues**2)
    )
    np.testing.assert_allclose(
        actual.omega_script_k_matrix, expected.omega_script_k_matrix
    )
    np.testing.assert_array_equal(actual.z_permittivity_matrix, eps_matrix)
    np.testing.assert_array_equal(actual.inverse_z_permittivity_matrix, eta_matrix)


def test_uniform_matrix_limit_matches_uniform_solver():
    epsilon = jnp.asarray(2.7 + 0.1j)
    expected = fmm.eigensolve_isotropic_media(
        wavelength=jnp.asarray(0.83),
        in_plane_wavevector=jnp.asarray([0.21, 0.07]),
        primitive_lattice_vectors=LATTICE,
        permittivity=epsilon.reshape(1, 1),
        expansion=EXPANSION,
        formulation=fmm.Formulation.FFT,
    )
    actual = fmm.eigensolve_isotropic_media_from_convolution_matrices(
        wavelength=jnp.asarray(0.83),
        in_plane_wavevector=jnp.asarray([0.21, 0.07]),
        primitive_lattice_vectors=LATTICE,
        permittivity_matrix=epsilon * jnp.eye(EXPANSION.num_terms),
        inverse_permittivity_matrix=jnp.eye(EXPANSION.num_terms) / epsilon,
        expansion=EXPANSION,
    )
    np.testing.assert_allclose(
        _sort(actual.eigenvalues**2), _sort(expected.eigenvalues**2)
    )
    np.testing.assert_allclose(
        actual.omega_script_k_matrix, expected.omega_script_k_matrix
    )


def test_constant_tangent_matches_sampled_vector_formulation():
    x = np.arange(256) / 256
    permittivity = jnp.asarray(np.where(x < 0.4, 4.0, 1.0)[:, None])
    eps_matrix, eta_matrix = _matrices(permittivity)

    def tangent_vector_field(permittivity, expansion, primitive_lattice_vectors):
        del expansion, primitive_lattice_vectors
        return jnp.zeros_like(permittivity), jnp.ones_like(permittivity)

    expected = fmm.eigensolve_isotropic_media(
        wavelength=jnp.asarray(0.73),
        in_plane_wavevector=jnp.asarray([0.13, 0.0]),
        primitive_lattice_vectors=LATTICE,
        permittivity=permittivity,
        expansion=EXPANSION,
        formulation=tangent_vector_field,
    )
    actual = fmm.eigensolve_isotropic_media_from_convolution_matrices(
        wavelength=jnp.asarray(0.73),
        in_plane_wavevector=jnp.asarray([0.13, 0.0]),
        primitive_lattice_vectors=LATTICE,
        permittivity_matrix=eps_matrix,
        inverse_permittivity_matrix=eta_matrix,
        expansion=EXPANSION,
        tangent_vector=jnp.asarray([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        _sort(actual.eigenvalues**2), _sort(expected.eigenvalues**2)
    )
    np.testing.assert_allclose(
        actual.omega_script_k_matrix, expected.omega_script_k_matrix
    )
    small_tangent = fmm.eigensolve_isotropic_media_from_convolution_matrices(
        wavelength=jnp.asarray(0.73),
        in_plane_wavevector=jnp.asarray([0.13, 0.0]),
        primitive_lattice_vectors=LATTICE,
        permittivity_matrix=eps_matrix,
        inverse_permittivity_matrix=eta_matrix,
        expansion=EXPANSION,
        tangent_vector=jnp.asarray([0.0, 1e-20]),
    )
    np.testing.assert_allclose(
        _sort(small_tangent.eigenvalues**2), _sort(actual.eigenvalues**2)
    )


def test_analytic_lamellar_matrices_remove_sampling_error():
    args = dict(
        fill_fraction=jnp.asarray(0.37),
        center=jnp.asarray(0.13),
        expansion=EXPANSION,
    )
    eps_matrix = fft.binary_lamellar_convolution_matrix(4.0, 1.0, **args)
    eta_matrix = fft.binary_lamellar_convolution_matrix(0.25, 1.0, **args)
    exact = fmm.eigensolve_isotropic_media_from_convolution_matrices(
        wavelength=jnp.asarray(0.73),
        in_plane_wavevector=jnp.asarray([0.13, 0.0]),
        primitive_lattice_vectors=LATTICE,
        permittivity_matrix=eps_matrix,
        inverse_permittivity_matrix=eta_matrix,
        expansion=EXPANSION,
        tangent_vector=jnp.asarray([0.0, 1.0]),
    )
    exact_values = _sort(exact.eigenvalues**2)
    scale = np.max(np.abs(exact_values))
    errors = []
    for samples in (128, 2048):
        x = np.arange(samples) / samples
        distance = (x - 0.13 + 0.5) % 1 - 0.5
        sampled = jnp.asarray(np.where(np.abs(distance) < 0.37 / 2, 4.0, 1.0)[:, None])
        sampled_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(0.73),
            in_plane_wavevector=jnp.asarray([0.13, 0.0]),
            primitive_lattice_vectors=LATTICE,
            permittivity=sampled,
            expansion=EXPANSION,
            formulation=lambda p, e, l: (jnp.zeros_like(p), jnp.ones_like(p)),
        )
        errors.append(
            np.max(np.abs(_sort(sampled_result.eigenvalues**2) - exact_values)) / scale
        )
    assert errors[1] < errors[0]
    assert errors[1] < 5e-4


def test_batching_jit_and_directional_gradient():
    def solve(fill):
        eps_matrix = fft.binary_lamellar_convolution_matrix(
            jnp.asarray([[[4.0], [2.5]]]), 1.0, fill, 0.13, EXPANSION
        )
        eta_matrix = fft.binary_lamellar_convolution_matrix(
            jnp.asarray([[[0.25], [0.4]]]), 1.0, fill, 0.13, EXPANSION
        )
        return fmm.eigensolve_isotropic_media_from_convolution_matrices(
            wavelength=jnp.asarray([0.71, 0.83])[:, None, None],
            in_plane_wavevector=jnp.asarray([0.13, 0.0]),
            primitive_lattice_vectors=LATTICE,
            permittivity_matrix=eps_matrix,
            inverse_permittivity_matrix=eta_matrix,
            expansion=EXPANSION,
            tangent_vector=jnp.asarray([0.0, 1.0]),
        )

    result = jax.jit(solve)(jnp.asarray(0.37))
    assert result.eigenvalues.shape == (2, 2, 1, 10)

    def loss(fill):
        values = solve(fill).eigenvalues
        return jnp.sum(jnp.abs(values) ** 2)

    derivative = jax.grad(loss)(jnp.asarray(0.37))
    for step in (1e-4, 3e-5, 1e-5):
        finite_difference = (loss(0.37 + step) - loss(0.37 - step)) / (2 * step)
        np.testing.assert_allclose(derivative, finite_difference, rtol=2e-4)


def _cusolver_available():
    implementation = getattr(jax.lax.linalg, "EigImplementation", None)
    implementation = getattr(implementation, "CUSOLVER", None)
    return implementation is not None and any(
        device.platform == "gpu" for device in jax.devices()
    )


@pytest.mark.skipif(not _cusolver_available(), reason="cuSOLVER is unavailable")
def test_cusolver_patterned_solve_is_device_native_and_differentiable():
    x = np.arange(256) / 256
    permittivity = jnp.asarray(
        np.where(((x - 0.17) % 1) < 0.37, 3.4 + 0.2j, 1.2 + 0.05j)[:, None]
    )

    def solve(permittivity):
        return fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(0.83),
            in_plane_wavevector=jnp.asarray([0.21, 0.07]),
            primitive_lattice_vectors=LATTICE,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
            eig_backend=utils.EigBackend.CUSOLVER,
        )

    solve_jit = jax.jit(solve)
    actual = solve_jit(permittivity)
    expected = fmm.eigensolve_isotropic_media(
        wavelength=jnp.asarray(0.83),
        in_plane_wavevector=jnp.asarray([0.21, 0.07]),
        primitive_lattice_vectors=LATTICE,
        permittivity=permittivity,
        expansion=EXPANSION,
        formulation=fmm.Formulation.FFT,
    )
    assert next(iter(actual.eigenvalues.devices())).platform == "gpu"
    compiler_ir = str(solve_jit.lower(permittivity).compiler_ir())
    assert "cusolver_geev_ffi" in compiler_ir
    assert "xla_ffi_python_gpu_callback" not in compiler_ir
    np.testing.assert_allclose(
        _sort(actual.eigenvalues**2), _sort(expected.eigenvalues**2), rtol=1e-10
    )

    def loss(fill):
        eps_matrix = fft.binary_lamellar_convolution_matrix(
            4.0, 1.0, fill, 0.13, EXPANSION
        )
        eta_matrix = fft.binary_lamellar_convolution_matrix(
            0.25, 1.0, fill, 0.13, EXPANSION
        )
        result = fmm.eigensolve_isotropic_media_from_convolution_matrices(
            wavelength=jnp.asarray(0.73),
            in_plane_wavevector=jnp.asarray([0.13, 0.0]),
            primitive_lattice_vectors=LATTICE,
            permittivity_matrix=eps_matrix,
            inverse_permittivity_matrix=eta_matrix,
            expansion=EXPANSION,
            tangent_vector=jnp.asarray([0.0, 1.0]),
            eig_backend=utils.EigBackend.CUSOLVER,
        )
        return jnp.sum(jnp.abs(result.eigenvalues) ** 2)

    derivative = jax.jit(jax.grad(loss))(jnp.asarray(0.37))
    for step in (1e-4, 3e-5, 1e-5):
        finite_difference = (loss(0.37 + step) - loss(0.37 - step)) / (2 * step)
        np.testing.assert_allclose(derivative, finite_difference, rtol=2e-4)


@pytest.mark.parametrize(
    "permittivity_shape,inverse_shape,error",
    [
        ((5,), (5, 5), "trailing shape"),
        ((4, 4), (5, 5), "trailing shape"),
        ((2, 5, 5), (3, 5, 5), "batch-compatible"),
    ],
)
def test_matrix_shape_validation(permittivity_shape, inverse_shape, error):
    with pytest.raises(ValueError, match=error):
        fmm.eigensolve_isotropic_media_from_convolution_matrices(
            wavelength=jnp.asarray(0.73),
            in_plane_wavevector=jnp.asarray([0.0, 0.0]),
            primitive_lattice_vectors=LATTICE,
            permittivity_matrix=jnp.ones(permittivity_shape),
            inverse_permittivity_matrix=jnp.ones(inverse_shape),
            expansion=EXPANSION,
        )


def test_transverse_matrix_rejects_broadcasting_over_matrix_axes():
    with pytest.raises(ValueError, match="square"):
        fmm_matrices.transverse_permittivity_from_convolution_matrices(
            jnp.eye(5), jnp.ones((5, 1)), jnp.asarray([0.0, 1.0])
        )


def _chapter_10_tm_reflection(half_width, eig_backend=utils.EigBackend.DEFAULT):
    orders = np.column_stack(
        (np.arange(-half_width, half_width + 1), np.zeros(2 * half_width + 1, int))
    )
    expansion = basis.Expansion(orders)
    wavevector = jnp.asarray([2 * jnp.pi / jnp.sqrt(2), 0.0])

    def uniform():
        return fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(1.0),
            in_plane_wavevector=wavevector,
            primitive_lattice_vectors=LATTICE,
            permittivity=jnp.ones((1, 1)),
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )

    grating = fmm.eigensolve_isotropic_media_from_convolution_matrices(
        wavelength=jnp.asarray(1.0),
        in_plane_wavevector=wavevector,
        primitive_lattice_vectors=LATTICE,
        permittivity_matrix=fft.binary_lamellar_convolution_matrix(
            12.96, 1.0, 0.28, 0.0, expansion
        ),
        inverse_permittivity_matrix=fft.binary_lamellar_convolution_matrix(
            1 / 12.96, 1.0, 0.28, 0.0, expansion
        ),
        expansion=expansion,
        tangent_vector=jnp.asarray([0.0, 1.0]),
        eig_backend=eig_backend,
    )
    layers = [uniform(), grating, uniform()]
    s_matrix = scattering.stack_s_matrix(
        layers,
        [jnp.asarray(0.0), jnp.asarray(1 / (2 * jnp.sqrt(2))), jnp.asarray(0.0)],
    )
    num_terms = expansion.num_terms
    identity = jnp.eye(2 * num_terms, dtype=complex)
    zero = jnp.zeros_like(identity)
    electric, _ = fields.fields_from_wave_amplitudes(identity, zero, layers[0])
    electric_map = jnp.concatenate(electric[:2], axis=0)
    target = jnp.zeros((2 * num_terms, 1), dtype=complex).at[half_width, 0].set(1.0)
    incident = jnp.linalg.solve(electric_map, target)
    reflected, transmitted = s_matrix.s21 @ incident, s_matrix.s11 @ incident
    pin, _ = fields.amplitude_poynting_flux(
        incident, jnp.zeros_like(incident), layers[0]
    )
    _, pr = fields.amplitude_poynting_flux(
        jnp.zeros_like(reflected), reflected, layers[0]
    )
    pt, _ = fields.amplitude_poynting_flux(
        transmitted, jnp.zeros_like(transmitted), layers[-1]
    )
    transmitted_efficiency, _ = fields.diffraction_efficiencies(
        transmitted, jnp.zeros_like(transmitted), layers[-1], jnp.sum(pin)
    )
    _, reflected_efficiency = fields.diffraction_efficiencies(
        jnp.zeros_like(reflected), reflected, layers[0], jnp.sum(pin)
    )
    reflected_by_order = reflected_efficiency[:, 0]
    total = (-jnp.sum(pr) + jnp.sum(pt)) / jnp.sum(pin)
    np.testing.assert_allclose(
        jnp.sum(reflected_efficiency) + jnp.sum(transmitted_efficiency), total
    )
    return reflected_by_order[half_width], total


def test_chapter_10_tm_benchmark_convergence():
    # Gralak, chapter 10, section 10.6 reports R_0(TM)=0.9487 (four digits) for
    # this dielectric grating. This is an independent scattering target, not an
    # exact golden value; verify Fourier-order convergence and lossless flux too.
    coarse, coarse_total = _chapter_10_tm_reflection(20)
    fine, fine_total = _chapter_10_tm_reflection(40)
    assert abs(fine - 0.9487) < abs(coarse - 0.9487)
    np.testing.assert_allclose(fine, 0.9487, atol=5e-4)
    np.testing.assert_allclose(coarse_total, 1.0, atol=1e-10)
    np.testing.assert_allclose(fine_total, 1.0, atol=1e-10)


@pytest.mark.skipif(not _cusolver_available(), reason="cuSOLVER is unavailable")
def test_cusolver_matches_default_scattering_and_flux():
    expected_reflection, expected_total = _chapter_10_tm_reflection(8)
    actual_reflection, actual_total = _chapter_10_tm_reflection(
        8, utils.EigBackend.CUSOLVER
    )
    np.testing.assert_allclose(actual_reflection, expected_reflection, rtol=1e-10)
    np.testing.assert_allclose(actual_total, expected_total, rtol=1e-10)
    np.testing.assert_allclose(actual_total, 1.0, atol=1e-10)

    converged_reflection, converged_total = _chapter_10_tm_reflection(
        40, utils.EigBackend.CUSOLVER
    )
    np.testing.assert_allclose(converged_reflection, 0.9487, atol=5e-4)
    np.testing.assert_allclose(converged_total, 1.0, atol=1e-10)
