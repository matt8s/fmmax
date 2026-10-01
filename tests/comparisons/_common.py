"""Shared normal-incidence, air-clad optical comparison conventions."""

import jax
import jax.numpy as jnp
import numpy as np

from fmmax import basis, fields, fmm, scattering

jax.config.update("jax_enable_x64", True)


def orders_1d(half_width):
    return np.column_stack(
        (np.arange(-half_width, half_width + 1), np.zeros(2 * half_width + 1, int))
    )


def fmmax_result(
    epsilon,
    wavelength,
    thickness,
    half_width,
    polarization,
    polar_angle_degrees=0.0,
    azimuthal_angle_degrees=0.0,
):
    expansion = basis.Expansion(basis_coefficients=orders_1d(half_width))
    lattice = basis.LatticeVectors(u=jnp.array([1.0, 0.0]), v=jnp.array([0.0, 1.0]))

    polar_angle = jnp.deg2rad(polar_angle_degrees)
    azimuthal_angle = jnp.deg2rad(azimuthal_angle_degrees)
    in_plane_wavevector = jnp.asarray(
        [
            2 * jnp.pi / wavelength * jnp.sin(polar_angle) * jnp.cos(azimuthal_angle),
            2 * jnp.pi / wavelength * jnp.sin(polar_angle) * jnp.sin(azimuthal_angle),
        ]
    )

    def solve(eps):
        eps = jnp.asarray(eps, dtype=complex)
        if eps.ndim == 0:
            eps = eps.reshape(1, 1)
        return fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=lattice,
            permittivity=eps,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )

    layers = [solve(1.0), solve(epsilon), solve(1.0)]
    smat = scattering.stack_s_matrix(
        layers, [jnp.asarray(0.0), jnp.asarray(thickness), jnp.asarray(0.0)]
    )
    n = expansion.num_terms
    identity = jnp.eye(2 * n, dtype=complex)
    electric, _ = fields.fields_from_wave_amplitudes(
        identity, jnp.zeros_like(identity), layers[0]
    )
    electric_map = jnp.concatenate(electric[:2], axis=0)
    component = 1 if polarization == "TE" else 0
    polarization_xy = (
        jnp.asarray([-jnp.sin(azimuthal_angle), jnp.cos(azimuthal_angle)])
        if polarization == "TE"
        else jnp.asarray([jnp.cos(azimuthal_angle), jnp.sin(azimuthal_angle)])
    )
    target = jnp.zeros((2 * n, 1), dtype=complex)
    target = target.at[half_width, 0].set(polarization_xy[0])
    target = target.at[half_width + n, 0].set(polarization_xy[1])
    incident = jnp.linalg.solve(electric_map, target)
    reflected, transmitted = smat.s21 @ incident, smat.s11 @ incident
    zero = jnp.zeros_like(incident)
    pin, _ = fields.amplitude_poynting_flux(incident, zero, layers[0])
    _, pr = fields.amplitude_poynting_flux(zero, reflected, layers[0])
    pt, _ = fields.amplitude_poynting_flux(transmitted, zero, layers[-1])
    er, _ = fields.fields_from_wave_amplitudes(zero, reflected, layers[0])
    et, _ = fields.fields_from_wave_amplitudes(transmitted, zero, layers[-1])
    return (
        np.asarray(-(pr[:n, 0] + pr[n:, 0]) / jnp.sum(pin)),
        np.asarray((pt[:n, 0] + pt[n:, 0]) / jnp.sum(pin)),
        np.asarray(er[component][:, 0]),
        np.asarray(et[component][:, 0]),
    )
