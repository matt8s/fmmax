"""Independent S4 comparisons with matched diffraction-order sets."""

import numpy as np
import pytest

from ._common import fmmax_result, orders_1d

S4 = pytest.importorskip("S4")


def _s4(
    epsilon,
    wavelength,
    thickness,
    half_width,
    polarization,
    patterned=False,
    polar_angle_degrees=0.0,
    azimuthal_angle_degrees=0.0,
):
    sim = S4.New(Lattice=1.0, NumBasis=2 * half_width + 1)
    sim.SetOptions(
        PolarizationDecomposition=False,
        DiscretizedEpsilon=False,
        LanczosSmoothing=False,
        SubpixelSmoothing=False,
    )
    sim.SetMaterial(Name="air", Epsilon=1.0)
    sim.SetMaterial(Name="high", Epsilon=complex(epsilon))
    sim.AddLayer(Name="input", Thickness=0.0, Material="air")
    sim.AddLayer(
        Name="film", Thickness=thickness, Material="air" if patterned else "high"
    )
    if patterned:
        sim.SetRegionRectangle(
            Layer="film",
            Material="high",
            Center=(0.0, 0.0),
            Angle=0.0,
            Halfwidths=(0.2, 0.5),
        )
    sim.AddLayer(Name="output", Thickness=0.0, Material="air")
    sim.SetFrequency(1 / wavelength)
    sim.SetExcitationPlanewave(
        IncidenceAngles=(polar_angle_degrees, azimuthal_angle_degrees),
        sAmplitude=float(polarization == "TE"),
        pAmplitude=float(polarization == "TM"),
        Order=0,
    )
    orders = [tuple(order) for order in sim.GetBasisSet()]
    assert set(orders) == set(map(tuple, orders_1d(half_width)))
    permutation = [orders.index(tuple(order)) for order in orders_1d(half_width)]
    incident, _ = sim.GetPowerFlux(Layer="input")
    reflected = np.asarray(sim.GetPowerFluxByOrder(Layer="input"))[:, 1]
    transmitted = np.asarray(sim.GetPowerFluxByOrder(Layer="output"))[:, 0]
    powers = (
        -reflected.real[permutation] / incident.real,
        transmitted.real[permutation] / incident.real,
    )
    if (
        half_width == 0
        and not patterned
        and polar_angle_degrees == 0
        and azimuthal_angle_degrees == 0
    ):
        # Sample outside interfaces and remove free-space propagation to recover
        # Cartesian electric amplitudes at the entrance and exit reference planes.
        offset = 0.037
        k0 = 2 * np.pi / wavelength
        component = 1 if polarization == "TE" else 0
        electric_in, _ = sim.GetFields(0.0, 0.0, -offset)
        electric_out, _ = sim.GetFields(0.0, 0.0, thickness + offset)
        phase = np.exp(-1j * k0 * offset)
        r = (electric_in[component] - phase) * phase
        t = electric_out[component] * phase
        return (*powers, np.atleast_1d(r), np.atleast_1d(t))
    return powers


@pytest.mark.parametrize("epsilon", [2.25, 2.25 + 0.12j, -5.0 + 0.5j])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_film(epsilon, polarization):
    result = _s4(epsilon, 0.83, 0.19, 0, polarization)
    reference = fmmax_result(epsilon, 0.83, 0.19, 0, polarization)
    for actual, expected in zip(result, reference):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("epsilon", [2.25, 2.25 + 0.12j, -5.0 + 0.5j])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_oblique_film(epsilon, polarization):
    angle = 31.0
    result = _s4(
        epsilon,
        0.83,
        0.19,
        0,
        polarization,
        polar_angle_degrees=angle,
    )
    reference = fmmax_result(
        epsilon,
        0.83,
        0.19,
        0,
        polarization,
        polar_angle_degrees=angle,
    )
    for actual, expected in zip(result, reference[:2]):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
@pytest.mark.parametrize("epsilon", [4.0, -5.0 + 0.5j])
def test_grating_sampling_convergence(polarization, epsilon):
    reference = _s4(epsilon, 0.73, 0.23, 5, polarization, patterned=True)
    errors = []
    for samples in (256, 4096, 16384):
        x = (np.arange(samples) / samples + 0.5) % 1 - 0.5
        grid = np.where(np.abs(x) < 0.2, epsilon, 1.0)[:, None]
        result = fmmax_result(grid, 0.73, 0.23, 5, polarization)
        errors.append(max(np.max(np.abs(a - b)) for a, b in zip(result[:2], reference)))
    # S4 uses analytic geometry; refinement controls the sampled-interface error.
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 1e-3


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_oblique_dielectric_grating_sampling_convergence(polarization):
    angle = 27.0
    reference = _s4(
        4.0,
        0.73,
        0.23,
        5,
        polarization,
        patterned=True,
        polar_angle_degrees=angle,
    )
    errors = []
    for samples in (256, 4096, 16384):
        x = (np.arange(samples) / samples + 0.5) % 1 - 0.5
        grid = np.where(np.abs(x) < 0.2, 4.0, 1.0)[:, None]
        result = fmmax_result(
            grid,
            0.73,
            0.23,
            5,
            polarization,
            polar_angle_degrees=angle,
        )
        errors.append(max(np.max(np.abs(a - b)) for a, b in zip(result[:2], reference)))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 1e-3


@pytest.mark.parametrize("epsilon", [2.25, 2.25 + 0.12j, -5.0 + 0.5j])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_conical_film(epsilon, polarization):
    polar_angle = 31.0
    azimuthal_angle = 23.0
    result = _s4(
        epsilon,
        0.83,
        0.19,
        0,
        polarization,
        polar_angle_degrees=polar_angle,
        azimuthal_angle_degrees=azimuthal_angle,
    )
    reference = fmmax_result(
        epsilon,
        0.83,
        0.19,
        0,
        polarization,
        polar_angle_degrees=polar_angle,
        azimuthal_angle_degrees=azimuthal_angle,
    )
    for actual, expected in zip(result, reference[:2]):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_conical_dielectric_grating_sampling_convergence(polarization):
    polar_angle = 27.0
    azimuthal_angle = 19.0
    reference = _s4(
        4.0,
        0.73,
        0.23,
        5,
        polarization,
        patterned=True,
        polar_angle_degrees=polar_angle,
        azimuthal_angle_degrees=azimuthal_angle,
    )
    errors = []
    for samples in (256, 4096, 16384):
        x = (np.arange(samples) / samples + 0.5) % 1 - 0.5
        grid = np.where(np.abs(x) < 0.2, 4.0, 1.0)[:, None]
        result = fmmax_result(
            grid,
            0.73,
            0.23,
            5,
            polarization,
            polar_angle_degrees=polar_angle,
            azimuthal_angle_degrees=azimuthal_angle,
        )
        errors.append(max(np.max(np.abs(a - b)) for a, b in zip(result[:2], reference)))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 1e-3
