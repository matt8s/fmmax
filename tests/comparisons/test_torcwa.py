"""End-to-end optical comparisons with the optional torcwa solver."""

import numpy as np
import pytest

from ._common import fmmax_result as _fmmax
from ._common import orders_1d as _orders

torch = pytest.importorskip("torch")
torcwa = pytest.importorskip("torcwa")


def _torcwa(epsilon, wavelength, thickness, half_width, polarization):
    sim = torcwa.rcwa(
        freq=1 / wavelength,
        order=[half_width, 0],
        L=[1.0, 1.0],
        dtype=torch.complex128,
        device=torch.device("cpu"),
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(
        thickness=thickness, eps=torch.as_tensor(epsilon, dtype=torch.complex128)
    )
    sim.solve_global_smatrix()
    pin = "y" if polarization == "TE" else "x"
    powers, amplitudes = [], []
    for port in ("reflection", "transmission"):
        power = sum(
            sim.S_parameters(
                _orders(half_width),
                port=port,
                direction="forward",
                polarization=pout + pin,
                power_norm=True,
            )
            .abs()
            .square()
            for pout in ("x", "y")
        )
        amplitude = sim.S_parameters(
            _orders(half_width),
            port=port,
            direction="forward",
            polarization=pin + pin,
            power_norm=False,
        )
        powers.append(power.detach().cpu().numpy())
        amplitudes.append(amplitude.detach().cpu().numpy())
    return (*powers, *amplitudes)


@pytest.mark.parametrize("epsilon", [2.25, 2.25 + 0.12j, -5.0 + 0.5j])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_film_against_analytic(epsilon, polarization):
    wavelength, thickness = 0.83, 0.19
    index = np.sqrt(complex(epsilon))
    r01 = (1 - index) / (1 + index)
    phase = np.exp(2j * np.pi * index * thickness / wavelength)
    r = r01 * (1 - phase**2) / (1 - r01**2 * phase**2)
    t = (1 - r01**2) * phase / (1 - r01**2 * phase**2)
    expected = (abs(r) ** 2, abs(t) ** 2, r, t)
    for solver in (_fmmax, _torcwa):
        result = solver(epsilon, wavelength, thickness, 0, polarization)
        for actual, reference in zip(result, expected):
            np.testing.assert_allclose(actual, reference, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("half_width", [3, 5])
@pytest.mark.parametrize("epsilon", [4.0, 4.0 + 0.1j, -5.0 + 0.5j])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
@pytest.mark.parametrize("shift", [0.0, 0.125])
def test_sampled_grating(half_width, epsilon, polarization, shift):
    x = (np.arange(512) / 512 + 0.5) % 1 - 0.5
    grid = np.where(np.abs(x) < 0.2, epsilon, 1.0)[:, None]
    original_grid = grid
    grid = np.roll(grid, int(512 * shift), axis=0)
    actual = _fmmax(grid, 0.73, 0.23, half_width, polarization)
    reference = _torcwa(grid, 0.73, 0.23, half_width, polarization)
    # Compare propagating powers and Cartesian E amplitudes (including evanescent
    # amplitudes). Both solvers use the same sampled geometry and FFT formulation.
    for a, b in zip(actual, reference):
        np.testing.assert_allclose(a, b, atol=2e-7, rtol=2e-7)
    total = np.sum(actual[0]) + np.sum(actual[1])
    if np.imag(epsilon) == 0:
        np.testing.assert_allclose(total, 1.0, atol=1e-10)
    else:
        assert 0 < total < 1
    if shift:
        original = _fmmax(original_grid, 0.73, 0.23, half_width, polarization)
        phase = np.exp(-2j * np.pi * _orders(half_width)[:, 0] * shift)
        for shifted_amplitude, amplitude in zip(actual[2:], original[2:]):
            np.testing.assert_allclose(shifted_amplitude, amplitude * phase, atol=1e-10)
