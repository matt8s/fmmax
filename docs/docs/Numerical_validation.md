# Cross-solver numerical validation

Optional end-to-end comparisons are in `tests/comparisons`. They supplement the
existing grcwa component tests with independently assembled optical simulations.

## Reproduction

The torcwa reference is
[`kch3782/torcwa` at `51c0d24`](https://github.com/kch3782/torcwa/tree/51c0d24abcd5f12dbae310614844d108b2dcb8a1).
Install it in an environment with the FMMAX test dependencies:

```sh
python -m pip install "torcwa @ git+https://github.com/kch3782/torcwa@51c0d24abcd5f12dbae310614844d108b2dcb8a1"
python -m pytest tests/comparisons/test_torcwa.py
```

The initial CPU run used Python 3.10.21, JAX 0.4.38, NumPy 2.2.6 and PyTorch
2.14.0. Thirty tests passed: both linear polarizations, lossless dielectric,
absorbing dielectric and metallic uniform films, and binary gratings with 7 and
11 Fourier orders. The film tests
check complex electric reflection/transmission against Fresnel/Airy formulas.
The grating tests check per-order power and complex electric amplitudes using
identical sampled permittivity, plus lossless conservation or passive absorption.
Translated gratings additionally check the complex diffraction-order phase
`exp(-2j*pi*m*shift)`, avoiding ambiguities hidden by a symmetric centered profile.
These cases establish agreement for their specified parameters, not universal
equivalence or converged accuracy for every grating.
The torcwa cases also pass with Python 3.14.6, JAX 0.11.2, NumPy 2.5.3,
PyTorch 2.14.0+cpu and the optional jeig 0.5.1 backend.

The S4 reference is
[`matt8s/S4` at `db74ede`](https://github.com/matt8s/S4/tree/db74ede8b207052b411ce105736e0d46230594c2).
Build its Python extension following that repository's instructions, using a
consistent native compiler/runtime and BLAS/LAPACK installation, then run:

```sh
python -m pytest tests/comparisons/test_s4.py
```

Ten S4 cases passed in the same Python 3.10 environment, including metallic films
and gratings. For the dielectric films, the maximum
difference over complex amplitudes and powers was 3.4e-16. At fixed 11-order
truncation, refining the FMMAX grating raster from 256 to 4096 samples reduced
the maximum per-order power difference from S4 as follows:

| Polarization | 256 samples | 4096 samples |
| --- | ---: | ---: |
| TE | 1.48e-3 | 9.02e-5 |
| TM | 1.42e-3 | 8.91e-5 |

This is evidence of material-sampling convergence at fixed modal truncation;
it does not establish convergence with the number of Fourier orders.
The metallic TM case (`epsilon = -5 + 0.5j`) still differed by 2.31e-3 at 4096
samples. Further refinement to 16384 samples reduced that difference below 1e-3;
the regression requires decreasing error through all three resolutions. The
4096-sample discrepancy is therefore not treated as a solver defect.

## Conventions

- Wavelength is the FMMAX input; comparator frequency is `1 / wavelength`.
- Forward propagation is increasing z, with spatial phase `exp(+1j * kz * z)`.
  Passive internal materials have positive imaginary permittivity.
- Match explicit integer reciprocal-order pairs. A torcwa order `[M, 0]` has
  `2*M+1` terms; a nominal basis-size request is not an order-set specification.
- At normal incidence, TE has electric field along y and TM along x.
- FMMAX `s11` transmits and `s21` reflects. Their entries are modal amplitudes;
  convert with `fields_from_wave_amplitudes` before comparing Cartesian electric
  fields. torcwa's reflected p basis can have a different sign, so compare its
  `xx`/`yy` amplitudes with `power_norm=False`.
- Set exterior-layer thicknesses to zero to match interface reference planes.
- Normalize each solver by its own incident flux. Reflected power is the negative
  backward flux. Compare propagating powers separately from evanescent amplitudes.

For a uniform isotropic exterior, `fmmax.fields.diffraction_efficiencies` combines
the two transverse components belonging to each reciprocal-lattice order and
normalizes forward and backward powers by incident flux. The backward result is
reported as a positive efficiency for power traveling toward decreasing z. This
helper is not a raw modal-S unitarity test and should not be applied to a patterned
or generally anisotropic layer where modal components do not map directly to
external diffraction orders.

## Observed implementation differences

The pinned torcwa source defines `pi = 3.141592652589793`, approximately 1e-9
below NumPy's pi. This introduces a small phase error; the analytic film tests
use 1e-8 tolerances. For the lossless TE film in the test, the maximum error
against the analytic result was 6.29e-10. A separate diagnostic that temporarily
replaced only `torcwa.rcwa.pi` with NumPy's pi reduced it to 4.01e-16, isolating
this discrepancy to torcwa's constant. The regression suite tests the original,
unmodified comparator. No corresponding FMMAX correction is needed.

FMMAX reconstructs patterned-layer longitudinal electric fields using the
convolution matrix of reciprocal permittivity. torcwa uses the inverse of the
permittivity convolution matrix in its field reconstruction. Those operations
are different at finite truncation. External scattering agreement does not
settle near-interface internal-field convergence; no solver defect is assigned
from that distinction alone.

S4 can use analytic geometric Fourier coefficients while these torcwa/FMMAX
tests use a sampled profile. Such comparisons must refine both the material
sampling and Fourier truncation. Conservation and reciprocity alone are not
proof of accuracy; independent analytic or converged references are needed.
