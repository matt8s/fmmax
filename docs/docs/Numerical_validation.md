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

Twenty-six S4 cases pass, including metallic films and gratings at normal
incidence, oblique and conical dielectric/absorbing/metallic films, and oblique
and conical dielectric gratings. For the normal-incidence dielectric films, the maximum
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

For incidence at 31 degrees in the periodic x-z plane, the maximum per-order power
difference for uniform dielectric, absorbing, and metallic films was 7.8e-16.
For the dielectric grating at 27 degrees, fixed 11-order truncation gave:

| Polarization | 256 samples | 4096 samples | 16384 samples |
| --- | ---: | ---: | ---: |
| TE | 1.91e-2 | 1.14e-3 | 2.85e-4 |
| TM | 9.70e-3 | 5.99e-4 | 1.50e-4 |

Again, the regression checks monotonic raster convergence and a final error below
`1e-3`; it does not hide the coarse-grid discrepancy by relaxing the gate.

For conical incidence, uniform-film per-order powers agreed to 1.0e-15 at polar
and azimuthal angles of 31 and 23 degrees. A dielectric grating at polar and
azimuthal angles of 27 and 19 degrees gave:

| Polarization | 256 samples | 4096 samples | 16384 samples |
| --- | ---: | ---: | ---: |
| TE | 2.02e-2 | 1.24e-3 | 3.09e-4 |
| TM | 1.52e-2 | 9.31e-4 | 2.32e-4 |

## Conventions

- Wavelength is the FMMAX input; comparator frequency is `1 / wavelength`.
- Forward propagation is increasing z, with spatial phase `exp(+1j * kz * z)`.
  Passive internal materials have positive imaginary permittivity.
- Match explicit integer reciprocal-order pairs. A torcwa order `[M, 0]` has
  `2*M+1` terms; a nominal basis-size request is not an order-set specification.
- At normal incidence, TE has electric field along y and TM along x. The oblique
  S4 cases use incidence in the periodic x-z plane, so TE remains along y and TM
  remains in the x-z plane. For conical incidence, the FMMAX incident Cartesian
  field is rotated into S4's s/p basis before normalizing by incident power.
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

## Accelerator precision

JAX may use reduced-precision matrix multiplication for complex64 calculations on
some GPUs. This is separate from the host eigensolver precision. Use 64-bit arrays
for strict cross-solver validation, or set the JAX matrix-multiplication precision
to `highest` when checking complex64 residuals:

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
```

GPU timings must synchronize results with `block_until_ready()` and report
compilation/warmup separately from steady-state execution.

A focused NVIDIA A100 run used Python 3.14.6, JAX/jaxlib 0.11.2 with the CUDA 13
plugin, NumPy 2.5.3, SciPy 1.18.1, and optional jeig 0.5.1. It verified that the
production host eigencallback returned GPU-resident results, then ran the callback,
analytic-matrix, gradient/flux, and all 26 S4 cases: 81 tests and 62 subtests
passed. This is accelerator correctness and placement evidence, not a performance
claim; a speedup requires synchronized timing on an otherwise comparable,
uncontended CPU/GPU workload.

An initial A100 CPU/GPU timing comparison using the optional jeig default backend
used 64-bit arrays, highest matrix-multiplication precision, one host BLAS/OpenMP
thread, explicit JIT warmup, and synchronized every timed result with
`block_until_ready()`. Three fresh processes per backend gave the following
medians of process medians:

| Workload | CPU | GPU | CPU time / GPU time |
| --- | ---: | ---: | ---: |
| Patterned 129-term eigensolve, 2048-point raster | 107.1 ms | 108.3 ms | 0.99 |
| Uniform-layer field reconstruction, 4096 sources | 72.4 ms | 7.51 ms | 9.65 |

The default patterned solve is host-eigensolver-bound and does not accelerate.
Field reconstruction is a large accelerator-native matrix workload and is about
9.6 times faster. Compilation reverses the result for a first call:
compile-plus-first execution was 1.34 and 1.48 times slower on GPU for the
eigensolve and field workloads, respectively. GPU acceleration therefore benefits
repeated or batched work rather than every call.

FMMAX also provides an explicit, optional device-native cuSOLVER path:

```python
import jax.numpy as jnp
import numpy as np

from fmmax import basis, fmm, utils

orders = np.column_stack((np.arange(-64, 65), np.zeros(129, dtype=int)))
expansion = basis.Expansion(orders)
lattice = basis.LatticeVectors(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
x = jnp.arange(2048) / 2048
permittivity = jnp.where(x[:, None] < 0.4, 4.0, 1.0)

layer = fmm.eigensolve_isotropic_media(
    wavelength=jnp.asarray(0.73),
    in_plane_wavevector=jnp.asarray([0.21, 0.07]),
    primitive_lattice_vectors=lattice,
    permittivity=permittivity,
    expansion=expansion,
    formulation=fmm.Formulation.FFT,
    eig_backend=utils.EigBackend.CUSOLVER,
)
```

This calls `jax.lax.linalg.eig` with `EigImplementation.CUSOLVER`. It requires an
NVIDIA GPU, a sufficiently recent JAX version exposing that implementation, and
cuSOLVER 11.7.1 or newer. It does not depend on jeig or PyTorch. Unsupported
configurations raise an error rather than silently moving the decomposition to the
CPU. Lowered JAX IR contains `cusolver_geev_ffi` and contains neither a
`pure_callback` nor `xla_ffi_python_gpu_callback`, confirming that the matrix
construction and nonsymmetric eigendecomposition remain in the compiled GPU
computation.

The same synchronized protocol measured the end-to-end patterned solve, including
Fourier-matrix construction, dense matrix products, and eigendecomposition:

| Fourier terms | Eigensystem dimension | CPU | A100 cuSOLVER | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 129 | 258 | 111.8 ms | 65.2 ms | 1.71 |
| 257 | 514 | 731.7 ms | 130.4 ms | 5.61 |
| 513 | 1026 | 3.587 s | 368.5 ms | 9.73 |

These are medians of three fresh-process medians with five synchronized steady
calls per process. The crossover is workload- and system-dependent: native GPU
execution does not guarantee acceleration for small matrices, while the cubic
dense eigensolve increasingly favors the GPU as truncation grows.
The 258-dimensional result was more load-sensitive in earlier exploratory runs;
the larger-matrix speedups are the stronger evidence for useful scaling.
Compile-plus-first medians for dimensions 258, 514, and 1026 were respectively
1.020, 1.068, and 1.648 seconds on GPU versus 0.929, 1.397, and 4.170 seconds on
CPU. The smallest one-shot solve therefore still favored CPU.

All CPU/GPU field arrays agreed within `3.8e-16` relative error. Patterned-layer
mode ordering differed, as eigensolver ordering is not physical; after minimum-cost
mode matching, all 258 longitudinal eigenvalues agreed within `1.3e-11` relative
and `1.2e-11` absolute error. The benchmark also exposed backend-dependent signs
for numerically real propagation constants. FMMAX now chooses positive real part
when the imaginary part is within a matrix-dimension-scaled backward-error bound,
while retaining positive imaginary part for genuinely evanescent modes. For the
1026-mode case, all CPU/GPU propagation constants matched after permutation within
`5.9e-11` relative and `8.1e-11` absolute error. The device-native path also passes
batched residual, directional-gradient, lossless-flux, and converged chapter-10
scattering checks.

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
