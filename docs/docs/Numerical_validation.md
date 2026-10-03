# Cross-solver numerical validation

## Overview

The optional end-to-end comparisons in `tests/comparisons` test complete optical simulations against pinned versions of torcwa and S4. They supplement the existing grcwa component tests.

The comparisons exercise complex field amplitudes, per-order powers, material sampling, Fourier-order conventions, flux normalization, and phase-reference choices. Agreement applies to the specified cases and numerical settings; it does not imply universal solver equivalence or convergence for every grating.

## Reproducing the comparisons

### torcwa

The torcwa reference is [`kch3782/torcwa` at `51c0d24`](https://github.com/kch3782/torcwa/tree/51c0d24abcd5f12dbae310614844d108b2dcb8a1). Install it in an environment containing the FMMAX test dependencies and run:

```sh
python -m pip install "torcwa @ git+https://github.com/kch3782/torcwa@51c0d24abcd5f12dbae310614844d108b2dcb8a1"
python -m pytest tests/comparisons/test_torcwa.py
```

A CPU validation environment used Python 3.10.21, JAX 0.4.38, NumPy 2.2.6, and PyTorch 2.14.0. The cases also pass with Python 3.14.6, JAX 0.11.2, NumPy 2.5.3, PyTorch 2.14.0+cpu, and the optional jeig 0.5.1 backend.

### S4

The S4 reference is [`matt8s/S4` at `db74ede`](https://github.com/matt8s/S4/tree/db74ede8b207052b411ce105736e0d46230594c2). Build its Python extension following that repository's instructions, using a consistent native compiler/runtime and BLAS/LAPACK installation, then run:

```sh
python -m pytest tests/comparisons/test_s4.py
```

## What is compared

The torcwa suite contains 30 tests covering both linear polarizations, a lossless dielectric, absorbing dielectric and metallic uniform films, and binary gratings with 7 and 11 Fourier orders. Film tests compare complex electric reflection and transmission with Fresnel/Airy formulas. Grating tests compare per-order power and complex electric amplitudes using identical sampled permittivity, together with lossless conservation or passive absorption.

Translated-grating cases also test the complex diffraction-order phase

```text
exp(-2j*pi*m*shift)
```

so that phase agreement is not hidden by a symmetric centered profile.

The S4 suite contains 26 cases: metallic films and gratings at normal incidence; oblique and conical dielectric, absorbing, and metallic films; and oblique and conical dielectric gratings. For normal-incidence dielectric films, the maximum difference over complex amplitudes and powers was 3.4e-16.

## Conventions required for comparison

Cross-solver agreement requires physical quantities and reciprocal orders to be matched explicitly:

- Wavelength is the FMMAX input; comparator frequency is `1 / wavelength`.
- Forward propagation is increasing z, with spatial phase `exp(+1j * kz * z)`. Passive internal materials have positive imaginary permittivity.
- Match explicit integer reciprocal-order pairs. A torcwa order `[M, 0]` contains `2*M+1` terms; a nominal basis-size request alone does not specify the order set.
- At normal incidence, TE has electric field along y and TM along x. The oblique S4 cases use incidence in the periodic x-z plane, so TE remains along y and TM remains in the x-z plane. For conical incidence, rotate the FMMAX incident Cartesian field into S4's s/p basis before normalizing by incident power.
- FMMAX `s11` transmits and `s21` reflects. Their entries are modal amplitudes, not Cartesian electric-field components. Convert them with `fields_from_wave_amplitudes` before comparing Cartesian fields.
- torcwa's reflected p basis can have a different sign. Its `xx`/`yy` amplitudes are therefore compared with `power_norm=False`.
- Set exterior-layer thicknesses to zero so that the solvers use the same interface reference planes.
- Normalize each solver by its own incident flux. Reflected power is the negative backward flux.
- Compare propagating powers separately from evanescent amplitudes. Evanescent amplitudes can be physically useful, but they are not propagating diffraction efficiencies.

For a uniform isotropic exterior, `fmmax.fields.diffraction_efficiencies` combines the two transverse components belonging to each reciprocal-lattice order and normalizes forward and backward powers by incident flux. It reports backward power as a positive efficiency for power traveling toward decreasing z.

This helper is designed for external diffraction orders, not as a raw modal-S unitarity test. In a patterned or generally anisotropic layer, modal components do not map directly to external diffraction orders.

## Grating convergence and comparison results

### Normal incidence

For a period-1 dielectric grating with permittivity 4 in air, a 0.4-period stripe width, wavelength 0.73, thickness 0.23, and fixed 11-order truncation, refining the FMMAX grating raster from 256 to 4096 samples reduced the maximum per-order power difference from S4:

| Polarization | 256 samples | 4096 samples |
| --- | ---: | ---: |
| TE | 1.48e-3 | 9.02e-5 |
| TM | 1.42e-3 | 8.91e-5 |

This demonstrates convergence with material sampling at a fixed modal truncation. Convergence with the number of Fourier orders is a separate study.

For the metallic TM case with `epsilon = -5 + 0.5j`, the difference at 4096 samples was 2.31e-3. Refining to 16384 samples reduced it below 1e-3, and the regression requires decreasing error through all three resolutions. The observed material-grid convergence accounts for the 4096-sample difference.

### Oblique incidence

For incidence at 31 degrees in the periodic x-z plane, the maximum per-order power difference for uniform dielectric, absorbing, and metallic films was 7.8e-16.

For a dielectric grating at 27 degrees and fixed 11-order truncation:

| Polarization | 256 samples | 4096 samples | 16384 samples |
| --- | ---: | ---: | ---: |
| TE | 1.91e-2 | 1.14e-3 | 2.85e-4 |
| TM | 9.70e-3 | 5.99e-4 | 1.50e-4 |

The regression checks monotonic raster convergence and a final error below `1e-3`, retaining the coarse-grid difference rather than absorbing it into a looser comparison tolerance.

### Conical incidence

For uniform films at polar and azimuthal angles of 31 and 23 degrees, per-order powers agreed to 1.0e-15.

A dielectric grating at polar and azimuthal angles of 27 and 19 degrees gave:

| Polarization | 256 samples | 4096 samples | 16384 samples |
| --- | ---: | ---: | ---: |
| TE | 2.02e-2 | 1.24e-3 | 3.09e-4 |
| TM | 1.52e-2 | 9.31e-4 | 2.32e-4 |

These raster studies should be interpreted separately from Fourier-order convergence. Increasing the spatial resolution improves the representation of the material profile at the selected order set; increasing the order set tests the modal truncation itself.

## GPU precision and eigensolver backends

JAX can use reduced-precision matrix multiplication for complex64 calculations on some GPUs. This behavior is separate from the precision of a host eigensolver. Use 64-bit arrays for strict cross-solver validation, or request the highest matrix-multiplication precision when checking complex64 residuals:

```python
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
```

GPU results must be synchronized with `block_until_ready()` for timing. Compilation and warmup should be reported separately from steady-state execution.

A focused NVIDIA A100 run used Python 3.14.6, JAX/jaxlib 0.11.2 with the CUDA 13 plugin, NumPy 2.5.3, SciPy 1.18.1, and optional jeig 0.5.1. It verified that the production host eigencallback returned GPU-resident results and exercised the callback, analytic-matrix, gradient/flux, and all 26 S4 cases. In total, 81 tests and 62 subtests passed. This establishes accelerator correctness and placement for those cases; timing requires a separate benchmark protocol.

FMMAX provides two eigensolver choices:

- `utils.EigBackend.DEFAULT` preserves the host-compatible behavior. It uses jeig when available and otherwise the JAX path; on a GPU, the nonsymmetric eigendecomposition can remain host-bound even when surrounding arrays and operations are on the device.
- `utils.EigBackend.CUSOLVER` explicitly requests JAX's device-native nonsymmetric cuSOLVER implementation. This is useful when the supported GPU environment and matrix size justify keeping the eigendecomposition in the compiled device computation.

Select the cuSOLVER path with `eig_backend`:

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

This calls `jax.lax.linalg.eig` with `EigImplementation.CUSOLVER`. It requires an NVIDIA GPU, a sufficiently recent JAX version that exposes this implementation, and cuSOLVER 11.7.1 or newer. It does not depend on jeig or PyTorch. Unsupported configurations raise an error rather than silently moving the decomposition to the CPU.

The lowered JAX IR contains `cusolver_geev_ffi` and contains neither a `pure_callback` nor `xla_ffi_python_gpu_callback`, confirming that matrix construction and the nonsymmetric eigendecomposition remain in the compiled GPU computation.

## Performance results

The timing tables below are machine-specific recorded measurements from one NVIDIA A100 PCIe GPU and a dual-socket Intel Xeon Gold 6326 host with 32 physical CPU cores. The benchmark limited BLAS and OpenMP to one host thread. CPU measurements used the default JAX CPU eigensolver, while the device-native table explicitly selected `EigBackend.CUSOLVER` on the GPU. The results show how workload size and backend choice affect this system rather than predicting performance on every host or accelerator.

### Default backend and field reconstruction

The NVIDIA A100 CPU/GPU comparison used the optional jeig default backend, 64-bit arrays, highest matrix-multiplication precision, one host BLAS/OpenMP thread, explicit JIT warmup, and `block_until_ready()` for every timed result. Three fresh processes were run per backend; the table reports medians of the process medians.

| Workload | CPU | GPU | CPU time / GPU time |
| --- | ---: | ---: | ---: |
| Patterned 129-term eigensolve, 2048-point raster | 107.1 ms | 108.3 ms | 0.99 |
| Uniform-layer field reconstruction, 4096 sources | 72.4 ms | 7.51 ms | 9.65 |

The default patterned solve is host-eigensolver-bound and had similar CPU and GPU times for this workload. Field reconstruction is a large accelerator-native matrix workload and was about 9.6 times faster.

Compilation changes the one-call comparison: compile-plus-first execution was 1.34 and 1.48 times slower on the GPU for the eigensolve and field workloads, respectively. GPU acceleration is therefore most useful for repeated or batched work rather than every individual call.

### Device-native cuSOLVER

The same synchronized protocol measured the complete patterned solve, including Fourier-matrix construction, dense matrix products, and eigendecomposition:

| Fourier terms | Eigensystem dimension | CPU | A100 cuSOLVER | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 129 | 258 | 111.8 ms | 65.2 ms | 1.71 |
| 257 | 514 | 731.7 ms | 130.4 ms | 5.61 |
| 513 | 1026 | 3.587 s | 368.5 ms | 9.73 |

These values are medians of three fresh-process medians, with five synchronized steady calls per process. The crossover depends on the workload and system: native GPU execution does not guarantee acceleration for small matrices, while the cubic dense eigensolve increasingly favors the GPU as truncation grows.

The 258-dimensional result was more load-sensitive in exploratory runs, making the larger-matrix speedups the stronger evidence for useful scaling. Compile-plus-first medians for dimensions 258, 514, and 1026 were respectively 1.020, 1.068, and 1.648 seconds on GPU, compared with 0.929, 1.397, and 4.170 seconds on CPU. The smallest one-shot solve therefore favored the CPU.

All CPU/GPU field arrays agreed within `3.8e-16` relative error. Patterned-layer mode ordering differed because eigensolver ordering is not physical. After minimum-cost mode matching, all 258 longitudinal eigenvalues agreed within `1.3e-11` relative and `1.2e-11` absolute error.

The comparison also exposed backend-dependent signs for numerically real propagation constants. FMMAX chooses a positive real part when the imaginary part is within a matrix-dimension-scaled backward-error bound, while retaining a positive imaginary part for genuinely evanescent modes. For the 1026-mode case, all CPU/GPU propagation constants matched after permutation within `5.9e-11` relative and `8.1e-11` absolute error.

The device-native path also passes batched residual, directional-gradient, lossless-flux, and converged chapter-10 scattering checks.

## Solver-specific notes

### torcwa's value of pi

The pinned torcwa source defines

```text
pi = 3.141592652589793
```

which is approximately 1e-9 below NumPy's pi. This introduces a small phase error, so the analytic film tests use 1e-8 tolerances. For the lossless TE film in the test, the maximum error against the analytic result was 6.29e-10.

As a diagnostic, replacing only `torcwa.rcwa.pi` temporarily with NumPy's pi reduced the error to 4.01e-16, isolating the discrepancy to torcwa's constant. The regression suite uses the original, unmodified comparator; FMMAX requires no corresponding correction.

### Patterned-layer internal fields

FMMAX reconstructs patterned-layer longitudinal electric fields using the convolution matrix of reciprocal permittivity. torcwa instead uses the inverse of the permittivity convolution matrix. These two finite-truncation operations are different.

Agreement in external scattering therefore does not settle convergence of internal fields near material interfaces. That factorization distinction alone is not evidence of a defect in either solver.

### S4 geometry representation

S4 can use analytic geometric Fourier coefficients, whereas the torcwa/FMMAX tests described here use a sampled profile. A comparison between these representations must distinguish material-raster refinement from Fourier-order convergence and test both where appropriate.

## Interpretation checklist

When using these comparisons or constructing a new one:

1. Match the explicit integer reciprocal-order pairs, not only a nominal basis size.
2. Align propagation, field, and polarization conventions before comparing complex amplitudes.
3. Match interface reference planes; otherwise propagation phase can appear as disagreement.
4. Normalize each solver with its own incident flux and apply the correct sign to backward flux.
5. Compare propagating efficiencies separately from evanescent amplitudes.
6. Refine the material raster independently of the Fourier-order truncation.
7. Synchronize accelerator calculations and separate compilation from steady-state timing.
8. Match modes before comparing eigensystems because raw mode ordering is not physical.
9. Use analytic references or demonstrably converged calculations when judging accuracy.

Flux conservation and reciprocity are important consistency checks, but they are not proof of accuracy on their own. Independent analytic or converged references are needed.
