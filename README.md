# TinyEigvals.jl

`TinyEigvals.jl` is a small fixed-size complex eigenvalue solver for tiny matrices. It follows the high-level structure of LAPACK's nonsymmetric eigenvalue pipeline, but keeps only the tiny-matrix, eigenvalues-only core in pure Julia using `StaticArrays.jl`. The package is intended as a lightweight backend for batched tiny-matrix eigenvalue problems.



## Usage

The package exposes two main entry points:

- `tiny_eigvals!(M::MMatrix)` for in-place computation
- `tiny_eigvals(M::SMatrix)` or `tiny_eigvals(M::MMatrix)` for a non-mutating convenience call

Supported matrix sizes are compile-time fixed `N x N` matrices with `1 <= N <= 15`, and the current API is intended for **complex** matrices. The returned eigenvalues are an `NTuple{N,T}`.

Currently, install the package directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/weishansu011017/TinyEigvals.jl")
```

Once the package is registered in Julia's General registry, installation will simply be:

```julia
using Pkg
Pkg.add("TinyEigvals")
```



After installation, a typical in-place usage looks like:

```julia
using TinyEigvals
using StaticArrays
using Random

# Define a tiny complex eigenvalue problem
Random.seed!(1234)
N = 4
T = ComplexF64
M = MMatrix{N,N,T}(1e8 .* (randn(T, N, N) .- 1.0))

# Solve in-place; M is modified by the algorithm
W = tiny_eigvals!(M)
typeof(W)  # NTuple{4, ComplexF64}
```

If you want to keep the input matrix unchanged, use `tiny_eigvals` instead:

```julia
using TinyEigvals
using StaticArrays
using Random

Random.seed!(1234)
N = 4
T = ComplexF64
A = SMatrix{N,N,T}(1e8 .* (randn(T, N, N) .- 1.0)) # Or MMatrix

# Solve without modifying A
W = tiny_eigvals(A)
```

For larger or dynamically sized matrices, use `LinearAlgebra.eigvals` instead.



## Algorithm outline

This solver follows a simplified LAPACK-inspired eigenvalue pipeline, but is implemented entirely in Julia using fixed-size arrays from `StaticArrays.jl`. The computation proceeds through the following stages:

1. Scaling: scales the matrix to improve numerical robustness.
2. Balancing: applies a simplified diagonal scaling step corresponding to the scaling-only part of `ZGEBAL`; no permutation/isolation is performed, so the active window remains the full matrix.
3. Hessenberg reduction: reduces the balanced matrix to upper Hessenberg form using an in-place unblocked Householder scheme, closer in spirit to the small-matrix `ZGEHD2` / `ZGEHRD` path than to the full blocked LAPACK implementation.
4. Schur/QR eigenvalues: runs an eigenvalues-only Hessenberg QR iteration on the reduced matrix. This is closest to the small-matrix `ZHSEQR('E','N')` path when LAPACK falls back to `ZLAHQR`, rather than the full `ZLAQR0` multishift/AED machinery used for larger problems.
5. Unscaling: rescales the eigenvalues to undo the initial matrix scaling.

Compared with a full LAPACK implementation, this solver intentionally simplifies several parts:

1. Tiny fixed sizes only: methods are generated only for compile-time matrix sizes up to `15 x 15`, rather than supporting arbitrary matrix dimensions.
2. Static-array-only interface: the implementation is built around `StaticArrays.jl` (`SMatrix`/`MMatrix`) instead of general dense array types.
3. Complex eigenvalues only: the current API is specialized for complex matrices and does not try to cover the broader nonsymmetric eigensolver family.
4. Simplified scaling: the preprocessing step is a single global matrix scaling based on the maximum entry magnitude.
5. `ZGEBAL` reduced to its scaling-only core: LAPACK's permutation/isolation stage and balancing metadata are omitted, so `ilo, ihi` stay at `1, N`.
6. No explicit `Q` representation from Hessenberg reduction: LAPACK stores Householder data so that `Q` can later be generated or applied (`ZUNGHR` / `ZUNMHR`-style). Here, the matrix is only reduced in place to Hessenberg form for the purpose of eigenvalue extraction.
7. No blocked `ZGEHRD` machinery: the Hessenberg reduction uses direct unblocked tiny-matrix kernels and fixed workspace instead of LAPACK's blocked Level-3 BLAS path.
8. No Schur form or Schur vectors: the QR stage is strictly eigenvalues-only (`WANTT = false`, `WANTZ = false`).
9. No large-matrix `ZHSEQR` stack: LAPACK switches to `ZLAQR0` with multishift QR and aggressive early deflation for larger problems; this implementation keeps only a compact tiny-matrix `ZLAHQR`-style iteration.
10. No downstream eigenvector/backtransform stages: routines analogous to `ZTREVC`, `ZHSEIN`, or `ZGEBAK` are intentionally absent.
11. Simplified interface and error reporting: the routine returns an `NTuple` of eigenvalues and reports QR nonconvergence by returning `NaN` entries instead of LAPACK-style `INFO` codes and workspace-query behavior.

This solver is well suited for large batches of tiny dense eigenvalue problems. Note that **eigenvectors are *NOT* computed by this routine**.



## CUDA Benchmark

`tiny_eigvals` is designed to work with `StaticArrays`, which also makes it suitable for GPU kernels when the input matrices are already available on device. A representative run of

`test/tiny_eigvals_gpu/benchmark_tiny_eigvals_gpu.jl`

gave the following batch-performance results for `8 x 8` complex eigenvalue problems:

| Grid | # Problems | CPU serial | CPU 12T | GPU | GPU/serial | GPU/thread |
|---|---:|---:|---:|---:|---:|---:|
| `128 x 128` | 16,384 | 101.59 ms | 12.8 ms | 5.76 ms | 17.6x | 2.2x |
| `256 x 256` | 65,536 | 405.62 ms | 49.35 ms | 10.635 ms | 38.1x | 4.6x |
| `512 x 512` | 262,144 | 1626.11 ms | 195.7 ms | 38.284 ms | 42.5x | 5.1x |
| `1024 x 1024` | 1,048,576 | 6505.63 ms | 791.79 ms | 147.568 ms | 44.1x | 5.4x |

These numbers show that `tiny_eigvals` can run correctly inside a CUDA kernel and that, for large batches of tiny matrices already resident on GPU, the GPU implementation can significantly outperform both serial and threaded CPU execution.

This benchmark measures the batched eigensolver kernel itself. It should not be interpreted as an end-to-end workflow speedup unless matrix construction and host-device transfer costs are also included in the surrounding application.
