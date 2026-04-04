# `tiny_eigvals` Performance Report: CPU vs CUDA GPU

> **Date:** 2026-04-04  
> **Package:** TinyEigvals.jl  
> **Source:** `test/tiny_eigvals_gpu/benchmark_tiny_eigvals_gpu.jl`

---

## 1. Purpose

This benchmark checks two things:

1. `tiny_eigvals` can execute correctly inside a CUDA kernel.
2. Batched `8x8` complex eigensolves benefit from GPU execution relative to serial and threaded CPU runs.

The target workload is the batch of tiny eigenvalue problems used in streaming-instability parameter sweeps.

---

## 2. Test Environment

| Component | Specification |
|---|---|
| CPU threads | 12 Julia threads |
| GPU | NVIDIA GeForce RTX 5070 |
| CUDA runtime | 12.8.0 |
| Julia | 1.12.5 |
| OS | Windows |

---

## 3. Benchmark Setup

Each eigenproblem is an `8x8` complex matrix built from the linearised streaming-instability system with fixed parameters:

- `St = 0.1`
- `eps = 3.0`

The benchmark samples wavenumber grids of size:

- `128 x 128`
- `256 x 256`
- `512 x 512`
- `1024 x 1024`

This produces `N^2` independent eigenproblems per run.

The following execution modes are compared:

- CPU serial
- CPU threaded with `Threads.@threads` on 12 threads
- GPU with one CUDA thread per eigenproblem and 256 threads per block

Timings are median values from `BenchmarkTools`.

---

## 4. Correctness Checks

The benchmark script includes:

- a single-matrix CPU vs GPU comparison
- a single-matrix comparison against the expected reference growth rate
- batched GPU vs CPU agreement checks for every tested grid size

All correctness checks passed in the benchmark run.

Single-matrix reference values from the run:

- CPU max `Re(lambda) = 0.41900913231353887`
- GPU max `Re(lambda) = 0.41900913231349685`
- expected reference `~= 0.4190204`

---

## 5. Performance Results

### 5.1 Timing Table

| Grid | # Problems | CPU serial | CPU 12T | GPU | GPU/serial | GPU/thread |
|---|---:|---:|---:|---:|---:|---:|
| `128x128` | 16,384 | 101.59 ms | 12.80 ms | 5.760 ms | 17.6x | 2.2x |
| `256x256` | 65,536 | 405.62 ms | 49.35 ms | 10.635 ms | 38.1x | 4.6x |
| `512x512` | 262,144 | 1626.11 ms | 195.70 ms | 38.284 ms | 42.5x | 5.1x |
| `1024x1024` | 1,048,576 | 6505.63 ms | 791.79 ms | 147.568 ms | 44.1x | 5.4x |

### 5.2 Throughput

| Grid | CPU serial | CPU 12T | GPU |
|---|---:|---:|---:|
| `128x128` | 161 K/s | 1280 K/s | 2844 K/s |
| `256x256` | 162 K/s | 1328 K/s | 6162 K/s |
| `512x512` | 161 K/s | 1340 K/s | 6848 K/s |
| `1024x1024` | 161 K/s | 1324 K/s | 7106 K/s |

---

## 6. Interpretation

- The GPU path is already faster at the smallest tested batch, but the advantage there is modest because launch overhead is still visible.
- From `256x256` onward, the GPU advantage stabilises and the speedup over the 12-thread CPU path stays around `4.6x` to `5.4x`.
- The largest tested case reaches about `7.1 million` eigenproblems per second on the GPU.
- The serial CPU throughput stays near `161 K/s`, while the threaded CPU path stays near `1.3 M/s`, so the GPU remains clearly ahead for large batches.

This supports the claim that `tiny_eigvals` is a good fit for GPU-resident batched tiny-matrix eigensolves.

---

## 7. Scope of the Result

This benchmark measures the eigensolver kernel workload itself. It does **not** include:

- matrix construction overhead in the surrounding application
- host-device transfer costs
- end-to-end workflow timing

So the results support GPU acceleration for the batched eigensolver stage, but should not be read as a full application speedup without additional profiling.
