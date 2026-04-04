# ============================================================================ #
#  Benchmark: tiny_eigvals - CPU (serial / threaded) vs CUDA GPU
# ============================================================================ #
#
#  This script verifies that `tiny_eigvals` runs inside a CUDA kernel and
#  compares throughput between CPU and GPU for batches of 8x8 complex matrices,
#  matching the matrix size used by the streaming-instability solver.
#
#  Usage:
#    julia --project=. -t 16 test/tiny_eigvals_gpu/benchmark_tiny_eigvals_gpu.jl
#    julia --project=. -t auto test/tiny_eigvals_gpu/benchmark_tiny_eigvals_gpu.jl
#
#  Requirements:
#  - CUDA.jl installed
#  - a CUDA-capable GPU
#
# ============================================================================ #

using BenchmarkTools
using CUDA
using Logging
using StaticArrays
using Statistics
using Test
using TinyEigvals

const HLR = 0.05
const ETA_PARAM = 0.0025
const ETA_VK_OVER_CS = ETA_PARAM / HLR
const INV_ETA_VK_OVER_CS = inv(ETA_VK_OVER_CS)

@inline H(keta_r::Float64) = INV_ETA_VK_OVER_CS * keta_r
@inline denom(eps::Float64, st::Float64) = (1 + eps)^2 + st^2
@inline uxlc(eps::Float64, st::Float64) = ETA_VK_OVER_CS * (2 * eps * st) / denom(eps, st)
@inline uylc(eps::Float64, st::Float64) = -ETA_VK_OVER_CS * (1 + eps * st^2 / denom(eps, st)) / (1 + eps)
@inline wxlc(eps::Float64, st::Float64) = -ETA_VK_OVER_CS * (2 * st) / denom(eps, st)
@inline wylc(eps::Float64, st::Float64) = -ETA_VK_OVER_CS * (1 - st^2 / denom(eps, st)) / (1 + eps)

@inline function max_realpart(eigs)
    val = -Inf
    @inbounds for k in SOneTo(length(eigs))
        r = real(eigs[k])
        val = ifelse(r > val, r, val)
    end
    return val
end

function build_si_matrix(st::Float64, eps::Float64, kx::Float64, kz::Float64)
    vx = uxlc(eps, st)
    vy = uylc(eps, st)
    wx = wxlc(eps, st)
    wy = wylc(eps, st)

    inv_st = inv(st)
    eps_inv_st = eps * inv_st
    rx = eps_inv_st * (wx - vx)
    ry = eps_inv_st * (wy - vy)
    a = -im * kx * wx
    b = -im * kx * vx

    M = zero(MMatrix{8, 8, ComplexF64})
    @inbounds begin
        M[1, 1] = a
        M[6, 1] = rx
        M[7, 1] = ry

        M[1, 2] = -im * kx
        M[2, 2] = a - inv_st
        M[3, 2] = -0.5
        M[6, 2] = eps_inv_st

        M[2, 3] = 2
        M[3, 3] = a - inv_st
        M[7, 3] = eps_inv_st

        M[1, 4] = -im * kz
        M[4, 4] = a - inv_st
        M[8, 4] = eps_inv_st

        M[5, 5] = b
        M[6, 5] = (-im * kx) - rx
        M[7, 5] = -ry
        M[8, 5] = -im * kz

        M[2, 6] = inv_st
        M[5, 6] = -im * kx
        M[6, 6] = b - eps_inv_st
        M[7, 6] = -0.5

        M[3, 7] = inv_st
        M[6, 7] = 2
        M[7, 7] = b - eps_inv_st

        M[4, 8] = inv_st
        M[5, 8] = -im * kz
        M[8, 8] = b - eps_inv_st
    end
    return SMatrix{8, 8, ComplexF64}(M)
end

function gpu_eigvals_kernel!(output, mat)
    eigs = tiny_eigvals(mat)
    @inbounds output[1] = max_realpart(eigs)
    return nothing
end

function cpu_batch_serial!(results::Vector{Float64}, mats::Vector{SMatrix{8, 8, ComplexF64, 64}})
    @inbounds for i in eachindex(mats)
        results[i] = max_realpart(tiny_eigvals(mats[i]))
    end
    return nothing
end

function cpu_batch_threaded!(results::Vector{Float64}, mats::Vector{SMatrix{8, 8, ComplexF64, 64}})
    Threads.@threads for i in eachindex(mats)
        @inbounds results[i] = max_realpart(tiny_eigvals(mats[i]))
    end
    return nothing
end

function gpu_batch_kernel!(results, mats, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds results[idx] = max_realpart(tiny_eigvals(mats[idx]))
    end
    return nothing
end

function print_header()
    println("=" ^ 78)
    println("  tiny_eigvals - CPU (serial / threaded) vs CUDA GPU Benchmark")
    println("=" ^ 78)
    println()
    println("CPU Threads: ", Threads.nthreads())
    println("GPU Device : ", CUDA.name(CUDA.device()))
    println("CUDA       : ", CUDA.runtime_version())
    println("Julia      : ", VERSION)
    println()
end

function run_single_gpu_correctness_test()
    println("-" ^ 72)
    println("  Test 1: Verify tiny_eigvals runs correctly inside a CUDA kernel")
    println("-" ^ 72)

    test_mat = build_si_matrix(0.1, 3.0, H(30.0), H(30.0))
    cpu_maxreal = max_realpart(tiny_eigvals(test_mat))
    println("  CPU max Re(lambda) = ", cpu_maxreal)
    println("  Expected ~= 0.4190204 (linA growth rate)")

    gpu_output = CUDA.zeros(Float64, 1)
    try
        @cuda threads=1 gpu_eigvals_kernel!(gpu_output, test_mat)
        CUDA.synchronize()
        gpu_maxreal = Array(gpu_output)[1]
        println("  GPU max Re(lambda) = ", gpu_maxreal)

        @testset "GPU tiny_eigvals correctness" begin
            @test gpu_maxreal ≈ cpu_maxreal rtol = 1e-10
            @test gpu_maxreal ≈ 0.4190204 rtol = 1e-3
        end
        println("  GPU result matches CPU!\n")
    finally
        CUDA.unsafe_free!(gpu_output)
    end
end

function run_batch_benchmark(problem_dims::Vector{Int})
    println("-" ^ 78)
    println("  Test 2: Batch performance - CPU (serial / threaded) vs GPU")
    println("-" ^ 78)

    st_val = 0.1
    eps_val = 3.0
    nthreads = Threads.nthreads()

    summary_header = "  " * rpad("NxN", 12) * rpad("Nproblems", 12) *
                     rpad("CPU serial", 16) * rpad("CPU $(nthreads)T", 16) *
                     rpad("GPU", 16) * rpad("GPU/serial", 14) *
                     rpad("GPU/thread", 14)
    summary_lines = String[]

    for n_per_dim in problem_dims
        nproblems = n_per_dim * n_per_dim
        kxs = range(H(1.0), H(100.0), length = n_per_dim)
        kzs = range(H(1.0), H(100.0), length = n_per_dim)
        all_mats = [build_si_matrix(st_val, eps_val, Float64(kx), Float64(kz)) for kx in kxs, kz in kzs]
        mats_flat = vec(all_mats)

        println()
        println("  -- $n_per_dim x $n_per_dim = $nproblems problems --")

        cpu_results = Vector{Float64}(undef, nproblems)
        cpu_batch_serial!(cpu_results, mats_flat)
        print("    CPU (serial)      : ")
        cpu_bench = @benchmark cpu_batch_serial!($cpu_results, $mats_flat) samples = 20 evals = 1
        display(cpu_bench)
        println()

        cpu_t_results = Vector{Float64}(undef, nproblems)
        cpu_batch_threaded!(cpu_t_results, mats_flat)
        print("    CPU ($nthreads threads) : ")
        cpu_t_bench = @benchmark cpu_batch_threaded!($cpu_t_results, $mats_flat) samples = 20 evals = 1
        display(cpu_t_bench)
        println()

        gpu_mats = CuArray(mats_flat)
        gpu_results = CUDA.zeros(Float64, nproblems)

        try
            threads_per_block = 256
            nblocks = cld(nproblems, threads_per_block)

            @cuda threads = threads_per_block blocks = nblocks gpu_batch_kernel!(gpu_results, gpu_mats, nproblems)
            CUDA.synchronize()

            gpu_results_host = Array(gpu_results)
            @testset "GPU correctness N=$n_per_dim" begin
                @test all(isapprox.(gpu_results_host, cpu_results, rtol = 1e-6))
            end

            print("    GPU (CUDA)        : ")
            gpu_bench = @benchmark begin
                @cuda threads=$threads_per_block blocks=$nblocks gpu_batch_kernel!($gpu_results, $gpu_mats, $nproblems)
                CUDA.synchronize()
            end samples = 20 evals = 1
            display(gpu_bench)
            println()

            t_serial = median(cpu_bench.times) / 1e6
            t_thread = median(cpu_t_bench.times) / 1e6
            t_gpu = median(gpu_bench.times) / 1e6

            line = "  " * rpad("$(n_per_dim)x$(n_per_dim)", 12) *
                   rpad("$nproblems", 12) *
                   rpad("$(round(t_serial, digits = 2)) ms", 16) *
                   rpad("$(round(t_thread, digits = 2)) ms", 16) *
                   rpad("$(round(t_gpu, digits = 3)) ms", 16) *
                   rpad("$(round(t_serial / t_gpu, digits = 1))x", 14) *
                   rpad("$(round(t_thread / t_gpu, digits = 1))x", 14)
            push!(summary_lines, line)
        finally
            CUDA.unsafe_free!(gpu_mats)
            CUDA.unsafe_free!(gpu_results)
        end
    end

    println()
    println("=" ^ 100)
    println("  Summary (CPU threads = $nthreads)")
    println("=" ^ 100)
    println(summary_header)
    println("  " * "-" ^ 96)
    for line in summary_lines
        println(line)
    end
    println("=" ^ 100)
end

function main(args = ARGS)
    isempty(args) || @warn "This benchmark script ignores command-line arguments." args

    CUDA.functional() || error("CUDA is not functional in this environment.")

    print_header()
    run_single_gpu_correctness_test()
    run_batch_benchmark([128, 256, 512, 1024])
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
