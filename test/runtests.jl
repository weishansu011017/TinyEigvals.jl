# ============================================================================ #
#  TinyEigvals.jl - Test Suite Entry Point
# ============================================================================ #
#
#  Run with:  julia --project -e 'using Pkg; Pkg.test()'
#             or: include("test/runtests.jl") from the REPL
#
#  Ordering convention
#  -------------------
#  1. Solver / numerics    (fixed-size eigenvalue solver)
#
# ============================================================================ #

using Test
using TinyEigvals

# 1. Solver / numerics ------------------------------------------------------- #
include("tiny_eigvals_tests.jl")
