module TinyEigvals

using Logging
using StaticArrays

# Add implementation includes here, for example:
include(joinpath(@__DIR__, "julia", "scaling", "scaling.jl"))
include(joinpath(@__DIR__, "julia", "balancing", "balancing.jl"))
include(joinpath(@__DIR__, "julia", "hessenberg_reduction", "hessenberg_reduction.jl"))
include(joinpath(@__DIR__, "julia", "schur_eigenvals", "schur_eigenvals.jl"))
include(joinpath(@__DIR__, "julia", "eigsolvers", "tiny_eigvals.jl"))

# Package metadata helpers.
version() = pkgversion(@__MODULE__)

function about()
    @info "TinyEigvals Module\n  Version: $(version())\n  Made by Wei-Shan Su, Apr 2026"
    return nothing
end

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
