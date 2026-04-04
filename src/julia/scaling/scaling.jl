@inline function _scale!(M :: MMatrix{N, N, T}) where {N, T <: Complex}
    RT = Base._realtype(T)

    # Find the maximum absolute value in M i.e. max_ij ||A||_ij
    maxnum = _maxabs(M)

    # Define the small number & large number
    smlnum  = _smlnum(RT)
    bignum  = _bignum(RT)

    # Comparison
    scale = one(RT)
    if isfinite(maxnum) && maxnum > 0
        if maxnum < smlnum
            scale = smlnum / maxnum
        elseif maxnum > bignum
            scale = bignum / maxnum
        end
    end

    # Apply to the matrix
    @inbounds for i in eachindex(M)
        M[i] *= scale
    end

    return scale
end

@inline function _unscale!(W::MVector{N,T}, α) where {N,T<:Complex}
    @inbounds for i in 1:N
        W[i] /= α
    end
    return nothing
end

# Toolbox
@inline _smlnum(::Type{T}) where {T<:AbstractFloat} = floatmin(T) / eps(T)
@inline _bignum(::Type{T}) where {T<:AbstractFloat} = inv(_smlnum(T))

"""
    _maxabs(array::A) where {T<:Complex, A<:AbstractArray{T}}

Compute the maximum absolute value of a complex array using a single-pass
squared-magnitude reduction:

    ‖A‖ₘₐₓ = maxᵢ |aᵢ| = √( maxᵢ abs2(aᵢ) )

# Parameters
- `array::AbstractArray{T}`: Complex-valued array (`T <: Complex`).

# Returns
- `Real`: The maximum absolute value of the elements of `array`,
  with type `Base._realtype(T)`.
"""
@inline function _maxabs(array :: A) where {T <: Complex, A <: AbstractArray{T}}
    RT = Base._realtype(T)
    max2 = zero(RT)
    @inbounds @simd for z in array
        absz2 = abs2(z)
        max2 = ifelse(absz2 > max2, absz2, max2)
    end
    return sqrt(max2)
end