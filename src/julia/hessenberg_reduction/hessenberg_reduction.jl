@inline function _hessenberg_reduce!(M :: MMatrix{N,N,T}, ilo :: Int, ihi :: Int, work :: MVector{L,T}) where {N, L, T <: Complex}
    # tau stored in work[1:N-1]
    # tmp stored in work[N:(N+N-1)] = work[N:2N-1]
    woff = N - 1                                                    
    @inbounds for i in ilo:(ihi-1)
        # Generate reflector parameters (τ, β) and overwrite M[i+2:ihi, i] with v_tail.
        τ, β = _Householder_reflector_col!(M, i, ihi)
        work[i] = τ
        M[i+1, i] = β   # β is real (imag=0); becomes the first subdiagonal entry of H

        # Apply H(i) from the right:  M[1:ihi, i+1:ihi] ← M[1:ihi, i+1:ihi] * H(i)
        _apply_reflector_right!(M, i, ihi, τ, work, woff)

        # Apply H(i)ᴴ from the left: M[i+1:ihi, i+1:N] ← H(i)ᴴ * M[i+1:ihi, i+1:N]
        _apply_reflector_left!(M, i, ihi, τ, work, woff)
    end
    return nothing
end

# Toolbox
@inline function _ladiv(a::T, b::T) where {T<:Complex}
    # Stable complex division: a / b
    ar = real(a); ai = imag(a)
    br = real(b); bi = imag(b)

    if br == 0 && bi == 0
        RT = Base._realtype(T)
        nan = RT(NaN)
        return T(nan, nan)
    end

    if abs(br) >= abs(bi)
        # scale by br
        r   = bi / br
        den = br + bi*r
        return T((ar + ai*r) / den, (ai - ar*r) / den)
    else
        # scale by bi
        r   = br / bi
        den = bi + br*r
        return T((ai + ar*r) / den, (-ar + ai*r) / den)
    end
end
@inline function _Householder_reflector_col!(M :: MMatrix{N, N, T}, i :: Int, ihi :: Int) where {N, T <: Complex}
    RT = Base._realtype(T)

    # Tail length L = ihi - i - 1, reflector order nref = ihi - i = L+1
    nref = ihi - i
    α = M[i+1, i]

    if nref ≤ 0
        return zero(T), α
    end

    # xnorm = ||x||_2, x is M[i+2:ihi, i]
    xnorm2 = zero(RT)
    @inbounds for k in (i+2):ihi
        xnorm2 += abs2(M[k, i])
    end

    αre = real(α)
    αim = imag(α)

    if xnorm2 == zero(RT) && αim == zero(RT)
        return zero(T), α
    end

    beta = -copysign(sqrt(αre * αre + αim * αim + xnorm2), αre)

    safmin = _smlnum(RT)
    rsafmn = _bignum(RT)

    knt = 0
    if abs(beta) < safmin
        while abs(beta) < safmin && knt < 20
            knt += 1
            @inbounds for k in (i+2):ihi
                M[k, i] *= rsafmn
            end
            beta *= rsafmn
            αre  *= rsafmn
            αim  *= rsafmn
        end

        xnorm2 = zero(RT)
        @inbounds for k in (i+2):ihi
            xnorm2 += abs2(M[k, i])
        end

        beta = -copysign(sqrt(αre * αre + αim * αim + xnorm2), αre)
    end

    τ = T((beta - αre) / beta, (-αim) / beta)

    # Scale the Householder vector tail: x := x / (α - beta)
    # We want invden = 1 / (dr + i*di) where:
    #   dr = αre - beta,  di = αim
    #
    # DO NOT use conj(denom)/abs2(denom) via (dr^2+di^2) directly:
    #   - dr^2 + di^2 can overflow/underflow
    # Use a stable ratio form (same idea as LAPACK's robust complex division).

    dr = αre - beta
    di = αim

    # Rare pathological case: α == beta (denom == 0). Keep behavior explicit.
    # In well-behaved Hessenberg reduction this should basically never trigger.
    if dr == zero(RT) && di == zero(RT)
        nan = RT(NaN)
        invden = T(nan, nan)
    else
        adr = abs(dr)
        adi = abs(di)

        if adr >= adi
            # r = di/dr, den = dr + di*r = dr*(1+r^2)
            r   = di / dr
            den = dr + di*r
            invden = T(inv(den), -r * inv(den))          # (1 - i r)/den
        else
            # r = dr/di, den = di + dr*r = di*(1+r^2)
            r   = dr / di
            den = di + dr*r
            invden = T(r * inv(den), -inv(den))          # (r - i)/den
        end
    end

    @inbounds for k in (i+2):ihi
        M[k, i] *= invden
    end

    if knt != 0
        @inbounds for _ in 1:knt
            beta *= safmin
        end
    end

    β = T(beta, zero(RT))
    return τ, β
end

@inline function _apply_reflector_right!(A :: MMatrix{N, N, T}, i :: Int, ihi :: Int, τ :: T, work :: MVector{L, T}, woff :: Int) where {N, L, T<:Complex}

    m = ihi - i
    if m <= 0 || τ == zero(T)
        return nothing
    end

    @inbounds for r in 1:ihi
        s = A[r, i+1]  # v1 = 1
        for t in 2:m
            v = A[i+t, i]
            s += A[r, i+t] * v
        end
        work[woff + r] = s
    end

    @inbounds for r in 1:ihi
        wr = work[woff + r]
        A[r, i+1] -= wr * τ
        for t in 2:m
            v = A[i+t, i]
            A[r, i+t] -= wr * τ * conj(v)
        end
    end
    return nothing
end

@inline function _apply_reflector_left!(A :: MMatrix{N, N, T}, i :: Int, ihi :: Int, τ :: T, work :: MVector{L, T}, woff :: Int) where {N, L, T<:Complex}

    m = ihi - i
    ncol = N - i
    if m <= 0 || ncol <= 0 || τ == zero(T)
        return nothing
    end

    τc = conj(τ)

    @inbounds for j in 1:ncol
        col = i + j
        s = A[i+1, col]  # conj(v1)=1
        for t in 2:m
            v = A[i+t, i]
            s += conj(v) * A[i+t, col]
        end
        work[woff + j] = s
    end

    @inbounds for j in 1:ncol
        col = i + j
        tj = work[woff + j]
        A[i+1, col] -= τc * tj
        for t in 2:m
            v = A[i+t, i]
            A[i+t, col] -= τc * v * tj
        end
    end
    return nothing
end