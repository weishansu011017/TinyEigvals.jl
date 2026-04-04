@inline function _balance!(M :: MMatrix{N, N, T}) where {N, T <: Complex}
    RT = Base._realtype(T)
    zeropfive = RT(0.5)
    factor = RT(0.95)
    two = RT(2)

    # Simplified ZGEBAL-style scaling only:
    # keep the full active window and iteratively rescale rows/columns.
    K = 1
    L = N
    bal = ones(MVector{N,RT})
    
    # Start iteration 
    noconv = true
    while noconv
        noconv = false

        for i in K:L
            # Get the 2-norm of column and row at (i, i)
            C, R = _rowcol_norm(M, i)
            
            # Guard against zero C or R due to underflow
            if C == 0 || R == 0
                continue
            end

            S = C + R
            F = one(RT)

            # while smaller
            while C < (zeropfive * R)
                F *= two
                C *= two
                R *= zeropfive 
            end

            # while larger
            while (C * zeropfive) >= R
                F *= zeropfive
                C *= zeropfive 
                R *= two
            end

            # Now balance
            canbalance = (F != one(RT)) && ((C + R) < factor * S)
            if canbalance
                bal[i] *= F
                invF = one(RT) / F

                @inbounds begin
                    for j in 1:i-1
                        M[j,i] *= F
                        M[i,j] *= invF
                    end
                    for j in i+1:N
                        M[j,i] *= F
                        M[i,j] *= invF
                    end
                end

                noconv = true
            end

        end
    end

    return K, L
end

# Toolbox
@inline function _rowcol_norm(M :: MMatrix{N,N,T}, i :: Int) where {N,T<:Complex}
    RT = real(T)
    C2 = zero(RT)
    R2 = zero(RT)

    @inbounds begin
        for j in 1:i-1
            C2 += abs2(M[j,i])
            R2 += abs2(M[i,j])
        end
        for j in i+1:N
            C2 += abs2(M[j,i])
            R2 += abs2(M[i,j])
        end
    end
    C = sqrt(C2)
    R = sqrt(R2)
    return C, R
end
