@inline function _schur_eigvals!(M :: MMatrix{N, N, T}, ilo :: Int, ihi :: Int, α :: RT, work :: MVector{L, T}) where {N, L, RT <: AbstractFloat, T <: Complex{RT}}
    # ----------------------------------------------------------------------
    # _schur_eigvals_scaled! : small-N Hessenberg QR eigenvalue iteration (eigs-only)
    #
    # Intended LAPACK correspondence:
    #   - Implements the small-matrix branch of ZHSEQR (N <= NTINY=15) where ZHSEQR
    #     delegates to ZLAHQR for QR iteration/deflation on a Hessenberg matrix H.
    #
    # Assumptions / contracts:
    #   - M is an upper Hessenberg matrix H on entry (after ZGEHRD-like reduction).
    #   - The current balancing step does not perform LAPACK-style permutation/
    #     isolation, so in normal use `ilo=1` and `ihi=N` and the active window is
    #     the full matrix.
    #   - This routine may overwrite M inside the active window during QR steps.
    #
    # Output behavior (eigenvalues-only):
    #   - We do NOT compute Schur vectors (WANTZ=false) and do NOT form the full Schur
    #     form (WANTT=false). Only eigenvalues are produced.
    #   - Eigenvalues are returned as an `NTuple{N,T}`; in the normal path we return
    #     `diag(M)` after convergence/deflation (optionally scaled by `invα`).
    #
    # Algorithm overview (ZLAHQR core):
    #   1) Hessenberg housekeeping in the active region:
    #        - clear entries below the first subdiagonal
    #        - normalize phases so H(k,k-1) are real (>=0)
    #   2) Deflation loop from bottom to top:
    #        - find negligible subdiagonal (deflation split) using ZLAHQR criteria
    #          (including Ahues–Tisseur conservative test)
    #        - if a 1×1 block splits off, shrink the active index `i`
    #        - otherwise choose shift μ (Wilkinson / exceptional), choose start row m,
    #          and apply one implicit single-shift Hessenberg QR step (bulge chasing)
    #        - re-normalize bottom subdiagonal phase to remain real
    #   3) On success, eigenvalues are taken from the diagonal entries H(k,k).
    #
    # Failure policy:
    #   - LAPACK returns INFO>0 on nonconvergence. Here we DO NOT throw; instead we
    #     return a tuple filled with NaNs to mark failure.
    # ----------------------------------------------------------------------
    invα = inv(α)
    @inbounds begin
        # [ZHSEQR] Quick return if possible:
        #   IF( ILO.EQ.IHI ) THEN W(ILO)=H(ILO,ILO); RETURN
        # Also handle degenerate "empty active window" (ilo>ihi).
        if ilo ≥ ihi
            return ntuple(k -> M[k,k] * invα, Val(N))
        end

        # ------------------------------------------------------------------
        # ZLAHQR prologue (Hessenberg housekeeping and phase normalization)
        # ------------------------------------------------------------------

        # [ZLAHQR] "clear out the trash"
        # Ensures strictly-below-subdiagonal entries in the active region are zero,
        # restoring/maintaining the Hessenberg structure before QR iterations.
        _hessenberg_clear_trash_small!(M, ilo, ihi)

        # [ZLAHQR] "ensure that subdiagonal entries are real"
        # Applies a unit-modulus scaling to make H(i,i-1) real (>=0) for i=ilo+1:ihi,
        # improving numerical behavior and matching LAPACK assumptions used later
        # (e.g., DBLE(H(m+1,m)) in shift/start-row tests).
        _make_subdiagonal_real!(M, ilo, ihi)

        # ------------------------------------------------------------------
        # Constants / iteration limits (ZLAHQR uses NH fixed for the routine)
        # ------------------------------------------------------------------

        # [ZLAHQR] NH = IHI-ILO+1 (fixed for the duration of the routine)
        nh = ihi - ilo + 1

        # [ZLAHQR] ITMAX = 30 * MAX(10, NH)
        itmax = 30 * max(10, nh)

        # [ZLAHQR] KDEFL counts iterations since a deflation event
        kdefl = 0

        # ------------------------------------------------------------------
        # Main ZLAHQR-style deflation loop (working from bottom to top)
        # ------------------------------------------------------------------

        i = ihi
        while i >= ilo
            converged = false

            # [ZLAHQR] DO ITS = 0, ITMAX
            for _ in 0:itmax

                # (A) Deflation test: find negligible subdiagonal entry -> split index Lsplit
                #     ZLAHQR: "Look for a single small subdiagonal element."
                #     Must implement:
                #       - CABS1 test against SMLNUM
                #       - Ahues–Tisseur conservative deflation criterion
                Lsplit = _hessenberg_deflation_split(M, ilo, ihi, i, nh)

                # If H(L,L-1) is negligible, set it to zero (split / deflation)
                if Lsplit > ilo
                    M[Lsplit, Lsplit-1] = zero(T)
                    kdefl = 0
                end

                # (B) If a 1×1 block split off at the bottom, record eigenvalue and shrink
                if Lsplit >= i
                    kdefl = 0
                    i = Lsplit - 1
                    converged = true
                    break
                end

                # (C) Choose shift μ (Wilkinson or exceptional shift)
                kdefl += 1
                μ = _hessenberg_select_shift(M, Lsplit, i, kdefl)

                # (D) Choose start row m for the single-shift QR step
                m = _hessenberg_qr_start(M, Lsplit, i, nh, μ)

                # (E) One implicit single-shift Hessenberg QR step (bulge chasing)
                _hessenberg_single_shift_qr_step!(M, Lsplit, i, nh, m, μ, work)

                # (F) Ensure bottom subdiagonal H(i,i-1) is real again (phase cleanup)
                temp = M[i,i-1]
                if imag(temp) != 0
                    _hessenberg_fix_bottom_subdiag_phase!(M, Lsplit, i)
                end
            end

            if !converged
                nan = RT(NaN)
                bad = T(nan, nan)
                return ntuple(_ -> bad, Val(N))
            end
        end
    end
    return ntuple(k -> M[k,k] * invα, Val(N))
end

# Toolbox
@inline _cabs1(x::T) where {T<:Complex} = abs(real(x)) + abs(imag(x))
@inline function _larfg2(v1::T, v2::T) where {T<:Complex}
    RT  = Base._realtype(T)
    zRT = zero(RT)

    x = v2
    xnorm2= abs2(x)

    # Identity reflector if x == 0 and v1 is real
    if xnorm2 == zRT && imag(v1) == zRT
        return (zero(T), v1, zero(T))
    end

    ar = real(v1); ai = imag(v1)
    βmag = sqrt(ar * ar + ai * ai + xnorm2)
    sgn  = (ar >= zRT) ? one(RT) : -one(RT)
    beta_r = -sgn * βmag

    # LAPACK-style scaling guard:
    # use SMLNUM = SAFMIN/ULP (your project definition) as the threshold,
    # not SAFMIN itself, to avoid scaling into overflow.
    smlnum = _smlnum(RT)        # = floatmin(RT)/eps(RT)
    safmin = floatmin(RT)
    rsafmn = _bignum(RT)
    knt = 0

    v1s = v1
    xs  = x
    while abs(beta_r) < smlnum
        knt += 1
        v1s *= rsafmn
        xs  *= rsafmn
        ar = real(v1s); ai = imag(v1s)
        xnorm2 = abs2(xs)
        βmag = sqrt(ar * ar + ai * ai + xnorm2)
        sgn  = (ar >= zRT) ? one(RT) : -one(RT)
        beta_r = -sgn * βmag
        # Optional hard stop to avoid pathological loops:
        if knt > 20
            break
        end
    end

    beta = T(beta_r, zRT)

    # tau = (beta - alpha) / beta
    inv_beta_r = inv(beta_r)              # RT
    tau = (beta - v1s) * inv_beta_r       # Complex * Real

    # v2 := x / (alpha - beta)
    denom = v1s - beta
    v2out = _ladiv(xs, denom)

    # rescale beta back
    if knt != 0
        for _ in 1:knt
            beta *= safmin
        end
    end

    return (tau, beta, v2out)
end

@inline function _copy_isolated_eigvals!(W :: MVector{N, T}, H :: MMatrix{N, N, T}, ilo :: Int, ihi :: Int) where {N, T <: Complex}
    # LAPACK ZHSEQR:
    #   IF(ILO>1)  W(1:ILO-1)     = diag(H)(1:ILO-1)
    #   IF(IHI<N)  W(IHI+1:N)     = diag(H)(IHI+1:N)

    @inbounds begin
        if ilo > 1
            for i in 1:(ilo-1)
                W[i] = H[i, i]
            end
        end
        if ihi < N
            for i in (ihi+1):N
                W[i] = H[i, i]
            end
        end
    end
    return nothing
end

@inline function _hessenberg_clear_trash_small!(H :: MMatrix{N, N, T}, ilo :: Int, ihi :: Int) where {N, T <: Complex}
    # LAPACK ZLAHQR:
    #   DO J = ILO, IHI-3
    #      H(J+2,J) = 0
    #      H(J+3,J) = 0
    #   END DO
    #   IF( ILO <= IHI-2 ) H(IHI, IHI-2) = 0

    z = zero(T)
    @inbounds begin
        for j in ilo:(ihi - 3)
            H[j + 2, j] = z
            H[j + 3, j] = z
        end
        if ilo <= ihi - 2
            H[ihi, ihi - 2] = z
        end
    end
    return nothing
end
@inline function _make_subdiagonal_real!(H :: MMatrix{N, N, T}, ilo :: Int, ihi :: Int) where {N, T <: Complex}
    # LAPACK ZLAHQR (eigenvalues-only path: WANTT=false, WANTZ=false):
    #   JLO = ILO, JHI = IHI
    #   For i = ILO+1:IHI, if imag(H(i,i-1)) != 0:
    #     SC  = H(i,i-1) / CABS1(H(i,i-1))
    #     SC  = conj(SC) / abs(SC)          (unit-modulus)
    #     H(i,i-1) = abs(H(i,i-1))          (real >= 0)
    #     scale row i:      H(i, i:JHI)         *= SC
    #     scale column i:   H(JLO:min(JHI,i+1), i) *= conj(SC)

    RT = Base._realtype(T)
    zRT = zero(RT)
    jlo = ilo
    jhi = ihi

    @inbounds for i in (ilo + 1):ihi
        hij = H[i, i-1]
        if imag(hij) != zRT
            n1 = _cabs1(hij)
            if n1 == zRT
                continue
            end

            sc = hij / n1
            asc = abs(sc)
            if asc == zRT
                continue
            end
            sc = conj(sc) / asc

            # set subdiagonal entry real (>=0)
            H[i, i-1] = T(abs(hij), zRT)

            # scale row i, columns i:jhi
            for j in i:jhi
                H[i, j] *= sc
            end

            # scale column i, rows jlo:min(jhi,i+1)
            scconj = conj(sc)
            rmax = min(jhi, i + 1)
            for r in jlo:rmax
                H[r, i] *= scconj
            end
        end
    end

    return nothing
end
@inline function _hessenberg_deflation_split(H :: MMatrix{N, N, T}, ilo :: Int, ihi :: Int, i :: Int, nh :: Int) where {N, T <: Complex}
    # LAPACK ZLAHQR: "Look for a single small subdiagonal element."
    #
    # DO K = I, L+1, -1  (here L is always ILO in the eigenvalues-only outer loop)
    #   if CABS1(H(K,K-1)) <= SMLNUM -> split at K
    #   else compute TST and apply Ahues–Tisseur conservative deflation criterion
    # end
    # If no split found, return ILO.

    RT  = Base._realtype(T)
    zRT = zero(RT)

    # ULP = DLAMCH('PRECISION')
    ulp = eps(RT)

    # SMLNUM = SAFMIN*(NH/ULP) = NH*(SAFMIN/ULP)
    smlnum = nh * _smlnum(RT)

    @inbounds for k in i:-1:(ilo + 1)
        hsub = H[k, k-1]

        # Hard tiny test: if subdiagonal is extremely tiny, deflate immediately.
        if _cabs1(hsub) <= smlnum
            return k
        end

        # TST = CABS1(H(k-1,k-1)) + CABS1(H(k,k))
        tst = _cabs1(H[k-1, k-1]) + _cabs1(H[k, k])

        # If TST == 0, add neighboring subdiagonals for scale (as in LAPACK).
        if tst == zRT
            if k - 2 >= ilo
                tst += abs(real(H[k-1, k-2]))          # subdiagonal assumed real after normalization
            end
            if k + 1 <= ihi
                tst += abs(real(H[k+1, k]))            # IMPORTANT: upper bound is IHI (routine scope)
            end
        end

        # Ahues–Tisseur conservative test (LAWN 122, 1997).
        # Note: after phase normalization, H(k,k-1) is treated as real.
        if abs(real(hsub)) <= ulp * tst
            ab = max(_cabs1(H[k,   k-1]), _cabs1(H[k-1, k]))
            ba = min(_cabs1(H[k,   k-1]), _cabs1(H[k-1, k]))
            aa = max(_cabs1(H[k, k]),
                     _cabs1(H[k-1, k-1] - H[k, k]))
            bb = min(_cabs1(H[k, k]),
                     _cabs1(H[k-1, k-1] - H[k, k]))
            s = aa + ab

            # Avoid 0/0; if s==0 we can safely treat as deflatable.
            if s == zRT
                return k
            end

            if ba * (ab / s) <= max(smlnum, ulp * (bb * (aa / s)))
                return k
            end
        end
    end

    return ilo
end

@inline function _hessenberg_select_shift(H :: MMatrix{N, N, T}, Lsplit :: Int, i :: Int, kdefl :: Int) where {N, T <: Complex}
    # ------------------------------------------------------------------
    # LAPACK ZLAHQR: shift selection (Exceptional shifts / Wilkinson shift)
    #
    # This implements the block:
    #
    #   IF( MOD(KDEFL,2*KEXSH).EQ.0 ) THEN
    #      S = DAT1*ABS(DBLE(H(I,I-1)))
    #      T = S + H(I,I)
    #   ELSE IF( MOD(KDEFL,KEXSH).EQ.0 ) THEN
    #      S = DAT1*ABS(DBLE(H(L+1,L)))
    #      T = S + H(L,L)
    #   ELSE
    #      T = H(I,I)
    #      U = SQRT(H(I-1,I))*SQRT(H(I,I-1))
    #      S = CABS1(U)
    #      IF( S.NE.0 ) THEN
    #         X  = HALF*(H(I-1,I-1)-T)
    #         SX = CABS1(X)
    #         S  = MAX(S, CABS1(X))
    #         Y  = S*SQRT((X/S)**2 + (U/S)**2)
    #         IF( SX.GT.0 ) THEN
    #            IF( DBLE(X/SX)*DBLE(Y) + DIMAG(X/SX)*DIMAG(Y) .LT. 0 ) Y=-Y
    #         END IF
    #         T = T - U*ZLADIV(U, X+Y)
    #      END IF
    #   END IF
    #
    # Notes / assumptions:
    # - Subdiagonal entries H(k,k-1) are assumed real after phase normalization
    #   (i.e., real(H[i,i-1]) is meaningful, mirroring DBLE(...) in Fortran).
    # - kdefl counts iterations since last deflation and controls when to use
    #   exceptional shifts to break stagnation.
    # - _ladiv implements the stable complex division used by ZLADIV.
    # ------------------------------------------------------------------

    RT = Base._realtype(T)
    zRT = zero(RT)

    kexsh = 10                  # KEXSH in ZLAHQR
    dat1  = RT(3) / RT(4)       # DAT1 in ZLAHQR (3/4)
    half  = RT(1) / RT(2)       # HALF

    @inbounds begin
        # --------------------------------------------------------------
        # Exceptional shifts (triggered periodically when deflation stalls)
        # --------------------------------------------------------------
        if mod(kdefl, 2 * kexsh) == 0
            # S = DAT1*ABS(DBLE(H(I,I-1)));  T = S + H(I,I)
            s = dat1 * abs(real(H[i, i-1]))
            return H[i, i] + T(s, zRT)
        elseif mod(kdefl, kexsh) == 0
            # S = DAT1*ABS(DBLE(H(L+1,L)));  T = S + H(L,L)
            s = dat1 * abs(real(H[Lsplit + 1, Lsplit]))
            return H[Lsplit, Lsplit] + T(s, zRT)
        end

        # --------------------------------------------------------------
        # Wilkinson shift (default choice; derived from trailing 2×2 block)
        # --------------------------------------------------------------
        t = H[i, i]

        # U = sqrt(H(I-1,I))*sqrt(H(I,I-1))   (ZLAHQR form)
        u = sqrt(H[i-1, i] * H[i, i-1])

        # If U is zero, shift is just the bottom-right diagonal entry.
        if _cabs1(u) == zRT
            return t
        end

        # X = 0.5*(H(I-1,I-1) - T)
        x  = half * (H[i-1, i-1] - t)
        sx = _cabs1(x)

        # S = MAX(CABS1(U), CABS1(X))
        s = max(_cabs1(u), _cabs1(x))

        # Y = S*sqrt((X/S)^2 + (U/S)^2)
        y = s * sqrt((x / s)^2 + (u / s)^2)

        # Choose sign of Y to avoid cancellation (Wilkinson sign choice)
        if sx > zRT
            xs = x / sx
            if real(xs) * real(y) + imag(xs) * imag(y) < zRT
                y = -y
            end
        end

        # T = T - U*ZLADIV(U, X+Y)
        return t - u * _ladiv(u, x + y)
    end
end
@inline function _hessenberg_qr_start(H :: MMatrix{N, N, T}, Lsplit :: Int, i :: Int, nh :: Int, μ :: T) where {N, T <: Complex}
    # LAPACK ZLAHQR: "Look for two consecutive small subdiagonal elements."
    # Determine the row m at which to start the single-shift QR step so that
    # the step is likely to make H(m,m-1) negligible.
    #
    # Notes:
    # - Subdiagonal entries are assumed real after _make_subdiagonal_real!.
    # - This is the eigenvalues-only path, so I1=Lsplit, I2=i in later updates.
    # - nh is not used here (kept for signature stability / symmetry with other helpers).

    RT  = Base._realtype(T)
    zRT = zero(RT)
    ulp = eps(RT)

    @inbounds begin
        # Scan m = i-1, i-2, ..., Lsplit+1
        for m in (i - 1):-1:(Lsplit + 1)
            h11  = H[m,   m]
            h22  = H[m+1, m+1]
            h11s = h11 - μ
            h21  = real(H[m+1, m])                  # should be real
            s    = _cabs1(h11s) + abs(h21)

            # Normalize (h11s, h21) by s (if s==0, leave as zero)
            if s != zRT
                h11s /= s
                h21  /= s
            else
                h11s = zero(T)
                h21  = zRT
            end

            h10 = real(H[m, m-1])                   # should be real

            # If starting at row m would make H(m,m-1) negligible, choose m
            if abs(h10) * abs(h21) <= ulp * (_cabs1(h11s) * (_cabs1(h11) + _cabs1(h22)))
                return m
            end
        end

        # Fallback: start at the top of the active block
        return Lsplit
    end
end
@inline function _hessenberg_single_shift_qr_step!(H :: MMatrix{N, N, T}, Lsplit :: Int, i :: Int, nh :: Int, m :: Int, μ :: T, work :: MVector{L, T}) where {N, L, T <: Complex}
    # LAPACK ZLAHQR: "Single-shift QR step" (DO K = M, I-1)
    #
    # Eigenvalues-only path (WANTT=false, WANTZ=false):
    #   I1 = Lsplit, I2 = i
    #
    # Requires an external 2-vector Householder generator equivalent to ZLARFG:
    #   (tau, beta, v2) = _larfg2(v1, v2)
    # such that the reflector is G = I - tau * [1; v2] * [1, conj(v2)].
    #
    # Uses work[1:2] as scratch (no allocations).

    RT = Base._realtype(T)
    zRT = zero(RT)

    i1 = Lsplit
    i2 = i

    @inbounds for k in m:(i-1)
        # ------------------------------------------------------------
        # Form the 2-vector V = (v1, v2) used to build the 2×2 reflector
        # ------------------------------------------------------------
        v1 = work[1]
        v2 = work[2]

        if k == m
            # Rebuild V from (H(m,m)-μ, H(m+1,m)), normalized as in ZLAHQR.
            h11s = H[m, m] - μ
            h21  = real(H[m+1, m])                    # subdiagonal assumed real
            s    = _cabs1(h11s) + abs(h21)
            if s != zRT
                v1 = h11s / s
                v2 = T(h21 / s, zRT)
            else
                v1 = zero(T)
                v2 = zero(T)
            end
        else
            v1 = H[k,   k-1]
            v2 = H[k+1, k-1]
        end

        # ------------------------------------------------------------
        # ZLARFG(2, V1, V2, 1, T1)
        # Return:
        #   tau = T1
        #   beta = new V1 (stored back to H(k,k-1) when k>m)
        #   v2   = new V2 (Householder vector component)
        # ------------------------------------------------------------
        tau, beta, v2 = _larfg2(v1, v2)

        if k > m
            H[k,   k-1] = beta
            H[k+1, k-1] = zero(T)
        end

        # --------------------------------------------------------------------
        # Apply the 2×2 Householder reflector to chase the bulge.
        #
        # We use the standard complex Householder form:
        #     G = I - τ * v * vᴴ,   where v = [1; v2].
        #
        # IMPORTANT:
        # - Do NOT assume v2 is real. In LAPACK ZLAHQR some paths can keep v2 real,
        #   but in our implementation v2 generally becomes complex (e.g. from _ladiv).
        # - Therefore we apply the reflector using the full complex formulas.
        #
        # Notation:
        # - Left-apply updates rows (k,k+1):    [Hk; Hk+1] ← Gᴴ * [Hk; Hk+1]
        # - Right-apply updates cols (k,k+1):   [H·k  H·k+1] ← [H·k  H·k+1] * G
        # --------------------------------------------------------------------

        conjv2 = conj(v2)

        # ------------------------------------------------------------
        # Apply Gᴴ from the left to rows k,k+1 on columns k:I2
        #
        # For each column j, let x = [H[k,j]; H[k+1,j]].
        # Then:
        #   x ← (I - conj(τ) * v * vᴴ) * x
        # where vᴴ x = H[k,j] + conj(v2)*H[k+1,j].
        #
        # Update:
        #   w = vᴴ x
        #   t = conj(τ) * w
        #   H[k,  j]   -= t
        #   H[k+1,j]   -= t * v2
        # ------------------------------------------------------------
        conjtau = conj(tau)
        for j in k:i2
            w = H[k, j] + conjv2 * H[k+1, j]
            t = conjtau * w
            H[k,   j] -= t
            H[k+1, j] -= t * v2
        end

        # ------------------------------------------------------------
        # Apply G from the right to columns k,k+1 on rows I1:min(k+2,i)
        #
        # For each row j, let y = [H[j,k]  H[j,k+1]].
        # Then:
        #   y ← y * (I - τ * v * vᴴ)
        #
        # Compute y*v = H[j,k] + v2*H[j,k+1], then:
        #   t = (y*v) * τ
        #   H[j,k]   -= t
        #   H[j,k+1] -= t * conj(v2)
        # ------------------------------------------------------------
        jmax = min(k + 2, i)
        for j in i1:jmax
            w = H[j, k] + v2 * H[j, k+1]
            t = w * tau
            H[j, k]   -= t
            H[j, k+1] -= t * conjv2
        end

        # ------------------------------------------------------------
        # Extra scaling when k == m and m > Lsplit (ZLAHQR special case)
        # Ensures H(m,m-1) remains real after starting the step at m>L.
        # ------------------------------------------------------------
        if k == m && m > Lsplit
            temp = one(T) - tau
            tr = real(temp); ti = imag(temp)
            rtemp = sqrt(tr * tr + ti * ti)
            temp /= rtemp                         # unit-modulus

            H[m+1, m] *= conj(temp)
            if m + 2 <= i
                H[m+2, m+1] *= temp
            end
            conjtemp = conj(temp)
            for j in m:i
                if j != m + 1
                    # Scale row segment H(j, j+1:I2) by TEMP
                    if i2 > j
                        for c in (j+1):i2
                            H[j, c] *= temp
                        end
                    end
                    # Scale column segment H(I1:j-1, j) by conj(TEMP)
                    if j > i1
                        for r in i1:(j-1)
                            H[r, j] *= conjtemp
                        end
                    end
                end
            end
        end
    end
    return nothing
end
@inline function _hessenberg_fix_bottom_subdiag_phase!(H :: MMatrix{N, N, T}, Lsplit :: Int, i :: Int) where {N, T <: Complex}
    # LAPACK ZLAHQR: "Ensure that H(I,I-1) is real."
    #
    # Eigenvalues-only path: I1 = Lsplit, I2 = i
    #
    # Fortran logic:
    #   TEMP = H(I,I-1)
    #   IF (DIMAG(TEMP) != 0) THEN
    #     RTEMP = ABS(TEMP)
    #     H(I,I-1) = RTEMP
    #     TEMP = TEMP / RTEMP
    #     IF (I2 > I) scale row I:      H(I, I+1:I2) *= conj(TEMP)
    #     scale column I:               H(I1:I-1, I) *= TEMP
    #   END IF

    RT = Base._realtype(T)
    zRT = zero(RT)

    i1 = Lsplit
    i2 = i

    @inbounds begin
        temp = H[i, i-1]
        if imag(temp) != zRT
            tr = real(temp); ti = imag(temp)
            rtemp = sqrt(tr*tr + ti*ti)
            if rtemp != zRT
                H[i, i-1] = T(rtemp, zRT)
                temp /= rtemp

                # scale row i, columns (i+1):i2 by conj(temp)
                if i2 > i
                    for c in (i+1):i2
                        H[i, c] *= conj(temp)
                    end
                end

                # scale column i, rows i1:(i-1) by temp
                if i > i1
                    for r in i1:(i-1)
                        H[r, i] *= temp
                    end
                end
            end
        end
    end

    return nothing
end
