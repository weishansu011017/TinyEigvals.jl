"""
    tiny_eigvals!(M::MMatrix{N,N,T}) where {N, T<:Complex}

Compute the eigenvalues of a tiny complex square matrix `M` (compile-time size `N`)
using a pure-Julia, fixed-size pipeline that mirrors a simplified LAPACK approach,
specialized for `N ≤ 15`.

Methods are defined for each `N ∈ 1:15` (via metaprogramming) and return the
eigenvalues as a `Tuple{Vararg{T,N}}` to ensure an `isbits` return type and avoid
heap allocation. The input matrix `M` is overwritten in-place by the reduction steps.

Algorithm outline (LAPACK-inspired, simplified, pure Julia):
1. Scaling: scales the matrix to improve numerical robustness.
2. Balancing: diagonal scaling inspired by the scaling-only part of `ZGEBAL`;
   no permutation/isolation is performed, so the active window remains `1:N`.
3. Hessenberg reduction: reduces the balanced matrix to upper Hessenberg form
   using an in-place unblocked Householder scheme (`ZGEHD2`/`ZGEHRD`-like for tiny `N`),
   with a small fixed workspace `work`.
4. Schur/QR eigenvalues: QR iterations on the Hessenberg form to obtain eigenvalues
   via a tiny-matrix, eigenvalues-only `ZLAHQR`/`ZHSEQR('E','N')`-style path.
5. Unscaling: rescales the eigenvalues to undo the initial matrix scaling.

# Parameters
- `M::MMatrix{N,N,T}`: Input matrix (modified in-place). `T` must be a complex type.

# Keyword Arguments
None.

# Returns
- `Wout::Tuple{Vararg{T,N}}`: Eigenvalues of `M` as a length-`N` tuple.

# Notes
- Supported sizes: `1 ≤ N ≤ 15`. For other sizes, no method is defined; use
  `LinearAlgebra.eigvals` (LAPACK-backed) instead.
- This is **not** a full LAPACK reimplementation; it keeps only the tiny-matrix,
  eigenvalues-only core of the pipeline and omits Schur vectors, eigenvectors,
  blocked large-matrix machinery, and the full `ZGEBAL` permutation stage.
- Workspace is fixed-size and stack-allocated via `StaticArrays` to minimize
  allocations in tight loops / batched usage.
"""
function tiny_eigvals! end

"""
    tiny_eigvals(M::SMatrix{N,N,T}) where {N, T<:Complex}

Compute the eigenvalues of a tiny complex square matrix `M` (compile-time size `N`)
without modifying the input, by copying `M` into a mutable `MMatrix` workspace and
running [`tiny_eigvals!`](@ref) on that workspace. This provides a convenience API
for immutable static matrices.

This method is specialized for `N ≤ 15` (methods are generated for each `N ∈ 1:15`)
and returns the eigenvalues as a `Tuple{Vararg{T,N}}` to keep the return type
`isbits`. The input `M` is **not** modified.

# Parameters
- `M::SMatrix{N,N,T}`: Input matrix (not modified). `T` must be a complex type.

# Keyword Arguments
None.

# Returns
- `Wout::Tuple{Vararg{T,N}}`: Eigenvalues of `M` as a length-`N` tuple.

# Notes
- Supported sizes: `1 ≤ N ≤ 15`. For other sizes, no method is defined; use
  `LinearAlgebra.eigvals` (LAPACK-backed) instead.
- Internally allocates a mutable `MMatrix{N,N,T}` copy of `M`. This is required
  because the pipeline overwrites the matrix in-place.
- The algorithm stages are identical to `tiny_eigvals!`: global scaling,
  simplified balancing, in-place Hessenberg reduction, eigenvalues-only QR,
  and unscaling.
"""
function tiny_eigvals end


for N in 1:15
    @eval begin
        @inline function tiny_eigvals!(M::MMatrix{$N, $N, T}) where {T <: Complex}
            # WORK: length 2N
            work  = zero(MVector{$(2N) ,T})

            # Step 0: scaling (matrix scaling only)
            s = _scale!(M)

            # Step 1: simplified balancing (scaling-only ZGEBAL-like step)
            ilo, ihi = _balance!(M)

            # Step 2: in-place Hessenberg reduction (tiny-N ZGEHD2/ZGEHRD-like)
            # tau stored in work[itau : itau+N-1], scratch in work[iwrk : end]
            _hessenberg_reduce!(M, ilo, ihi, work)

            # Step 3: eigenvalues-only Hessenberg QR (tiny-N ZLAHQR/ZHSEQR-like)
            # Step 4: undo scaling on eigenvalues
            Wout = _schur_eigvals!(M, ilo, ihi, s, work)
            return Wout
        end
        function tiny_eigvals(M::MMatrix{$N, $N, T}) where {T <: Complex}
            Mmod = zero(MMatrix{$N, $N, T})
            @inbounds begin
                Mmod .= M
            end
            Wout = tiny_eigvals!(Mmod)
            return Wout
        end
        function tiny_eigvals(M::SMatrix{$N, $N, T}) where {T <: Complex}
            Mmod = zero(MMatrix{$N, $N, T})
            @inbounds begin
                Mmod .= M
            end
            Wout = tiny_eigvals!(Mmod)
            return Wout
        end
    end
end


