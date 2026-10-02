# Arnoldi iteration, from GMRES.jl.
#
# Builds an orthonormal basis Q of the Krylov space of the operator A from b, one vector per call
# to `step!`, and the upper Hessenberg matrix H with A Qₙ = Qₙ₊₁ H.
#
# Interface: the vectors broadcast and support `norm`, `dot` and `similar`; the operator is called
# in place as `A(out, v)`.

mutable struct ArnoldiIteration{X, OP}
    A::OP              # linear operator, called as A(out, v)
    Q::Vector{X}       # Arnoldi vectors
    H::Matrix{Float64} # upper Hessenberg matrix, (n + 1) × n after n steps

    # start from b, normalised in place
    function ArnoldiIteration(A::OP, b::X) where {X, OP}
        b ./= norm(b)
        return new{X, OP}(A, X[b], zeros(1, 0))
    end
end

# one Arnoldi step: a new basis vector and a new column of H
function step!(arn::ArnoldiIteration)
    Q = arn.Q
    n = length(Q)

    # ---- H grows by one row and one column ----
    H                = zeros(n + 1, n)
    H[1:n, 1:n - 1] .= arn.H
    arn.H            = H

    # ---- new direction, orthogonal to the basis ----
    v = similar(Q[n])
    arn.A(v, Q[n])

    _orthogonalise!(v, Q, view(H, 1:n, n))

    # ---- normalisation ----
    H[n + 1, n] = norm(v)
    v ./= H[n + 1, n]
    push!(Q, v)

    return Q, H
end

# Orthogonalise v against the basis Q, accumulating the projections in h. Each pass is modified
# Gram–Schmidt: every projection is computed on the vector already cleared of the previous ones.
#
# Why two passes: when v lies almost in the span of Q, as it does once the Krylov space captures
# the operator, the subtractions cancel most of v and rounding errors leave components along Q
# that grow relative to what remains of v, so the basis slowly loses orthogonality. A second pass
# removes these residual components; two passes are enough to keep Q orthonormal to rounding
# ("twice is enough", Giraud et al. 2005; http://slepc.upv.es/documentation/reports/str1.pdf).
function _orthogonalise!(v::V,
                         Q::Vector{V},
                         h::AbstractVector) where {V}
    for _ in 1:2
        for j in eachindex(h)
            hj    = dot(v, Q[j])
            v   .-= hj .* Q[j]
            h[j] += hj
        end
    end

    return v
end

"""
    lincomb!(out, Q, y) -> out

The vector `Σⱼ yⱼ Qⱼ` of the Krylov space, from its coordinates `y` on the first `length(y)`
Arnoldi vectors.
"""
function lincomb!(out::V,
                    Q::Vector{V},
                    y::AbstractVector) where {V}
    out .= 0 .* out
    for (j, yj) in enumerate(y)
        out .+= yj .* Q[j]
    end

    return out
end
