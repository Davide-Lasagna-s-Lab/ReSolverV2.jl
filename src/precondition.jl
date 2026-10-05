# Preconditioning: an invertible operator B on orbits, given by the user, that defines the metric
#
#     ⟨p, q⟩_B = ⟨B p, B q⟩ = ⟨p, M q⟩,     M = B⁺ B,
#
# in which both methods work (see the Preconditioning page of the documentation). L-BFGS applies
# M⁻¹ = B⁻¹ B⁻⁺, the hookstep B⁻¹ as a right preconditioner. The search only applies the inverses, through two
# functions that a preconditioner type extends:
#
#     precondition!(out, B, p)            out = B⁻¹ p
#     precondition_adjoint!(out, B, p)    out = B⁻⁺ p, with B⁻⁺ the adjoint of B⁻¹
#
# A multiple of the identity, `I` by default, leaves both methods as without preconditioning.

"""
    precondition!(out::Orbit, B, p::Orbit) -> out

Apply the inverse of the preconditioner `B` to the orbit `p`: `out = B⁻¹ p`. Extend it, together
with [`precondition_adjoint!`](@ref), for each preconditioner type.
"""
precondition!(out::Orbit, B::UniformScaling, p::Orbit) = (out .= p ./ B.λ)

"""
    precondition_adjoint!(out::Orbit, B, p::Orbit) -> out

Apply the adjoint of the inverse of the preconditioner `B` to the orbit `p`: `out = B⁻⁺ p`, in
the inner product of the orbits. For a self-adjoint `B` it is [`precondition!`](@ref).
"""
precondition_adjoint!(out::Orbit, B::UniformScaling, p::Orbit) = (out .= p ./ conj(B.λ))

# applications of B⁻¹ and B⁻⁺ through a System, counted, and projected back onto the invariant
# subspace of the System, which a preconditioner that does not commute with the symmetry would leave
function _precondition!(out::Orbit, F, p::Orbit)
    F.evaluations.precondition += 1
    precondition!(out, F.B, p)
    F.project(out.a)

    return out
end

function _precondition_adjoint!(out::Orbit, F, p::Orbit)
    F.evaluations.precondition += 1
    precondition_adjoint!(out, F.B, p)
    F.project(out.a)

    return out
end

# out = M⁻¹ p = B⁻¹ B⁻⁺ p, with tmp as scratch
function _metric_inverse!(out::Orbit, F, p::Orbit, tmp::Orbit)
    _precondition_adjoint!(tmp, F, p)
    _precondition!(out, F, tmp)

    return out
end
