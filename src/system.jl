# System: a dynamical system ∂t a = N(a) written in space-time form, and the quantities the
# methods need:
#
#     residual    r = ω ∂s a - Σᵢ cᵢ ∂ᵢ a - N(a)
#     objective   R = ½‖r‖²                                   both methods
#     gradient    ∇R, from the adjoint operator                L-BFGS
#     jacobian    𝒥 δx, from the linearised operator           Newton–Krylov hookstep
#
# With x = (a, ρ, c), ρ = log ω the unknown for the frequency, and ∂ᵢ the derivative along the
# i-th drift direction,
#
#     ∇ₐ R       = -ω ∂s r + Σᵢ cᵢ ∂ᵢ r - L⁺ r
#     ∂R/∂ρ      =  ω ⟨∂s a, r⟩,     since ∂ω/∂ρ = ω
#     ∂R/∂cᵢ     = -⟨∂ᵢ a, r⟩
#
# The gradient uses the skew-adjointness of the derivatives in the inner product, ∂⁺ = -∂.
#
# The operators and the derivatives are given by the user and opaque to the search, whatever the
# system and its discretisation. What they must satisfy, with a the field of an Orbit:
#
#     nl(out, a)               nonlinear operator N(a)
#     lin(out, b)              linearised operator L b, about the point set by linearise!
#     adj(out, b)              adjoint operator L⁺ b, about the same point
#     linearise!(op, a)        sets the linearisation point of op, lin or adj, to a
#     dds!(out, a)             derivative along the rescaled time s ∈ [0, 2π)
#     dd!(out, a)              derivative along a drift direction, one function per direction
#     B                        preconditioner, optional (precondition.jl)
#
# All write into `out`, of the type of `a`, and return it; the derivatives must be skew-adjoint in
# `dot`, ⟨∂a, b⟩ = -⟨a, ∂b⟩, as Fourier derivatives are. The fields broadcast and support `dot`
# and `similar`.
#
# The Newton system is square: the unknowns ω and cᵢ are matched by phase conditions that remove
# the neutral directions of time and space translations,
#
#     ⟨∂s a, δa⟩ = 0,    ⟨∂ᵢ a, δa⟩ = 0.

# Number of evaluations made through a System, cumulative over its lifetime.
mutable struct Evaluations
        residual::Int # residuals, one nonlinear operator each
        gradient::Int # gradients, one adjoint operator each
        jacobian::Int # Jacobian actions, one linearised operator each
    precondition::Int # applications of B⁻¹ or B⁻⁺

    Evaluations() = new(0, 0, 0, 0)
end

# a snapshot of the counters, as a named tuple
counts(e::Evaluations) = (residual     = e.residual,
                          gradient     = e.gradient,
                          jacobian     = e.jacobian,
                          precondition = e.precondition)

"""
    System(nl, lin, adj, dds!, x::Orbit; linearise!, ddi=(), B=I)

The system with nonlinear, linearised and adjoint operators `nl`, `lin` and `adj`, rescaled-time
derivative `dds!` and function `linearise!(op, a)` setting the linearisation point of `lin` or
`adj`, on orbits shaped like `x`, e.g. for the Kuramoto–Sivashinsky example

```julia
System(KSNonlinear(g), KSLinearised(g), KSLinearised(g; adjoint=true), dds!, x;
       linearise!, ddi=(ddx!,))
```

`ddi` is a tuple of functions `dd!(out, a) -> out`, the derivatives along the directions of
translational symmetry in which the orbit drifts with an unknown speed, one per drift speed in
`x.p`. The default `()` searches for periodic orbits. Leave out the directions in which a symmetry
of the problem forbids drifting.

`B` is the preconditioner, an invertible operator on orbits for which
[`precondition!`](@ref) and [`precondition_adjoint!`](@ref) are defined; the identity `I` by
default.
"""
struct System{NL, LIN, ADJ, DS, DD, LN, PB, X, O<:Orbit}
            nl::NL           # nonlinear operator N(a)
           lin::LIN          # linearised operator L b, about the linearisation point
           adj::ADJ          # adjoint operator L⁺ b, about the same point
          dds!::DS           # derivative in rescaled time s
           ddi::DD           # derivatives along the drift directions
    linearise!::LN           # sets the linearisation point of lin or adj
             B::PB           # preconditioner
         cache::NTuple{2, X} # scratch fields
      residual::O            # residual of the last call to objective
   evaluations::Evaluations  # counters of the evaluations

    function System(        nl,
                           lin,
                           adj,
                          dds!,
                             x::Orbit;
                    linearise!,
                           ddi::Tuple=(),
                             B=I)

        # ---- input checks ----
        length(ddi) == ndrift(x) ||
            throw(ArgumentError("one drift derivative per drift speed in x.p"))

        # ---- assemble ----
        cache    = (similar(x.a), similar(x.a))
        residual = similar(x)

        return new{typeof(nl),
                   typeof(lin),
                   typeof(adj),
                   typeof(dds!),
                   typeof(ddi),
                   typeof(linearise!),
                   typeof(B),
                   typeof(x.a),
                   typeof(x)}(nl,
                              lin,
                              adj,
                              dds!,
                              ddi,
                              linearise!,
                              B,
                              cache,
                              residual,
                              Evaluations())
    end
end


# ---------------------------------------------------------------------------- #
# residual and objective                                                       #
# ---------------------------------------------------------------------------- #
"""
    residual!(r::Orbit, F::System, x::Orbit) -> r

The augmented residual at `x`: the field `r.a = ω ∂s a - Σᵢ cᵢ ∂ᵢ a - N(a)`, and zero parameters
`r.p`, the phase conditions, which vanish at the current point. Minus `r` is the right-hand side
of the Newton system.
"""
function residual!(r::Orbit, F::System, x::Orbit)
    tmp = F.cache[1]
    F.evaluations.residual += 1

    # ---- time derivative ----
    F.dds!(tmp, x.a)
    r.a .= frequency(x) .* tmp

    # ---- drift ----
    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, x.a)
        r.a .-= drift(x, i) .* tmp
    end

    # ---- nonlinear operator ----
    F.nl(tmp, x.a)
    r.a .-= tmp

    # ---- phase conditions ----
    fill!(r.p, 0)

    return r
end

"""
    objective(F::System, x::Orbit) -> R

`R = ½‖r‖²`. The residual is left in `F.residual`.
"""
function objective(F::System, x::Orbit)
    r = residual!(F.residual, F, x)
    return norm(r)^2 / 2
end


# ---------------------------------------------------------------------------- #
# gradient, for L-BFGS                                                         #
# ---------------------------------------------------------------------------- #
"""
    gradient!(g::Orbit, F::System, x::Orbit) -> R

Write the gradient of `R = ½‖r‖²` into `g` and return `R`. Moves the linearisation point of the
operators to `x`.
"""
function gradient!(g::Orbit, F::System, x::Orbit)
    tmp = F.cache[1]
    F.evaluations.gradient += 1

    # ---- residual and linearisation point ----
    R = objective(F, x)
    r = F.residual.a
    F.linearise!(F.adj, x.a)

    # ---- field: -ω ∂s r + Σᵢ cᵢ ∂ᵢ r - L⁺ r ----
    F.dds!(g.a, r)
    g.a .*= -frequency(x)

    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, r)
        g.a .+= drift(x, i) .* tmp
    end

    F.adj(tmp, r)
    g.a .-= tmp

    # ---- log-frequency ω ⟨∂s a, r⟩ and drift speeds -⟨∂ᵢ a, r⟩ ----
    F.dds!(tmp, x.a)
    g.p[1] = frequency(x) * dot(tmp, r)

    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, x.a)
        g.p[1 + i] = -dot(tmp, r)
    end

    return R
end


# ---------------------------------------------------------------------------- #
# Jacobian, for Newton–Krylov                                                  #
# ---------------------------------------------------------------------------- #
"""
    jacobian!(out::Orbit, F::System, x::Orbit, δx::Orbit) -> out

The Jacobian of the residual at `x`, with the phase conditions as extra rows, applied to `δx`.
Assumes the linearisation point was set to `x` with `F.linearise!(F.lin, x.a)`.
"""
function jacobian!(out::Orbit, F::System, x::Orbit, δx::Orbit)
    tmp = F.cache[1]
    ∂a  = F.cache[2]
    F.evaluations.jacobian += 1

    # ---- operator on δa: ω ∂s δa - Σᵢ cᵢ ∂ᵢ δa - L δa ----
    F.dds!(tmp, δx.a)
    out.a .= frequency(x) .* tmp

    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, δx.a)
        out.a .-= drift(x, i) .* tmp
    end

    F.lin(tmp, δx.a)
    out.a .-= tmp

    # ---- log-frequency column ω ∂s a δρ and time phase condition ----
    F.dds!(∂a, x.a)
    out.a .+= (frequency(x) * δx.p[1]) .* ∂a
    out.p[1] = dot(∂a, δx.a)

    # ---- drift columns and space phase conditions ----
    for (i, dd!) in enumerate(F.ddi)
        dd!(∂a, x.a)
        out.a .-= drift(δx, i) .* ∂a
        out.p[1 + i] = dot(∂a, δx.a)
    end

    return out
end


# ---------------------------------------------------------------------------- #
# entry point, with one method per search                                      #
# ---------------------------------------------------------------------------- #
"""
    solve!(x::Orbit, F::System, method) -> x

Search for an orbit of the system `F`, starting from `x` and overwriting it with the result.
`method` is an [`LBFGS`](@ref) object, which minimises `½‖r‖²` and is robust far from a solution,
or a [`NewtonHookstep`](@ref) object, which solves `r = 0` and converges fast close to one. Both
hold their own options. A typical search runs L-BFGS first and refines with the hookstep:

```julia
solve!(x, F, LBFGS(maxiter=300))
solve!(x, F, NewtonHookstep(krylov_dim=150))
```
"""
function solve! end
