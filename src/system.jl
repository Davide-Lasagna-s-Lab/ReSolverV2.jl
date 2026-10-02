# System: a dynamical system ∂t a = N(a) written in space-time form, and the quantities the
# methods need:
#
#     residual    r = ω ∂s a - Σᵢ cᵢ ∂ᵢ a - N(a)
#     objective   J = ½‖r‖²                                   both methods
#     gradient    ∇J, from the adjoint operator                L-BFGS
#     jacobian    J δx, from the linearised operator           Newton–Krylov hookstep
#
# With x = (a, ω, c) and ∂ᵢ the derivative along the i-th drift direction,
#
#     ∇ₐ J       = -ω ∂s r + Σᵢ cᵢ ∂ᵢ r - L⁺ r
#     ∂J/∂ω      =  ⟨∂s a, r⟩
#     ∂J/∂cᵢ     = -⟨∂ᵢ a, r⟩
#
# The gradient uses the skew-adjointness of the derivatives in the inner product, ∂⁺ = -∂.
#
# The operators and the derivatives are given by the user and opaque to the search: the projected
# Navier–Stokes equations of ReSolverEquations, P N(u₀ + E a) on modal coefficients a, as well as
# Kuramoto–Sivashinsky on Fourier coefficients. What they must satisfy, with a the field of an
# Orbit:
#
#     nl(out, a)               nonlinear operator N(a)
#     lin(out, b)              linearised operator L b, about the point set by linearise!
#     adj(out, b)              adjoint operator L⁺ b, about the same point
#     linearise!(op, a)        sets the linearisation point of op, lin or adj, to a
#     dds!(out, a)             derivative along the phase s ∈ [0, 2π)
#     dd!(out, a)              derivative along a drift direction, one function per direction
#
# All write into `out`, of the type of `a`, and return it; the derivatives must be skew-adjoint in
# `dot`, ⟨∂a, b⟩ = -⟨a, ∂b⟩, as Fourier derivatives are. The fields broadcast and support `dot`
# and `similar`.
#
# The Newton system is square: the unknowns ω and cᵢ are matched by phase conditions that remove
# the neutral directions of time and space translations,
#
#     ⟨∂s a, δa⟩ = 0,    ⟨∂ᵢ a, δa⟩ = 0.

"""
    System(nl, lin, adj, dds!, x::Orbit; linearise!, ddi=())

The system with nonlinear, linearised and adjoint operators `nl`, `lin` and `adj`, phase
derivative `dds!` and function `linearise!(op, a)` setting the linearisation point of `lin` or
`adj`, on orbits shaped like `x`. For the projected Navier–Stokes equations of ReSolverEquations
on ReSolverFlowsBase fields,

```julia
System(construct_equations(...)..., dds!, x; linearise! = linearise_about!, ddi=(ddx1!,))
```

`ddi` is a tuple of functions `dd!(out, a) -> out`, the derivatives along the
periodic directions in which the orbit drifts with an unknown speed, one per drift speed in `x.p`:
`(ddx1!,)` for a streamwise drift in a channel, `(ddx1!, ddx3!)` for streamwise and spanwise
drifts. The default `()` searches for periodic orbits. Leave out the directions in which a
symmetry of the problem forbids drifting.
"""
struct System{NL, LIN, ADJ, DS, DD, LN, X}
            nl::NL           # nonlinear operator N(a)
           lin::LIN          # linearised operator L b, about the linearisation point
           adj::ADJ          # adjoint operator L⁺ b, about the same point
          dds!::DS           # derivative along the phase s
           ddi::DD           # derivatives along the drift directions
    linearise!::LN           # sets the linearisation point of lin or adj
         cache::NTuple{3, X} # two scratch fields and the residual

    function System(        nl,
                           lin,
                           adj,
                          dds!,
                             x::Orbit;
                    linearise!,
                           ddi::Tuple=())

        # ---- input checks ----
        length(ddi) == ndrift(x) ||
            throw(ArgumentError("one drift derivative per drift speed in x.p"))

        # ---- assemble ----
        cache = (similar(x.a), similar(x.a), similar(x.a))

        return new{typeof(nl),
                   typeof(lin),
                   typeof(adj),
                   typeof(dds!),
                   typeof(ddi),
                   typeof(linearise!),
                   typeof(x.a)}(nl,
                                lin,
                                adj,
                                dds!,
                                ddi,
                                linearise!,
                                cache)
    end
end


# ---------------------------------------------------------------------------- #
# residual and objective                                                       #
# ---------------------------------------------------------------------------- #
"""
    residual!(r, F::System, x::Orbit) -> r

The residual `r = ω ∂s a - Σᵢ cᵢ ∂ᵢ a - N(a)`.
"""
function residual!(r, F::System, x::Orbit)
    tmp = F.cache[1]

    # ---- time derivative ----
    F.dds!(tmp, x.a)
    r .= frequency(x) .* tmp

    # ---- drift ----
    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, x.a)
        r .-= drift(x, i) .* tmp
    end

    # ---- nonlinear operator ----
    F.nl(tmp, x.a)
    r .-= tmp

    return r
end

"""
    objective(F::System, x::Orbit) -> J

`J = ½‖r‖²`. The residual is left in `F.cache[3]`.
"""
function objective(F::System, x::Orbit)
    r = residual!(F.cache[3], F, x)
    return norm(r)^2 / 2
end


# ---------------------------------------------------------------------------- #
# gradient, for L-BFGS                                                         #
# ---------------------------------------------------------------------------- #
"""
    gradient!(g::Orbit, F::System, x::Orbit) -> J

Write the gradient of `J = ½‖r‖²` into `g` and return `J`. Moves the linearisation point of the
operators to `x`.
"""
function gradient!(g::Orbit, F::System, x::Orbit)
    tmp = F.cache[1]

    # ---- residual and linearisation point ----
    J = objective(F, x)
    r = F.cache[3]
    F.linearise!(F.adj, x.a)

    # ---- coefficients: -ω ∂s r + Σᵢ cᵢ ∂ᵢ r - P L⁺ E r ----
    F.dds!(g.a, r)
    g.a .*= -frequency(x)

    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, r)
        g.a .+= drift(x, i) .* tmp
    end

    F.adj(tmp, r)
    g.a .-= tmp

    # ---- frequency ⟨∂s a, r⟩ and drift speeds -⟨∂ᵢ a, r⟩ ----
    F.dds!(tmp, x.a)
    g.p[1] = dot(tmp, r)

    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, x.a)
        g.p[1 + i] = -dot(tmp, r)
    end

    return J
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

    # ---- operator on δa: ω ∂s δa - Σᵢ cᵢ ∂ᵢ δa - P L E δa ----
    F.dds!(tmp, δx.a)
    out.a .= frequency(x) .* tmp

    for (i, dd!) in enumerate(F.ddi)
        dd!(tmp, δx.a)
        out.a .-= drift(x, i) .* tmp
    end

    F.lin(tmp, δx.a)
    out.a .-= tmp

    # ---- frequency column and time phase condition ----
    F.dds!(∂a, x.a)
    out.a .+= frequency(δx) .* ∂a
    out.p[1] = dot(∂a, δx.a)

    # ---- drift columns and space phase conditions ----
    for (i, dd!) in enumerate(F.ddi)
        dd!(∂a, x.a)
        out.a .-= drift(δx, i) .* ∂a
        out.p[1 + i] = dot(∂a, δx.a)
    end

    return out
end
