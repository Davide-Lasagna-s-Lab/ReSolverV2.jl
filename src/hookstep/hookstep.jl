# Newton–Krylov hookstep (Viswanath 2007, https://arxiv.org/abs/0809.1498), as in GMRES.jl and
# NKSearch.
#
# Newton's method solves r(x) = 0 by repeatedly solving the linear system
#
#     J δx = -r,                                                                   (1)
#
# with J the Jacobian of the residual at x, augmented with the phase conditions so that (1) is
# square (see system.jl). Far from a solution the full Newton step is unreliable, so the step is
# restricted to a trust region ‖δx‖ ≤ Δ, inside which the linear model r + J δx is trusted:
#
#     min ‖r + J δx‖   subject to   ‖δx‖ ≤ Δ.                                      (2)
#
# Krylov space. J is never formed: only its action on a vector, `jacobian!`, is available. The
# Arnoldi iteration started from b = -r builds an orthonormal basis Qₙ of the Krylov space
# span{b, J b, …, Jⁿ⁻¹ b} and the (n+1) × n Hessenberg matrix H with J Qₙ = Qₙ₊₁ H. For
# δx = Qₙ y, and since Qₙ₊₁ has orthonormal columns with first column b/β, β = ‖b‖,
#
#     ‖r + J δx‖ = ‖Qₙ₊₁ (g - H y)‖ = ‖g - H y‖,     g = (β, 0, …, 0),
#
# so (2) restricted to the Krylov space is a small problem in y, ‖y‖ = ‖δx‖:
#
#     min ‖g - H y‖   subject to   ‖y‖ ≤ Δ.                                        (3)
#
# Hookstep. Problem (3) is solved exactly from the SVD of H (`_hookstep` below). Inside the trust
# region its solution is the GMRES least-squares solution; on the boundary it bends from the
# Newton direction towards steepest descent, hence the name. As in GMRES.jl, (3) is solved after
# every Arnoldi step, and the Krylov space stops growing when the residual ‖g - H y‖ of the
# hookstep falls below `krylov_tol` β, or at dimension `krylov_dim`.
#
# Trust-region update. The linear model predicts the reduction ½(β² - ‖g - H y‖²) of ½‖r‖²; the
# step is evaluated by the ratio ρ of the actual reduction to the predicted one:
#
#     ρ < 1/4                  poor model: Δ shrinks by 4,
#     ρ > 3/4 on the boundary  good model, step limited by Δ: Δ doubles, up to `Δmax`,
#     ρ > eta                  the step is accepted, otherwise it is solved again from (3) with
#                              the smaller Δ in the same Krylov space, without new Jacobian actions.


"""
    NewtonHookstep(; krylov_dim=50, krylov_tol=1e-3, Δ=1e-2, Δmax=1, eta=1e-3,
                     maxiter=20, tol=1e-10, verbose=true, io=stdout)

Newton–Krylov hookstep solution of `r = 0`, for [`solve!`](@ref). Krylov spaces grow up to
dimension `krylov_dim`, or until the hookstep residual is below `krylov_tol` times `‖r‖`; the
trust-region radius starts at `Δ` and never exceeds `Δmax`; steps are accepted when the actual
reduction of `½‖r‖²` is at least `eta` times the predicted one. Stops when `‖r‖ < tol` or after
`maxiter` iterations. Fast near a solution.
"""
Base.@kwdef struct NewtonHookstep
    krylov_dim::Int  = 50     # largest dimension of the Krylov space
    krylov_tol::Real = 1e-3   # relative residual of (3) that stops the Krylov space
             Δ::Real = 1e-2   # initial trust-region radius
          Δmax::Real = 1      # largest trust-region radius
           eta::Real = 1e-3   # smallest ratio of actual to predicted reduction of an accepted step
       maxiter::Int  = 20     # largest number of Newton iterations
           tol::Real = 1e-10  # stop when ‖r‖ < tol
       verbose::Bool = true   # print one line per iteration
            io::IO   = stdout # where to print
end


# ---------------------------------------------------------------------------- #
# Newton iterations                                                            #
# ---------------------------------------------------------------------------- #
function solve!(     x::Orbit,
                     F::System,
                method::NewtonHookstep)

    (; krylov_dim, krylov_tol, Δmax, eta, maxiter, tol, verbose, io) = method

    # ---- workspace ----
    b  = similar(x) # right-hand side of (1), (-r, 0): no phase condition on the step
    δx = similar(x) # Newton step Qₙ y
    xn = similar(x) # trial point x + δx

    # ---- initial residual and trust region ----
    J = objective(F, x)
    Δ = method.Δ

    verbose && _print_newton_header(io)
    verbose && _print_newton_row(io, 0, J, frequency(x), Δ, NaN, 0)

    for iter in 1:maxiter
        sqrt(2J) < tol && break

        # ---- Newton system (1) at x ----
        # the Jacobian acts about x: set the linearisation point of the operators
        F.linearise!(F.lin, x.a)

        residual!(b.a, F, x)
        b.a .*= -1
        fill!(b.p, 0)
        β = norm(b)

        # ---- Krylov space and hookstep (3), one Arnoldi step at a time ----
        # Arnoldi normalises its starting vector in place, hence the copy of b
        arn = ArnoldiIteration((out, v) -> jacobian!(out, F, x, v), copy(b))
        g   = [β]

        H           = arn.H
        y           = zeros(0)
        at_boundary = false

        for _ in 1:krylov_dim
            # one more basis vector, one more column of H, one more entry of g
            _, H = step!(arn)
            push!(g, 0.0)

            # hookstep in the current space; stop once the linear model is solved well enough
            y, at_boundary = _hookstep(H, g, Δ)
            norm(g - H * y) < krylov_tol * β && break
        end

        # ---- trust region: accept the step, or shrink Δ and solve (3) again ----
        ρ = 0.0
        while true
            # step in the full space and trial point
            lincomb!(δx, arn.Q, y)
            xn .= x .+ δx

            # actual and predicted reductions of ½‖r‖²
            Jn        = objective(F, xn)
            predicted = (β^2 - norm(g - H * y)^2) / 2
            ρ         = (J - Jn) / predicted

            # radius update
            if ρ < 1/4
                Δ /= 4
            elseif ρ > 3/4 && at_boundary
                Δ = min(2Δ, Δmax)
            end

            # accepted: move to the trial point
            if ρ > eta
                x .= xn
                J  = Jn
                break
            end

            # rejected: give up if the trust region has collapsed, otherwise a shorter hookstep
            Δ < eps() * norm(x) && return x
            y, at_boundary = _hookstep(H, g, Δ)
        end

        verbose && _print_newton_row(io, iter, J, frequency(x), Δ, ρ, size(H, 2))
    end

    # ---- leave the residual of x in the cache ----
    objective(F, x)

    return x
end


# ---------------------------------------------------------------------------- #
# trust-region step in the Krylov space                                        #
# ---------------------------------------------------------------------------- #
# Solve (3), min ‖g - H y‖ with ‖y‖ ≤ Δ, as in the hookstep of GMRES.jl.
#
# With the SVD H = U D Vᵀ and p = Uᵀg, the problem decouples along the singular directions. Its
# solution is
#
#     y(μ) = V (p d / (d² + μ)),     ‖y(μ)‖ = ‖p d / (d² + μ)‖,
#
# where μ ≥ 0 is the Lagrange multiplier of the constraint: μ = 0 gives the least-squares (GMRES)
# solution V (p / d), and larger μ damp the directions of small singular values, shortening the
# step and turning it towards steepest descent. If the least-squares solution lies inside the trust
# region it is the answer; otherwise μ is the root of ‖y(μ)‖ = Δ, which decreases monotonically in
# μ and is found by bisection on [0, ‖p‖ max(d) / Δ], an interval where ‖y(μ)‖ ≤ ‖p‖ max(d) / μ
# guarantees a sign change. Returns y and whether it lies on the boundary.
function _hookstep(H::Matrix, g::Vector, Δ::Real)
    U, d, V = svd(H)
    p       = U' * g

    # ---- ‖y(μ)‖ - Δ, decreasing in μ ----
    fun(μ) = norm(p .* d ./ (d.^2 .+ μ)) - Δ

    # ---- least-squares solution, if inside the trust region ----
    fun(0.0) <= 0 && return V * (p ./ d), false

    # ---- boundary: root of fun by bisection ----
    lo = 0.0
    hi = norm(p) * maximum(d) / Δ
    for _ in 1:100
        μ = (lo + hi) / 2
        fun(μ) > 0 ? (lo = μ) : (hi = μ)
    end

    # hi keeps ‖y‖ ≤ Δ
    return V * (p .* d ./ (d.^2 .+ hi)), true
end


# ---------------------------------------------------------------------------- #
# output                                                                       #
# ---------------------------------------------------------------------------- #
_print_newton_header(io::IO) =
    println(io, "  iter      ‖r‖          ω          radius       ρ       krylov")

_print_newton_row(io::IO, iter, J, ω, Δ, ρ, k) =
    @printf(io, "%6d  %.4e  %.6e  %.3e  %8.3f  %6d\n", iter, sqrt(2J), ω, Δ, ρ, k)
