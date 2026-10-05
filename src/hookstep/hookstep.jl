# Newton–Krylov hookstep (Viswanath 2007, https://arxiv.org/abs/0809.1498), as in GMRES.jl and
# NKSearch.
#
# Newton's method solves r(x) = 0 by repeatedly solving the linear system
#
#     𝒥 δx = -r,                                                                   (1)
#
# with 𝒥 the Jacobian of the residual at x, augmented with the phase conditions so that (1) is
# square (see system.jl). Far from a solution the full Newton step is unreliable, so the step is
# restricted to a trust region ‖δx‖ ≤ Δ, inside which the linear model r + 𝒥 δx is trusted:
#
#     min ‖r + 𝒥 δx‖   subject to   ‖δx‖ ≤ Δ.                                      (2)
#
# Krylov space. 𝒥 is never formed: only its action on a vector, `jacobian!`, is available. The
# Arnoldi iteration started from b = -r builds an orthonormal basis Qₙ of the Krylov space
# span{b, 𝒥 b, …, 𝒥ⁿ⁻¹ b} and the (n+1) × n Hessenberg matrix H with 𝒥 Qₙ = Qₙ₊₁ H. For
# δx = Qₙ y, and since Qₙ₊₁ has orthonormal columns with first column b/β, β = ‖b‖,
#
#     ‖r + 𝒥 δx‖ = ‖Qₙ₊₁ (g - H y)‖ = ‖g - H y‖,     g = (β, 0, …, 0),
#
# so (2) restricted to the Krylov space is a small problem in y, ‖y‖ = ‖δx‖:
#
#     min ‖g - H y‖   subject to   ‖y‖ ≤ Δ.                                        (3)
#
# Hookstep. Problem (3) is solved exactly from the SVD of H (`_hookstep` below). Inside the trust
# region its solution is the GMRES least-squares solution; on the boundary it bends from the
# Newton direction towards steepest descent, hence the name. The Krylov space stops growing when
# the residual of the GMRES solution, the unconstrained minimiser of ‖g - H y‖, falls below
# `krylov_tol` β, or at dimension `krylov_dim`, and (3) is then solved in that space (Viswanath
# 2007). The criterion is on the unconstrained problem on purpose: when the trust region is active,
# the residual of the hookstep cannot fall below a level set by Δ, and a criterion on it would grow
# the Krylov space to `krylov_dim` at every step limited by the trust region.
#
# Preconditioning. With a preconditioner B the Arnoldi iteration runs on 𝒥 B⁻¹ and the step is
# δx = B⁻¹ Qₙ y (right preconditioning): the residual of the model is unchanged, and the trust
# region ‖y‖ ≤ Δ bounds ‖B δx‖, the length of the step in the metric of B
# (README, Methodology §4).
#
# Trust-region update. The linear model predicts the reduction ½(β² - ‖g - H y‖²) of ½‖r‖²; the
# step is evaluated by the ratio ρ of the actual reduction to the predicted one:
#
#     ρ < 1/4                  poor model: Δ shrinks by 4,
#     ρ > 3/4 on the boundary  good model, step limited by Δ: Δ doubles, up to `Δmax`,
#     ρ > eta                  the step is accepted, otherwise it is solved again from (3) with
#                              the smaller Δ in the same Krylov space, without new Jacobian actions.
#
# The output reports, for every iteration, whether the accepted step is the full Newton step, inside
# the trust region, or a hookstep on its boundary, and how many trial steps were rejected first.


"""
    NewtonHookstep(; krylov_dim=50, krylov_tol=1e-3, Δ=1e-2, Δmax=1, eta=1e-3,
                     maxiter=20, tol=1e-10, verbose=true, io=stdout,
                     callback=info -> false)

Newton–Krylov hookstep solution of `r = 0`, for [`solve!`](@ref). Krylov spaces grow up to
dimension `krylov_dim`, or until the GMRES solution of the Newton system has a residual below
`krylov_tol` times `‖r‖`, and the hookstep is then computed in that space; the
trust-region radius starts at `Δ` and never exceeds `Δmax`; steps are accepted when the actual
reduction of `½‖r‖²` is at least `eta` times the predicted one. Stops when `‖r‖ < tol`, after
`maxiter` iterations, or when `callback(info)`, called at the start and after every iteration,
returns `true`. `info` is a named tuple with the iteration `iter`, the orbit `x`, the residual
norm `res`, the cumulative `evaluations` of the system, `krylov`, the relative residual of the
GMRES solution of the Newton system after each Arnoldi step of the iteration, and `step`, `"newton"` or `"hook"`, the kind
of the accepted step. A [`Trace`](@ref) records them. Fast near a
solution.
"""
Base.@kwdef struct NewtonHookstep{CB}
    krylov_dim::Int  = 50                        # largest dimension of the Krylov space
    krylov_tol::Real = 1e-3                      # relative GMRES residual stopping the Krylov space
             Δ::Real = 1e-2                      # initial trust-region radius
          Δmax::Real = 1                         # largest trust-region radius
           eta::Real = 1e-3                      # smallest actual/predicted reduction to accept
       maxiter::Int  = 20                        # largest number of Newton iterations
           tol::Real = 1e-10                     # stop when ‖r‖ < tol
       verbose::Bool = true                      # print one line per iteration
            io::IO   = stdout                    # where to print
      callback::CB   = info -> false             # called every iteration; true stops
end


# ---------------------------------------------------------------------------- #
# Newton iterations                                                            #
# ---------------------------------------------------------------------------- #
function solve!(     x::Orbit,
                     F::System,
                method::NewtonHookstep)

    # Δ is the trust-region radius: it starts from the option and changes at every iteration
    (; krylov_dim, krylov_tol, Δ, Δmax, eta, maxiter, tol, verbose, io, callback) = method

    # ---- workspace ----
    b  = similar(x) # right-hand side of (1), (-r, 0): no phase condition on the step
    z  = similar(x) # preconditioned step Qₙ y
    δx = similar(x) # Newton step B⁻¹ Qₙ y
    xn = similar(x) # trial point x + δx
    w  = similar(x) # B⁻¹ v inside the Arnoldi operator

    # ---- initial residual ----
    R = objective(F, x)

    verbose && _print_newton_header(io)
    verbose && _print_newton_row(io, 0, R, frequency(x), Δ, NaN, 0, NaN, "", 0)
    stop = callback(_info(0, x, R, F))

    for iter in 1:maxiter
        (stop || sqrt(2R) < tol) && break

        # ---- Newton system (1) at x ----
        # the Jacobian acts about x: set the linearisation point of the operators
        F.linearise!(F.lin, x.a)

        residual!(b, F, x)
        b .*= -1
        β = norm(b)

        # ---- Krylov space, one Arnoldi step at a time ----
        # right preconditioning: the Arnoldi operator is 𝒥 B⁻¹, and the step δx = B⁻¹ z
        # Arnoldi normalises its starting vector in place, hence the copy of b
        A!  = (out, v) -> jacobian!(out, F, x, _precondition!(w, F, v))
        arn = ArnoldiIteration(A!, copy(b))
        g   = [β]

        H      = arn.H
        krylov = Float64[] # relative residual of the GMRES solution after each Arnoldi step

        for _ in 1:krylov_dim
            # one more basis vector, one more column of H, one more entry of g
            _, H = step!(arn)
            push!(g, 0.0)

            # least-squares (GMRES) solution in the current space: stop once it solves the Newton
            # system well enough. The criterion ignores the trust region: on its boundary the
            # residual of the hookstep stalls at a level set by Δ, which no larger space can lower
            push!(krylov, norm(g - H * (H \ g)) / β)
            krylov[end] < krylov_tol && break

            # breakdown: the new direction lies in the space, which no further step can enlarge
            H[end, end] <= 1e-12 * β && break
        end

        # hookstep (3) in the final space
        y, at_boundary = _hookstep(H, g, Δ)

        # ---- trust region: accept the step, or shrink Δ and solve (3) again ----
        ρ        = 0.0
        rejected = 0 # trial steps rejected before the accepted one
        while true
            # step in the full space and trial point
            lincomb!(z, arn.Q, y)
            _precondition!(δx, F, z)
            xn .= x .+ δx

            # actual and predicted reductions of ½‖r‖²
            Rn        = objective(F, xn)
            predicted = (β^2 - norm(g - H * y)^2) / 2
            ρ         = (R - Rn) / predicted

            # radius update; a ratio that is not a number, e.g. from a vanishing predicted
            # reduction or a residual that overflows, counts as a poor step
            if !(ρ >= 1/4)
                Δ /= 4
            elseif ρ > 3/4 && at_boundary
                Δ = min(2Δ, Δmax)
            end

            # accepted: move to the trial point
            if ρ > eta
                x .= xn
                R  = Rn
                break
            end

            # rejected: give up if the trust region has collapsed, otherwise a shorter hookstep
            if Δ < eps() * norm(x)
                verbose && println(io, "  trust region collapsed: no step in the Krylov space reduces ‖r‖")
                objective(F, x)
                return x
            end
            y, at_boundary = _hookstep(H, g, Δ)
            rejected += 1
        end

        # the accepted step: the Newton (GMRES) step inside the trust region, or a hookstep on
        # its boundary
        step = at_boundary ? "hook" : "newton"

        verbose && _print_newton_row(io, iter, R, frequency(x), Δ, ρ, size(H, 2), krylov[end], step, rejected)
        stop = callback(_info(iter, x, R, F, krylov, step))
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
# radius: trust-region radius after the update; krylov: dimension of the Krylov space; linear:
# relative residual of the GMRES solution of the Newton system; step: newton, the full step inside the trust
# region, or hook, on its boundary; rejected: trial steps rejected before it, each shrinking the radius
_print_newton_header(io::IO) =
    println(io, "  iter      ‖r‖          ω          radius       ρ      krylov    linear    step   rejected")

function _print_newton_row(io::IO, iter, R, ω, Δ, ρ, k, lin, step, rejected)
    @printf(io, "%6d  %.4e  %.6e  %.3e  %8.3f  %6d  %.2e  %6s  %8s\n",
            iter, sqrt(2R), ω, Δ, ρ, k, lin, step, iter == 0 ? "" : string(rejected))
end
