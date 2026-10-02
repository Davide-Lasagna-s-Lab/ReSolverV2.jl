# Limited-memory BFGS on J = ½‖r‖², from ResolverOptimAlgorithms.
#
# The search direction is -H g, with H the inverse-Hessian approximation of the last m curvature
# pairs (s, y) = (x₊ - x, g₊ - g), applied by the two-loop recursion. A pair is kept only if its
# curvature ⟨y, s⟩ is positive enough (cautious update), so that H stays positive definite. If the
# direction is not a descent direction, steepest descent is used instead.

const CAUTIOUS     = 1e-10      # smallest relative curvature ⟨y, s⟩ / (‖s‖‖y‖) of a kept pair
const SCALE_BOUNDS = (1e-8, 1e8) # bounds of the initial inverse-Hessian scaling ⟨y, s⟩ / ⟨y, y⟩


"""
    LBFGS(; memory=10, maxiter=100, tol=1e-10, verbose=true, io=stdout)

L-BFGS minimisation of `½‖r‖²`, keeping `memory` curvature pairs, for [`solve!`](@ref). Stops when
`‖r‖ < tol` or after `maxiter` iterations. Robust far from a solution.
"""
Base.@kwdef struct LBFGS
     memory::Int  = 10     # largest number of curvature pairs
    maxiter::Int  = 100    # largest number of iterations
        tol::Real = 1e-10  # stop when ‖r‖ < tol
    verbose::Bool = true   # print one line per iteration
         io::IO   = stdout # where to print
end


# ---------------------------------------------------------------------------- #
# curvature pairs                                                              #
# ---------------------------------------------------------------------------- #
struct LBFGSMemory{X<:Orbit}
    s::Vector{X}       # steps x₊ - x, oldest first
    y::Vector{X}       # gradient changes g₊ - g
    ρ::Vector{Float64} # 1 / ⟨y, s⟩
    α::Vector{Float64} # coefficients of the first loop
    q::X               # vector transformed by the first loop
    m::Int             # largest number of pairs
end

LBFGSMemory(x::Orbit, m::Int) = LBFGSMemory(typeof(x)[], typeof(x)[], Float64[], zeros(m), similar(x), m)

# keep the pair (s, y) if its curvature is positive enough, dropping the oldest beyond m pairs
function _push_pair!(mem::LBFGSMemory, s::Orbit, y::Orbit)
    ys = dot(y, s)

    # ---- cautious update ----
    ys <= max(eps(), CAUTIOUS * norm(s) * norm(y)) && return mem

    # ---- limited memory ----
    if length(mem.s) == mem.m
        popfirst!(mem.s)
        popfirst!(mem.y)
        popfirst!(mem.ρ)
    end

    push!(mem.s, copy(s))
    push!(mem.y, copy(y))
    push!(mem.ρ, 1 / ys)

    return mem
end

# p = -H g, by the two-loop recursion
function _direction!(p::Orbit, g::Orbit, mem::LBFGSMemory)
    n = length(mem.s)

    # ---- no pairs yet: steepest descent ----
    n == 0 && return (p .= .-g)

    # ---- first loop, newest pair first ----
    mem.q .= g
    for i in n:-1:1
        mem.α[i] = mem.ρ[i] * dot(mem.s[i], mem.q)
        mem.q .-= mem.α[i] .* mem.y[i]
    end

    # ---- initial inverse Hessian γ I, γ = ⟨y, s⟩ / ⟨y, y⟩ of the newest pair ----
    ys = dot(mem.y[end], mem.s[end])
    yy = dot(mem.y[end], mem.y[end])
    γ  = clamp(ys / yy, SCALE_BOUNDS...)

    # ---- second loop, oldest pair first ----
    p .= γ .* mem.q
    for i in 1:n
        β = mem.ρ[i] * dot(mem.y[i], p)
        p .+= (mem.α[i] - β) .* mem.s[i]
    end

    p .*= -1

    return p
end


# ---------------------------------------------------------------------------- #
# iterations                                                                   #
# ---------------------------------------------------------------------------- #
function solve!(     x::Orbit,
                     F::System,
                method::LBFGS)

    (; memory, maxiter, tol, verbose, io) = method

    # ---- workspace ----
    mem = LBFGSMemory(x, memory)
    g   = similar(x) # gradient at x
    gt  = similar(x) # gradient at the trial point
    p   = similar(x) # search direction
    xt  = similar(x) # trial point
    s   = similar(x) # step
    y   = similar(x) # gradient change

    # ---- initial point ----
    J = gradient!(g, F, x)
    α = 0.0

    verbose && _print_lbfgs_header(io)
    verbose && _print_lbfgs_row(io, 0, J, norm(g), α, frequency(x))

    for iter in 1:maxiter
        sqrt(2J) < tol && break

        # ---- quasi-Newton direction, or steepest descent if it does not descend ----
        _direction!(p, g, mem)
        dot(g, p) >= 0 && (p .= .-g)

        # ---- step along the direction ----
        α, _ = _linesearch!(xt, F, x, p, g, J)
        Jt   = gradient!(gt, F, xt)

        # ---- curvature pair ----
        s .= xt .- x
        y .= gt .- g
        _push_pair!(mem, s, y)

        # ---- accept ----
        x .= xt
        g .= gt
        J  = Jt

        verbose && _print_lbfgs_row(io, iter, J, norm(g), α, frequency(x))
    end

    # ---- leave the residual of x in the cache ----
    objective(F, x)

    return x
end


# ---------------------------------------------------------------------------- #
# output                                                                       #
# ---------------------------------------------------------------------------- #
_print_lbfgs_header(io::IO) =
    println(io, "  iter      ‖r‖         ‖∇J‖        step          ω")

_print_lbfgs_row(io::IO, iter, J, gnorm, α, ω) =
    @printf(io, "%6d  %.4e  %.4e  %.3e  %.6e\n", iter, sqrt(2J), gnorm, α, ω)
