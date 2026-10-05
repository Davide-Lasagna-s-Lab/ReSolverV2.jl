# Limited-memory BFGS on R = ½‖r‖², from ResolverOptimAlgorithms.
#
# The search direction is -H g, with H the inverse-Hessian approximation of the last m curvature
# pairs (s, y) = (x₊ - x, g₊ - g), applied by the two-loop recursion. A pair is kept only if its
# curvature ⟨y, s⟩ is positive enough (cautious update), so that H stays positive definite. If the
# direction is not a descent direction, steepest descent is used instead.
#
# With a preconditioner B the recursion works in the metric M = B⁺B: the initial inverse Hessian is
# θ M⁻¹ instead of θ I, and steepest descent is -M⁻¹ g (README, Methodology §4). With
# B = I both reduce to the plain recursion.

const CAUTIOUS     = 1e-10      # smallest relative curvature ⟨y, s⟩ / (‖s‖‖y‖) of a kept pair
const SCALE_BOUNDS = (1e-8, 1e8) # bounds of the initial scaling θ = ⟨y, s⟩ / ⟨y, M⁻¹ y⟩


"""
    LBFGS(; memory=10, maxiter=100, tol=1e-10, verbose=true, io=stdout,
            callback=info -> false)

L-BFGS minimisation of `½‖r‖²`, keeping `memory` curvature pairs, for [`solve!`](@ref). Stops when
`‖r‖ < tol`, after `maxiter` iterations, or when `callback(info)`, called at the start and after
every iteration, returns `true`. `info` is a named tuple with the iteration `iter`, the orbit `x`,
the residual norm `res`, the cumulative `evaluations` of the system and `krylov`, empty here. A
[`Trace`](@ref) records them. Robust far from a solution.
"""
Base.@kwdef struct LBFGS{CB}
      memory::Int  = 10                        # largest number of curvature pairs
     maxiter::Int  = 100                       # largest number of iterations
         tol::Real = 1e-10                     # stop when ‖r‖ < tol
     verbose::Bool = true                      # print one line per iteration
          io::IO   = stdout                    # where to print
    callback::CB   = info -> false             # called every iteration; true stops
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
    t::NTuple{2, X}    # scratch for the metric M⁻¹
    m::Int             # largest number of pairs
end

LBFGSMemory(x::Orbit, m::Int) = LBFGSMemory(typeof(x)[],
                                            typeof(x)[],
                                            Float64[],
                                            zeros(m),
                                            similar(x),
                                            (similar(x), similar(x)),
                                            m)

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

# p = -H g, by the two-loop recursion in the metric of the preconditioner of F
function _direction!(p::Orbit, g::Orbit, mem::LBFGSMemory, F)
    n = length(mem.s)
    t = mem.t

    # ---- no pairs yet: steepest descent in the metric, -M⁻¹ g ----
    if n == 0
        _metric_inverse!(p, F, g, t[1])
        p .*= -1
        return p
    end

    # ---- first loop, newest pair first ----
    mem.q .= g
    for i in n:-1:1
        mem.α[i] = mem.ρ[i] * dot(mem.s[i], mem.q)
        mem.q .-= mem.α[i] .* mem.y[i]
    end

    # ---- initial inverse Hessian θ M⁻¹, θ = ⟨y, s⟩ / ⟨y, M⁻¹ y⟩ of the newest pair ----
    _metric_inverse!(t[2], F, mem.y[end], t[1])

    ys = dot(mem.y[end], mem.s[end])
    yy = dot(mem.y[end], t[2])
    θ  = clamp(ys / yy, SCALE_BOUNDS...)

    # ---- second loop, oldest pair first ----
    _metric_inverse!(p, F, mem.q, t[1])
    p .*= θ
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

    (; memory, maxiter, tol, verbose, io, callback) = method

    # ---- workspace ----
    mem = LBFGSMemory(x, memory)
    g   = similar(x) # gradient at x
    gt  = similar(x) # gradient at the trial point
    p   = similar(x) # search direction
    xt  = similar(x) # trial point
    s   = similar(x) # step
    y   = similar(x) # gradient change

    # ---- initial point ----
    R = gradient!(g, F, x)
    α = 0.0

    verbose && _print_lbfgs_header(io)
    verbose && _print_lbfgs_row(io, 0, R, norm(g), α, frequency(x))
    stop = callback(_info(0, x, R, F))

    for iter in 1:maxiter
        (stop || sqrt(2R) < tol) && break

        # ---- quasi-Newton direction, or steepest descent if it does not descend ----
        _direction!(p, g, mem, F)
        dot(g, p) >= 0 && (p .= .-g)

        # ---- step along the direction ----
        α, _ = _linesearch!(xt, F, x, p, g, R)
        Rt   = gradient!(gt, F, xt)

        # ---- curvature pair ----
        s .= xt .- x
        y .= gt .- g
        _push_pair!(mem, s, y)

        # ---- accept ----
        x .= xt
        g .= gt
        R  = Rt

        verbose && _print_lbfgs_row(io, iter, R, norm(g), α, frequency(x))
        stop = callback(_info(iter, x, R, F))
    end

    # ---- leave the residual of x in the cache ----
    objective(F, x)

    return x
end


# ---------------------------------------------------------------------------- #
# output                                                                       #
# ---------------------------------------------------------------------------- #
_print_lbfgs_header(io::IO) =
    println(io, "  iter      ‖r‖         ‖∇R‖        step          ω")

_print_lbfgs_row(io::IO, iter, R, gnorm, α, ω) =
    @printf(io, "%6d  %.4e  %.4e  %.3e  %.6e\n", iter, sqrt(2R), gnorm, α, ω)
