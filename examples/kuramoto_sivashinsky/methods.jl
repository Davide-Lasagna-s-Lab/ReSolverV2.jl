# L-BFGS against the Newton–Krylov hookstep for the orbit of the Kuramoto–Sivashinsky equation of
# example.jl.
#
#     julia --project=examples examples/kuramoto_sivashinsky/methods.jl
#
# writes methods.png and memory.png next to this file.
#
# The cost of the two methods is compared in operator applications, one nonlinear, linearised or
# adjoint operator each, and in wall-clock time. The first measure is fair only if the three
# operators cost about the same, which is checked first. Two searches are compared: from the
# near-recurrence of example.jl, far from the solution, on the base grid; and from the converged
# orbit perturbed as in convergence.jl, close to the solution, on a grid refined four times.
# L-BFGS runs with the linear and with the jacobian preconditioner, the hookstep with the jacobian
# one.
#
# methods.png: residual against the operator applications and the time, for the two searches.
#
# memory.png: the search from the near-recurrence with a limited number M of stored orbits: M/2
# curvature pairs for L-BFGS, a Krylov space of M - 1 vectors for the hookstep.

using Random
using Printf
using CairoMakie

include("ks.jl")

import ReSolverV2: residual!, gradient!, jacobian!

Random.seed!(1)


# ============================================================================ #
# Initial guess and converged orbit, as in example.jl                          #
# ============================================================================ #

g  = KSGrid(22, 33, 49)
Δt = 0.25
u₀ = 0.1 .* randn(g.Nx)
u₀ .-= sum(u₀) / g.Nx
U  = integrate(u₀, g, 0.05, 4000; every=4000)
U  = integrate(U[:, end], g, 0.05, 8000; every=5)

e, i, m, ℓ = recurrence(U, g, Δt, (12, 20))
x₀         = initial_orbit(U, g, Δt, i, m, ℓ)

# the system on the grid of the orbit y, with a preconditioner of the given kind
system(y::Orbit, kind::Symbol) =
    System(KSNonlinear(y.a.g), KSLinearised(y.a.g), KSLinearised(y.a.g; adjoint=true), dds!, y;
           linearise!, ddi=(ddx!,), B=KSPreconditioner(y; kind))

x★ = copy(x₀)
solve!(x★, system(x★, :jacobian), NewtonHookstep(maxiter=50, krylov_dim=150, Δ=0.1, Δmax=10,
                                                 tol=1e-12, verbose=false))

@printf "orbit: T = %.4f\n" 2π / ReSolverV2.frequency(x★)


# ============================================================================ #
# Cost of the operators                                                        #
# ============================================================================ #

# seconds per call of f(), the best of five batches of n calls, after a call to compile
function seconds(f::Function; n::Int=200)
    f()
    return minimum(@elapsed(for _ in 1:n; f(); end) for _ in 1:5) / n
end

# the operators on a grid refined f times, about the converged orbit
for f in (1, 4)
    gf  = KSGrid(22, 32f + 1, 48f + 1)
    y   = Orbit(resample(x★.a, gf), copy(x★.p))
    v   = Orbit(resample(x★.a, gf), copy(x★.p))
    out = similar(y)
    Fy  = system(y, :jacobian)
    nl  = KSNonlinear(gf)
    lin = linearise!(KSLinearised(gf), y.a)
    adj = linearise!(KSLinearised(gf; adjoint=true), y.a)

    @printf "%3d × %3d: nonlinear %.1f μs, linearised %.1f μs, adjoint %.1f μs\n" gf.Nx gf.Ns 1e6 * seconds(() -> nl(out.a, y.a)) 1e6 * seconds(() -> lin(out.a, v.a)) 1e6 * seconds(() -> adj(out.a, v.a))
    @printf "           residual %.1f μs, gradient %.1f μs, Jacobian action %.1f μs, preconditioner %.1f μs\n" 1e6 * seconds(() -> residual!(out, Fy, y)) 1e6 * seconds(() -> gradient!(out, Fy, y)) 1e6 * seconds(() -> jacobian!(out, Fy, y, v)) 1e6 * seconds(() -> precondition!(out, Fy.B, v))
end


# ============================================================================ #
# Searches                                                                     #
# ============================================================================ #

# ---- close start: the converged orbit perturbed as in convergence.jl, on a grid refined 4 times ----
δ = resample(KSField(KSGrid(22, 7, 7), randn(7, 7)), g)
δ.data .*= 0.01 * norm(x★.a) / norm(δ)
g₄ = KSGrid(22, 129, 193)
x₄ = Orbit(resample(x★.a, g₄) .+ resample(δ, g₄), x★.p .+ 0.01)

# operator applications: nonlinear operators, in the residuals, adjoint operators, in the
# gradients, and linearised operators, in the Jacobian actions
applications(e::NamedTuple) = e.residual + e.gradient + e.jacobian

# a trace that stops the search after tmax seconds
function stopper(trace::Trace, tmax::Real)
    return info -> (trace(info); trace.time[end] > tmax)
end

# one search with a method and a preconditioner, from y, for at most tmax seconds; the method is
# run once briefly first, so that the time does not include compilation
function search(y::Orbit, method::Symbol, kind::Symbol, tol::Real, tmax::Real;
                memory::Int=10, krylov_dim::Int=150)
    F     = system(y, kind)
    trace = Trace()

    make(maxiter, callback) = method === :lbfgs ?
        LBFGS(; memory, maxiter, tol, verbose=false, callback) :
        NewtonHookstep(; krylov_dim, maxiter, tol, Δ=0.1, Δmax=10, verbose=false, callback)

    solve!(copy(y), F, make(2, info -> false))
    F = system(y, kind)
    solve!(copy(y), F, make(100_000, stopper(trace, tmax)))

    return trace
end

starts = (("far: near-recurrence, 33 × 49", x₀, 1e-11, 20.0),
          ("close: perturbed orbit, 129 × 193", x₄, 1e-9, 60.0))
runs   = (("L-BFGS, linear", :lbfgs, :linear), ("L-BFGS, jacobian", :lbfgs, :jacobian),
          ("hookstep, jacobian", :hookstep, :jacobian))
colors = Dict("L-BFGS, linear" => :royalblue, "L-BFGS, jacobian" => :seagreen,
              "hookstep, jacobian" => :crimson)

traces = Dict()
for (sname, y, tol, tmax) in starts, (rname, method, kind) in runs
    trace = search(y, method, kind, tol, tmax)
    traces[(sname, rname)] = trace

    @printf "%-34s %-20s ‖r‖ = %.1e → %.1e in %5d iterations, %6d operator applications, %6.2f s\n" sname rname trace.res[1] trace.res[end] length(trace) - 1 applications(trace.evaluations[end]) trace.time[end]
    flush(stdout)
end


# ============================================================================ #
# Limited memory                                                               #
# ============================================================================ #

budgets = (5, 10, 20, 50, 100, 200)
memtr   = Dict()
for M in budgets
    memtr[(M, :lbfgs)]    = search(x₀, :lbfgs, :linear, 1e-11, 20.0; memory=M ÷ 2)
    memtr[(M, :hookstep)] = search(x₀, :hookstep, :jacobian, 1e-11, 20.0; krylov_dim=M - 1)

    for method in (:lbfgs, :hookstep)
        t = memtr[(M, method)]
        @printf "M = %3d, %-8s ‖r‖ = %.1e in %5d iterations, %6d operator applications, %6.2f s\n" M method t.res[end] length(t) - 1 applications(t.evaluations[end]) t.time[end]
    end
    flush(stdout)
end


# ============================================================================ #
# Figures                                                                      #
# ============================================================================ #

# Logarithmic ticks at integer powers of ten, with minor ticks at 2–9 times them; every third
# decade over a wide range.
function logticks!(ax, lo::Real, hi::Real; axis::Symbol=:x)
    k₁ = floor(Int, log10(lo))
    k₂ = ceil(Int, log10(hi))
    st = k₂ - k₁ > 6 ? 3 : 1

    ks    = filter(k -> mod(k, st) == 0, k₁:k₂)
    major = (10.0 .^ ks, [rich("10", superscript(string(k))) for k in ks])
    minor = st == 1 ? [m * 10.0^k for k in k₁:k₂ - 1 for m in 2:9] : 10.0 .^ (k₁:k₂)

    setproperty!(ax, Symbol(axis, :ticks), major)
    setproperty!(ax, Symbol(axis, :minorticks), minor)
    setproperty!(ax, Symbol(axis, :minorticksvisible), true)
    setproperty!(ax, Symbol(axis, :minorgridvisible), true)

    return ax
end

# a logarithmic axis spanning whole decades around vals
function decades!(ax, vals; axis::Symbol=:x)
    vals = filter(v -> isfinite(v) && v > 0, vals)
    lo   = 10.0^floor(log10(minimum(vals)))
    hi   = 10.0^ceil(log10(maximum(vals)))
    axis === :x ? xlims!(ax, lo, hi) : ylims!(ax, lo, hi)
    logticks!(ax, lo, hi; axis)
    return ax
end

# ---- methods.png: one row per start, residual against applications and time ----
fig = Figure(size=(1100, 800), fontsize=13)

for (row, (sname, _, tol, _)) in enumerate(starts)
    ax1 = Axis(fig[row, 1]; title=sname, xlabel="operator applications", ylabel="‖r‖",
               xscale=log10, yscale=log10)
    ax2 = Axis(fig[row, 2]; title=sname, xlabel="time [s]", ylabel="‖r‖", xscale=log10,
               yscale=log10)

    xs, ts, rs = Float64[], Float64[], Float64[]
    for (rname, _, _) in runs
        tr = traces[(sname, rname)]
        n  = applications.(tr.evaluations)
        t  = max.(tr.time, 1e-4)

        lines!(ax1, max.(n, 1), tr.res; color=colors[rname], label=rname)
        lines!(ax2, t, tr.res; color=colors[rname], label=rname)
        append!(xs, max.(n, 1)); append!(ts, t); append!(rs, tr.res)
    end
    hlines!(ax1, tol; color=:black, linestyle=:dash)
    hlines!(ax2, tol; color=:black, linestyle=:dash)

    decades!(ax1, xs); decades!(ax2, ts)
    decades!(ax1, [rs; tol]; axis=:y); decades!(ax2, [rs; tol]; axis=:y)
    linkyaxes!(ax1, ax2)
    row == 1 && axislegend(ax1; position=:lb)
end

save(joinpath(@__DIR__, "methods.png"), fig)

# ---- memory.png: residual against time for each budget, and the cost against the budget ----
fig = Figure(size=(1300, 420), fontsize=13)
cmap = cgrad(:viridis, length(budgets); categorical=true)

axes = [Axis(fig[1, j]; title, xlabel="time [s]", ylabel="‖r‖", xscale=log10, yscale=log10)
        for (j, title) in enumerate(("L-BFGS, linear, M/2 curvature pairs",
                                     "hookstep, jacobian, M - 1 Krylov vectors"))]
ts, rs = Float64[], Float64[]
for (k, M) in enumerate(budgets), (ax, method) in zip(axes, (:lbfgs, :hookstep))
    tr = memtr[(M, method)]
    t  = max.(tr.time, 1e-4)
    lines!(ax, t, tr.res; color=cmap[k], label="M = $M")
    append!(ts, t); append!(rs, tr.res)
end
for ax in axes
    hlines!(ax, 1e-11; color=:black, linestyle=:dash)
    decades!(ax, ts); decades!(ax, [rs; 1e-11]; axis=:y)
end
linkyaxes!(axes...)
axislegend(axes[2]; position=:lb)

# time to reach the tolerance against the budget; missing if not reached
ax = Axis(fig[1, 3]; title="time to ‖r‖ < 10⁻¹¹", xlabel="stored orbits M", ylabel="time [s]",
          xscale=log10, yscale=log10)
ys = Float64[]
for (method, color, label) in ((:lbfgs, :royalblue, "L-BFGS, linear"),
                               (:hookstep, :crimson, "hookstep, jacobian"))
    y = [memtr[(M, method)].res[end] < 1e-11 ? memtr[(M, method)].time[end] : NaN for M in budgets]
    scatterlines!(ax, collect(budgets), y; color, label, markersize=9)
    append!(ys, y)
end
logticks!(ax, 1, 1000)
xlims!(ax, 3, 300)
decades!(ax, ys; axis=:y)
axislegend(ax; position=:rt)

save(joinpath(@__DIR__, "memory.png"), fig)
