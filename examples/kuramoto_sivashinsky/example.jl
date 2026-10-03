# An orbit of the Kuramoto–Sivashinsky equation on L = 22, the domain studied by Cvitanović,
# Davidchack and Siminos (2010), searched for as a relative periodic orbit. The initial guess is a
# near-recurrence of a chaotic trajectory, away from the equilibria, with period between 12 and 20;
# L-BFGS brings the residual down, and the Newton–Krylov hookstep converges it. The search runs
# twice, without and with the diagonal Fourier preconditioner of ks.jl. It finds the shortest
# pre-periodic orbit of the system, traversed twice: T = 20.5057, zero drift.
#
#     julia --project=examples examples/kuramoto_sivashinsky/example.jl
#
# writes example.png next to this file.

using Random
using Printf
using CairoMakie

include("ks.jl")

Random.seed!(1)


# ---- grid: 33 points along x, 49 in rescaled time ----
g = KSGrid(22, 33, 49)

# ---- chaotic trajectory: random snapshot of zero mean, which KS conserves, transient of 200,
#      then 400 time units ----
Δt = 0.25
u₀ = 0.1 .* randn(g.Nx)
u₀ .-= sum(u₀) / g.Nx
U  = integrate(u₀, g, 0.05, 4000; every=4000)
U  = integrate(U[:, end], g, 0.05, 8000; every=5)

# ---- initial guess: the best near-recurrence with period in [12, 20], with drift ----
e, i, m, ℓ = recurrence(U, g, Δt, (12, 20))
x₀         = initial_orbit(U, g, Δt, i, m, ℓ)

@printf "initial guess: recurrence error %.3f, T = %.2f, shift = %.2f\n" e m * Δt ℓ

# ---- operators ----
nl  = KSNonlinear(g)
lin = KSLinearised(g)
adj = KSLinearised(g; adjoint=true)

# ---- search, without and with preconditioner ----
runs = map((("no preconditioner", I), ("preconditioner", KSPreconditioner(x₀)))) do (name, B)
    F     = System(nl, lin, adj, dds!, x₀; linearise!, ddi=(ddx!,), B)
    x     = copy(x₀)
    trace = Trace()

    solve!(x, F, LBFGS(maxiter=300, verbose=false, callback=trace))
    nlbfgs = length(trace)

    solve!(x, F, NewtonHookstep(maxiter=20, krylov_dim=150, Δ=0.1, Δmax=10, tol=1e-11,
                                verbose=false, callback=trace))

    T = 2π / ReSolverV2.frequency(x)
    @printf "%-18s ‖r‖ = %.2e after L-BFGS, %.2e after hookstep; T = %.4f, shift = %.4f\n" name trace.res[nlbfgs] trace.res[end] T -x.p[2] * T

    (; name, x, trace, nlbfgs)
end


# ============================================================================ #
# Figure                                                                       #
# ============================================================================ #

# Logarithmic ticks on an axis of ax: major ticks at integer powers of ten, labelled 10ⁿ, every
# decade or every third decade, at multiples of three, over a wide range; minor ticks at 2–9 times the powers of ten, or
# at every decade over a wide range.
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

# the orbit over one period in the fixed frame, u(x', t) = a(x' - c t, ωt)
function fixed_frame(x::Orbit)
    T = 2π / ReSolverV2.frequency(x)
    c = x.p[2]
    t = (0:g.Ns - 1) .* (T / g.Ns)
    u = reduce(hcat, [shift(x.a.data[:, l], -c * t[l], g) for l in 1:g.Ns])

    return t, u
end

fig = Figure(size=(1200, 420), fontsize=14)

# ---- initial guess and converged orbit ----
for (j, (title, x)) in enumerate((("initial guess", x₀), ("converged orbit", runs[2].x)))
    t, u = fixed_frame(x)
    ax   = Axis(fig[1, j]; title, xlabel="x", ylabel="t")
    hm   = heatmap!(ax, xpoints(g), t, u; colormap=:balance, colorrange=(-3, 3))
    j == 2 && Colorbar(fig[1, 3], hm; label="u")
end

# ---- residual histories ----
ax = Axis(fig[1, 4]; title="residual; dashed: L-BFGS → hookstep", xlabel="iteration",
          ylabel="‖r‖", yscale=log10)
for (run, color) in zip(runs, (:gray40, :crimson))
    lines!(ax, 0:length(run.trace) - 1, run.trace.res; color, label=run.name)
    vlines!(ax, run.nlbfgs - 1; color, linestyle=:dash)
end
axislegend(ax; position=:lb)

res = reduce(vcat, [run.trace.res for run in runs])
logticks!(ax, minimum(res), maximum(res); axis=:y)

colsize!(fig.layout, 4, Relative(0.4))
save(joinpath(@__DIR__, "example.png"), fig)
