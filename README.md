# ReSolverV2.jl

Space-time search for periodic and relative periodic orbits of a dynamical system `∂t a = N(a)`.
The unknowns are the space-time field `a`, the frequency `ω = 2π/T` and optional drift speeds
`cᵢ` along periodic directions. The residual is

```
r(a, ω, c) = ω ∂s a - Σᵢ cᵢ ∂ᵢ a - N(a),
```

with phase `s ∈ [0, 2π)`. The operators — `N`, its linearisation and its adjoint — are given by
the user and opaque to the search: for flows, the projected Navier–Stokes operators
`P N(u₀ + E a)` on modal coefficients `a`, from [ReSolverEquations](https://github.com/Davide-Lasagna-s-Lab/ReSolver-Equations.jl);
grids and flow cases come from [ReSolverCases](https://github.com/Davide-Lasagna-s-Lab/ReSolver-Cases.jl).

## Usage

```julia
using ReSolverFlowsBase, ReSolverEquations, ReSolverCases, ReSolverV2

g   = ChannelGrid(33, 65, 33; Nt=33, α=1.14, β=2.5)
eqs = PlaneCouetteFlow(g, 400)                 # (nl, lin, adj)

x = Orbit(ProjectedField(g, a₀, Ψ), [ω₀, c₀])  # coefficients on modes Ψ, frequency, drift speed
F = System(eqs..., dds!, x;                    # nl, lin, adj and the phase derivative
           linearise! = linearise_about!,      # linearisation point of lin and adj
           ddi=(ddx1!,))                       # one drift speed, along x1

solve!(x, F, LBFGS(maxiter=1000))              # far from a solution: minimise ½‖r‖²
solve!(x, F, NewtonHookstep(maxiter=20))       # close to it: Newton–Krylov hookstep
```

## Methods

| method | solves | uses |
|---|---|---|
| `LBFGS` | `min ½‖r‖²` | gradient from the adjoint operator, L-BFGS with backtracking Armijo line search |
| `NewtonHookstep` | `r = 0` | Jacobian from the linearised operator, Arnoldi Krylov space, hookstep trust region (Viswanath 2007) |

Both stop when `‖r‖ < tol`. The Newton system is made square by phase conditions,
`⟨∂s a, δa⟩ = 0` and `⟨∂ᵢ a, δa⟩ = 0`, which fix the time and space translations matched to `ω`
and `cᵢ`.

## Structure

| file | content |
|---|---|
| `src/orbit.jl` | `Orbit`: field and parameters as one vector; broadcasting, `dot` |
| `src/system.jl` | `System`: operators and derivatives given by the user; residual, objective `½‖r‖²`, gradient, Jacobian action |
| `src/lbfgs/` | `LBFGS` and its `solve!`, with the line search; a local copy of ResolverOptimAlgorithms |
| `src/hookstep/` | `NewtonHookstep` and its `solve!`: Arnoldi iteration (from GMRES.jl), trust-region step from the SVD of the Hessenberg matrix |

## Status

A sketch: the gradient and the Jacobian are verified against finite differences, and both methods
reduce the residual on a small plane Couette problem. Open points:

- modes: build the basis from resolvent modes (ChannelResolvent) and load the Gibson orbits;
- preconditioning: weighted inner products (`FarazmandWeight` in ReSolverFlowsBase) for both
  methods;
- frequency: optimise `log ω` to keep it positive, as in SpaceTimeResiduals;
- output: iteration callback, trace and saving of intermediate orbits;
- continuation in the Reynolds number and other parameters.
