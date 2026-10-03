# ReSolverV2.jl

*Space-time search for periodic and relative periodic orbits of dynamical systems.*

Developed by Davide Lasagna, University of Southampton.

Many dynamical systems, from low-dimensional models to turbulent flows, organise their chaotic
dynamics around unstable periodic orbits: solutions that repeat exactly after a period ``T``, or
repeat up to a translation along a direction of symmetry, the *relative* periodic orbits. Finding
them is hard because they are unstable: integrating forward in time from a nearby state, as a
shooting method does, amplifies every error exponentially over the period, and the longer or more
unstable the orbit, the worse the problem.

ReSolverV2 avoids time integration altogether. It treats a whole orbit as a single unknown: the
field over one period, written in terms of the rescaled time ``s = \omega t \in [0, 2\pi)``, together
with the frequency ``\omega = 2\pi/T`` and the drift speeds ``c_i``. An orbit is a zero of the
space-time residual

```math
r(u, \omega, c) = \omega\,\partial_s u - \sum_i c_i\,\partial_i u - N(u) ,
```

which measures, at every point in space and rescaled time, how far the candidate field is from
satisfying the equation ``\partial_t u = N(u)``. Two methods drive this residual to zero:

| method | solves | uses | best |
|---|---|---|---|
| [`LBFGS`](@ref) | ``\min \tfrac12\lVert r\rVert^2`` | gradient from the adjoint operator, line search | far from a solution |
| [`NewtonHookstep`](@ref) | ``r = 0`` | Jacobian from the linearised operator, Krylov space, trust region | close to a solution |

## Features

- **Generic.** The search knows nothing about the system beyond a handful of functions supplied by
  the user: the nonlinear operator, its linearisation and adjoint, and the derivatives in rescaled
  time and along the drift directions. The same code serves an ODE such as the Lorenz system, a PDE
  such as the Kuramoto–Sivashinsky equation, or a large flow solver, in any discretisation.
- **Relative periodic orbits.** Drift speeds along any set of symmetry directions are unknowns of
  the search, with the phase conditions that make the Newton system square.
- **Preconditioning.** A preconditioner changes the metric in which both methods work, through a
  two-function interface; the examples show that it decides whether the search converges at all.
- **Tracing.** A [`Trace`](@ref) records the residual, the parameters, the evaluations of the
  operators and the convergence of every inner solve.
- **Self-contained.** The package depends only on the Julia standard library; L-BFGS and the
  Arnoldi iteration are local, readable implementations.

## A first example

An orbit of the Kuramoto–Sivashinsky equation on a periodic domain of length ``L = 22``,
discretised with Fourier modes in space and rescaled time, searched for as a relative periodic orbit
from a near-recurrence of a chaotic trajectory; the search finds the shortest pre-periodic orbit of
the system, traversed twice (see [Kuramoto–Sivashinsky equation](@ref) for the full analysis):

```julia
using ReSolverV2
include("examples/kuramoto_sivashinsky/ks.jl")

g = KSGrid(22, 33, 49)                          # L = 22, 33 points in x, 49 in rescaled time

U          = integrate(u₀, g, 0.05, 8000; every=5)   # chaotic trajectory
e, i, m, ℓ = recurrence(U, g, 0.25, (12, 20))        # its best near-recurrence
x          = initial_orbit(U, g, 0.25, i, m, ℓ)      # Orbit: field, [log ω, c]

F = System(KSNonlinear(g), KSLinearised(g), KSLinearised(g; adjoint=true), dds!, x;
           linearise!, ddi=(ddx!,), B=KSPreconditioner(x))

solve!(x, F, LBFGS(maxiter=300))
solve!(x, F, NewtonHookstep(krylov_dim=150))
```

![Shortest pre-periodic orbit of Kuramoto–Sivashinsky on L = 22](assets/ks_example.png)

## Contents

```@contents
Pages = ["usage.md",
         "theory/formulation.md", "theory/optimisation.md", "theory/root_finding.md",
         "theory/preconditioning.md",
         "examples/lorenz.md", "examples/kuramoto_sivashinsky.md",
         "api.md"]
Depth = 1
```

## A note on how this package was written

This package was written with AI assistance (Claude, by Anthropic). Developing it would have taken
me about two months, with a great deal of reasoning, manual derivations and reading. Claude did it
in a day. — Davide Lasagna
