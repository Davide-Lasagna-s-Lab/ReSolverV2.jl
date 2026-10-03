# Usage

## Installation

ReSolverV2 is not registered. From the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/ReSolverV2.jl")
```

The package depends only on the Julia standard library. The examples have their own environment,
`examples/Project.toml`, with FFTW and CairoMakie; from the root of the repository,

```
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/kuramoto_sivashinsky/example.jl
```

## The three objects

A search involves three objects.

- An [`Orbit`](@ref) holds the unknowns: the space-time field `a`, in any discretisation, and the
  parameter vector `p = [log ω, c₁, …, c_m]`, the logarithm of the frequency followed by one drift
  speed per drift direction. `ReSolverV2.frequency(x)` returns ``\omega``.
- A [`System`](@ref) holds the functions that define the problem: the nonlinear operator, its
  linearisation and adjoint, the derivatives, and optionally a preconditioner.
- A method, [`LBFGS`](@ref) or [`NewtonHookstep`](@ref), holds the options of the search, and
  [`solve!`](@ref) runs it.

```julia
x = Orbit(a₀, [log(2π / T₀), c₀])                    # initial guess
F = System(nl, lin, adj, dds!, x; linearise!, ddi=(ddx!,), B=P)

solve!(x, F, LBFGS(maxiter=300))                     # far from a solution
solve!(x, F, NewtonHookstep(krylov_dim=150))         # close to it
```

`solve!` overwrites `x` with the result.

## Defining a system

The search knows nothing about the system beyond the functions passed to [`System`](@ref):

```julia
System(nl, lin, adj, dds!, x; linearise!, ddi=(), B=I)
```

With `a` the field of an `Orbit`:

| argument | call | does |
|---|---|---|
| `nl` | `nl(out, a)` | nonlinear operator ``N(a)`` |
| `lin` | `lin(out, b)` | linearised operator ``L\{u_0\}\, b``, about the point set by `linearise!` |
| `adj` | `adj(out, b)` | adjoint operator ``L^+\{u_0\}\, b``, about the same point |
| `linearise!` | `linearise!(op, a)` | sets the linearisation point of `op`, `lin` or `adj`, to `a` |
| `dds!` | `dds!(out, a)` | derivative ``\partial_s`` in rescaled time |
| `ddi` | `dd!(out, a)` for each `dd!` in the tuple | derivative ``\partial_i`` along each drift direction |
| `B` | `precondition!(out, B, p)`, `precondition_adjoint!(out, B, p)` | apply ``B^{-1}`` and ``B^{-+}`` to an orbit |

All write into `out` and return it. `x` is an orbit of the right shape, used to allocate the
workspace; its parameter vector fixes the number of drift directions, which must equal
`length(ddi)`.

The field type of the orbit must

- broadcast, with `.=`, `.+`, scalar multiplication and the like, writing in place;
- support `similar`, `copy` and `LinearAlgebra.dot`, the inner product of the fields, real valued;
- make the derivatives exactly skew-adjoint in that inner product, ``\langle \partial u, v\rangle =
  -\langle u, \partial v\rangle``, and `adj` the exact adjoint of `lin`, for the gradient and the
  Jacobian to be exact.

A plain `Array` works; the examples define their own field types, `KSField` with values on a
space-time grid and `LorenzField` with Fourier coefficients in time, to keep the transforms planned
and the operators allocation-free.

### Checking a system

Before a search, check the three properties that the methods rely on, with random orbits `x`, `v`,
`w`:

```julia
linearise!(lin, x.a); linearise!(adj, x.a)

# linearisation: central difference of nl, exact for a quadratic operator
ε  = 1e-3
FD = (nl(similar(x.a), x.a .+ ε .* v.a) .- nl(similar(x.a), x.a .- ε .* v.a)) ./ 2ε
norm(FD .- lin(similar(x.a), v.a)) / norm(FD)              # ≈ 1e-10 or less

# adjoint: ⟨w, L v⟩ = ⟨L⁺ w, v⟩
dot(w.a, lin(similar(x.a), v.a)) - dot(adj(similar(x.a), w.a), v.a)   # ≈ rounding

# skew-adjoint derivative
dot(dds!(similar(x.a), x.a), v.a) + dot(x.a, dds!(similar(v.a), v.a)) # ≈ rounding
```

The test suite of the package performs these checks on the Kuramoto–Sivashinsky model, together
with finite-difference checks of the gradient and of the Jacobian of the residual.

### Drift directions

`ddi` lists the derivatives along the directions in which the orbit drifts with an unknown speed.
Leave it empty, `ddi=()`, and give `p = [log ω]` for a periodic orbit; give one derivative and one
speed per drift direction for a relative periodic orbit. Include only directions of translational
symmetry that the solution is free to drift along: a direction forbidden by a further symmetry makes
the Newton system singular (see [Search by root finding](@ref)).

## Preconditioners

A preconditioner is any type for which two functions are extended, applying the inverse of ``B``
and its adjoint in the inner product of the orbits:

```julia
struct MyPreconditioner … end

ReSolverV2.precondition!(out::Orbit, B::MyPreconditioner, p::Orbit)         = …  # out = B⁻¹ p
ReSolverV2.precondition_adjoint!(out::Orbit, B::MyPreconditioner, p::Orbit) = …  # out = B⁻⁺ p

F = System(nl, lin, adj, dds!, x; linearise!, ddi, B=MyPreconditioner(…))
```

Both act on the field and on the parameters of the orbit. The default `B = I` applies no
preconditioning. For a real, positive operator diagonal in an orthogonal basis, as the preconditioners
of the examples, ``B^{-+} = B^{-1}`` and the second function can call the first. The choice of ``B``
and its effect are discussed in [Preconditioning](@ref) and measured in the examples.

## Methods and options

[`LBFGS`](@ref) minimises ``\tfrac12\lVert r\rVert^2``:

| option | default | meaning |
|---|---|---|
| `memory` | `10` | number of curvature pairs |
| `maxiter` | `100` | largest number of iterations |
| `tol` | `1e-10` | stop when ``\lVert r\rVert`` < `tol` |
| `verbose`, `io` | `true`, `stdout` | one line per iteration, and where to print it |
| `callback` | `info -> false` | called at the start and after every iteration; `true` stops |

[`NewtonHookstep`](@ref) solves ``r = 0``:

| option | default | meaning |
|---|---|---|
| `krylov_dim` | `50` | largest dimension of the Krylov space |
| `krylov_tol` | `1e-3` | relative residual of the Newton system that stops the Krylov space |
| `Δ`, `Δmax` | `1e-2`, `1` | initial and largest trust-region radius |
| `eta` | `1e-3` | smallest ratio of actual to predicted reduction of an accepted step |
| `maxiter` | `20` | largest number of Newton iterations |
| `tol` | `1e-10` | stop when ``\lVert r\rVert`` < `tol` |
| `verbose`, `io` | `true`, `stdout` | one line per iteration, and where to print it |
| `callback` | `info -> false` | called at the start and after every iteration; `true` stops |

With `verbose=true` the hookstep prints, for every Newton iteration, the residual norm, the
frequency, the trust-region radius, the ratio of actual to predicted reduction, the dimension of the
Krylov space and the relative residual ``\lVert g - Hy\rVert/\beta`` reached by the inner solve.

Some practical guidance, from the examples:

- start with L-BFGS from a poor guess, and switch to the hookstep once ``\lVert r\rVert`` is small;
- with a good preconditioner a loose inner tolerance, `krylov_tol` between ``10^{-1}`` and
  ``10^{-4}``, minimises the total cost; solving the Newton system exactly is wasteful;
- the radius `Δ` is measured in the metric of the preconditioner, so its scale changes with ``B``;
- the best preconditioner need not be the same for the two methods: a positive, self-adjoint ``B``
  bounded below gives a balanced metric for L-BFGS, while the hookstep gains most from a ``B`` close
  to the space-time operator, phase included. Build a second `System` with the same functions and a
  different `B` for the hookstep phase (see [Kuramoto–Sivashinsky equation](@ref)).

## Tracing a search

The callback of both methods receives a named tuple `info` with:

| field | content |
|---|---|
| `iter` | the iteration number of the method |
| `x` | the current orbit |
| `res` | the residual norm ``\lVert r\rVert`` |
| `evaluations` | cumulative numbers of residuals, gradients, Jacobian actions and preconditioner applications of the system |
| `krylov` | for the hookstep, the relative residual of the Newton system after each Arnoldi step of the iteration; empty otherwise |

A [`Trace`](@ref) passed as callback records all of them, with the elapsed time, and can follow
several calls to `solve!`:

```julia
trace = Trace()
solve!(x, F, LBFGS(maxiter=300, callback=trace))
solve!(x, F, NewtonHookstep(callback=trace))

trace.res          # residual norms
trace.time         # seconds since the first record
trace.evaluations  # cumulative evaluations
trace.krylov       # inner-solve histories
```

All the convergence figures of the examples are drawn from such traces.
