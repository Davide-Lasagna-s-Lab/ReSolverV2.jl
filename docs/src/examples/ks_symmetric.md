# Kuramoto–Sivashinsky in the odd subspace

On domains larger than ``L = 22`` the Kuramoto–Sivashinsky equation is often studied in the
subspace of odd solutions, ``u(-x) = -u(x)``, in which the dynamics is still chaotic but the
continuous translation symmetry is absent: Lan & Cvitanović (2008) and Lasagna (2018) computed
large sets of periodic orbits there, on ``L = 38.5``. This page searches for a periodic orbit in the
odd subspace on ``L = 39``, with the keyword `project` of [`System`](@ref) (see
[Symmetric subspaces](../usage.md#Symmetric-subspaces)), and compares the search with searches in the full space. The script
is `examples/kuramoto_sivashinsky/symmetric.jl`; the model is that of the
[Kuramoto–Sivashinsky equation](@ref) page.

## The odd subspace

The equation is equivariant under the reflection ``u(x) \to -u(-x)``, and the fields fixed by it,
the odd fields, form a subspace that the dynamics never leaves: if ``u`` is odd, so are
``u\,\partial_x u``, ``\partial_x^2 u`` and ``\partial_x^4 u``. Odd fields are sine series,
``\hat u_k`` purely imaginary, and have zero mean, so the pinning of the mean is inactive. A
translation ``u(x + \ell)`` of an odd field is odd only for ``\ell = 0`` or ``L/2``: there is no
continuous drift, and the orbits of the subspace are periodic, ``c = 0``. The neutral direction of
translations, ``\partial_x u``, is an even field, outside the subspace.

The grid has ``N_x = 65`` points in space and ``N_s = 73`` in rescaled time. An odd field on it is
determined by ``(N_x - 1)/2 = 32`` values per time level, so the problem in the subspace has
``32 \times 73 + 1 = 2\,337`` real unknowns, against ``65 \times 73 + 2 = 4\,747`` in the full space with
the drift speed.

## Implementation

The projection on the odd fields, ``P u = (u - R u)/2`` with ``R u(x) = u(-x)``, is computed on the
grid, which is symmetric under the reflection: the point ``x_j = (j - 1) L/N_x`` is mapped to the
point ``j' = N_x - j + 2``, modulo ``N_x``, and ``x = 0`` is fixed:

```julia
function odd!(u::KSField)
    Nx = u.g.Nx

    for l in axes(u.data, 2), j in 1:(Nx + 1) ÷ 2
        j′ = mod(Nx - j + 1, Nx) + 1
        a  = u.data[j, l]
        b  = u.data[j′, l]

        u.data[j, l]  = (a - b) / 2
        u.data[j′, l] = (b - a) / 2
    end

    return u
end
```

``P`` is an orthogonal projection in the inner product of the fields, and the result is exactly
odd, to the last bit. The test suite checks that ``P`` is idempotent and self-adjoint, that the
residual of an odd orbit is odd without any projection, and that a projected search stays odd.
The search is then

```julia
F = System(KSNonlinear(g), KSLinearised(g), KSLinearised(g; adjoint=true), dds!, x;
           linearise!, B=KSPreconditioner(x; kind=:jacobian), project=odd!)
```

with `x` an orbit with the log-frequency only, `ddi=()` by default. The jacobian preconditioner
commutes with the reflection: without drift its multipliers satisfy
``\hat A_0(k, -n) = \overline{\hat A_0(k, n)}``, which is how the reflection acts on the Fourier
coefficients.

**The trajectory must be kept odd too.** The odd subspace is unstable in the full space: a
trajectory integrated from an odd initial condition leaves it, the even part growing from rounding
errors as ``e^{0.09 t}`` approximately, from ``10^{-15}`` to ``10^{-12}`` at ``t = 100``,
``10^{-8}`` at ``t = 200`` and order one by ``t = 500``. The integrator of the example therefore
removes the cosine part of the Fourier coefficients at every time step (`integrate(...; odd=true)`).

## The orbit

A chaotic trajectory in the subspace is integrated from a random sine series for 800 time units
after a transient of 200, and its best near-recurrence with period between 20 and 30 is sought,
without shift: relative error 0.013, with ``T_0 = 25.50``. From it, the hookstep with the jacobian
preconditioner, Krylov spaces of at most 200 vectors and ``\tau = 10^{-3}``, converges in 4 Newton
iterations and 214 Jacobian actions, to ``\lVert r\rVert = 3.3 \times 10^{-13}``: a periodic orbit of
period ``T = 25.370562``, odd to the last bit.

![Periodic orbit of Kuramoto–Sivashinsky in the odd subspace, L = 39](../assets/ks_symmetric.png)

*Top: the initial guess and the converged orbit over one period; the dashed line marks
``x = L/2``, about which odd periodic fields are also antisymmetric. Bottom: the search in the odd
subspace; left, the residual against the Newton iterations, all full Newton steps; right, the
relative residual of the GMRES solution of the Newton system against the Arnoldi steps, one curve
per Newton iteration, with the tolerance ``10^{-3}`` dashed.*

## Projected search against the full space

How much does the projection matter? The script repeats the search from the same, exactly odd,
initial guess in the full space, in two ways:

- **full space, periodic orbit**: the same [`System`](@ref) without `project`, `ddi=()` and the
  log-frequency only. The odd fields are invariant under the operators, so in exact arithmetic this
  search is identical to the projected one; it shows the effect of rounding errors alone.
- **full space, relative periodic orbit**: `ddi=(ddx!,)`, the drift speed as an unknown starting
  from zero, without `project`: the search one would run without exploiting the symmetry, as for the
  orbit on ``L = 22``.

| search | `ddi` | `project` | Newton iterations | Jacobian actions | Arnoldi steps per iteration | ``\lVert (I - P)u\rVert / \lVert u\rVert`` at convergence |
|---|---|---|---|---|---|---|
| odd subspace | `()` | `odd!` | 4 | 214 | 56, 51, 52, 55 | 0 |
| full space, periodic orbit | `()` | `identity` | 4 | 281 | 61, 65, 76, 79 | ``1.8 \times 10^{-10}`` |
| full space, relative periodic orbit | `(ddx!,)` | `identity` | 4 | 307 | 63, 73, 82, 89 | ``1.8 \times 10^{-7}`` |

*The last column is the distance of the converged orbit from the odd subspace.*

All three converge, to the same orbit and period, at the same rate. The full-space searches cost
31% and 43% more, and their inner solves grow from one Newton iteration to the next, while those of
the projected search do not. In exact arithmetic the three Krylov spaces would be the same, built
from the odd right-hand side by operators that preserve odd fields. In floating point, rounding
errors put even components into the Krylov basis, and the Jacobian of the full space amplifies them
along its nearly singular even direction, the translation ``\partial_x u``. Without a drift speed
this direction is a null vector of the full-space Jacobian, removed by no phase condition: the
Newton steps contain arbitrary small translations, and the orbit leaves the subspace by ``10^{-7}``
after the first iteration. With a drift speed the phase condition removes it, but the drift speed
fluctuates around zero during the iterations, ``-3 \times 10^{-15}`` at convergence, and translates
the orbit by ``10^{-7}`` to ``10^{-6}``. GMRES spends Arnoldi steps on these even components, which
play no role in the solution.

The projected search avoids all of this: its Jacobian, restricted to the odd fields, is nonsingular
without a phase condition for translations, the even components are removed as soon as they appear,
and the orbit is exactly odd at every iteration. It does not reduce the memory or the cost of an
operator, since the fields keep their full size; a discretisation in sine series would halve both,
at the price of a different field type and different operators.

## References

- Y. Lan and P. Cvitanović, *Unstable recurrent patterns in Kuramoto–Sivashinsky dynamics*,
  Phys. Rev. E 78, 026208 (2008).
- D. Lasagna, *Sensitivity analysis of chaotic systems using unstable periodic orbits*, SIAM J. Appl.
  Dyn. Syst. 17, 547–580 (2018).
