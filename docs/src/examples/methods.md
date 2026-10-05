# L-BFGS against the hookstep

The two methods of the package solve the same problem in different ways: [`LBFGS`](@ref) minimises
``R = \tfrac12\lVert r\rVert^2`` with gradients, [`NewtonHookstep`](@ref) solves ``r = 0`` with
Jacobian actions. Counted in iterations the comparison is meaningless, since a Newton iteration
contains a whole Krylov solve. This page compares them at equal cost, on both examples, with the
preconditioner that captures the whole linear space-time operator, and studies two practical
questions: how the methods behave when the memory available for stored orbits is limited, and
whether a hybrid search, L-BFGS first and the hookstep after, pays off. The scripts are
`examples/lorenz/methods.jl` and `examples/kuramoto_sivashinsky/methods.jl`.

## Measuring the cost

**Operator applications.** The cost of a search is dominated by the operators of the system, each
of which acts on a whole space-time field. The figures count *operator applications*: every
application of the nonlinear operator ``N``, of the linearised operator ``L`` or of the adjoint
operator ``L^+`` counts as one. In the two methods,

| operation | operators | where |
|---|---|---|
| residual ``r`` | one ``N`` | every trial point of the L-BFGS line search and of the trust region, and the right-hand side of every Newton system |
| gradient ``\nabla R`` | one ``N`` and one ``L^+`` | every accepted L-BFGS iterate |
| Jacobian action ``\mathcal{J}\,\delta p`` | one ``L`` | every Arnoldi step of the hookstep |

so that an L-BFGS iteration costs at least three applications, one ``N`` and one ``L^+`` for the
gradient and one ``N`` for the first trial point of the line search, and a Newton iteration with a
Krylov space of ``n`` vectors costs ``n`` applications of ``L`` and two or more of ``N``. Derivatives,
preconditioner applications, inner products and the small dense problems of the two methods are not
counted, but are included in the wall-clock time.

**Are the three operators equally expensive?** Counting them alike is fair only if they cost about
the same, which depends on their implementation, not on the methods. The scripts measure them first
(best of five batches, after compilation):

| system | ``N`` | ``L`` | ``L^+`` | residual | gradient | Jacobian action | ``B^{-1}`` |
|---|---|---|---|---|---|---|---|
| Lorenz, ``K = 40`` | OPS_LOR |
| KS, ``33 \times 49`` | OPS_KS1 |
| KS, ``129 \times 193`` | OPS_KS4 |

*Times in μs per call. Residual, gradient and Jacobian action are the functions of the package,
which add the derivatives, the inner products and, for the gradient, a residual and a linearisation
to the operators.*

The check found two inefficiencies in the first versions of the examples, both fixed. In the Lorenz
model the linearised and adjoint operators allocated a ``3 \times 3`` matrix at every grid point and
cost 2.7 times the nonlinear operator; written out component by component they cost the same. In
the Kuramoto–Sivashinsky model the adjoint operator computed ``\partial_x w`` with a transform pair
of its own, five FFTs against three for ``N`` and ``L``; computing it from the coefficients of ``w``,
already available, brings it to four, the minimum for ``U\,\partial_x w``. With the operators
balanced, a gradient costs about two residuals, and a Jacobian action about one and a half.

## The two preconditioners of L-BFGS

With a preconditioner ``B``, L-BFGS works in the metric ``\langle p, q\rangle_B = \langle Bp, Bq\rangle``:
its first direction is ``-M^{-1}\nabla R`` with ``M = B^+B``, and its initial inverse Hessian
``\theta M^{-1}`` (see [Preconditioning](@ref)). Equivalently, it runs plain L-BFGS on the variables
``q = Bp``, in which the Gauss–Newton Hessian of ``R`` is ``(\mathcal{J}B^{-1})^+(\mathcal{J}B^{-1})``. Its
convergence is therefore governed by the *singular values* of ``\mathcal{J}B^{-1}``, the square roots of
the eigenvalues of that Hessian, and not by the clustering of its eigenvalues, which governs GMRES.

For Kuramoto–Sivashinsky two preconditioners built on the symbol
``\hat A_0 = \mathrm{i}(\omega_0 n - c_0 k) + k^4 - k^2`` of the linear space-time operator are compared
(see [Preconditioners](kuramoto_sivashinsky.md#Preconditioners)):

- **linear**, ``B = 1 + |\hat A_0|``, real and positive: it rescales each Fourier mode by the size of
  the operator, so that L-BFGS works in the metric with symbol ``(1 + |\hat A_0|)^2``;
- **jacobian**, ``B = \hat A_0``, complex: the operator itself, metric with symbol ``|\hat A_0|^2``.

For the hookstep the second is three times cheaper; for L-BFGS it is worse, because ``|\hat A_0|`` is
small, between ``0.075`` and ``0.22``, on the linearly unstable wavenumbers ``0 < k < 1`` at ``n = 0``,
where the advection by the orbit, which the preconditioner ignores, is not small. On these modes
``\mathcal{J}B^{-1}`` has singular values up to about 15, and its condition number is
``1.6 \times 10^{3}``, against ``2.4 \times 10^{2}`` with the linear preconditioner. For the Lorenz
system the jacobian preconditioner, ``B = \mathrm{i}\,\omega_0 n\,I - J(\bar u)`` on the mode ``n``, is
used for both methods.

## Kuramoto–Sivashinsky

Two starts, each on the base grid and on a grid refined four times: *far*, the near-recurrence of
the [first search](kuramoto_sivashinsky.md#The-shortest-pre-periodic-orbit), with ``\lVert r\rVert = 7.8 \times 10^{-2}``,
and *close*, the converged orbit perturbed as in the convergence study, with
``\lVert r\rVert = 1.6 \times 10^{-2}``. The searches stop at ``\lVert r\rVert < 10^{-6}``, a residual
small enough to identify the orbit, or after 20 s (base grid) or 60 s (fine grid); L-BFGS keeps 10
curvature pairs, the hookstep up to 150 Krylov vectors.

![L-BFGS against the hookstep, Kuramoto–Sivashinsky](../assets/ks_methods.png)

*Residual against the operator applications (left) and the wall-clock time (right), for the four
searches. Dashed: the tolerance ``10^{-6}``.*

KS_METHODS

## Lorenz system

Two starts, each with ``K = 20`` and ``K = 80`` modes: *far*, the converged orbit perturbed by 10% on
its five lowest modes and by ``0.05`` in the log-frequency, with ``\lVert r\rVert = 45``, and *close*,
perturbed by 1% and ``0.01`` as in the convergence study, with ``\lVert r\rVert = 4.6``. The searches
stop at ``\lVert r\rVert < 10^{-12}`` or after 10 s; L-BFGS keeps 10 curvature pairs, the hookstep up to
100 Krylov vectors.

![L-BFGS against the hookstep, Lorenz](../assets/lorenz_methods.png)

*Residual against the operator applications (left) and the wall-clock time (right), for the four
searches. Dashed: the tolerance ``10^{-12}``.*

LOR_METHODS

## Limited memory

Both methods store orbits: L-BFGS two per curvature pair, ``s_k`` and ``y_k``, the hookstep one per
Krylov vector. For a large problem, a flow on a fine grid for instance, this memory, not the
arithmetic, may limit the search. The figures below repeat the far search with a budget of ``M``
stored orbits: ``M/2`` curvature pairs for L-BFGS, a Krylov space of at most ``M - 1`` vectors for the
hookstep, for ``M`` from 5 to 200. The few work orbits of each method, about six, are not counted.

![Limited memory, Kuramoto–Sivashinsky](../assets/ks_memory.png)

*Kuramoto–Sivashinsky, far start, base grid, tolerance ``10^{-6}``. Left and centre: residual against
time for each budget. Right: time to reach the tolerance; missing points did not reach it.*

![Limited memory, Lorenz](../assets/lorenz_memory.png)

*Lorenz, far start, ``K = 40``, tolerance ``10^{-12}``.*

MEMORY

## A hybrid search

Far from the solution the hookstep spends its first iterations on hooksteps, limited by the trust
region; L-BFGS, robust and cheap per iteration, makes fast progress there, and slows down close to
the solution, where Newton converges in a few iterations. A hybrid search runs L-BFGS until the
residual falls below a threshold ``r_T``, then switches to the hookstep. For Lorenz both phases use
the jacobian preconditioner; for Kuramoto–Sivashinsky L-BFGS uses the linear one, the better metric,
and the hookstep the jacobian one, the better preconditioner of the Newton systems, each phase with
its own [`System`](@ref).

![Hybrid search, Kuramoto–Sivashinsky](../assets/ks_hybrid.png)

*Kuramoto–Sivashinsky, far start, base grid: L-BFGS, the hookstep and the hybrid with thresholds
``r_T = 3 \times 10^{-2}``, ``2 \times 10^{-2}`` and ``10^{-2}`` (dashed).*

![Hybrid search, Lorenz](../assets/lorenz_hybrid.png)

*Lorenz, far starts with ``K = 20`` and ``K = 80``: L-BFGS, the hookstep and the hybrid with
thresholds ``r_T = 10``, ``5``, ``1`` and ``0.1`` (dashed).*

HYBRID

## Conclusions

CONCLUSIONS
