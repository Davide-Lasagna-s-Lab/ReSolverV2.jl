# Search by root finding

Close to a solution, Newton's method converges far faster than any minimisation of the residual.
The second method, [`NewtonHookstep`](@ref), solves ``r = 0`` with a Newton–Krylov iteration
globalised by the hookstep trust region of Viswanath (2007). Equations of the
[Space-time formulation](@ref) are cited as (F*n*).

## Newton step

At the current orbit ``p`` Newton's method looks for the step ``\delta p = (\delta u, \delta\rho,
\delta c)`` that cancels the linearised residual, ``r(p + \delta p) \approx r(p) + \delta r = 0``,
and moves to ``p + \delta p``. From (F6), the linearised residual is

```math
\delta r = A\,\delta u + \omega\,\partial_s u\;\delta\rho - \sum_{i=1}^{m} \partial_i u\;\delta c_i ,
\qquad
A = \omega\,\partial_s - \sum_i c_i\,\partial_i - L , \tag{1}
```

with ``A`` the linearised space-time operator, ``L`` taken about ``u``, and ``m`` the number of drift
directions.

## Two defects and the phase conditions

The equation ``\delta r = -r`` cannot be solved as it stands.

1. *Too many unknowns.* It has as many equations as field unknowns ``\delta u``, but ``1 + m`` more
   unknowns, ``\delta\rho`` and the ``\delta c_i``.
2. *A singular operator.* By the symmetries, a solution ``u`` shifted in rescaled time,
   ``u(x, s + \sigma)``, or along a drift direction, ``u(x + \eta\, e_i, s)``, is again a solution
   with the same ``\omega`` and ``c``. Differentiating ``r = 0`` with respect to ``\sigma`` and
   ``\eta`` at zero gives

```math
A\,\partial_s u = 0, \qquad A\,\partial_i u = 0 , \tag{2}
```

   so at a solution ``A`` is singular, with null space spanned by the *neutral directions*
   ``\partial_s u`` and ``\partial_i u``. Close to a solution ``A`` is nearly singular, and a step can
   wander along these directions without reducing the residual.

Both defects are removed by requiring the step to be orthogonal to the neutral directions, the
*phase conditions*

```math
\langle \partial_s u, \delta u\rangle = 0, \qquad \langle \partial_i u, \delta u\rangle = 0,
\quad i = 1, \dots, m . \tag{3}
```

These are ``1 + m`` equations, as many as the missing ones, and they fix the origin of ``s`` and the
position of the orbit along the drift directions, which the residual cannot determine.

## Block system

Together, the linearised residual and the phase conditions form the Newton system

```math
\underbrace{\begin{bmatrix}
A                                  & \omega\,\partial_s u & -\partial_1 u & \cdots & -\partial_m u \\
\langle \partial_s u, \cdot\,\rangle & 0                    & 0             & \cdots & 0             \\
\langle \partial_1 u, \cdot\,\rangle & 0                    & 0             & \cdots & 0             \\
\vdots                             & \vdots               & \vdots        & \ddots & \vdots        \\
\langle \partial_m u, \cdot\,\rangle & 0                    & 0             & \cdots & 0
\end{bmatrix}}_{\mathcal{J}}
\begin{bmatrix}
\delta u \\ \delta\rho \\ \delta c_1 \\ \vdots \\ \delta c_m
\end{bmatrix}
=
-\begin{bmatrix}
r \\ 0 \\ 0 \\ \vdots \\ 0
\end{bmatrix} . \tag{4}
```

The first block row is the linearised residual, the others the phase conditions. The right-hand
side is minus the augmented residual ``(r, 0, \dots, 0)`` returned by `residual!`, whose parameter
components, the phase conditions, vanish at the current orbit. The operator ``\mathcal{J}`` maps an
orbit to an orbit, the space of the unknowns, so the system is square; `jacobian!` computes its
action on ``\delta p``. The singular ``A`` is bordered by its own null vectors: at an isolated
relative periodic orbit, whose only neutral directions are those of the symmetries, ``\mathcal{J}``
is nonsingular, and Newton's method converges quadratically.

**Which drift directions.** Every drift direction adds a row and a column to ``\mathcal{J}``, and
must be a genuine neutral direction of the problem. If the fields are restricted to a symmetric
subspace that a translation would leave, as the odd solutions ``u(-x) = -u(x)`` of the
Kuramoto–Sivashinsky equation, ``\partial_i u`` has no component in that subspace: its row and
column of ``\mathcal{J}`` vanish and the system is singular. Such directions are left out of `ddi`.

## Krylov space

**Only actions are available.** The operator ``\mathcal{J}`` acts on orbits, whose dimension is the
number of field unknowns plus ``1 + m``: far too large, for a PDE, to be formed or factorised.
What is available is its action on an orbit, ``\delta p \mapsto \mathcal{J}\,\delta p``, computed
at the cost of one linearised operator and a few derivatives. The Newton system is therefore solved
in a Krylov space, built from these actions alone (Saad & Schultz 1986; Knoll & Keyes 2004).

**Arnoldi iteration.** Starting from the right-hand side ``b = -(r, 0, \dots, 0)``, of norm
``\beta = \lVert b\rVert``, the Arnoldi iteration builds an orthonormal basis
``Q_n = [q_1, \dots, q_n]`` of the Krylov space

```math
\mathcal{K}_n = \mathrm{span}\{\, b,\ \mathcal{J} b,\ \dots,\ \mathcal{J}^{\,n-1} b \,\} ,
\qquad q_1 = b/\beta , \tag{5}
```

one vector per Jacobian action. At step ``n`` it applies ``\mathcal{J}`` to the last vector,
``v = \mathcal{J} q_n``, removes its components ``h_{jn} = \langle q_j, v\rangle`` along the basis,
and normalises what remains, ``q_{n+1} = v / h_{n+1,n}`` with ``h_{n+1,n}`` its norm. The
coefficients ``h_{jn}`` fill an ``(n+1) \times n`` upper Hessenberg matrix ``H``, and the construction
is summarised by the Arnoldi relation

```math
\mathcal{J}\, Q_n = Q_{n+1}\, H . \tag{6}
```

The orthogonalisation is modified Gram–Schmidt, done twice: a single pass loses orthogonality once
the new vector lies almost in the span of the basis, as it does when the Krylov space captures the
dominant action of ``\mathcal{J}``; a second pass restores it to rounding ("twice is enough",
Giraud, Langou & Rozložník 2005). If ``h_{n+1,n}`` vanishes the Krylov space is invariant
(breakdown), and it stops growing: the least-squares solution below is then exact.

**Reduction to a small problem.** Look for the step in the Krylov space, ``\delta p = Q_n y`` with
coordinates ``y \in \mathbb{R}^n``, where ``Q_n y = \sum_{j=1}^{n} y_j\, q_j``. Both terms of the
residual of the linear model are combinations of the first ``n + 1`` Arnoldi vectors: the
right-hand side is the first of them, ``b = \beta\, q_1 = Q_{n+1}\, g``, with
``g = (\beta, 0, \dots, 0) \in \mathbb{R}^{n+1}``, and by the Arnoldi relation
``\mathcal{J}\,\delta p = \mathcal{J}\, Q_n y = Q_{n+1}\, H y``. Hence

```math
b - \mathcal{J}\,\delta p = Q_{n+1}\,(g - H y) . \tag{7}
```

The norm of an orbit written on an orthonormal basis is the Euclidean norm of its coordinates: for
any ``z \in \mathbb{R}^{n+1}``,

```math
\lVert Q_{n+1} z\rVert^2
= \sum_{j,k} z_j\, z_k\, \langle q_j, q_k\rangle
= \sum_j z_j^2
= \lVert z\rVert^2 , \tag{8}
```

since ``\langle q_j, q_k\rangle = \delta_{jk}``. With ``z = g - H y``, and with ``z = y`` on the first
``n`` vectors,

```math
\lVert b - \mathcal{J}\,\delta p \rVert = \lVert g - H y \rVert ,
\qquad \lVert \delta p\rVert = \lVert y\rVert . \tag{9}
```

Solving the Newton system in ``\mathcal{K}_n`` is the ``(n+1) \times n`` least-squares problem
``\min_y \lVert g - H y\rVert``, whose solution is the GMRES step. The Krylov dimension is tens to a
few hundred, orders of magnitude below the number of unknowns of a PDE.

## Hookstep

**Why a trust region.** The Newton step solves the linear model exactly, but the model is accurate
only near the current orbit. Far from a solution, and with ``\mathcal{J}`` nearly singular along
directions that the phase conditions do not remove, the GMRES step can be very long and move to a
point where the residual is larger. The step is therefore restricted to a trust region
``\lVert \delta p\rVert \le \Delta``, which by (9) reads, in the Krylov space,

```math
\min_y \lVert g - H y\rVert \quad\text{subject to}\quad \lVert y\rVert \le \Delta . \tag{10}
```

**Solution.** The minimiser satisfies ``(H^\top H + \mu I)\, y = H^\top g`` for a Lagrange multiplier
``\mu \ge 0``, with ``\mu = 0`` if the constraint is inactive and ``\lVert y\rVert = \Delta``
otherwise. With the singular value decomposition ``H = U \Sigma V^\top``, singular values
``\sigma_1, \dots, \sigma_n``, and ``\hat g = U^\top g \in \mathbb{R}^n``, the problem decouples along
the singular directions:

```math
y(\mu) = V\,\hat y(\mu), \qquad
\hat y_k(\mu) = \frac{\sigma_k\, \hat g_k}{\sigma_k^2 + \mu}, \qquad
\lVert y(\mu)\rVert^2 = \sum_{k=1}^{n} \frac{\sigma_k^2\, \hat g_k^2}{(\sigma_k^2 + \mu)^2} . \tag{11}
```

If the GMRES step ``y(0)``, with ``\hat y_k = \hat g_k / \sigma_k``, lies inside the region, it is the
step. Otherwise ``\mu`` is the root of ``\lVert y(\mu)\rVert = \Delta``: the norm decreases
monotonically in ``\mu`` and is bounded by ``\lVert y(\mu)\rVert \le \max_k \sigma_k\, \lVert \hat
g\rVert / \mu``, so the root is unique and lies in ``[0,\ \max_k \sigma_k\, \lVert \hat g\rVert /
\Delta]``, where it is found by bisection.

**Why "hook".** As ``\mu`` grows from zero, ``y(\mu)`` moves from the Newton (GMRES) step,
``\mu = 0``, towards the steepest-descent direction of ``\lVert g - H y\rVert^2``,
``y \approx H^\top g / \mu`` for large ``\mu``, along a curved path. The multiplier damps the
components along small singular values, ``\hat y_k \approx \sigma_k \hat g_k / \mu`` when
``\sigma_k^2 \ll \mu``: these are the nearly singular directions of ``\mathcal{J}``, where the Newton
step is long and least reliable. A short trust region keeps the well-determined part of the step
and discards the rest.

## Trust-region update

The model predicts the reduction ``\tfrac12\big(\beta^2 - \lVert g - Hy\rVert^2\big)`` of
``J = \tfrac12\lVert r\rVert^2``; the trial point ``p + \delta p`` gives the actual one, at the cost
of one nonlinear operator. Their ratio ``\gamma`` drives the radius:

| ratio | meaning | action |
|---|---|---|
| ``\gamma < 1/4``, or not a number | poor model | ``\Delta \leftarrow \Delta/4`` |
| ``\gamma > 3/4``, step on the boundary | good model, step limited by ``\Delta`` | ``\Delta \leftarrow \min(2\Delta, \Delta_{\max})`` |
| ``\gamma > \eta`` | sufficient reduction | accept the step |

A rejected step is recomputed with the smaller radius in the same Krylov space, without new
Jacobian actions. Close to a solution the steps fall inside the region and the hookstep is the
Newton step.

## Inexact Newton

The Krylov space grows until the residual of the model falls below a fraction of the right-hand
side,

```math
\lVert g - H y\rVert \le \tau\, \beta , \tag{12}
```

with ``\tau`` the option `krylov_tol`, or until it reaches `krylov_dim` vectors. This is an
*inexact Newton* method (Dembo, Eisenstat & Steihaug 1982): each step reduces the residual of the
linear model by the factor ``\tau`` only, and close to the solution the Newton iteration converges
linearly with a rate about ``\tau``, rather than quadratically. Solving the Newton system exactly,
``\tau \to 0``, restores quadratic convergence but spends Arnoldi steps on accuracy that the next
Newton iteration does not need; a large ``\tau`` makes each step cheap but needs more Newton
iterations. The best ``\tau`` balances the two, and adaptive choices that tighten ``\tau`` as the
residual decreases are possible (Eisenstat & Walker 1996). The examples measure the cost of
convergence against ``\tau``.

## Cost

Each Newton iteration costs one linearisation, one linearised operator per Krylov vector, one
nonlinear operator per trial step, and the orthogonalisation of each new vector against the basis,
``O(n N)`` operations for ``n`` vectors of ``N`` unknowns. The cost of a search is dominated by the
number of Jacobian actions, which depends on how fast the Krylov solver converges: on the spectrum
of ``\mathcal{J}``, and so on [Preconditioning](@ref).
