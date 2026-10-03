# Search by optimisation

Far from a solution, a Newton step built on a linear model is unreliable. The first method instead
minimises the size of the residual, a smooth functional of the orbit, with a quasi-Newton method:
[`LBFGS`](@ref). Equations of the [Space-time formulation](@ref) are cited as (F*n*).

## Objective

The search minimises

```math
J(p) = \tfrac12 \lVert r(p) \rVert^2 , \tag{1}
```

with ``r`` the residual (F6) and ``\lVert\cdot\rVert`` the norm of the fields. The global minimum
``J = 0`` is attained exactly at the orbits; local minima with ``J > 0`` also exist, and a search
that stalls on one needs a different initial guess.

## Gradient

The gradient of ``J`` is the orbit ``\nabla J`` given by the Riesz representation theorem in the
inner product (F10): for every variation ``\delta p = (\delta u, \delta\rho, \delta c)``,

```math
J(p + \delta p) = J(p) + \langle \nabla J, \delta p\rangle + o(\lVert \delta p\rVert) . \tag{2}
```

The variation changes the residual by

```math
\delta r = \omega\,\partial_s\delta u - \sum_i c_i\,\partial_i\delta u - L\,\delta u
         + \omega\,\partial_s u\;\delta\rho - \sum_i \partial_i u\;\delta c_i , \tag{3}
```

where the frequency term follows from ``\delta\omega = \omega\,\delta\rho``, and ``J`` changes, to
first order, by ``\langle r, \delta r\rangle``, in the inner product of the fields. Moving the
operators onto ``r``, with the skew-adjointness (F12) of the derivatives and the adjoint ``L^+``,

```math
\langle r, \delta r\rangle
= \Big\langle -\omega\,\partial_s r + \sum_i c_i\,\partial_i r - L^{+} r,\ \delta u \Big\rangle
+ \omega\,\langle \partial_s u, r\rangle\,\delta\rho
- \sum_i \langle \partial_i u, r\rangle\,\delta c_i . \tag{4}
```

Matching this with ``\langle \nabla J, \delta p\rangle`` in the inner product (F10) gives the
components of ``\nabla J = (g_u, g_\rho, g_c)``:

```math
g_u = -\omega\,\partial_s r + \sum_i c_i\,\partial_i r - L^{+} r, \qquad
g_\rho = \omega\,\langle \partial_s u, r\rangle, \qquad
g_{c_i} = -\langle \partial_i u, r\rangle . \tag{5}
```

One evaluation of ``J`` costs one nonlinear operator; one gradient costs in addition one adjoint
operator, about the current orbit. The gradient is exact for the discretised problem when the
discrete derivatives are exactly skew-adjoint and the adjoint operator is the exact adjoint of the
discrete linearised operator; the examples check both to rounding.

## Limited-memory BFGS

``J`` is minimised by limited-memory BFGS (Nocedal & Wright 2006, ch. 7). The search direction is

```math
d = -H g , \tag{6}
```

with ``g = \nabla J`` and ``H`` an approximation of the inverse Hessian built from the last
`memory` pairs

```math
s_k = p_{k+1} - p_k, \qquad y_k = g_{k+1} - g_k ,
```

applied by the two-loop recursion, starting from the scaled identity ``\theta I``, with
``\theta = \langle y, s\rangle / \langle y, y\rangle`` of the newest pair. A pair is kept only if its
curvature is positive enough (cautious update),

```math
\langle y_k, s_k\rangle > 10^{-10}\, \lVert s_k\rVert\, \lVert y_k\rVert , \tag{7}
```

so that ``H`` stays positive definite; if ``d`` is not a descent direction, steepest descent is used
instead. The step ``\alpha`` along ``d`` is found by backtracking from ``\alpha = 1``, halving it until
the Armijo condition

```math
J(p + \alpha d) \le J(p) + 10^{-4}\, \alpha\, \langle g, d\rangle \tag{8}
```

holds. L-BFGS is robust from a poor initial guess, but converges only linearly, at a rate set by the
spread of the spectrum of the Hessian of ``J``, which in the Gauss–Newton approximation is
``\mathcal{J}_r^+ \mathcal{J}_r``, with ``\mathcal{J}_r`` the Jacobian of the residual (3). For a
stiff problem this spread is large, and L-BFGS needs a preconditioner (see
[Preconditioning](@ref)).

## Cost

Each iteration costs one gradient, that is one nonlinear and one adjoint operator, plus one
nonlinear operator per rejected line-search trial; with a preconditioner, two applications of
``B^{-1}`` or ``B^{-+}`` for the initial matrix of the recursion.
