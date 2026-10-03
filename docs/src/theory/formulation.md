# Space-time formulation

This page sets up the problem solved by ReSolverV2: the search for a periodic or relative periodic
orbit written as the zero of a residual over space and time, and the objects from which both search
methods are built.

## Periodic and relative periodic orbits

Consider a dynamical system

```math
\partial_t u = N(u) \tag{1}
```

for a field ``u(x', t)`` that depends on the spatial coordinates ``x'`` of the fixed frame and on
time. For brevity the notation shows a single variable ``x'``, but the method applies to any number
of spatial coordinates, and to ordinary differential equations, which have none. Suppose that
``N`` is equivariant under translations along some of the coordinates, the directions ``x'_i``: if
``u`` is a solution, so is its translate ``\tau_\ell u = u(x' + \ell, t)``, with ``\ell = (\ell_i)``
a shift along those directions. In a channel flow these are the streamwise and spanwise directions;
in a pipe, the axial and azimuthal ones; for the Kuramoto–Sivashinsky equation on a periodic
domain, the only spatial direction.

A *periodic orbit* is a solution that repeats after a period ``T``,

```math
u(x', t + T) = u(x', t) . \tag{2}
```

Because of the symmetry, a solution may instead repeat only up to a translation ``\ell_i`` along
the symmetry directions: a *relative periodic orbit*,

```math
u(x', t + T) = u(x' + \ell, t) . \tag{3}
```

It is periodic in a frame moving with the drift speeds ``c_i = -\ell_i/T``, of coordinates

```math
x = x' - c\,t . \tag{4}
```

## Rescaled time and the residual

Express the solution in the moving coordinates ``x`` and in the *rescaled time*
``s = \omega t \in [0, 2\pi)``, with ``\omega = 2\pi/T`` the frequency, keeping the symbol ``u``:
in the new variables ``u = u(x, s)`` is periodic in ``s`` with period ``2\pi``, whatever the period
``T`` of the orbit, and by the chain rule the time derivative becomes

```math
\partial_t \;\longrightarrow\; \omega\,\partial_s - \sum_i c_i\,\partial_i , \tag{5}
```

with ``\partial_i`` the derivative along ``x_i``. Equation (1) becomes the *space-time residual*

```math
r(u, \omega, c) \;=\; \omega\,\partial_s u \;-\; \sum_i c_i\,\partial_i u \;-\; N(u) \;=\; 0 , \tag{6}
```

where, by equivariance, ``N`` acts in the moving frame as in the fixed one. Periodic orbits are the
case ``c = 0``, for which the two frames coincide. Only directions of translational symmetry can
carry a drift, and a further symmetry of the solution can forbid even some of those (see
[Search by root finding](@ref)): the drift directions are chosen by the user.

The whole orbit is represented at once. The period is no longer an integration time, as in a
shooting method, but an unknown, like the drift speeds; no time integration is ever performed, so
the exponential amplification of errors along an unstable orbit, which limits shooting methods to
short or weakly unstable orbits, never arises.

## Space-time fields

The unknown field is an element of the space

```math
\mathcal{U} = \big\{\, u(x, s) \ :\ u \text{ real, } 2\pi\text{-periodic in } s \,\big\} \tag{7}
```

of space-time fields over the spatial domain and one period of rescaled time, with the boundary
conditions of the problem and periodicity along the symmetry directions. ``\mathcal{U}`` carries the
inner product

```math
\langle u, v\rangle = \frac{1}{2\pi\,|\Omega|} \int_0^{2\pi} \!\! \int_\Omega u(x, s)\, v(x, s)\;
\mathrm{d}x\, \mathrm{d}s , \tag{8}
```

with ``\Omega`` the spatial domain, or any discretisation of it, and the norm
``\lVert u\rVert^2 = \langle u, u\rangle``. The residual (6) maps a field and the parameters to a
field, ``r : \mathcal{U} \times \mathbb{R} \times \mathbb{R}^m \to \mathcal{U}``, with ``m`` the
number of drift directions.

## Unknowns

The frequency enters through its logarithm,

```math
\rho = \log\omega, \qquad \omega = e^{\rho} . \tag{9}
```

``\omega`` stays positive whatever the step, and a step in ``\rho`` is a relative change of the
frequency, so that its scale does not depend on the period of the orbit. The unknowns form an
[`Orbit`](@ref), ``p = (u, \rho, c_1, \dots, c_m)``, an element of the space
``\mathcal{P} = \mathcal{U} \times \mathbb{R} \times \mathbb{R}^m``. For two orbits
``p = (u, \rho, c)`` and ``q = (v, \varrho, e)`` the inner product is

```math
\langle p, q\rangle = \langle u, v\rangle + \rho\,\varrho + \sum_i c_i\, e_i , \tag{10}
```

where ``\langle u, v\rangle`` is the inner product (8) of ``\mathcal{U}``. A discretisation with
``N_u`` real degrees of freedom for the field has ``N_u + 1 + m`` unknowns.

## Space-time operators

Both methods are built from five operators on ``\mathcal{U}``. They act on a whole orbit at once,
mapping a space-time field to a space-time field:

- ``N : \mathcal{U} \to \mathcal{U}``, the *nonlinear operator*: the right-hand side of (1)
  extended to space-time fields.
- ``L\{u_0\} : \mathcal{U} \to \mathcal{U}``, the *linearised operator* about a point
  ``u_0 \in \mathcal{U}``, the derivative of ``N`` there:

```math
N(u_0 + \delta u) = N(u_0) + L\{u_0\}\,\delta u + O(\lVert \delta u\rVert^2) . \tag{11}
```

  The braces mark the dependence on ``u_0``, the *linearisation point*, which the methods set
  before using the operator. In the following, ``L`` without braces is taken about the current
  orbit ``u``.
- ``L^+\{u_0\} : \mathcal{U} \to \mathcal{U}``, the *adjoint* of ``L\{u_0\}`` in the inner product
  of ``\mathcal{U}``, defined by ``\langle v, L\{u_0\}\, w\rangle = \langle L^+\{u_0\}\, v, w\rangle``
  for all ``v, w \in \mathcal{U}``.
- ``\partial_s : \mathcal{U} \to \mathcal{U}``, the derivative in rescaled time. On the Fourier
  coefficients of ``u`` in ``s`` it multiplies the mode ``n`` by ``\mathrm{i}\,n``.
- ``\partial_i : \mathcal{U} \to \mathcal{U}``, the derivative along the drift direction ``x_i`` of
  the moving frame. On the Fourier coefficients it multiplies the mode of wavenumber ``k_i`` by
  ``\mathrm{i}\,k_i``.

Being multiplications by imaginary numbers, mode by mode, ``\partial_s`` and ``\partial_i`` are
skew-adjoint in the inner product of ``\mathcal{U}``,

```math
\langle \partial u, v\rangle = -\langle u, \partial v\rangle , \tag{12}
```

so that each derivative is its own adjoint up to a sign. In a discretisation this property must hold
exactly, not only up to discretisation errors, for the gradient of the next page to be exact: Fourier
derivatives on an odd number of points, without the unpaired Nyquist mode, satisfy it.

The user provides the five operators as functions, together with the inner product of the fields;
the search uses nothing else about the system (see [Usage](@ref)).
