# API

## Unknowns and system

```@docs
Orbit
ReSolverV2.frequency
System
```

## Methods

```@docs
solve!
LBFGS
NewtonHookstep
```

## Tracing

```@docs
Trace
```

## Preconditioners

```@docs
ReSolverV2.precondition!
ReSolverV2.precondition_adjoint!
```

## Residual, gradient and Jacobian

These functions are used by the methods and are not exported; they are useful to check a system.

```@docs
ReSolverV2.residual!
ReSolverV2.objective
ReSolverV2.gradient!
ReSolverV2.jacobian!
ReSolverV2.jacobian_adjoint!
```
