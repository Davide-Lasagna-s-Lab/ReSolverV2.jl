# Trace: a simple record of a search, passed as the callback of a method.

# the argument of the callbacks
_info(iter::Int, x::Orbit, J::Real, F, krylov::Vector{Float64}=Float64[]) =
    (iter        = iter,
     x           = x,
     res         = sqrt(2J),
     evaluations = counts(F.evaluations),
     krylov      = krylov)

"""
    Trace()

History of a search, filled when passed as `callback` to [`LBFGS`](@ref) or
[`NewtonHookstep`](@ref). At the start and after every iteration it records:

- `iter`, the iteration number of the method;
- `res`, the residual norm `‖r‖`;
- `p`, the parameters `[log ω, c₁, …]`;
- `time`, the seconds elapsed since the first record;
- `evaluations`, the cumulative numbers of residuals, gradients, Jacobian actions and
  preconditioner applications of the system;
- `krylov`, for the hookstep, the relative residual of the linear model after each Arnoldi step
  of the iteration; empty for L-BFGS and at the start.

The same trace can follow several calls to `solve!`, e.g. L-BFGS and then the hookstep: the
iteration numbers restart at zero with each call.

```julia
trace = Trace()
solve!(x, F, LBFGS(callback=trace))
solve!(x, F, NewtonHookstep(callback=trace))

trace.res # residual norms, in order
```
"""
struct Trace
           iter::Vector{Int}             # iteration number of the method
            res::Vector{Float64}         # residual norm ‖r‖
              p::Vector{Vector{Float64}} # parameters [log ω, c₁, …]
           time::Vector{Float64}         # seconds since the first record
    evaluations::Vector{NamedTuple}      # cumulative evaluations of the system
         krylov::Vector{Vector{Float64}} # relative residuals of the linear model, hookstep only
             t₀::Base.RefValue{Float64}  # wall-clock time of the first record

    Trace() = new(Int[], Float64[], Vector{Float64}[], Float64[], NamedTuple[],
                  Vector{Float64}[], Ref(0.0))
end

# record one iteration; never stops the search
function (trace::Trace)(info::NamedTuple)
    isempty(trace.time) && (trace.t₀[] = time())

    push!(trace.iter, info.iter)
    push!(trace.res, info.res)
    push!(trace.p, copy(info.x.p))
    push!(trace.time, time() - trace.t₀[])
    push!(trace.evaluations, info.evaluations)
    push!(trace.krylov, copy(info.krylov))

    return false
end

Base.length(trace::Trace) = length(trace.res)
