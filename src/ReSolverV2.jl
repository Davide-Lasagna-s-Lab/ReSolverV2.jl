module ReSolverV2

# Space-time search for periodic and relative periodic orbits of a dynamical system ∂t a = N(a),
#
#     r(a, ω, c) = ω ∂s a - Σᵢ cᵢ ∂ᵢ a - N(a) = 0,
#
# with space-time field a, frequency ω = 2π/T, phase s ∈ [0, 2π) and drift speeds cᵢ along
# periodic directions. Two methods share one residual:
#
#     LBFGS            minimise ½‖r‖² with L-BFGS, gradient from the adjoint operator
#     NewtonHookstep   solve r = 0 with Newton–Krylov hookstep, Jacobian from the linearised operator
#
# The operators N, L and L⁺ are given by the user, e.g. the three projected Navier–Stokes
# operators of ReSolverEquations.construct_equations.
# L-BFGS (from ResolverOptimAlgorithms) and the Arnoldi iteration (from GMRES.jl) are local
# copies, free to be adapted.

using LinearAlgebra
using Printf

export Orbit, System, LBFGS, NewtonHookstep, solve!

include("orbit.jl")
include("system.jl")
include("lbfgs/linesearch.jl")
include("lbfgs/lbfgs.jl")
include("hookstep/arnoldi.jl")
include("hookstep/hookstep.jl")

end
