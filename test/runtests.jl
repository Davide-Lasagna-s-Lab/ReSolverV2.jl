using Test
using LinearAlgebra
using Random
using FFTW
using ReSolverFlowsBase
using ReSolverEquations
using ReSolverCases
using ReSolverV2

import ReSolverV2: objective, gradient!, jacobian!

Random.seed!(1)

# ---- orthonormal random modes ----

# M modes per wavenumber over the three velocity components, orthonormal in the grid inner product
# (weights w over the wall-normal points), with ψ(-k) = conj(ψ(k)) on the plane kx = 0 so that
# the expanded fields are real
function random_modes(g, M)
    Ny = size(g, 1)
    w  = repeat(vec(ReSolverFlowsBase.weights(g)), 3)
    sz = transform_size(g)

    Ψ = ntuple(_ -> zeros(ComplexF64, M, sz...), 3)

    for I in CartesianIndices(sz[2:end])
        # orthonormal columns in the weighted inner product: W^(-1/2) Q
        Q = Matrix(qr(randn(ComplexF64, 3Ny, M)).Q)[:, 1:M] ./ sqrt.(w)

        for n in 1:3
            Ψ[n][:, :, I] .= transpose(Q[(n - 1) * Ny + 1:n * Ny, :])
        end
    end

    # Hermitian symmetry on kx = 0: copy each wavenumber to its opposite, conjugated
    Nz, Nt = sz[3], sz[4]
    for l in 1:Nz, n in 1:Nt
        lm = mod(1 - l, Nz) + 1
        nm = mod(1 - n, Nt) + 1
        (lm, nm) < (l, n) && continue
        for c in 1:3
            if (lm, nm) == (l, n)
                Ψ[c][:, :, 1, l, n] .= real.(Ψ[c][:, :, 1, l, n])
            else
                Ψ[c][:, :, 1, lm, nm] .= conj.(Ψ[c][:, :, 1, l, n])
            end
        end
    end

    return Ψ
end

# random orbit of small amplitude on the modes Ψ, with frequency ω and drift speeds c
function random_orbit(g, Ψ, ω, c...)
    a = ProjectedField(g, 1e-1 .* randn(ComplexF64, size(ProjectedField(g, Ψ))), Ψ)
    return Orbit(a, [ω, c...])
end


# ---- problem: plane Couette flow, one drift direction ----
g   = ChannelGrid(7, 13, 5; Nt=5, width=5)
Ψ   = random_modes(g, 4)
eqs = PlaneCouetteFlow(g, 50; fftw_flags=FFTW.ESTIMATE)
x   = random_orbit(g, Ψ, 1.3, 0.2)
F   = System(eqs..., dds!, x; linearise! = linearise_about!, ddi=(ddx1!,))


@testset "Orbit                                                   " begin
    y = copy(x)

    @test norm(y .- x) == 0
    @test (2 .* y).p == 2 .* x.p
    @test dot(x, y) ≈ dot(x.a, x.a) + sum(abs2, x.p)
end


@testset "Gradient                                                " begin
    g₁ = similar(x)
    J  = gradient!(g₁, F, x)

    @test J ≈ objective(F, x)

    # ---- directional derivative against central differences ----
    δ  = random_orbit(g, Ψ, 0.7, -0.3)
    ε  = 1e-6
    FD = (objective(F, x .+ ε .* δ) - objective(F, x .- ε .* δ)) / 2ε

    @test dot(g₁, δ) ≈ FD rtol=1e-6
end


@testset "Jacobian                                                " begin
    δ = random_orbit(g, Ψ, 0.7, -0.3)

    F.linearise!(F.lin, x.a)
    Jδ = jacobian!(similar(x), F, x, δ)

    # ---- residual part against central differences ----
    ε  = 1e-6
    rp = copy(ReSolverV2.residual!(similar(x.a), F, x .+ ε .* δ))
    rm = copy(ReSolverV2.residual!(similar(x.a), F, x .- ε .* δ))

    @test norm(Jδ.a .- (rp .- rm) ./ 2ε) / norm(Jδ.a) < 1e-6
end


@testset "solve!                                                  " begin
    for method in (LBFGS(maxiter=5, verbose=false), NewtonHookstep(maxiter=5, verbose=false))
        y  = copy(x)
        J₀ = objective(F, y)

        solve!(y, F, method)

        @test objective(F, y) < J₀
    end
end
