# Build the documentation:
#
#     julia --project=docs docs/make.jl
#
# The figures are produced by the scripts in examples/ and copied into docs/src/assets.

using Documenter
using ReSolverV2

# ---- figures of the examples ----
assets = joinpath(@__DIR__, "src", "assets")
mkpath(assets)

for (folder, prefix) in (("kuramoto_sivashinsky", "ks"), ("lorenz", "lorenz"), ("metric", "metric"))
    dir = joinpath(@__DIR__, "..", "examples", folder)
    for file in filter(endswith(".png"), readdir(dir))
        cp(joinpath(dir, file), joinpath(assets, "$(prefix)_$(file)"); force=true)
    end
end

# ---- pages ----
makedocs(sitename = "ReSolverV2.jl",
         authors  = "Davide Lasagna",
         modules  = [ReSolverV2],
         format   = Documenter.HTML(mathengine = Documenter.KaTeX(),
                                    prettyurls = true,
                                    canonical  = "https://davide-lasagna-s-lab.github.io/ReSolverV2.jl/",
                                    size_threshold = 400_000),
         pages    = ["Home"     => "index.md",
                     "Usage"    => "usage.md",
                     "Theory"   => ["theory/formulation.md",
                                    "theory/optimisation.md",
                                    "theory/root_finding.md",
                                    "theory/preconditioning.md"],
                     "Examples" => ["examples/lorenz.md",
                                    "examples/kuramoto_sivashinsky.md",
                                    "examples/methods.md"],
                     "API"      => "api.md"],
         checkdocs = :exports,
         warnonly  = [:missing_docs, :cross_references])
