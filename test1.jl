using Turing
using Distributions
using Plots
using StatsPlots
using LinearAlgebra
using Random
using AdvancedHMC
using LogDensityProblems
using LogDensityProblemsAD
using ProgressMeter
using MicroCanonicalHMC
using MuseInference
using LaTeXStrings
using AbstractDifferentiation
using StatsBase
using StatsFuns
using CSV, DataFrames
using MCMCDiagnosticTools
using Zygote: @adjoint
using Zygote
using ChainRules.ChainRulesCore
using Healpix
using BenchmarkTools
using Pathfinder
include("../../HEALPIX_utils.jl")
include("../../UTILITIES.jl")
include("funcs.jl")

ProgressMeter.ijulia_behavior(:clear)
Random.seed!(1123)

nside = 8
lmax = 15

realiz_Dl = CSV.read("Capse_Cl.csv", DataFrame)[1:lmax-1,1]
realiz_Cl = dl2cl(realiz_Dl, 2)
realiz_Cl[1] += 1e-10
realiz_Cl[2] += 1e-10
realiz_map = synfast(realiz_Cl, nside)
realiz_alm = map2alm(realiz_map, lmax=lmax)
realiz_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([realiz_alm], lmax, 1), lmax, 1), Cl2Kl(realiz_Cl))

ϵ=10
N = Diagonal(ϵ*ones(nside2npix(nside)))
inv_N = inv(N)
e = rand(MvNormal(zeros(nside2npix(nside)), N))
gen_map = HealpixMap{Float64,RingOrder}(deepcopy(realiz_map) + e)
gen_alm = map2alm(gen_map, lmax=lmax)
gen_Cl = anafast(gen_map, lmax=lmax)
gen_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([gen_alm], lmax, 1), lmax, 1), Cl2Kl(gen_Cl))

alm₀ = synalm(gen_Cl)
Cℓ₀ = alm2cl(alm₀)
θ₀ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([alm₀], lmax, 1), lmax, 1), Cl2Kl(Cℓ₀))

g = gradient(x->NegLogPosterior(x), θ₀)

d = length(θ₀)

struct LogTargetDensity
    dim::Int
end

LogDensityProblemsAD.logdensity(p::LogTargetDensity, θ) = -NegLogPosterior(θ)
LogDensityProblemsAD.dimension(p::LogTargetDensity) = p.dim
LogDensityProblemsAD.capabilities(::Type{LogTargetDensity}) = LogDensityProblemsAD.LogDensityOrder{1}()

ℓπ = LogTargetDensity(d)
n_LF = 50

n_samples, n_adapts = 3_000, 2_000

metric = DiagEuclideanMetric(d)
ham = Hamiltonian(metric, ℓπ, Zygote)
initial_ϵ = 0.1
integrator = Leapfrog(initial_ϵ)

kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(n_LF)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.75, integrator))

t = time()
samples_HMC, stats_HMC = sample(ham, kernel, θ₀, n_samples, adaptor, n_adapts; drop_warmup = true, progress=true, verbose=true)
HMC_t = time()-t

HMC_ess, HMC_rhat = Summarize(samples_HMC)
println(mean(HMC_ess), "\n")
println(mean(HMC_t), "\n")
h = histogram(HMC_rhat, label="HMC", color="coral2")
display(h)
