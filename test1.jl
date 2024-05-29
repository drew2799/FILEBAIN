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
using Threads
include("HEALPIX_utils.jl")
include("UTILITIES.jl")
include("funcs.jl")

ProgressMeter.ijulia_behavior(:clear)
Random.seed!(1123)

nside = 64
lmax = 127

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

g = gradient(x->NegLogPosterior(x), gen_θ)

d = length(gen_θ)

struct LogTargetDensity
    dim::Int
end

LogDensityProblemsAD.logdensity(p::LogTargetDensity, θ) = -NegLogPosterior(θ)
LogDensityProblemsAD.dimension(p::LogTargetDensity) = p.dim
LogDensityProblemsAD.capabilities(::Type{LogTargetDensity}) = LogDensityProblemsAD.LogDensityOrder{1}()

ℓπ = LogTargetDensity(d)
n_LF = 50

n_samples, n_adapts, n_chains = 10_000, 10_000, 2

metric = DiagEuclideanMetric(d)
ham = Hamiltonian(metric, ℓπ, Zygote)
initial_ϵ = 0.1
integrator = Leapfrog(initial_ϵ)

kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(n_LF)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.75, integrator))

Threads.@threads for i in 1:nchains

    alm₀ = synalm(gen_Cl)
    Cℓ₀ = alm2cl(alm₀)
    θ₀ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([alm₀], lmax, 1), lmax, 1), Cl2Kl(Cℓ₀))

    t = time()
    samples_HMC, stats_HMC = sample(ham, kernel, θ₀, n_samples, adaptor, n_adapts; 
                                    drop_warmup = true)#, progress=true, verbose=true)
    HMC_t = time()-t

    CSV.write("unmask_HMC_stats_n64_chain$i.csv", DataFrame(stats_HMC[1:10_000]))
    CSV.write("unmask_HMC_samples_n64_chain$i.csv", permutedims(DataFrame(samples_HMC[1:10_000], :auto)))

    HMC_ess, HMC_rhat = Summarize(samples_HMC)
    CSV.write("unmask_HMC_perf_n64_chain$i.csv", DataFrame(N_adapt=n_adapt, N_samples=n_samples, 
    ESS=mean(HMC_ess), time=HMC_t))

    samples_HMC, stats_HMC = 0
    
end
