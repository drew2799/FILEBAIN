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

include("HEALPIX_utils.jl")
include("UTILITIES.jl")
include("INFERENCE_funcs.jl")

ProgressMeter.ijulia_behavior(:clear)
Random.seed!(1123)

# SET UP
nside = 8
lmax = 15

# Realization
realiz_Dl = CSV.read("Capse_Cl.csv", DataFrame)[1:lmax-1,1]
realiz_Cl = dl2cl(realiz_Dl, 2)
realiz_Cl[1] += 1e-10
realiz_Cl[2] += 1e-10
realiz_map = synfast(realiz_Cl, nside)

realiz_alm = map2alm(realiz_map, lmax=lmax)
realiz_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([realiz_alm], lmax, 1), lmax, 1), Cl2Kl(realiz_Cl))

# Data generation
noise = 500*I
inv_noise = inv(noise)
e = rand(MvNormal(zeros(nside2npix(nside)), noise))
gen_map = HealpixMap{Float64,RingOrder}(deepcopy(realiz_map) + e)

gen_alm = map2alm(gen_map, lmax=lmax)
gen_Cl = anafast(gen_map, lmax=lmax)
gen_θ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([gen_alm], lmax, 1), lmax, 1), Cl2Kl(gen_Cl))

# Chain starting point
alm₀ = synalm(gen_Cl)
Cℓ₀ = alm2cl(alm₀)
θ₀ = vcat(x_vecmat2vec(from_healpix_alm_to_alm([alm₀], lmax, 1), lmax, 1), Cl2Kl(Cℓ₀))

g = gradient(x->NegLogPosterior(x), θ₀)

# INFERENCE
d = length(θ₀)

struct LogTargetDensity
    dim::Int
end

LogDensityProblemsAD.logdensity(p::LogTargetDensity, θ) = -NegLogPosterior(θ)
LogDensityProblemsAD.dimension(p::LogTargetDensity) = p.dim
LogDensityProblemsAD.capabilities(::Type{LogTargetDensity}) = LogDensityProblemsAD.LogDensityOrder{1}()

# Vanilla HMC
ℓπ = LogTargetDensity(d)
n_LF = 50

n_samples = 1_000

metric = DiagEuclideanMetric(d)
ham = Hamiltonian(metric, ℓπ, Zygote)
initial_ϵ = 0.1
integrator = Leapfrog(initial_ϵ)

kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(n_LF)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.75, integrator))

println("HMC sampling")
t = time()
samples_HMC, stats_HMC = sample(ham, kernel, θ₀, n_samples+2_000, adaptor, 2_000; drop_warmup = true, progress=true, verbose=true)
HMC_t = time()-t
CSV.write("HMC_chain_nside_$nside.csv", permutedims(DataFrame(samples_HMC[1:n_samples], :auto)))

# NUTS
metric = DiagEuclideanMetric(d)
ham = Hamiltonian(metric, ℓπ, Zygote)
initial_ϵ = find_good_stepsize(ham, θ₀)
integrator = Leapfrog(initial_ϵ)

kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.9, integrator))

println("NUTS sampling")
t = time()
samples_NUTS, stats_NUTS = sample(ham, kernel, θ₀, n_samples+500, adaptor, 500; drop_warmup = true, progress=true, verbose=true)
NUTS_t = time()-t
CSV.write("NUTS_chain_nside_$nside.csv", permutedims(DataFrame(samples_NUTS[1:n_samples], :auto)))

# MCHMC
mchmc_nsamples = 10_000
mchmc_nadapt = 10_000

function MCHMCℓπ(θ)
    return -NegLogPosterior(θ)
end

function MCHMCℓπ_grad(x)
    f, df = withgradient(MCHMCℓπ, x)
    return f, df[1]
end

target = CustomTarget(MCHMCℓπ, MCHMCℓπ_grad, θ₀)
spl = MicroCanonicalHMC.MCHMC(mchmc_nadapt, 0.0001, integrator="LF", adaptive=true, tune_eps=true, tune_L=false, tune_sigma=true, L=sqrt(d))

println("MCHMC sampling")
t = time()
samples_MCHMC = Sample(spl, target, mchmc_nsamples, init_params=θ₀, dialog=true)#, thinning=10)
MCHMC_t = time()-t
CSV.write("MCHMC_chain_nside_$nside.csv", permutedims(DataFrame(samples_MCHMC, :auto)))

# STATISTICS
HMC_ess, HMC_rhat = Summarize(samples_HMC)
NUTS_ess, NUTS_rhat = Summarize(samples_NUTS)
MCHMC_ess, MCHMC_rhat = Summarize(samples_MCHMC')

HMC_ess_s = mean(HMC_ess[1:end])/HMC_t
NUTS_ess_s = mean(NUTS_ess[1:end])/NUTS_t
MCHMC_ess_s = mean(MCHMC_ess[1:end-3])/MCHMC_t
println("ESS/s: ", "HMC = ", HMC_ess_s,", NUTS = ", NUTS_ess_s, ", MCHMC = ", MCHMC_ess_s)

NUTS_n_grad = sum(2 .^ [stats_NUTS[i][:tree_depth] for i in 1:n_samples])
HMC_ess_grad = mean(HMC_ess)/(n_LF*n_samples)
NUTS_ess_grad = mean(NUTS_ess)/NUTS_n_grad
MCHMC_ess_grad = mean(MCHMC_ess[1:end-3])/mchmc_nsamples
println("\n", "ESS/grad_eval: ", "HMC = ", HMC_ess_grad, ", NUTS = ", NUTS_ess_grad, ", MCHMC = ", MCHMC_ess_grad)

HMC_τ = n_samples/mean(HMC_ess)
NUTS_τ = n_samples/mean(NUTS_ess)
MCHMC_τ = mchmc_nsamples/mean(MCHMC_ess[1:end-3])
println("\n", "τ: ", "HMC = ", HMC_τ, ", NUTS = ", NUTS_τ, ", MCHMC = ", MCHMC_τ)

# PLOTS
p = plot(layout=(1,3), plot_title="Gelman-Rubin", size=(1000,400))
histogram!(p, HMC_rhat, label="HMC", subplot=1, color="coral2")
histogram!(p, NUTS_rhat, label="NUTS", subplot=2)
histogram!(p, MCHMC_rhat[1:end-3], label="MCHMC", subplot=3, color="blue2")
savefig("GR_plot_nside_$nside.pdf")

p = plot(layout=(1,3), plot_title="Log density trace plots", size=(1000,500))
plot!(p, [stats_HMC[i][:log_density] for i in 1:n_samples], label="HMC", subplot=1,color="coral2")#, ylim=(-1000,-920))
plot!(p, [stats_NUTS[i][:log_density] for i in 1:n_samples], label="NUTS", subplot=2)#, ylim=(-1000,-920))
plot!(p, samples_MCHMC[end,:], label="MCHMC", subplot=3,color="blue2")#, ylim=(-1000,-920))
savefig("traceplot_nside_$nside.pdf")



















