# NEG-LOG-LIKELIHOOD
function NegLogLikelihood(alm; data = gen_map, lmax=lmax, nside=nside, invN = inv_noise)
    hpix_alm = from_alm_to_healpix_alm(alm, lmax, 1)
    d = alm2map(hpix_alm[1], nside)
    loglike = 0.5*transpose(data-d)*invN*(data-d)
    return loglike
end

# NEG-LOG-PRIOR alm
function AlmPrior(alm, Kl; lmax=lmax)
    
    Cl = Kl2Cl(Kl)
    
    alm0 = [alm[l][1,1] for l in 1:lmax+1]
    p_alm0 = 0.5*sum((alm0.^2)./Cl) + 0.5*sum(log.(Cl))

    p_alm = 0.
    for l in 1:lmax+1
        p_alm += (sum((alm[l][:,2:end].^2)./Cl[l]) + (l-1)*log(Cl[l]/2.))
    end

    return p_alm0 + p_alm
end

@adjoint function AlmPrior(alm, Kl; lmax=lmax)
    
    y = AlmPrior(alm, Kl)
    
    function AlmPrior_PB(ȳ)

        Cl = Kl2Cl(Kl)
        
        ā = deepcopy(alm)
        for l in 1:lmax+1
            ā[l][1,1] = ȳ * alm[l][1,1]/(Cl[l])
            ā[l][:,2:end] = (2*ȳ/(Cl[l])) .* alm[l][:,2:end]
        end

        k̄ = zeros(lmax+1)
        for l in 1:lmax+1
            Al = (alm[l][1,1]^2)/2 + sum(alm[l][:,2:end].^2)
            k̄[l] = ((l-0.5)/Cl[l] - Al/(Cl[l]^2))*1_500*pdf(Normal(0,1), Kl[l])*ȳ
        end
        
        return (ā, k̄, nothing)
    end
    return y, AlmPrior_PB
end

# NEG-LOG-PRIOR Kl + JACOBIAN
function KlPrior(Kl)
    return 0.5*sum(Kl.^2)
end

@adjoint function KlPrior(Kl)
    
    y = KlPrior(Kl)

    function KlPrior_PB(ȳ)
        K̄l = ȳ .* Kl
        return (K̄l, )
    end

    return y, KlPrior_PB
end

# NEG-LOG-POSTERIOR
function NegLogPosterior(θ; data = gen_map, lmax=lmax, nside=nside, invN = inv_noise)

    alm_vec = θ[1:end-lmax-1]
    Kl = θ[end-lmax:end]

    alm = x_vec2vecmat(alm_vec, lmax, 1)

    l = NegLogLikelihood(alm; data = gen_map, lmax=lmax, nside=nside, invN = inv_noise)
    p_alm = AlmPrior(alm, Kl; lmax=lmax)
    p_kl = KlPrior(Kl) #jacobian

    return l+p_alm+p_kl
end

# REPARAMETERIZATION
function Cl2Kl(Cl)
    C̃ = Cl./1_500
    return norminvcdf.(0,1,C̃)
end

function Kl2Cl(Kl)
    return 1_500 .* normcdf.(0,1,Kl)
end

@adjoint function Kl2Cl(Kl)
    
    y = Kl2Cl(Kl)
    
    function Kl2Cl_PB(ȳ)
        x̄ = 1_500 .* pdf.(Normal(0,1), Kl) .* ȳ
        return (x̄,)
    end
    return y, Kl2Cl_PB
end  