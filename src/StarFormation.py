import numpy as np
import time as Ti
from scipy.integrate import simpson as simps

from IMF import *
from utils import *

### Calculate mean of clump mass distribution for more efficient clump mass drawing
def calc_mean_clump_mass( minMcl, maxMcl, ClMFslope=-2):
    if ClMFslope!=-2:
        return (ClMFslope+1)/(ClMFslope+2)*(maxMcl**(ClMFslope+2)-minMcl**(ClMFslope+2))/(maxMcl**(ClMFslope+1)-minMcl**(ClMFslope+1))
    else:
        return (ClMFslope+1)/(maxMcl**(ClMFslope+1)-minMcl**(ClMFslope+1))*(np.log(maxMcl/minMcl))
        
def calc_var_clump_mass(minMcl , maxMcl ,ClMFslope = -2):
    if ClMFslope !=-2:
        return (ClMFslope+1)/(ClMFslope+3)*(maxMcl**(ClMFslope+3)-minMcl**(ClMFslope+3))/(maxMcl**(ClMFslope+1)-minMcl**(ClMFslope+1)) - calc_mean_clump_mass(minMcl, maxMcl,ClMFslope)**2
    else:
        return maxMcl*minMcl - calc_mean_clump_mass(minMcl , maxMcl,ClMFslope )**2
        
def calc_sigma_on_clump_number(M, minMcl, maxMcl, ClMFslope = -2):
    return np.sqrt(M * calc_var_clump_mass(minMcl, maxMcl,ClMFslope)/calc_mean_clump_mass(minMcl, maxMcl,ClMFslope)**3)

def calc_number_of_clouds_mean(M, minMcl, maxMcl, ClMFslope = -2):
    return M/calc_mean_clump_mass(minMcl, maxMcl, ClMFslope)

def calc_Mgas_for_threshold(T, minMcl, maxMcl, ClMFslope = -2):
    return calc_var_clump_mass(minMcl, maxMcl, ClMFslope)/(calc_mean_clump_mass(minMcl, maxMcl, ClMFslope)*T**2)
    
def calc_Mgas_for_threshold_Mstar(T,n, minMcl, maxMcl, ClMFslope = -2):
    return calc_var_clump_mass(minMcl, maxMcl, ClMFslope)/(calc_mean_stellar_mass_from_clouds(n,minMcl, maxMcl)*T**2)

@jit()
def get_numPerMass_SalpeterIMF(Mlow, Mup, slope):
    return (Mup**(1.-slope) - Mlow**(1.-slope)) / (Mup**(2.-slope) - Mlow**(2.-slope)) * (2.-slope)/(1.-slope)
    
@jit()
def get_MstarLimit_SalpeterIMF(Mstar,NstarPerMass, Mlow = 0.1, Mup = 100., slope = 2.35):
    Nstar = NstarPerMass * Mstar
    return ((Mup**(1.-slope)*(Nstar-1) + Mlow**(1.-slope))/Nstar)**(1./(1.-slope))

@njit(fastmath = True)
def get_numPerMass_EvolvingIMF(Mlow, Mup, Mmid,fmassive, slope):
    lowPart = (1.-fmassive) * (2.-slope) / (1.-slope) * (Mmid**(1.-slope) - Mlow**(1.-slope))/(Mmid**(2.-slope) - Mlow**(2.-slope))
    upPart = fmassive * np.log(Mup/Mmid) / (Mup - Mmid)
    return lowPart + upPart
    
@njit(fastmath = True)
def get_MstarLimit_EvolvingIMF(Mstar,NstarPerMass,Mmid,fmassive, Mlow=0.1, Mup=100., slope=2.35):
    Nstar = NstarPerMass * Mstar
    resultLow = 0
    resultUp = Mup * np.exp(-1./Nstar * ((1. - (Mlow/Mmid)**(1.-slope)) / (1.-slope) + np.log(Mup/Mmid)))
    if(resultUp > Mmid):
        result = resultUp
        return result
    if fmassive<1.:
        C1 = (1 - slope) * Mmid**(slope - 1) / ((1-fmassive) * (1 - np.exp((1 - slope) * np.log(Mlow / Mmid)) + (1 - slope) * np.log(Mup / Mmid)))
        resultLow = (Mlow**(1.-slope) + (1.-slope) / ((1.-fmassive) * C1) * (Nstar - 1)/Nstar)**(1./(1.-slope))
        result = resultLow
    return result
    
@njit
def calc_max_cloud_mass(Mhalo,z,f):
    return Mhalo/1e11 * ((1+z)/10)**1.5 * f

### Draw clump masses from powerlaw distribution
def finv(N):
    return (N * (maxMcl**(ClMFslope+1)-minMcl**(ClMFslope+1))+minMcl**(ClMFslope+1))**(1/(ClMFslope+1))
   
def draw_clump_mass(Gal,Mgas, MaxMstar):
    if Mgas>Gal.simParam.minMcl:
        ### Initial draw of clouds. Draws number of clouds assuming mean mass will equal theoretical mean
        ### Adjusting for star formation efficiency to not draw too many clouds
        
        clumpListMaxIndex = Gal.clumpDrawList.shape[0]-1
        stellarMassToForm = 0
        i=0
        if Gal.simple_clouds:

            clumpMasses   = np.zeros(int(Gal.cloudLimit * 4))
            stellarMasses = np.zeros(int(Gal.cloudLimit * 4))
            if (Gal.IMF_type == 'salpeter_cloud'):
                stellarMasses, clumpMasses, MstarLimits, Gal.clumpListIndex = draw_clouds_salpeter_cloud_imf_simple(Gal.clumpDrawList, 
                                                                                                                    Gal.clumpListIndex, 
                                                                                                                    clumpListMaxIndex, 
                                                                                                                    MaxMstar,
                                                                                                                    Gal.Nstar, 
                                                                                                                    Gal.simParam.fcl,
                                                                                                                    Gal.cloudMassLimit[Gal.j],
                                                                                                                    4 * Gal.cloudLimit)
                
            elif (Gal.IMF_type == 'evolving_cloud'):
                stellarMasses, clumpMasses, MstarLimits, Gal.clumpListIndex = draw_clouds_evolving_cloud_imf_simple(Gal.clumpDrawList, 
                                                                                                                 Gal.clumpListIndex, 
                                                                                                                 clumpListMaxIndex, 
                                                                                                                 MaxMstar, 
                                                                                                                 Gal.fmassiveHistory[Gal.j], 
                                                                                                                 Gal.fmassiveClList, 
                                                                                                                 Gal.McutoffClList, 
                                                                                                                 Gal.Nstar, 
                                                                                                                 Gal.simParam.fcl,
                                                                                                                 Gal.cloudMassLimit[Gal.j],
                                                                                                                 4 * Gal.cloudLimit)

            else:
                stellarMasses, clumpMasses, Gal.clumpListIndex = draw_clouds_global_imf_simple(Gal.clumpDrawList, 
                                                                                               Gal.clumpListIndex,
                                                                                               clumpListMaxIndex,
                                                                                               MaxMstar,
                                                                                               Gal.simParam.fcl,
                                                                                               Gal.cloudMassLimit[Gal.j],
                                                                                               4 * Gal.cloudLimit)
        else:
            
            if (Gal.IMF_type == 'salpeter_cloud') + (Gal.IMF_type == 'evolving_cloud'):
                metalBin = where(Gal.simParam.ZZ,Gal.metallicity)
                if Gal.IMF_type == 'salpeter_cloud':
                    stellarMasses, clumpMasses, MstarLimits, Gal.clumpListIndex = draw_clouds_salpeter_cloud_imf(Gal.clumpDrawList, 
                                                                                                                 Gal.clumpListIndex, 
                                                                                                                 clumpListMaxIndex, 
                                                                                                                 MaxMstar, 
                                                                                                                 metalBin,
                                                                                                                 Gal.MclListMin, 
                                                                                                                 Gal.MclListFactor, 
                                                                                                                 Gal.MclListBins,
                                                                                                                 Gal.SFEgrid, 
                                                                                                                 Gal.maxStarMassGrid,
                                                                                                                 Gal.cloudMassLimit[Gal.j],
                                                                                                                 Gal.cloudDensityHistory[Gal.j],
                                                                                                                4 * Gal.cloudLimit)
                else:
                    stellarMasses, clumpMasses, MstarLimits, Gal.clumpListIndex = draw_clouds_evolving_cloud_imf(clumpDrawList = Gal.clumpDrawList, 
                                                                                                                 clumpListIndex = Gal.clumpListIndex, 
                                                                                                                 clumpListMaxIndex = clumpListMaxIndex, 
                                                                                                                 MaxMstar = MaxMstar, 
                                                                                                                 metalBin = metalBin,
                                                                                                                 MclListMin = Gal.MclListMin, 
                                                                                                                 MclListFactor = Gal.MclListFactor, 
                                                                                                                 MclListBins = Gal.MclListBins,
                                                                                                                 f_massive = Gal.fmassiveHistory[Gal.j], 
                                                                                                                 fmassiveClList = Gal.fmassiveClList, 
                                                                                                                 SFEgrid = Gal.SFEgrid, 
                                                                                                                 maxStarMassGrid = Gal.maxStarMassGrid,
                                                                                                                 cloudMassLimit = Gal.cloudMassLimit[Gal.j],
                                                                                                                 cloudDensity = Gal.cloudDensityHistory[Gal.j],
                                                                                                                 maxgal = 4 * Gal.cloudLimit)
            else:
                Xi = Gal.get_Xi_from_fmassive()
                T =calc_T_from_Z_Shaver1983(Gal.metallicity)
                vesc0 = calc_cs(T)
                stellarMasses, clumpMasses, Gal.clumpListIndex = draw_clouds_global_imf(Gal.clumpDrawList, 
                                                                                        Gal.clumpListIndex, 
                                                                                        clumpListMaxIndex, 
                                                                                        T, 
                                                                                        Xi, 
                                                                                        vesc0, 
                                                                                        MaxMstar,
                                                                                        Gal.cloudMassLimit[Gal.j],
                                                                                        Gal.cloudDensityHistory[Gal.j],
                                                                                        Gal.SFE_function,
                                                                                        4 * Gal.cloudLimit)

        if len(clumpMasses):   
            Gal.cloudLimit = max(Gal.cloudLimit,len(clumpMasses))
            freeFallTimes = np.zeros_like(clumpMasses)+calc_tff(Gal.cloudDensityHistory[Gal.j])
            if Gal.simple_clouds:
                starFormationTimes = freeFallTimes * 2.55
                starFormationStartTimes =  0
                if (Gal.IMF_type != 'evolving_cloud') * (Gal.IMF_type != 'salpeter_cloud'):
                    MstarLimits = np.zeros_like(clumpMasses)
            else:
                starFormationTimes = freeFallTimes * calc_tsf(Gal.cloudDensityHistory[Gal.j],clumpMasses)
                starFormationStartTimes = freeFallTimes * calc_tsf0(Gal.cloudDensityHistory[Gal.j],clumpMasses) + np.linspace(0,Gal.dt,len(clumpMasses)+1)[:-1]
                if (Gal.IMF_type != 'evolving_cloud') * (Gal.IMF_type != 'salpeter_cloud'):
                    MstarLimits = np.zeros_like(clumpMasses)

            Gal.Mmetal = Gal.Mmetal - Gal.metallicity *  np.sum(clumpMasses)
            Gal.Mdust = Gal.Mdust - Gal.dustToGasRatio * np.sum(clumpMasses)
            return clumpMasses, stellarMasses, starFormationTimes, starFormationStartTimes, MstarLimits, Mgas-np.sum(clumpMasses)
        else:
            return np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0],int),Mgas

    else:
        return np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0],int),Mgas

@njit()
def draw_clouds_salpeter_cloud_imf(clumpDrawList, clumpListIndex, clumpListMaxIndex, MaxMstar, metalBin, MclListMin, MclListFactor, MclListBins,  SFEgrid, maxStarMassGrid, cloudMassLimit, cloudDensity, maxgal):
    clumpMasses   = np.zeros(int(maxgal))
    stellarMasses = np.zeros(int(maxgal))
    MstarLimits   = np.zeros(int(maxgal), dtype = 'int')
    stellarMassToForm = 0
    i = 0
    listBin = MclListBins * metalBin
    ### Continue drawing clouds until the maximum allowed stellar mass is exceeded
    while True:
        clumpMass = clumpDrawList[clumpListIndex]
        if clumpMass <= cloudMassLimit:
            SFE,MstarLimit = get_MstarLimit_and_SFE(clumpMass, MclListMin, MclListFactor, listBin, SFEgrid, maxStarMassGrid,cloudDensity)
            stellarMass = clumpMass * SFE
            # Check if the new cloud makes the total stellar mass to be formed exceed the maximum
            if stellarMassToForm + stellarMass>MaxMstar:
                break
            stellarMassToForm += stellarMass
            clumpMasses[i]     = clumpMass
            stellarMasses[i]   = stellarMass
            MstarLimits[i]     = MstarLimit
            i += 1
        clumpListIndex     = (clumpListIndex + 1) % clumpListMaxIndex
    return stellarMasses[:i], clumpMasses[:i], MstarLimits[:i], clumpListIndex
@njit()
def draw_clouds_salpeter_cloud_imf_simple(clumpDrawList, clumpListIndex, clumpListMaxIndex, MaxMstar, Nstar, fcl, cloudMassLimit, maxgal):
    clumpMasses   = np.zeros(int(maxgal))
    stellarMasses = np.zeros(int(maxgal))
    MstarLimits   = np.zeros(int(maxgal), dtype = 'int')
    stellarMassToForm = 0
    i = 0
    while True:
        clumpMass = clumpDrawList[clumpListIndex]
        if clumpMass <= cloudMassLimit:
            stellarMass = clumpMass * fcl
            if stellarMassToForm + stellarMass>MaxMstar:
                break
            stellarMassToForm += stellarMass
            clumpMasses[i]   = clumpMass
            stellarMasses[i] = stellarMass
            MstarLimits[i]   = int((get_MstarLimit_SalpeterIMF(stellarMass,Nstar)+5)//10)-1
            i += 1
        clumpListIndex = (clumpListIndex + 1) % clumpListMaxIndex
    return stellarMasses[:i], clumpMasses[:i], MstarLimits[:i], clumpListIndex
### Find SFE and MstarLimit values in the lists
@njit()
def get_MstarLimit_and_SFE(clumpMass, MclListMin, MclListFactor, listBin, SFEgrid, maxStarMassGrid,cloudDensity):
    clumpMassIndex = int((np.log10(clumpMass)-MclListMin) * MclListFactor)
    index = clumpMassIndex + listBin
    logDensity = np.log10(cloudDensity)
    if np.isclose(logDensity%1,0):
        SFE        = SFEgrid[int(logDensity)-2][index]
        MstarLimit = maxStarMassGrid[int(logDensity)-2][index]
    else:
        SFElow  = SFEgrid[int(logDensity)-2][index]
        SFEhigh = SFEgrid[int(logDensity)-1][index]
        SFE     = (logDensity-int(logDensity)) * (SFEhigh - SFElow) + SFElow

        MstarLimitlow  = maxStarMassGrid[int(logDensity)-2][index]
        MstarLimithigh = maxStarMassGrid[int(logDensity)-1][index]
        MstarLimit     = int((logDensity-int(logDensity)) * (MstarLimithigh - MstarLimitlow) + MstarLimitlow)
    return SFE,MstarLimit

@njit()
def draw_clouds_evolving_cloud_imf_simple(clumpDrawList, clumpListIndex, clumpListMaxIndex, MaxMstar, f_massive, fmassiveClList, McutoffClList, Nstar, fcl, cloudMassLimit, maxgal):
    clumpMasses   = np.zeros(int(maxgal))
    stellarMasses = np.zeros(int(maxgal))
    MstarLimits   = np.zeros(int(maxgal), dtype = 'int')
    fmassiveBin = find_nearest(f_massive,fmassiveClList)
    Nstar = Nstar[fmassiveBin]
    fmass = fmassiveClList[fmassiveBin]
    Mcut  = McutoffClList[fmassiveBin]
    stellarMassToForm = 0
    i = 0
    while True:
        clumpMass = clumpDrawList[clumpListIndex]
        if clumpMass <= cloudMassLimit:
            stellarMass = clumpMass * fcl
            if stellarMassToForm + stellarMass>MaxMstar:
                break
            stellarMassToForm += stellarMass
            clumpMasses[i]   = clumpMass
            stellarMasses[i] = stellarMass
            MstarLimits[i]   = int((get_MstarLimit_EvolvingIMF(stellarMass,Nstar,Mcut,fmass)+5)//10)-1
            i += 1
        clumpListIndex = (clumpListIndex + 1) % clumpListMaxIndex
    return stellarMasses[:i], clumpMasses[:i], MstarLimits[:i], clumpListIndex
    
@njit()
def draw_clouds_evolving_cloud_imf(clumpDrawList, clumpListIndex, clumpListMaxIndex, MaxMstar, metalBin, MclListMin, MclListFactor, MclListBins, f_massive, fmassiveClList, SFEgrid, maxStarMassGrid, cloudMassLimit, cloudDensity, maxgal):
    clumpMasses   = np.zeros(int(maxgal))
    stellarMasses = np.zeros(int(maxgal))
    MstarLimits   = np.zeros(int(maxgal), dtype = 'int')
    stellarMassToForm = 0
    i = 0
    fmassiveBin = find_nearest(f_massive,fmassiveClList)
    listBin = MclListBins * fmassiveBin + MclListBins * fmassiveClList.shape[0] * metalBin
    ### Continue drawing clouds until the maximum allowed stellar mass is exceeded
    while True:
        clumpMass = clumpDrawList[clumpListIndex]
        if clumpMass <= cloudMassLimit:
            SFE,MstarLimit = get_MstarLimit_and_SFE(clumpMass, MclListMin, MclListFactor, listBin, SFEgrid, maxStarMassGrid, cloudDensity)
            stellarMass = clumpMass * SFE
            # Check if the new cloud makes the total stellar mass to be formed exceed the maximum
            if stellarMassToForm + stellarMass>MaxMstar:
                break
            stellarMassToForm += stellarMass
            clumpMasses[i]     = clumpMass
            stellarMasses[i]   = stellarMass
            MstarLimits[i]     = MstarLimit
            i += 1
        clumpListIndex     = (clumpListIndex + 1) % clumpListMaxIndex
        
    return stellarMasses[:i], clumpMasses[:i], MstarLimits[:i], clumpListIndex

@njit()
def draw_clouds_global_imf_simple(clumpDrawList, clumpListIndex, clumpListMaxIndex,MaxMstar,fcl, cloudMassLimit, maxgal):
    clumpMasses   = np.zeros(int(maxgal))
    stellarMasses = np.zeros(int(maxgal))
    stellarMassToForm = 0
    i = 0
    while True:
        clumpMass = clumpDrawList[clumpListIndex]
        if clumpMass <= cloudMassLimit:
            stellarMass = clumpMass * fcl
            if stellarMassToForm + stellarMass>MaxMstar:
                break
            stellarMassToForm += stellarMass
            clumpMasses[i] = clumpMass
            stellarMasses[i] = stellarMass
            i += 1
        clumpListIndex = (clumpListIndex + 1) % clumpListMaxIndex
    stellarMasses = stellarMasses[:i] 
    clumpMasses   = clumpMasses[:i]
    return stellarMasses[:i], clumpMasses[:i],clumpListIndex

@njit()
def draw_clouds_global_imf(clumpDrawList, clumpListIndex, clumpListMaxIndex, T, Xi, vesc0, MaxMstar, cloudMassLimit,cloudDensity, SFE_function, maxgal):
    clumpMasses   = np.zeros(int(maxgal))
    stellarMasses = np.zeros(int(maxgal))
    stellarMassToForm = 0
    i = 0
    ### Continue drawing clouds until the maximum allowed stellar mass is exceeded
    while True:
        clumpMass = clumpDrawList[clumpListIndex]
        if clumpMass <= cloudMassLimit:
            stellarMass = clumpMass * SFE_function(cloudDensity, clumpMass,T,Xi,vesc0)
            # Check if the new cloud makes the total stellar mass to be formed exceed the maximum
            if stellarMassToForm + stellarMass>MaxMstar:
                break
            stellarMassToForm += stellarMass
            clumpMasses[i] = clumpMass

            stellarMasses[i] = stellarMass
            i += 1
        clumpListIndex = (clumpListIndex + 1) % clumpListMaxIndex
    return stellarMasses[:i], clumpMasses[:i],clumpListIndex

# Calculate the star formation time in units of the free fall time as (Sigma/100)^(0.255)
# Model based on by eye fit to Kim&Ostriker2018 and Menon2024 data.
#@njit()
#def calc_tsf(n,M, a = 0.4, b = 0.17, c = -0.23,d = 0.05): 
#    return a * n ** b * M ** (c + np.log10(M) * d)
@njit()
def calc_tsf(n,M): 
    return 0.1566 * n ** 0.17 * M ** 0.085

# Calculate time when starformation starts in units of freefall time
# Model pased on powerlaw fit to M/R(t0) in Kim&Ostriker2018. 
@njit()
def calc_tsf0(n,M):
    return 4.239 * M**-0.17*n**-0.085

# Calculate freefall time of cloud in units of Myr
@njit()
def calc_tff(n):
    return 43.5 / np.sqrt(n)

# calculate surface density of cloud in units of Msun/pc^2
@njit()
def calc_Surface_density(n,M):
    return M * (M / n * 53.7877) **(-2/3)

# calculate radius of cloud in units iof pc
@njit()
def calc_cloud_radius(n,M):
    return (M / n *9.6596)**(1/3)

# calculate escape velocity from surface of spherically symmetric, uniform cloud, in units of km/s
@njit()
def calc_v_esc(n,M):
    return 0.04494 * (n*M*M)**(0.166667)
    #0.04494 = (4/3 *ac.G**3* (u.Msun)**2 * np.pi * u.cm**-3*ac.m_p)**(1/6)).to(u.km/u.s)
    
# Calculate radiation pressure limited star formation efficiency of a cloud
@njit()    
def calc_SFE_Kim2018_RP0(n,M,Xi = 5.05e46):    
    ep_ej = 0.13 # Fraction of initial gas mass ejected by turbulence, unavailable for star formation
    v_ej = 4.15 * M**0.143 # Mass weighted ejection velocity. based on fit to data points in Kim&Ostriker2018
    surfaceDensity = calc_Surface_density(n,M)
    p_over_m = 135 * (surfaceDensity * 0.01) ** (-0.74) # p/m = 135 * (S/100)^-0.74. Momentum ejection efficiency. formula found in Kim&Ostriker2018 based on their simulation results
    p_over_m_over_v = p_over_m / v_ej
    fcorrXi = (31-Xi*2.07e-47)*0.03334 # scales as 1-Xi/(30)
    epsilon_star = (1-ep_ej)/((1 + p_over_m_over_v ))* fcorrXi
    sigma = surfaceDensity/Xi * 1.8258e43
    epsilon_min = (np.sqrt(1 + 4 * sigma ** 2) - 1) / (2 * sigma) * (1-ep_ej)# minimum SFE required for radiation pressure to counterbalance gravity
    return np.maximum(epsilon_min,epsilon_star)
@njit()    
def calc_SFE_Kim2018_RP(n,M,Xi = 5.05e46):    
    ep_ej = 0.13 # Fraction of initial gas mass ejected by turbulence, unavailable for star formation
    v_ej = 4.15 * M**0.143 # Mass weighted ejection velocity. based on fit to data points in Kim&Ostriker2018
    surfaceDensity = calc_Surface_density(n,M)
    p_over_m = 135 * (surfaceDensity * 0.01) ** (-0.74) # p/m = 135 * (S/100)^-0.74. Momentum ejection efficiency. formula found in Kim&Ostriker2018 based on their simulation results
    p_over_m_over_v = p_over_m / v_ej
    fcorrXi = (31-Xi*2.07e-47)*0.03334 # scales as 1-Xi/(30)
    epsilon_star = (1-ep_ej)/((1 + p_over_m_over_v ))* fcorrXi
    sigma = surfaceDensity/Xi * 1.8258e43
    epsilon_min = 0#(np.sqrt(1 + 4 * sigma ** 2) - 1) / (2 * sigma) # minimum SFE required for radiation pressure to counterbalance gravity
    return np.maximum(epsilon_min,epsilon_star)
@njit()
def calc_phi_t_phi_ion(n,M):
    c1 = -2.89; c2 = 2.11; c3 = 25.3
    return c1 + c2 * np.log10(calc_Surface_density(n,M) + c3)
@njit()
def calc_xi_PH(n,M,Xi,T = 8000):
    SigmaIon = 140 * (Xi/5.05e46)**0.5 * (T/8000) ** 0.85
    xi = calc_Surface_density(n,M)/(calc_phi_t_phi_ion(n,M)*SigmaIon)
    return xi
### Calculate photoionisation limited star formation efficiency of a cloud. Based on analytic model in Kim&Ostriker2018
@njit()
def calc_SFE_Kim2018_PH(n,M,Xi=5.05e46,T=8000):
    epsilon0 = 0.13
    xi_times_epsilon0 = calc_xi_PH(n,M,Xi,T) * (1 - epsilon0)
    return (2 * xi_times_epsilon0 / (1 + np.sqrt(1 + 4 * xi_times_epsilon0 * xi_times_epsilon0)))**2


### Calculate log(O/H) + 12 from metallicity assuming solar abundance ratios
@njit()
def calc_OH12(Z):
    return np.log10(Z * 74.6269) + 8.69 
    ### 74.6269 = 1/Zsun = 1/0.0134
    ### log(O/H) + 12 = 8.69 for the Sun

### Calculate the ionised gas temperature from metallicity with results from Shaver et al. 1983
@njit()
def calc_T_from_Z_Shaver1983(Z):
    return 1e4/1.49 * (9.82 - calc_OH12(np.maximum(Z,0.0000134)))

### Calculate sound speed in gas of temperature T
@njit()
def calc_cs(T):
    return 0.12849 * np.sqrt(T)

### Calculate total star formation efficiency of a cloud, accounting for both radiation pressure and photoionisation
@njit()
def calc_SFE(n,M,T,Xi,vesc0,a = 2.3):
    vesc  = calc_v_esc(n,M)
    SFE = calc_SFE_Kim2018_PH(n,M,Xi,T) * (np.exp(-(vesc/vesc0)**a)) + calc_SFE_Kim2018_RP(n,M,Xi) * (1-np.exp(-(vesc/vesc0)**a))
    if (np.log10(M) + 0.75*np.log10(n)) > 9:
        SFE -= 0.8* (np.log10(M) + 0.75 * np.log10(n)-9) / np.log10(n)
    return SFE

@njit()
def calc_SFE_no_turn(n,M,T,Xi,vesc0,a = 2.3):
    vesc  = calc_v_esc(n,M)
    SFE = calc_SFE_Kim2018_PH(n,M,Xi,T) * (np.exp(-(vesc/vesc0)**a)) + calc_SFE_Kim2018_RP0(n,M,Xi) * (1-np.exp(-(vesc/vesc0)**a))
    return SFE

@njit(parallel = True)
def calc_SFE_array(n,M,T,Xi,vesc0,SFE_function,a=2.3):
    result = np.zeros_like(M)
    for i in prange(result.shape[0]):
        result[i] = SFE_function(n,M[i],T,Xi,vesc0,a)
    return result
    
def get_cloud_density(Gal):
    if not Gal.constantCloudDensity:
        Gal.cloudDensityHistory[Gal.j] = calc_cloud_density(Gal)
        if Gal.simParam.smoothMassThreshold!= 1:
            Gal.MsmoothHistory[Gal.j] = calc_Mgas_for_threshold(Gal.simParam.smoothMassThreshold, Gal.simParam.minMcl, Gal.simParam.maxMcl, Gal.simParam.ClMFslope)
        else:
            Gal.MsmoothHistory[Gal.j] = 0
def calc_cloud_density(Gal):
    dmgas = (Gal.simParam.haloMassSubstep[Gal.j]- Gal.simParam.haloMassSubstep[Gal.j-1]) * Gal.simParam.OB0/Gal.simParam.OM0
    
    n = (np.maximum((Gal.Mgas-dmgas),0) * Gal.simParam.MH_per_Msun / (Gal.simParam.RvirSubstep[Gal.j]**3 * 4 * np.pi / 3)) * 5e5 + 1000
    return min(n,1e5)
    #zmax = 20
    #zmin = 5
    #lognmax = 4
    #lognmin = 2
    #z = Gal.simParam.substepRedshiftList[Gal.j]
    #n = 10 ** ((z - zmin) / (zmax - zmin) * (lognmax - lognmin) + lognmin)
    #return max(min(n,1e4),1e2)

def clump_mass_function(M,Mmin,Mmax):
    return Mmax * Mmin / (Mmax - Mmin) * M ** -2
def get_smooth_clump_list_simple(Gal,MstarMaxTmp):
    simParam=Gal.simParam
    if (Gal.IMF_type == 'salpeter') + (Gal.IMF_type == 'evolving') + (Gal.IMF_type == 'evolving2'):
        clumps = np.array([[MstarMaxTmp/simParam.fcl,#Clump Gas masses
                            MstarMaxTmp,#Clump stellar masses to be formed
                            simParam.tff*2.5+Gal.dt,#Time to form the stars
                            simParam.tff*2.5+Gal.dt,#Time left to form the stars
                            Gal.j , #Time of formation
                            Gal.metallicity,#metallicity of the clumps
                            where(simParam.ZZ,Gal.metallicity),#Metallicity index for adding to grid
                            0,#(simParam.meanClumpMass / simParam.clumpMeanDensityMsunPerpc3 * 3/(4*np.pi))**(1/3),#Mean Clump Radius
                            0,#Ionized front radius
                            Gal.dustToGasRatio,# Dust fraction
                            Gal.dustToGasRatio * MstarMaxTmp/simParam.fcl,#Dust mass
                            Gal.get_cloud_lifetime(np.array([10])) ,
                            0,
                            9]])
    else:
        hasSmooth = False
        fmassiveBin = 0
        if Gal.IMF_type == 'evolving_cloud':
            fmassiveBin = find_nearest(Gal.fmassiveHistory[Gal.j],Gal.fmassiveClList)
        if (Gal.smoothMstarFractionExists[fmassiveBin] == True)*(Gal.cloudMassLimit[Gal.j]>simParam.maxMcl):
            maxStarFaction = Gal.smoothMstarFraction[fmassiveBin]
            maxCloudMass   = Gal.smoothMaxCloudMass[fmassiveBin]
            muSFEVals      = Gal.smoothMeanSFEVals[fmassiveBin]
            hasSmooth = True
        
        if hasSmooth:
            clumps = np.zeros((Gal.maxStarMassVals.shape[0],14))
            maxStarFraction = Gal.smoothMstarFraction[fmassiveBin]
            maxCloudMass   = Gal.smoothMaxCloudMass[fmassiveBin]
            muSFEVals      = Gal.smoothMeanSFEVals[fmassiveBin]
            
            MstarVals = maxStarFraction * MstarMaxTmp
            clumps[:,0][muSFEVals>0]  = MstarVals[muSFEVals>0] / muSFEVals[muSFEVals>0]
            clumps[:,1]  = MstarVals
            clumps[:,2]  = calc_tsf(simParam.clumpMeanDensityPercm3,maxCloudMass) * simParam.tff + Gal.dt
            clumps[:,12] = calc_tsf0(simParam.clumpMeanDensityPercm3,maxCloudMass) * simParam.tff
            clumps[:,3]  = clumps[:,2] + clumps[:,12]
            clumps[:,4] = Gal.j
            clumps[:,5] = Gal.metallicity
            clumps[:,6] = where(simParam.ZZ, Gal.metallicity)
            #clumps[:,7] = (simParam.meanClumpMass / simParam.clumpMeanDensityMsunPerpc3 * 3/(4*np.pi))**(1/3) # This one is not quite right, but it is also not used used right now, so I am not dealing with it right now
            clumps[:,9] = Gal.Mdust/ Gal.Mgas
            clumps[:,10] = clumps[:,0] * clumps[:,9]
            clumps[:,11] = clumps[:,12] + Gal.get_cloud_lifetime(Gal.maxStarMassVals)
            clumps[:,13] = np.arange(Gal.maxStarMassVals.shape[0])
            clumps = clumps[clumps[:,0]>0]

        else:
            fmassiveBin = 0
            maxMcl = min(Gal.cloudMassLimit[Gal.j],simParam.maxMcl)
            maxStarFaction = np.zeros(Gal.maxStarMassVals.shape[0])
            maxCloudMass = np.zeros(Gal.maxStarMassVals.shape[0])
            clumps = np.zeros((Gal.maxStarMassVals.shape[0],14))
            muSFEVals = np.zeros(Gal.maxStarMassVals.shape[0])
            MstarVals = np.zeros(Gal.maxStarMassVals.shape[0])
            if (Gal.IMF_type == 'salpeter_cloud'):
                MstarLimitVals = np.zeros_like(Gal.MclList)
                for i,m in enumerate(Gal.MclList):
                    MstarLimitVals[i] = int((get_MstarLimit_SalpeterIMF(m,Gal.Nstar)+5)//10)-1
            else:
                fmassiveBin = find_nearest(Gal.fmassiveHistory[Gal.j],Gal.fmassiveClList)
                Nstar = Gal.Nstar[fmassiveBin]
                fmass = Gal.fmassiveClList[fmassiveBin]
                Mcut  = Gal.McutoffClList[fmassiveBin]
                MstarLimitVals = np.zeros_like(Gal.MclList)
                for i,m in enumerate(Gal.MclList):
                    MstarLimitVals[i] = int((get_MstarLimit_EvolvingIMF(m,Nstar,Mcut,fmass)+5)//10)-1
            for i in range(Gal.maxStarMassVals.shape[0]):
                mask = (MstarLimitVals == i) * (Gal.MclList >= simParam.minMcl) * (Gal.MclList <= maxMcl)

                if np.sum(mask):
                    if mask[-1] == False:
                        mask[np.where(mask)[0][-1] +1] = True
                    Mcl = Gal.MclList[mask]
                    muMstar = simps(simParam.fcl * clump_mass_function(Mcl,Mcl[0],Mcl[-1]) * Mcl, x=Mcl)
                    muSFEVals[i] = muMstar / ((-1)/(Mcl[-1]**(-1)-Mcl[0]**(-1))*(np.log(Mcl[-1]/Mcl[0])))
                    maxStarFaction[i] = np.log(Mcl[-1]/Mcl[0])/np.log(maxMcl/simParam.minMcl)
                    MstarVals[i] = maxStarFaction[i] * MstarMaxTmp
                    maxCloudMass[i] = Mcl[-1]
                    clumps[i,0] = MstarVals[i] / muSFEVals[i]
                    clumps[i,2] = calc_tsf(simParam.clumpMeanDensityPercm3,Mcl[-1]) * Gal.simParam.tff + Gal.dt
                    clumps[i,12] = calc_tsf0(simParam.clumpMeanDensityPercm3,Mcl[-1]) * Gal.simParam.tff
                    clumps[i,3] = clumps[i,2]
            clumps[:,1] = MstarVals
            clumps[:,0] *= MstarMaxTmp / np.sum(MstarVals)
            clumps[:,1] *= MstarMaxTmp / np.sum(MstarVals)

            clumps[:,4] = Gal.j
            clumps[:,5] = Gal.metallicity
            clumps[:,6] = where(simParam.ZZ, Gal.metallicity)
            clumps[:,9] = Gal.dustToGasRatio
            clumps[:,10] = clumps[:,0] * clumps[:,9]
            clumps[:,11] = Gal.get_cloud_lifetime(Gal.maxStarMassVals)
            clumps[:,13] = np.arange(Gal.maxStarMassVals.shape[0])
            clumps = clumps[clumps[:,0]>0]
            Gal.smoothMstarFraction[fmassiveBin] = maxStarFaction
            Gal.smoothMaxCloudMass[fmassiveBin]  = maxCloudMass
            Gal.smoothMeanSFEVals[fmassiveBin]   = muSFEVals
            Gal.smoothMstarFractionExists[fmassiveBin] = True
    return clumps


def get_smooth_clump_list(Gal,MstarMax):
    simParam = Gal.simParam
    maxMcl = min(Gal.cloudMassLimit[Gal.j],simParam.maxMcl)
    if (Gal.IMF_type == 'salpeter') + (Gal.IMF_type == 'evolving') + (Gal.IMF_type == 'evolving2'):
        #print(f'Substep {Gal.j}, Tsmooth = {simParam.smoothMassThreshold}')
        if simParam.smoothMassThreshold< 1:
            
            clumps = np.zeros((1,14))
            if (Gal.hasSmoothGrid == True)*(maxMcl==simParam.maxMcl):
                if (Gal.constantCloudDensity) * (simParam.clumpMeanDensityPercm3 in [1e2,1e3,1e4,1e5]):
                    if (Gal.IMF_type == 'salpeter'):
                        muMstar = Gal.smoothSFgrid[where(simParam.ZZ,Gal.metallicity)]
                    else:
                        muMstar  = Gal.smoothSFgrid[where(Gal.fmassiveClList,Gal.fmassiveHistory[Gal.j]),where(simParam.ZZ,Gal.metallicity)]
                else:
                    if (Gal.IMF_type == 'salpeter'):
                        muMstar = simParam.smoothSFInterps[0](np.array([np.log10(Gal.cloudDensityHistory[Gal.j]),Gal.metallicity]))
                    else:
                        muMstar = simParam.smoothSFInterps[1](np.array([np.log10(Gal.cloudDensityHistory[Gal.j]), Gal.fmassiveHistory[Gal.j], Gal.metallicity]))
            else:
                T = calc_T_from_Z_Shaver1983(Gal.metallicity)
                vesc0 = calc_cs(T)
                mm = np.linspace(simParam.minMcl,maxMcl,1_000)
                Xi = Gal.get_Xi_from_fmassive()
                muMstar = simps(calc_SFE_array(Gal.cloudDensityHistory[Gal.j],mm,T,Xi,vesc0,Gal.SFE_function) * clump_mass_function(mm,simParam.minMcl,maxMcl) * mm,x = mm)
            muSFE = muMstar / calc_mean_clump_mass( simParam.minMcl, maxMcl)
            clumps[0,0] = MstarMax / muSFE
            clumps[0,1] = MstarMax
            clumps[0,2] = calc_tsf(Gal.cloudDensityHistory[Gal.j],maxMcl) * calc_tff(Gal.cloudDensityHistory[Gal.j]) + Gal.dt
            clumps[0,12] = calc_tsf0(Gal.cloudDensityHistory[Gal.j],maxMcl) * calc_tff(Gal.cloudDensityHistory[Gal.j])
            clumps[0,3] = clumps[0,2] + clumps[0,12]
            clumps[0,4:10] = [Gal.j, 
                              Gal.metallicity, 
                              where(simParam.ZZ,Gal.metallicity),
                              0,
                              0,
                              Gal.dustToGasRatio]
            clumps[0,10] = clumps[0,0] * clumps[0,9]
            clumps[0,11] = np.maximum(Gal.get_cloud_lifetime(0), clumps[0,2]) + clumps[0,12]
            
            clumps[0,13] = 100
            return clumps
        else:
            #print(f'Using proper description in substep {Gal.j}')
            clumps = np.zeros((1,14))
            clumps[0, 0] = MstarMax
            clumps[0, 1] = MstarMax
            clumps[0, 2] = Gal.dt-0.01
            clumps[0, 3] = Gal.dt-0.01
            clumps[0,4:10] = [Gal.j, 
                              Gal.metallicity, 
                              where(simParam.ZZ,Gal.metallicity),
                              0,
                              0,
                              Gal.dustToGasRatio]
            clumps[0,10] = clumps[0,0] * clumps[0,9]
            clumps[0,11] = Gal.dt-0.01
            
            clumps[0,13] = 100
            return clumps
    elif (Gal.IMF_type == 'salpeter_cloud') + (Gal.IMF_type == 'evolving_cloud'):
        if (Gal.hasSmoothGrid == False) + (maxMcl<simParam.maxMcl):
            clumps = np.zeros((Gal.maxStarMassVals.shape[0],14))
            metalBin = where(simParam.ZZ,Gal.metallicity)
            listBin = Gal.MclListBins * metalBin
            if Gal.IMF_type == 'evolving_cloud':
                fmassiveBin = find_nearest(Gal.fmassiveHistory[Gal.j],Gal.fmassiveClList)
                listBin = listBin * Gal.fmassiveClList.shape[0] + Gal.MclListBins * fmassiveBin
            logn =np.log10(Gal.cloudDensityHistory[Gal.j]) 
            if logn%1==0:
                MstarLimitVals = Gal.maxStarMassGrid[int(logn)-2,listBin:listBin + Gal.MclListBins]
                SFEvals = Gal.SFEgrid[int(logn)-2,listBin:listBin + Gal.MclListBins]
            else:
                Mtmp0 = Gal.maxStarMassGrid[int(logn)-2,listBin:listBin + Gal.MclListBins]
                Mtmp1 = Gal.maxStarMassGrid[int(logn)-1,listBin:listBin + Gal.MclListBins]
                MstarLimitVals = ((logn%1) * (Mtmp1-Mtmp0) + Mtmp0).astype(int)
                SFEtmp0 = Gal.SFEgrid[int(logn)-2,listBin:listBin + Gal.MclListBins]
                SFEtmp1 = Gal.SFEgrid[int(logn)-1,listBin:listBin + Gal.MclListBins]
                SFEvals = ((logn%1) * (SFEtmp1-SFEtmp0) + SFEtmp0)
            
            muSFEVals = np.zeros(Gal.maxStarMassVals.shape[0])
            MstarVals = np.zeros(Gal.maxStarMassVals.shape[0])
            for i in range(Gal.maxStarMassVals.shape[0]):
                
                mask = (MstarLimitVals == i) * (Gal.MclList >= simParam.minMcl) * (Gal.MclList <= maxMcl)

                if np.sum(mask):
                    if mask[-1] == False:
                        mask[np.where(mask)[0][-1] +1] = True
                    Mcl = Gal.MclList[mask]

                    muMstar = simps(SFEvals[mask] * clump_mass_function(Mcl,Mcl[0],Mcl[-1]) * Mcl, x=Mcl)
                    muSFEVals[i] = muMstar / ((-1)/(Mcl[-1]**(-1)-Mcl[0]**(-1))*(np.log(Mcl[-1]/Mcl[0])))
                    MstarVals[i] = np.log(Mcl[-1]/Mcl[0])/np.log(maxMcl/simParam.minMcl) * MstarMax
                    clumps[i,0] = MstarVals[i] / muSFEVals[i]
                    clumps[i,2] = calc_tsf(Gal.cloudDensityHistory[Gal.j],Mcl[-1]) * calc_tff(Gal.cloudDensityHistory[Gal.j]) + Gal.dt
                    clumps[i,12] = calc_tsf0(Gal.cloudDensityHistory[Gal.j],Mcl[-1]) * calc_tff(Gal.cloudDensityHistory[Gal.j])
                    clumps[i,3] = clumps[i,2] + clumps[i,12]


            #clumps[:,0] = MstarVals / muSFEVals
            
            clumps[:,1] = MstarVals
            clumps[:,0] *= MstarMax / np.sum(MstarVals)
            clumps[:,1] *= MstarMax / np.sum(MstarVals)

            clumps[:,4] = Gal.j
            clumps[:,5] = Gal.metallicity
            clumps[:,6] = where(simParam.ZZ, Gal.metallicity)
            clumps[:,9] = Gal.Mdust/ Gal.Mgas
            clumps[:,10] = clumps[:,0] * clumps[:,9]
            clumps[:,11] = np.maximum(clumps[:,12] + Gal.get_cloud_lifetime(Gal.maxStarMassVals),clumps[:,2])
            clumps[:,13] = np.arange(Gal.maxStarMassVals.shape[0])
            #print(clumps[:,2],clumps[:,3])
            return clumps[clumps[:,0]>0]
        else:
            
            clumps = np.zeros((Gal.maxStarMassVals.shape[0],14))
            metalBin = where(simParam.ZZ,Gal.metallicity)
            if Gal.IMF_type == 'evolving_cloud':
                fmassiveBin = find_nearest(Gal.fmassiveHistory[Gal.j],Gal.fmassiveClList)
                if (Gal.constantCloudDensity) * (simParam.clumpMeanDensityPercm3 in [1e2,1e3,1e4,1e5]):
                    smoothVals = Gal.smoothSFgrid[:,metalBin,fmassiveBin]
                else:
                    smoothVals = simParam.smoothSFInterps[3]((np.log10(Gal.cloudDensityHistory[Gal.j]),np.array([0,1,2,3])[:,None],simParam.ZZ[metalBin],Gal.fmassiveClList[fmassiveBin],Gal.maxStarMassVals))
            else:
                if (Gal.constantCloudDensity) * (simParam.clumpMeanDensityPercm3 in [1e2,1e3,1e4,1e5]):
                    smoothVals = Gal.smoothSFgrid[:,metalBin]
                else:
                    smoothVals = simParam.smoothSFInterps[2]((np.log10(Gal.cloudDensityHistory[Gal.j]),np.array([0,1,2,3])[:,None],simParam.ZZ[metalBin],Gal.maxStarMassVals))
            MclTotal = MstarMax / np.sum(smoothVals[1] * smoothVals[0])
            clumps[:,0]  = MclTotal * smoothVals[1]
            clumps[:,1]  = clumps[:,0] * smoothVals[0]
            clumps[:,2]  = calc_tsf(Gal.cloudDensityHistory[Gal.j],smoothVals[3]) * calc_tff(Gal.cloudDensityHistory[Gal.j]) + Gal.dt
            clumps[:,12] = calc_tsf0(Gal.cloudDensityHistory[Gal.j],smoothVals[3]) * calc_tff(Gal.cloudDensityHistory[Gal.j])
            clumps[:,3]  = clumps[:,2] + clumps[:,12]
            clumps[:,4] = Gal.j
            clumps[:,5] = Gal.metallicity
            clumps[:,6] = where(simParam.ZZ, Gal.metallicity)
            clumps[:,9] = Gal.Mdust/ Gal.Mgas
            clumps[:,10] = clumps[:,0] * clumps[:,9]
            clumps[:,11] = np.maximum(clumps[:,12] + Gal.get_cloud_lifetime(Gal.maxStarMassVals), clumps[:,2])
            clumps[:,13] = np.arange(Gal.maxStarMassVals.shape[0])
            return clumps[clumps[:,0]>0]

    else: 
        print(f'How did you get this far without breaking. {Gal.IMF_type} is not a valid IMF')
        
def calc_mean_stellar_mass_from_clouds(n,minMcl,maxMcl):
    a = 7016 * n**-0.493
    b = -0.39
    return maxMcl * minMcl / (maxMcl - minMcl) / b * np.log((maxMcl/minMcl)**b * (a * minMcl ** b + 1) / (a * maxMcl ** b + 1)) * 0.87
def calc_mean_sfe_from_clouds(n,minMcl,maxMcl):
    return calc_mean_stellar_mass_from_clouds(n,minMcl,maxMcl) / calc_mean_clump_mass(minMcl,maxMcl)
def calc_mean_tsf_from_clouds(n,minMcl,maxMcl):
    a = 0.1566 * n ** 0.17; b = 0.085
    return maxMcl * minMcl / (maxMcl - minMcl) * (a/(b-1)) * (maxMcl ** (b-1) - minMcl ** (b-1))
def calc_mean_tsf0_from_clouds(n,minMcl,maxMcl):
    a = 4.239 * n **-0.085; b = -0.16
    return maxMcl * minMcl / (maxMcl - minMcl) * (a/(b-1)) * (maxMcl ** (b-1) - minMcl ** (b-1))

def form_stars_from_clumps(Gal):
    if (Gal.IMF_type == 'salpeter_cloud') + (Gal.IMF_type == 'evolving_cloud'):
        Gal.starGrid, Gal.allClumps, metalsReturned, dustReturned, gasReturned, Gal.starsFormed = form_stars_cloud_imf(Gal.allClumps, 
                                                                                                                       Gal.starGrid, 
                                                                                                                       Gal.j, 
                                                                                                                       Gal.dt)
    else:
        Gal.starGrid, Gal.allClumps, metalsReturned, dustReturned, gasReturned, Gal.starsFormed = form_stars_global_imf(Gal.allClumps, 
                                                                                                                        Gal.starGrid, 
                                                                                                                        Gal.j, 
                                                                                                                        Gal.dt)
    return gasReturned, metalsReturned, dustReturned

@njit()
def form_stars_cloud_imf(allClumps,starGrid,j,dt):
    n = allClumps.shape[0] # Number of star forming clumps
    metalsReturned=0. # Metal mass to be returned to ISM from finished clumps
    dustReturned = 0.
    gasReturned = 0.
    starsFormed = 0.
    if n:
        dead_cloud_mask = allClumps[:,11]<dt
        dead_mask = (allClumps[:,3]<dt) * (allClumps[:,3]>0)
        new_mask  = (allClumps[:,12] > 0) * (allClumps[:,12] < dt)
        not_new_or_dead_mask = (allClumps[:,3]>=dt) * (allClumps[:,12] <= 0)
        new_dead_mask = dead_mask * new_mask
        new_or_dead_mask = dead_mask + new_mask
        new_not_dead_mask = (~dead_mask) * new_mask
        not_new_dead_mask = dead_mask * (~new_mask)
        extra_mask = (allClumps[:,3] <= 0) + (allClumps[:,12] >= dt)
        allDeadClumps = allClumps[dead_cloud_mask]
        for i in prange(allDeadClumps.shape[0]):
            gasReturned += allDeadClumps[i,0]-allDeadClumps[i,1]
            metalsReturned += (allDeadClumps[i,0]-allDeadClumps[i,1])*allDeadClumps[i,5]
            dustReturned += (allDeadClumps[i,0]-allDeadClumps[i,1])*allDeadClumps[i,9]
        
        deltaT = allClumps[:,3].copy()
        
        deltaT[not_new_or_dead_mask] = dt
        deltaT[new_not_dead_mask] = np.maximum(dt-allClumps[new_not_dead_mask,12],0)
        deltaT[new_dead_mask] = allClumps[new_dead_mask,2]
        deltaT[extra_mask] = 0
        deltaT*= 1/allClumps[:,2]
        
        #print(n,np.sum(new_dead_mask+not_new_or_dead_mask+new_not_dead_mask+not_new_dead_mask+extra_mask),np.sum(deltaT>1),np.sum(deltaT[new_dead_mask]>1), np.sum(deltaT[not_new_or_dead_mask]>1), np.sum(deltaT[new_not_dead_mask]>1), np.sum(deltaT[not_new_dead_mask]>1), np.sum(deltaT[extra_mask]>1))
        
        metalBins = np.zeros(allClumps.shape[0],dtype = 'int')
        metalBins[:] = allClumps[:,6]
        MstarLimitBins = np.zeros(allClumps.shape[0],dtype = 'int')
        MstarLimitBins[:] = allClumps[:,13]
        for i in prange(n):
            starGrid[j,metalBins[i],MstarLimitBins[i]] += allClumps[i,1] * deltaT[i]
        
            
        allClumps[~dead_mask,3] -= dt
        allClumps[:,12] -= dt
        allClumps[dead_mask,3] = 0
        starsFormed += starGrid[j].sum()
    return starGrid, allClumps, metalsReturned, dustReturned, gasReturned, starsFormed
@njit()
def form_stars_global_imf(allClumps,starGrid,j,dt):
    n = allClumps.shape[0] # Number of star forming clumps
    metalsReturned=0. # Metal mass to be returned to ISM from finished clumps
    dustReturned = 0.
    gasReturned = 0.
    starsFormed = 0.
    if n:
        dead_cloud_mask = allClumps[:,11]<dt
        dead_mask = (allClumps[:,3]<dt) * (allClumps[:,3]>0)
        new_mask  = (allClumps[:,12] > 0) * (allClumps[:,12] < dt)
        not_new_or_dead_mask = (allClumps[:,3]>=dt) * (allClumps[:,12] <= 0)
        new_dead_mask = dead_mask * new_mask
        new_or_dead_mask = dead_mask + new_mask
        new_not_dead_mask = (~dead_mask) * new_mask
        not_new_dead_mask = dead_mask * (~new_mask)
        extra_mask = (allClumps[:,3] <= 0) + (allClumps[:,12] >= dt)
        allDeadClumps = allClumps[dead_cloud_mask]
        for i in prange(allDeadClumps.shape[0]):
            gasReturned += allDeadClumps[i,0]-allDeadClumps[i,1]
            metalsReturned += (allDeadClumps[i,0]-allDeadClumps[i,1])*allDeadClumps[i,5]
            dustReturned += (allDeadClumps[i,0]-allDeadClumps[i,1])*allDeadClumps[i,9]
        
        deltaT = allClumps[:,3].copy()
        
        deltaT[not_new_or_dead_mask] = dt
        deltaT[new_not_dead_mask] = np.maximum(dt-allClumps[new_not_dead_mask,12],0)
        deltaT[new_dead_mask] = allClumps[new_dead_mask,2]
        deltaT[extra_mask] = 0
        deltaT*= 1/allClumps[:,2]
        
        metalBins = np.zeros(allClumps.shape[0],dtype = 'int')
        metalBins[:] = allClumps[:,6]
        for i in prange(n):
            starGrid[j,metalBins[i]] +=  allClumps[i,1]*deltaT[i]
            
            
        allClumps[~dead_mask,3] -= dt
        allClumps[:,12] -= dt
        allClumps[dead_mask,3] = 0
        starsFormed += starGrid[j].sum()
    return starGrid, allClumps, metalsReturned, dustReturned, gasReturned, starsFormed


### Function used to compute values for the lifetime indexes files
def lifetime_indexes(substepTimeList,lifetime,IMF_type = 'salpeter'):
    if IMF_type == 'salpeter':
        indexes = np.zeros((substepTimeList.shape[0],substepTimeList.shape[0],2),int)
        for step in range(1,substepTimeList.shape[0]):
            for prevstep in range(step):
                args = np.argwhere((lifetime[:500]<(substepTimeList[step]-substepTimeList[prevstep]))*(lifetime[:500]>(substepTimeList[step]-substepTimeList[prevstep+1])))
                if len(args):
                    indexes[step,prevstep] = [args[0,0],args[-1,0]]
                else:
                    indexes[step,prevstep] = -1
    return indexes

### Compute metal, dust, and gas yields of dyying stars
def kill_stars(Gal):
    simParam = Gal.simParam
    
    dm = simParam.MM[1]-simParam.MM[0] #DeltaM for IMF computations
    returnedGas = 0  #Gas to be returned from dying stars
    metals = 0 #Metals to be returned from dying stars
    dust = 0
    indexSN = np.argwhere(simParam.MM>=simParam.MSN)[0,0]
    time = simParam.substepTimeList[Gal.j]
    
    dustDestructionRate = (1-simParam.fracColdGas) * Gal.dustToGasRatio * simParam.MSNblastwave * simParam.dustDestrEfficiency
    dustGrainGrowthRate = (Gal.metallicity -Gal.dustToGasRatio) * simParam.fracColdGas * Gal.Mdust / (simParam.dustGrowthTimescale * 0.0134)
    dust += Gal.dt * 1000 * dustGrainGrowthRate
    indexes = np.argwhere(simParam.lifetimeIndexes[Gal.j,simParam.startStep:Gal.j,0]!=-1).T[0]+ simParam.startStep
    if len(indexes):
        if Gal.IMF_type == 'salpeter':
            metals, returnedGas, Dust, Gal.metalMask = kill_stars_salpeter(Gal.IMF_list, 
                                                                           Gal.metalMask, 
                                                                           Gal.starGrid, 
                                                                           Gal.j, 
                                                                           indexes, 
                                                                           simParam.lifetimeIndexes, 
                                                                           Gal.yields, 
                                                                           Gal.returns, 
                                                                           Gal.dustYields, 
                                                                           simParam.dustProductionRate-dustDestructionRate)
            dust += Dust

        elif Gal.IMF_type == 'salpeter_cloud':
            metals, returnedGas, Dust, Gal.metalMask = kill_stars_salpeter_cloud(Gal.IMF_list, 
                                                                                 Gal.metalMask, 
                                                                                 Gal.starGrid, 
                                                                                 Gal.j, 
                                                                                 indexes, 
                                                                                 simParam.lifetimeIndexes, 
                                                                                 Gal.yields, 
                                                                                 Gal.returns, 
                                                                                 Gal.dustYields, 
                                                                                 simParam.dustProductionRate-dustDestructionRate)
            dust += Dust

        elif Gal.IMF_type == 'evolving_cloud':
            metals, returnedGas, Dust, Gal.metalMask = kill_stars_evolving_cloud(Gal.IMF_list, 
                                                                                 Gal.metalMask, 
                                                                                 Gal.starGrid, 
                                                                                 Gal.j, 
                                                                                 indexes, 
                                                                                 simParam.lifetimeIndexes, 
                                                                                 Gal.fmassiveIndexes, 
                                                                                 Gal.yields, 
                                                                                 Gal.returns, 
                                                                                 Gal.dustYields, 
                                                                                 simParam.dustProductionRate-dustDestructionRate)
            dust += Dust
        elif Gal.IMF_type == 'evolving': 
            metals, returnedGas, Dust, Gal.metalMask = kill_stars_evolving(Gal.IMF_list, 
                                                                           Gal.metalMask, 
                                                                           Gal.starGrid, 
                                                                           Gal.j, 
                                                                           indexes, 
                                                                           simParam.lifetimeIndexes, 
                                                                           Gal.yields, 
                                                                           Gal.returns, 
                                                                           Gal.dustYields, 
                                                                           simParam.dustProductionRate-dustDestructionRate)
            dust += Dust
            #for i in indexes: 
            #    index = simParam.lifetimeIndexes[Gal.j,i]
            #    returns = np.multiply(Gal.IMF_list[i,index[0]:index[1]+1], Gal.returns[index[0]:index[1]+1])
            #    returnedGas += np.multiply(Gal.starGrid[i],returns).sum()
            #    yields = np.multiply(Gal.IMF_list[i,index[0]:index[1]+1], Gal.yields[index[0]:index[1]+1])
            #    metals += np.multiply(Gal.starGrid[i],yields).sum()
#
            #    if index[0]<79:
            #        dustYields = np.multiply(Gal.IMF_list[i,index[0]:min(index[1]+1,79)], Gal.dustYields[index[0]:min(index[1]+1,79)])
            #        dust += np.multiply(Gal.starGrid[i],yields).sum()
            #    if index[1]>=79:
            #        dust += np.multiply(Gal.starGrid[i], Gal.IMF_list[i,max(index[0],79):index[1]+1]).sum() * (simParam.dustProductionRate - dustDestructionRate)

        elif Gal.IMF_type == 'evolving2':
            metals, returnedGas, dust, Gal.metalMask, Gal.IMF_list = kill_stars_evolving2(Gal.IMF_list, 
                                                                                          Gal.metalMask, 
                                                                                          Gal.starGrid, 
                                                                                          Gal.j, 
                                                                                          indexes, 
                                                                                          simParam.lifetimeIndexes, 
                                                                                          Gal.yields, 
                                                                                          Gal.returns, 
                                                                                          Gal.dustYields, 
                                                                                          simParam.dustProductionRate-dustDestructionRate, 
                                                                                          simParam.MM, 
                                                                                          simParam.substepRedshiftList[Gal.j], 
                                                                                          Gal.Mgas/Gal.Mhalo, 
                                                                                          simParam.cutoffMass)
            
            
            #Gal.IMF_list[Gal.j] = IMF_evolving_fmassive(simParam.MM[:500],simParam.substepRedshiftList[Gal.j],Gal.Mgas/Gal.Mhalo,simParam.cutoffMass,simParam.OB0,simParam.OM0) * dm 
            #for i in indexes: 
            #    index = simParam.lifetimeIndexes[Gal.j,i]
#
            #    stellarMassList = np.outer(Gal.starGrid[i], Gal.IMF_list[i,index[0]:index[1]+1])
            #    returnedGas += np.multiply(stellarMassList,Gal.returns[:,index[0]:index[1]+1]).sum()
            #    metals += np.multiply(stellarMassList,Gal.yields[:,index[0]:index[1]+1]).sum()
            #    if index[0]<79:
            #        dust += np.multiply(stellarMassList[:,:min(index[1]+1,79)-index[0]],Gal.dustYields[:,index[0]:min(index[1]+1,79)]).sum()
            #    if index[1]>=79:
            #        dust += Gal.SFRHistory[i] * np.sum(Gal.IMF_list[i,max(index[0],79):index[1]+1] * simParam.MM[max(index[0],79):index[1]+1]) * (simParam.dustProductionRate-dustDestructionRate)
                
    return returnedGas,metals,dust

@njit()
def kill_stars_evolving2(imf, metalMask, starGrid, j, indexes, lifetimeIndexes, yields, returns, dustYields, dustRate, MM, redshift, gasDensity, cutoffMass):
    metalMask[j] = starGrid[j]>0
    imf[j] = IMF_evolving_fmassive(MM[:500],redshift,gasDensity,cutoffMass) * (MM[1] - MM[0]) 
    metals = 0.
    returnedGas = 0.
    dust = 0.
    for k in prange(indexes.shape[0]):
        i = indexes[k]
        index = lifetimeIndexes[j,i] 
        mask = metalMask[i]
        if np.any(mask):
            thisIMF = imf[i,index[0]:index[1]+1]
            thisStarGrid = starGrid[i,mask]
            theseYields = yields[mask,index[0]:index[1]+1]
            theseReturns = returns[mask,index[0]:index[1]+1]
            theseDustYields = dustYields[mask,index[0]:index[1]+1]
            for Zbin in prange(thisStarGrid.shape[0]):
                starsFormed = (thisIMF * thisStarGrid[Zbin])
                metals += (theseYields[Zbin] * starsFormed).sum()
                returnedGas += (theseReturns[Zbin] * starsFormed).sum()
                if index[1]>=79:
                    dust += (starsFormed[max(index[0],79)-index[0]:index[1]+1-index[0]]).sum() * (dustRate)
                if index[0]<79:
                    dust += (theseDustYields[Zbin,:min(index[1],79)+1-index[0]] * starsFormed[:min(index[1],79)+1-index[0]]).sum()
    return metals, returnedGas, dust, metalMask, imf

@njit()
def kill_stars_evolving(imf, metalMask, starGrid, j, indexes, lifetimeIndexes, yields, returns, dustYields, dustRate):
    metalMask[j] = starGrid[j]>0
    metals = 0.
    returnedGas = 0.
    dust = 0.
    for k in prange(indexes.shape[0]):
        i = indexes[k]
        index = lifetimeIndexes[j,i] 
        mask = metalMask[i]
        if np.any(mask):
            thisIMF = imf[i,mask,index[0]:index[1]+1]
            thisStarGrid = starGrid[i,mask]
            theseYields = yields[mask,index[0]:index[1]+1]
            theseReturns = returns[mask,index[0]:index[1]+1]
            theseDustYields = dustYields[mask,index[0]:index[1]+1]
            for Zbin in prange(thisStarGrid.shape[0]):
                starsFormed = (thisIMF[Zbin] * thisStarGrid[Zbin])
                metals += (theseYields[Zbin] * starsFormed).sum()
                returnedGas += (theseReturns[Zbin] * starsFormed).sum()
                if index[1]>=79:
                    dust += (starsFormed[max(index[0],79)-index[0]:index[1]+1-index[0]]).sum() * (dustRate)
                if index[0]<79:
                    dust += (theseDustYields[Zbin,:min(index[1],79)+1-index[0]] * starsFormed[:min(index[1],79)+1-index[0]]).sum()
    return metals, returnedGas, dust, metalMask

@njit()
def kill_stars_salpeter(imf, metalMask, starGrid, j, indexes, lifetimeIndexes, yields, returns, dustYields, dustRate):
    metalMask[j] = starGrid[j]>0
    metals = 0.
    returnedGas = 0.
    dust = 0.
    for k in prange(indexes.shape[0]):
        i = indexes[k]
        index = lifetimeIndexes[j,i] 
        mask = metalMask[i]
        if np.any(mask):
            thisStarGrid = starGrid[i][mask]
            theseYields = yields[mask,index[0]:index[1]+1]
            theseReturns = returns[mask,index[0]:index[1]+1]
            theseDustYields = dustYields[mask,index[0]:index[1]+1]
            for Zbin in prange(thisStarGrid.shape[0]):
                metals += (theseYields[Zbin] * thisStarGrid[Zbin]).sum()
                returnedGas += (theseReturns[Zbin] * thisStarGrid[Zbin]).sum()
                if index[1]>=79:
                    dust += thisStarGrid[Zbin] * (dustRate) * np.sum(imf[max(index[0],79):index[1]+1])
                if index[0]<79:
                    dust += (theseDustYields[Zbin,:min(index[1],79)+1-index[0]] * thisStarGrid[Zbin]).sum()
    return metals, returnedGas, dust, metalMask
       
@njit(parallel = True)
def kill_stars_salpeter_cloud(imf, metalMask, starGrid, j, indexes, lifetimeIndexes, yields, returns, dustYields, dustRate):
    metalMask[j] = starGrid[j].sum(axis = 1)>0
    metals = 0.
    returnedGas = 0.
    dust = 0.
    for k in prange(indexes.shape[0]):
        i = indexes[k]
        index = lifetimeIndexes[j,i] 
        mask = metalMask[i]
        if np.any(mask):
            thisIMF = imf[index[0]:index[1]+1]
            thisStarGrid = starGrid[i][mask]
            theseYields = yields[mask,index[0]:index[1]+1]
            theseReturns = returns[mask,index[0]:index[1]+1]
            theseDustYields = dustYields[mask,index[0]:index[1]+1]
            for Zbin in prange(thisStarGrid.shape[0]):
                starsFormed = (thisIMF * thisStarGrid[Zbin]).sum(axis = 1)
                metals += (theseYields[Zbin] * starsFormed).sum()
                returnedGas += (theseReturns[Zbin] * starsFormed).sum()
                if index[1]>=79:
                    dust += (starsFormed[max(index[0],79)-index[0]:index[1]+1-index[0]]).sum() * (dustRate)
                if index[0]<79:
                    dust += (theseDustYields[Zbin,:min(index[1],79)+1-index[0]] * starsFormed[:min(index[1],79)+1-index[0]]).sum()
    return metals, returnedGas, dust, metalMask
    
@njit(parallel = True)
def kill_stars_evolving_cloud(imf, metalMask, starGrid, j, indexes, lifetimeIndexes, fmassiveIndexes, yields, returns, dustYields, dustRate):
    metalMask[j] = starGrid[j].sum(axis = 1)>0
    metals = 0.
    returnedGas = 0.
    dust = 0.
    for k in prange(indexes.shape[0]):
        i = indexes[k]
        mask = metalMask[i]
        if np.any(mask):
            index = lifetimeIndexes[j,i]
            fmask = fmassiveIndexes[i][mask]
            thisIMF = imf[fmask,index[0]:index[1]+1]
            thisStarGrid = starGrid[i][mask]
            theseYields = yields[mask,index[0]:index[1]+1]
            theseReturns = returns[mask,index[0]:index[1]+1]
            theseDustYields = dustYields[mask,index[0]:index[1]+1]
            for Zbin in prange(thisIMF.shape[0]):
                starsFormed = (thisIMF[Zbin] * thisStarGrid[Zbin]).sum(axis = 1)
                metals += (theseYields[Zbin] * starsFormed).sum()
                returnedGas += (theseReturns[Zbin] * starsFormed).sum()
                if index[1]>=79:
                    dust += (starsFormed[max(index[0],79)-index[0]:index[1]+1-index[0]]).sum() * (dustRate)
                if index[0]<79:
                    dust += (theseDustYields[Zbin,:min(index[1],79)+1-index[0]] * starsFormed[:min(index[1],79)+1-index[0]]).sum()
    return metals, returnedGas, dust, metalMask