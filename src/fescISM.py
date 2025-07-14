import numpy as np
from scipy.integrate import simpson as simps

radius_dust_grains = 5e-6
density_dust_grains = 2.25
grams_per_solar_mass = 2e33
cm_per_kpc = 3.086e21


### --------------------------------------------------------------- ###
### ---------------------- NFW profile ---------------------------- ###
### --------------------------------------------------------------- ###

def n0_factor_NFW(simParam):
    c = simParam.concentration_c
    return (18 *np.pi ** 2)/3  * c ** 3 * np.exp( gamma2_NFW(c) ) / dtIntegral_NFW(c) / simParam.MH_per_Msun * (simParam.pc_cm*simParam.h0)**3 /1.22## Assuming T = Tvir. else an extra factor of V**2 = T_vir/T = V_c**2/c_s**2 in the exponent
    
def F_factor_NFW(t):
    return np.log(1 + t) - t / (1 + t)

def gamma2_NFW(c):
    return 2 * c / F_factor_NFW(c)

def dtIntegral_NFW(c):
    gamma = gamma2_NFW(c)
    if np.isscalar(c):
        cc = np.linspace(0,c,1000)
        return simps((1+cc)**(gamma/cc)*cc**2,x=cc)
    else:
        dt = np.zeros_like(c)
        for i,C in enumerate(c):
            cc = np.linspace(0,C,1000)
            dt[i] = simps((1+cc)**(gamma[i]/cc)*cc**2,x=cc)
        return dt
    
def eta_NFW(Gal,simParam):
    c  = simParam.concentration_c[Gal.j]
    if Gal.riHistory[Gal.j]>0:
        cx = c * Gal.riHistory[Gal.j] / simParam.RvirSubstep[Gal.j]   
        gamma2 = gamma2_NFW(c)
        return np.exp(-gamma2)*(1+cx)**(gamma2/cx)
    else:
        return 1.
    
def calc_n_gal_surface_NFW(Gal,simParam):
    eta = eta_NFW(Gal,simParam)
    n0 = simParam.n0_factor_NFW[Gal.j] * Gal.Mgas / (Gal.Mhalo +Gal.Mgas)
    return eta * n0

def calc_n2_gal_vol_NFW(Gal,simParam):
    c  = simParam.concentration_c[Gal.j]
    x  = Gal.riHistory[Gal.j] / simParam.RvirSubstep[Gal.j]  
    
    g = gamma2_NFW(c)
    n0 = simParam.n0_factor_NFW[Gal.j] * Gal.Mgas / (Gal.Mhalo +Gal.Mgas)
    #print(f'c = {c:5.3f}, g = {g:5.3f}')
    eta2int = 0
    if x>0:
        xx = np.linspace(0,Gal.riHistory[Gal.j] / simParam.RvirSubstep[Gal.j] ,1000)
        eta2int = simps(xx**2 * (1+c*xx)**(2 * g/(c*xx)),x=xx)
        return 3 *np.exp(-2*g) /x**3* n0**2 * eta2int
    else:
        return n0**2

def calc_HI_extinction_NFW(Gal,simParam):
    gasRadius = simParam.RvirSubstep[Gal.j]
    ri = Gal.riHistory[Gal.j]
    n0 = simParam.n0_factor_NFW[Gal.j] * Gal.Mgas / (Gal.Mhalo +Gal.Mgas)
    c  = simParam.concentration_c[Gal.j]
    g  = gamma2_NFW(c)
    MgasIon = 0
    if ri>0:
        xx = np.linspace(0,ri / simParam.RvirSubstep[Gal.j] ,1000)
        MgasIon = 4 * np.pi * n0 * simParam.mp_g * np.exp(-g)*gasRadius**3 *  simps(xx**2 * (1+c*xx)**(g/(c*xx)),x=xx)
    #print('mean n ion = ',MgasIon/ (4 * np.pi * ri ** 3 / 3*simParam.mp_g))
    #print('MgasIon= ',MgasIon/grams_per_solar_mass)
    #print('Mgas= ',Gal.Mgas)
    sigma = Gal.Mgas * grams_per_solar_mass / (np.pi * (gasRadius) **2) * ((Gal.Mgas*grams_per_solar_mass-MgasIon)/(Gal.Mgas * grams_per_solar_mass))
    intFactor = 0.5* ( gasRadius- Gal.riHistory[Gal.j]) * (1- (Gal.riHistory[Gal.j]-gasRadius*simParam.rminGal)/(gasRadius*(1-simParam.rminGal)))
    tau = 3/4 * sigma / (simParam.mp_g/simParam.sigmaHI) * 0.5 / (1 + simParam.rminGal) * intFactor
    fesc = 0
    if tau < 1e-8:
        return 1
    else:
        fesc += (1-np.exp(-tau))/tau
        fesc += Gal.riHistory[Gal.j] / gasRadius
        return fesc
        
### --------------------------------------------------------------- ###
### ------------------ Exponential profile ------------------------ ###
### --------------------------------------------------------------- ###

def calc_HI_extinction(Gal,simParam):
    simParam = Gal.simParam
    nIGM, nMin, R,rc,n0,nc,Reff = get_gas_and_halo_properties(Gal)
    gasRadius = calc_dust_radius(simParam.RvirSubstep[Gal.j],simParam.substepRedshiftList[Gal.j]) 
    ri = Gal.riHistory[Gal.j]
    MgasIon = 4 * np.pi * n0*simParam.mp_g * (2*rc**3-rc*(2*rc**2+2*rc*ri + ri**2)*np.exp(-ri/rc))
    #print(((Gal.Mgas*grams_per_solar_mass-MgasIon)/(Gal.Mgas*grams_per_solar_mass)))
    sigma = Gal.Mgas * grams_per_solar_mass / (np.pi * (gasRadius)**2) * ((Gal.Mgas*grams_per_solar_mass-MgasIon)/(Gal.Mgas*grams_per_solar_mass))
    intFactor = 0.5* ( gasRadius- Gal.riHistory[Gal.j]) * (1- (Gal.riHistory[Gal.j]-gasRadius*simParam.rminGal)/(gasRadius*(1-simParam.rminGal)))
    tau = 3/4 * sigma/(simParam.mp_g/Gal.simParam.sigmaHI) * 0.5/(1+Gal.simParam.rminGal) * intFactor
    
    #tau = calc_HI_tau(Gal,min(gasRadius,simParam.Rvir[Gal.snap]))
    fesc = 0
    if tau < 1e-8:
        return 1
    else:
        fesc += (1-np.exp(-tau))/tau
        fesc += Gal.riHistory[Gal.j] / gasRadius
        return fesc
#omega = 0.75 - 1 https://arxiv.org/pdf/1507.02981

def get_gas_and_halo_properties(Gal):
    simParam = Gal.simParam
    nIGM   = (P18.critical_density(simParam.substepRedshiftList[Gal.j])/ac.m_p * P18.Ob(simParam.substepRedshiftList[Gal.j])).to(u.cm**-3).value#mean number density in IGM in cm^-3
    nMin   = simParam.rminGal
    R      = simParam.RvirSubstep[Gal.j]
    Reff   = min(calc_dust_radius(R,simParam.substepRedshiftList[Gal.j]),R)
    a = 0.5
    rc = Reff * a
    n0     = Gal.Mgas * simParam.MH_per_Msun/(4*np.pi*Reff**3)/(2 * a** 3 - (2 * a ** 3 + 2 * a ** 2 + a) * np.exp(-1/a)) #mean number density in center of galaxy in cm^-3
    #n1     = Gal.Mgas * simParam.MH_per_Msun/(4*np.pi*Reff**3/3)
    #print(f'{n0:5.3e},{n1:5.3e}')
    nc     = 2 * n0 /(1+nMin)
    
    return nIGM, nMin, R,rc,n0,nc,Reff

def calc_n2_gal_vol(Gal,simParam): #Volume averaged squared number density within ionizzed region of galaxy for recombination
    simParam = Gal.simParam
    nIGM, nMin, R,rc,n0,nc,Reff = get_gas_and_halo_properties(Gal)
    r = min(Gal.riHistory[Gal.j],R)#simParam.Rvir[Gal.snap]*cm_per_kpc)
    if r>0:
        #For uniformly distributed density 
        #return n0**2#nIGM**2# n0 ** 2 * (nMin**2 + nMin + 1) * rc / 6 * (1-np.exp(-2*r/rc))
        return  n0**2 * (nMin**2 + nMin + 1) * rc * (rc*rc - (rc*rc + 2*rc*r + 2*r*r) * np.exp(-(2*r)/rc)) *np.pi/ (4 * r**3/np.pi)
        #return n0**2
        # For spherically symmetric galaxy
        #return 3 * rho0 / r**3 * (2 * rc**3 - rc * (2 * rc**2 + 2 * r * rc + r**2) * np.exp(-r/rc))
    else:
        return n0**2#n0**2#
    
def calc_fescHI(Gal):
    simParam = Gal.simParam
    nIGM, nMin, R,rc,n0,nc,Reff = get_gas_and_halo_properties(Gal)
    r  = min(Gal.riHistory[Gal.j+1],Reff)
    tau0 = nc *  simParam.sigmaHI*(R-r) #*rc* (np.exp(-r/rc)-np.exp(-R/rc))
    
    if tau0 < 1e-8:
        return 1.
    else:
        fesc = (np.exp(-tau0 * nMin) * (1 - np.exp(-tau0 * (1-nMin))))/(tau0 * (1-nMin))
        return fesc

def calc_n_gal_surface(Gal,simParam):
    nIGM, nMin, R,rc,n0,nc,Reff = get_gas_and_halo_properties(Gal)
    r        = min(Gal.riHistory[Gal.j],R)
    ns = nc * np.exp(-r/rc)
    return ns#nIGM #ns


### --------------------------------------------------------------- ###
### ------------------ Dust attenuation --------------------------- ###
### --------------------------------------------------------------- ###

def calc_dust_tau_ion(dust_radius,dust_mass, radius_dust_grains = radius_dust_grains, density_dust_grains = density_dust_grains):
    sigma = dust_mass * grams_per_solar_mass / ( np.pi * (dust_radius )**2) * 5/3 #Factor 5/3 is for increase in extinction from 1500 Å to ionising photons, read from figure 4 in https://iopscience.iop.org/article/10.1086/379118/pdf
    tau   = 3/4 * sigma/(radius_dust_grains * density_dust_grains) 
    return tau
#@njit
def calc_dust_radius(Rvir,redshift,spin_param=0.04,dust_to_gas_radius_ratio = 1):
    return dust_to_gas_radius_ratio * Rvir * 4.5 * spin_param * ((1+redshift)/6)**1.8
#@njit("float64[:](float64[:], float64[:], float64[:], float64, float64,int64)")
def calc_dust_extinction_ion(redshift,Mdust,rvir,radius_dust_grains = radius_dust_grains,density_dust_grains = density_dust_grains,model = 0):
    dustRadius = calc_dust_radius(rvir,redshift)
    tau = calc_dust_tau_ion(min(dustRadius,rvir),Mdust,radius_dust_grains,density_dust_grains)
    #print(f'rdust/rvir = ',dustRadius/rvir,rvir)
    if tau < 1e-8:
        return 1
    elif model == 1:
        return np.exp(-tau)
    else:
        return (1-np.exp(-tau))/tau


### --------------------------------------------------------------- ###
### ------------------ All together now --------------------------- ###
### --------------------------------------------------------------- ###

def calc_fesc(Gal,Q):
    simParam = Gal.simParam
    deltaT = (simParam.substepTimeList[Gal.j+1]-simParam.substepTimeList[Gal.j])*simParam.Myr_s
    rvir = simParam.RvirSubstep[Gal.j]
    #nGal = 3 * Gal.gasMassHistory[Gal.j] * simParam.MH_per_Msun / (4 * np.pi *rvir**3 * simParam.cm3Perkpc3)
    if Gal.ISM_type == 'exp':
        nGalVol2 = calc_n2_gal_vol(Gal,simParam)
        nGalS    = calc_n_gal_surface(Gal,simParam)
        Gal.riHistory[Gal.j+1] = min(calc_dust_radius(simParam.RvirSubstep[Gal.j+1],simParam.substepRedshiftList[Gal.j+1]),(((4 * np.pi * simParam.alphaB * nGalVol2 * min(Gal.riHistory[Gal.j],rvir)**3-3*Q) * (np.exp(-simParam.alphaB * deltaT * nGalVol2/nGalS)) + 3*Q) / (4 * np.pi * simParam.alphaB * nGalVol2))**(1/3))
    else:
        nGalVol2 = calc_n2_gal_vol_NFW(Gal,simParam)
        nGalS    = calc_n_gal_surface_NFW(Gal,simParam)
        Gal.riHistory[Gal.j+1] = min(simParam.RvirSubstep[Gal.j+1],(((4 * np.pi * simParam.alphaB * nGalVol2 * min(Gal.riHistory[Gal.j],rvir)**3-3*Q) * (np.exp(-simParam.alphaB * deltaT * nGalVol2/nGalS)) + 3*Q) / (4 * np.pi * simParam.alphaB * nGalVol2))**(1/3))
    if Gal.ISM_type == 'exp':
        fescHI = calc_HI_extinction(Gal,simParam)#calc_fescHI(Gal)
    else:
        fescHI = calc_HI_extinction_NFW(Gal,simParam)
    dustExtinction = calc_dust_extinction_ion(simParam.substepRedshiftList[Gal.j],Gal.Mdust,0.5 * rvir/(1-simParam.rminGal))
    Qabs = Q * (1-fescHI*dustExtinction)
    trec = 1e5 /nGalS
    if Gal.verbose:
        print('term= ',4 * np.pi * simParam.alphaB * nGalVol2 * min(Gal.riHistory[Gal.j],rvir )**3)
        print('DeltaT= ',deltaT)
        print('nGalV= ',np.sqrt(nGalVol2))
        print('nGalS= ',nGalS)
        if Gal.ISM_type == 'NFW':
            print('n0   = ',simParam.n0_factor_NFW[Gal.j] * Gal.Mgas / (Gal.Mhalo+Gal.Mgas))
        print('Rstromgren= ',(3/(4*np.pi)*Q/(nGalVol2*simParam.alphaB))**(1/3))
        print('GalRi=      ',Gal.riHistory[Gal.j+1])
        print('Rvir =      ',simParam.RvirSubstep[Gal.j])
        print(f'Q = {Q:5.3e}')
        print(f'HI and dust fesc: {fescHI:5.3e}, {dustExtinction:5.3e}')
    Gal.fescISMHistory[Gal.j+1] = fescHI
    Gal.fescDustHistory[Gal.j+1] = dustExtinction
    
    return fescHI * dustExtinction
