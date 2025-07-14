import numpy as np
from IMF import *

def calc_Q_and_N_Clouds(Gal):
    simParam = Gal.simParam
    ri,N = ri_t_and_N_in_clouds(Gal)
    #print('ri= ',ri)
    #print('N= ',N)
    fescMC = calc_fesc_MC(Gal,ri)
    Qclouds = np.sum(N * fescMC)
    #print('Qclouds= ',Qclouds)
    return Qclouds, np.sum(N)
    
# Compute value of psi
def calc_psi(omega):
    return 4/(7 - 2*omega)

def calc_theta(omega):
    return 4/(7-4*omega)
    
# Compute ionised front radius in clouds
def ri_t_and_N_in_clouds(Gal):
    simParam = Gal.simParam
    #Time clouds have been forming stars in seconds
    t  = (Gal.allClumps[:,2]-Gal.allClumps[:,3]+ Gal.allClumps[:,12])*simParam.Myr_s
    #time until clouds destroyed
    tc = Gal.allClumps[:,11] 
    M = Gal.allClumps[:,1] * (Gal.allClumps[:,2]-Gal.allClumps[:,3]+ Gal.allClumps[:,12])/(Gal.allClumps[:,2])
    rc = Gal.allClumps[:,7]/3. * simParam.pc_cm
    Mgas = Gal.allClumps[:,0]
    #Initial Nion of stars, and break time, by IMF
    
    if Gal.IMF_type == 'salpeter':
        #Time when first stars die
        tb = 10 ** 0.6 + Gal.allClumps[:,12]#*simParam.Myr_s + np.zeros_like(t)
        S0 = 10**46.56 * np.ones_like(t)
    elif Gal.IMF_type == 'evolving':
        #Time when first stars die. Slightly different due to fitting
        tb = 10 ** 0.425 + Gal.allClumps[:,12]#*simParam.Myr_s + np.zeros_like(t)
        redshifts = simParam.substepRedshiftList[Gal.j]+np.zeros_like(t)# simParam.substepRedshiftList[Gal.allClumps[:,4].astype(int)]
        metals    = simParam.ZZ[Gal.allClumps[:,6].astype(int)]
        lZ = np.zeros_like(metals)
        lZ[metals>1e-4] = np.log10(metals[metals>1e-4])
        lZ[metals<=1e-4] = -4
        S0 = 10 **( 45.93 - (lZ+2) * 1.17 + 0.064 * (lZ + 3.46) * redshifts)
    else:
        Z = simParam.ZZ[Gal.allClumps[:,6].astype(int)]
        fMassive = get_fmassive_from_TBD_time_and_gasdensity(simParam.substepRedshiftList[Gal.j],Gal.Mgas/Gal.Mhalo,simParam.OB0,simParam.OM0)
        ltb = 0.49 * (1-23.37 * Z)+2.64 * (fMassive - 0.209)**2 if (fMassive<0.209) else 0.47 * (1-17.15 * Z)+0.06 * (1 - np.exp(-7.5 * (fMassive-0.209)))
        ltb[ltb<0.477]=0.477
        tb = 10 ** ltb + Gal.allClumps[:,12]# * simParam.Myr_s
        S0 = 10 ** ((3.4e-9 * (1 + 0.62 * Z) + 53.66 * (1-0.32 * Z) * (fMassive + 0.051 * (1 + 227.6 * Z)) ** (0.00766 * (1 + 50.01 * Z)))-6)
        
    n0 = simParam.clumpMeanDensityPercm3#Mean density of clouds
    omega = simParam.omega # CLoud profile fitting parameter (0 for homogenous clouds)
    psi = simParam.psi    # Just good to have this compactly
    theta = simParam.theta
    alphaB = simParam.alphaB# Hydrogen recombination coefficient
    ci = simParam.ci  #cm/s. sound speed at ionised front
    
    ri = np.zeros_like(t)
    Rstall = np.zeros_like(t)
    #lmask = (t<tb)
    #hmask = (t>=tb)
    ri = (((3*S0*M ) / (4*np.pi*alphaB))**0.25 * (ci**2/(n0*rc**omega))**0.5 * (4*t/5) / psi)**psi
    #ri[lmask] = (((3*S0[lmask]*M[lmask] ) / (4*np.pi*alphaB))**0.25 * (ci**2/(n0*rc[lmask]**omega))**0.5 * (4*t[lmask]/5) / psi)**psi
    #ri[hmask] = (((3*S0[hmask]*M[hmask]*tb[hmask]) / (4*np.pi*alphaB*t[hmask]))**0.25 * (ci**2/(n0*rc[hmask]**omega))**0.5 * (t[hmask] - tb[hmask]/5) / psi)**psi
    Rstall  = (((3*S0*M) / (4*np.pi*alphaB))**0.25 * ci/(n0*rc**omega) * (4/5) * 9.014585e+14)**theta #9.014585e+14 =() X/m_H * 3/(8 pi G))^0.5
    ri[ri>Rstall] = Rstall[ri>Rstall]
    
    N = np.zeros_like(t)
    #N = S0 * M * 0.5# * t[t<tb]
    N[tc>Gal.dt] = S0[tc>Gal.dt] * M[tc>Gal.dt]*0.5# * t[t<tb]
    N[tc<=Gal.dt] = S0[tc<=Gal.dt] * (M[tc<=Gal.dt] * ((Gal.dt-tc[tc<=Gal.dt])) / t[tc<=Gal.dt])# * (t[t>=tb] - tb/2)
    
    return ri,N

def calc_fesc_MC(Gal,ri):
    simParam = Gal.simParam
    fesc = np.zeros_like(ri)
    smooth_mask = np.argwhere(Gal.smoothStarFormation[Gal.allClumps[:,4].astype(int)]==1).T[0]
    
    #Find Maximum and minimum radii of clouds
    rMax = simParam.rmax * Gal.allClumps[:,7] * simParam.pc_cm
    rMin = simParam.rmin * Gal.allClumps[:,7] * simParam.pc_cm
    #If ionised front radius larger than rMax, then fesc = 1
    fesc[ri>rMax] = 1
    
    #Else it is this
    fesc[(ri<rMax)*(ri>rMin)] = ((ri-rMin)/(rMax-rMin))[(ri<rMax)*(ri>rMin)]
    
    #For those clouds that evaporate in this timestep
    dead_cloud_mask = Gal.allClumps[:,11]<Gal.dt
    Gal.allClumps[:,11]-=Gal.dt
    #fesc[dead_cloud_mask] = (Gal.allClumps[dead_cloud_mask,11]-tb)/Gal.dt * (1 - fesc[dead_cloud_mask]) + fesc[dead_cloud_mask]
    #If ionised front radius is smaller than rMin, then fesc = 0, but that is automatic
    
    ### For smoothed star formation timesteps
    
    for i in smooth_mask:
        if Gal.IMF_type == 'salpeter':
            S0 = simParam.S0Salpeter 
        elif Gal.IMF_type == 'evolving':
            redshift = simParam.substepRedshiftList[Gal.j]#[Gal.allClumps[i,4].astype(int)]
            metal    = simParam.ZZ[Gal.allClumps[i,6].astype(int)]
            if metal<1e-4:
                lZ = -4
            else:
                lZ = np.log10(metal)
            S0 = 10 **( 45.93 - (lZ+2) * 1.17 + 0.064 * (lZ + 3.46) * redshift) * (Gal.allClumps[i,2]-Gal.allClumps[i,3])/(Gal.allClumps[i,2])
            #print('S0=',S0)
        else:
            Z = simParam.ZZ[Gal.allClumps[i,6].astype(int)]
            fMassive = get_fmassive_from_TBD_time_and_gasdensity(simParam.substepRedshiftList[Gal.j], Gal.Mgas/Gal.Mhalo,simParam.OB0,simParam.OM0)
            S0 = 10 ** ((3.4e-9 * (1 + 0.62 * Z) + 53.66 * (1-0.32 * Z) * (fMassive + 0.051 * (1 + 227.6 * Z)) ** (0.00766 * (1 + 50.01 * Z)))-6)
        if Gal.allClumps[i,2]-Gal.allClumps[i,3] > simParam.tStall:
            fesc[i] = simParam.mean_fesc_rstall(S0)
        else:
            fesc[i] = simParam.mean_fesc_ri(S0,(Gal.allClumps[i,2]-Gal.allClumps[i,3])*simParam.Myr_s)
    return fesc