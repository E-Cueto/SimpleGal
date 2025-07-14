import numpy as np
import phys_const as pc
from IMF import *
from numba import jit, njit, vectorize, prange
import time as Ti

@njit()
def LUV_salpeter(t,t_i,t_f,z,Z,sfr):
    t,t_i,t_f = t*1e6,t_i*1e6,t_f*1e6
    t_0,t_1,s0  = 10 ** 5.95, 10 ** 6.48, sfr/1e6
    N1     = 10 ** 38.821
    N2     = 10 ** 39.065
    slope0,slope1 = 0.4678,-1.25
    
    L_UV   = np.zeros_like(t)
    t_0u   = t_f
    t_0l   = np.maximum(t_i,np.minimum(t-t_0,t_f))
    
    t_1u = np.minimum(t_f,np.maximum(t-t_0,t_i))
    t_1l = np.maximum(t_i,np.minimum(t-t_1,t_f))
    
    t_2u = np.minimum(t_f,np.maximum(t-t_1,t_i))
    t_2l = t_i
    
    L_UV +=  s0 * N1 * (t_0u - t_0l) 
    L_UV += -s0 * N1 * t_0 ** (-slope0) / (slope0 + 1) * ((t - t_1u) ** (slope0 + 1) - (t - t_1l) ** (slope0 + 1))
    L_UV += -s0 * N2 * t_1 ** (-slope1) / (slope1 + 1) * ((t - t_2u) ** (slope1 + 1) - (t - t_2l) ** (slope1 + 1))
    return L_UV
@njit()
def Nion_salpeter(t,t_i,t_f,z,Z,sfr):
    t,t_i,t_f = t*1e6,t_i*1e6,t_f*1e6
    t_0,s0  = 10**6.48, sfr/1e6
    Nion = np.zeros_like(t)
    t_0u = t_f
    t_0l = np.maximum(t_i,np.minimum(t-t_0,t_f))
    t_1u = np.minimum(t_f,np.maximum(t-t_0,t_i))
    t_1l = t_i
    N1    =  10**52.7
    Nion +=  s0 * N1 * (t_0u - t_0l) 
    Nion +=  s0 * N1 * t_0**3.9 / 2.9 * ((t-t_1u)**(-2.9)-(t-t_1l)**(-2.9))
    return Nion
@njit(parallel = True,fastmath = True, nogil = True)
def LUV_salpeter_cloud(t,t_i,t_f,Mmax,sfr):
    t,t_i,t_f = t*1e6,t_i*1e6,t_f*1e6
    s0   =  sfr/1e6
    L_UV = np.zeros_like(t)
    aN, bN, cN = 0.77682, 31.291, 6.647
    at0, bt0, ct0 = -0.2761, 6.49, 9.827
    ab, bb, cb = 0.2878, -0.1182, -8.66
    slope1, lN20, tb20 = -1.25, 32.403,7.
    lN1    = aN * np.log10(Mmax - cN) + bN
    lt0    = at0 * np.log10(Mmax - ct0) + bt0
    slope0 = ab * np.log10(Mmax - cb) + bb
    lt1    = (lN1 - lN20+ slope1 * tb20 - slope0 * lt0) / (slope1 - slope0)
    lN2    = lN20 + (lt1-tb20) * slope1
    N1 = s0 * 10 ** lN1
    N2 = s0 * 10 ** lN2
    t_0 = 10 ** lt0
    t_1 = 10 ** lt1
    for i in prange(t_0.shape[0]):
        t_0u = t_f
        t_0l = t_0u - np.maximum(t_i,np.minimum(t-t_0[i],t_f))

        t_1u = t - np.minimum(t_f,np.maximum(t-t_0[i],t_i))
        t_1l = t - np.maximum(t_i,np.minimum(t-t_1[i],t_f))

        t_2u = t - np.minimum(t_f,np.maximum(t-t_1[i],t_i))
        t_2l = t - t_i
        
        L_UV +=  ( N1[i] * (t_0l - t_0[i]**(-slope0[i]) / (1+slope0[i]) * ((t_1u)**(slope0[i] + 1)-(t_1l)**(slope0[i] + 1)))) + (-N2[i] *         t_1[i]**(-slope1   ) / (1+slope1   ) * ((t_2u)**(slope1    + 1)-(t_2l)**(slope1    + 1)) )
    return L_UV

@njit(parallel = True,fastmath = True,nogil = True)
def Nion_salpeter_cloud(t,t_i,t_f,Mmax,sfr):
    t,t_i,t_f = t*1e6,t_i*1e6,t_f*1e6
    s0   =  sfr/1e6
    Nion = np.zeros_like(t)
    aN, bN, cN, slope, lN20, tb20 = 1.888936, 43.00595, 8.985, -3.9, 44.97419, 6.9253
    lN1 = aN * np.log10(Mmax - cN) + bN 
    N1 = s0 * 10 ** lN1 
    t_0 = 10 ** ((lN1 - lN20) / slope + tb20)
    t_0u = t_f
    for i in prange(t_0.shape[0]):
        t_0l = t_0u - np.maximum(t_i,np.minimum(t-t_0[i],t_f))
        
        t_1u = (t-np.minimum(t_f,np.maximum(t-t_0[i],t_i)))**(slope + 1)
        t_1l = (t-t_i) ** (slope + 1)
        a = ( N1[i] * (t_0l + t_0[i]**(-slope) / (-(1+slope)) * ((t_1u)-(t_1l)) ))
        Nion +=  ( N1[i] * (t_0l + t_0[i]**(-slope) / (-(1+slope)) * ((t_1u)-(t_1l)) ))
    return Nion


@njit()
def LUV_evolving(t,t_i,t_f,z,Z,sfr):
    t,t_i,t_f = t*1e6,t_i*1e6,t_f*1e6
    lZ = np.zeros_like(Z)
    mask =  Z == 0
    lZ[mask] = -4
    lZ[~mask] = np.log10(Z[~mask])
    t_0, t_1, s0  = 10**6.475, 10**7.4, sfr/1e6
    L_UV = np.zeros_like(t)
    N0u   =  10 ** 39.9
    N0l   =  10 ** 38.0
    N1    =  N0u * fmassive(z,Z) + N0l * (1-fmassive(z,Z))
    N2    =  N1 * (t_1/t_0)**((lZ-z*.3)/3.5)
    
    t_0u = t_f
    t_0l = np.maximum(t_i,np.minimum(t-t_0,t_f))

    t_1u = np.minimum(t_f,np.maximum(t-t_0,t_i))
    t_1l = np.maximum(t_i,np.minimum(t-t_1,t_f))

    t_2u = np.minimum(t_f,np.maximum(t-t_1,t_i))
    t_2l = t_i
    for i in range(s0.shape[0]):
        L_UV +=  s0[i] * N1[i] * (t_0u - t_0l) 
        L_UV += -s0[i] * N1[i] * 3.5 / (lZ[i]-z*.3+ 3.5) * ((t-t_1u) **((lZ[i]-z*.3+3.5)/3.5)-(t-t_1l) **((lZ[i]-z*.3+3.5)/3.5)) / (t_0 ** ((lZ[i]-z*.3)/3.5))
        L_UV +=  s0[i] * N2[i] * t_1**1.35 / .35 * ((t-t_2u)**(-.35)-(t-t_2l)**(-.35))
    return L_UV

@njit()
def Nion_evolving(t,t_i,t_f,z,Z,sfr):
    t,t_i,t_f = t*1e6,t_i*1e6,t_f*1e6
    lZ = np.zeros_like(Z)
    mask =  Z < 1e-4
    lZ[mask] = -4
    lZ[~mask] = np.log10(Z[~mask])
    t_0, s0  = 10**6.35, sfr/1e6
    e0   = 51.93 - (lZ+2) * 1.17 + 0.064 * (lZ + 3.46) * z
    e1   = -3.91 - (((lZ+2.97) * 0.051) + 0.0293) * z
    N0   = 10 **e0
    Nion = np.zeros_like(t)
    t_0u = t_f
    t_0l = np.maximum(t_i,np.minimum(t-t_0,t_f))

    t_1u = np.minimum(t_f,np.maximum(t-t_0,t_i))
    t_1l = t_i
    for i in range(s0.shape[0]):
        Nion +=  s0[i] * N0[i] * ((t_0u - t_0l) - (1/(e1[i]+1)) * (t_0**(-e1[i])) * ((t-t_1u)**(e1[i]+1)-(t-t_1l)**(e1[i]+1)))
        
    return Nion

@njit(parallel = True,fastmath = True, nogil = True)
def LUV_evolving_cloud(t,t_i,t_f,Mmax,fmassive,sfr):
    t,t_i,t_f = t*1e6,t_i*1e6,t_f*1e6
    s0   =  sfr/1e6
    L_UV = np.zeros_like(t)
    aN, bN, cN = 0.77682, 31.291, 6.647
    at0, bt0, ct0 = -0.2761, 6.49, 9.827
    ab, bb, cb = 0.2878, -0.1182, -8.66
    lN20, tb20 = 32.403,7.
    fmin   = get_fmassive_from_Mc(Mmax)
    factorN = (1 + (np.sqrt(Mmax)-5) * 0.035)
    deltafN = (Mmax-10)**0.5 * 0.002
    expN = 18 * Mmax **-0.778
    t_0    = 10 ** (at0 * np.log10(Mmax - ct0) + bt0)
    slope0 = ab * np.log10(Mmax - cb) + bb
    lt1    = np.log10(377 * Mmax**(-1.2) + 1.3)+6
    t_1 = 10 ** lt1
    N1    = np.zeros_like(s0)
    slope1 = np.zeros_like(s0)
    N2 = np.zeros_like(s0)
    for i in prange(fmassive.shape[0]):
        N1[i] = s0[i] * 10 ** (aN * np.log10(Mmax - cN) + bN + ((np.maximum(fmassive[i],fmin) +deltafN) ** expN - (fmin+deltafN)**expN) * factorN)
    for i in prange(fmassive.shape[0]):
        slope1[i] = (-1.25 - np.maximum((fmassive[i] - fmin),0)**1.31 * (1-0.016 * (np.sqrt(Mmax)+1.4)))
    for i in prange(fmassive.shape[0]):
        N2[i] = s0[i] * 10 ** (lN20 + (lt1-tb20) * slope1[i])
    
    t_0u = t_f
    for i in prange(fmassive.shape[0]):
        for j in prange(Mmax.shape[0]):
            t_0l = t_0u - np.maximum(t_i,np.minimum(t-t_0[j],t_f))

            t_1u = t - np.minimum(t_f,np.maximum(t-t_0[j],t_i))
            t_1l = t - np.maximum(t_i,np.minimum(t-t_1[j],t_f))

            t_2u = (t - np.minimum(t_f,np.maximum(t-t_1[j],t_i)))
            t_2l = t - t_i
            L_UV +=  ( N1[i,j] * (t_0l - t_0[j]**(-slope0[j]) / (1+slope0[j]) * ((t_1u)**(slope0[j] + 1)-(t_1l)**(slope0[j] + 1)))) + (-N2[i,j] * (t_1[j]**(-slope1[i,j]) / (1+slope1[i,j]) * ((t_2u)**(slope1[i,j] + 1)-(t_2l)**(slope1[i,j] + 1))))
    return L_UV

@njit(parallel = True,fastmath = True, nogil = True)
def Nion_evolving_cloud(t,t_i,t_f,Mmax,fmassive,sfr):
    t,t_i,t_f = t*1e6,t_i*1e6,t_f*1e6
    s0   =  sfr/1e6
    Nion = np.zeros_like(t)
    aN, bN, cN = 1.888936, 43.00595, 8.985
    fmin = get_fmassive_from_Mc(Mmax)
    t_0 = (1013 * Mmax**(-1.8) + 2.257 ) * 1e6
    expSlope = 7 - 2.75 * np.log10(Mmax+5)
    factorSlope = 1. - np.log10(Mmax)*0.07
    
    slope = np.zeros((fmassive.shape[0],Mmax.shape[0]))
    N1 = np.zeros((fmassive.shape[0],Mmax.shape[0]))
    for i in prange(fmassive.shape[0]):
        N1[i] = s0[i] * 10 ** (aN * np.log10(Mmax - cN) + bN + (np.maximum(fmassive[i]-fmin,0)/(1-fmin))**(2.4-np.log10(Mmax)) *(1+ np.sqrt(Mmax-10) * 0.025)) 
        slope[i] = -3.9 - (np.minimum(fmassive[i],0.905) ** expSlope - fmin ** expSlope) * factorSlope    
    
    t_0u = t_f
    t_1l = (t-t_i)
    for i in prange(fmassive.shape[0]):
        for j in prange(Mmax.shape[0]):
            t_0l = t_0u - np.maximum(t_i,np.minimum(t-t_0[j],t_f))
            t_1u = (t-np.minimum(t_f,np.maximum(t-t_0[j],t_i)))
            
            Nion +=  ( N1[i,j] * (t_0l + t_0[j]**(-slope[i,j]) / (-(1+slope[i,j])) * ((t_1u)**(slope[i,j] + 1)-(t_1l)**(slope[i,j] + 1)) ))
    return Nion

@njit()
def LUV_evolving_fmassive(t,t_i,t_f,fMassive,Z,SFR):
    t,t_i,t_f = t*1e6,t_i*1e6,t_f*1e6
    t_0 = 1.26e6
    t_1 = 3.5035e6
    s0 = SFR
    exp1  = np.zeros_like(Z)
    exp2  = np.zeros_like(Z)
    lN2   = np.zeros_like(Z)
    
    exp2[fMassive<=0.7]   = -1.920 + 0.123 * np.exp(-2e3*(Z[fMassive<=0.7]- 0.001)) - 0.564
    exp2[fMassive>0.7]    = -1.920 + 0.123 * np.exp(-2e3*(Z[fMassive>0.7] - 0.001)) + 0.564 * np.cos(-4.155 * fMassive[fMassive>0.7]-0.385)
    lN0                   = 36 * (1 + 0.1 * Z) + 3.82 * (1 + 1.1 * Z) * (fMassive + 0.13)**0.146 - 6
    exp1[fMassive<=0.283] = 0.064 * (1 + 8166 * (Z[fMassive<=0.283] - 0.0035) ** 2) * np.exp(-3.29 * (fMassive[fMassive<=0.283]-0.5 * (1 + 9.89 * (Z[fMassive<=0.283]-6e-3)**2))**2)
    exp1[fMassive>0.283]  = 0.044 * (1 + 9311 * (Z[fMassive>0.283] - 0.0041) ** 2) * np.exp(-4.50 * (1 + 99886*(Z[fMassive>0.283]-6e-3)**2)*(fMassive[fMassive>0.283]-0.283))
    lN2[fMassive<0.252]   = -0.072 * (1 + 98.64 * Z[fMassive<0.252]) + 4.92 * (fMassive[fMassive<0.252]-0.252)**2
    lN2[fMassive>0.452]   = 0.023 * (1 - 53691 * Z[fMassive>0.452]**2) + 0.06 * (1 - np.exp(-7.5 * (fMassive[fMassive>0.452]-0.452)))
    lN2[lN2<0.]           = 0.
    
    N0   = 10 ** lN0
    N2   = 10 ** lN2 * N0
    exp1 = exp1 * 26.077
    L_UV = np.zeros_like(t)

    t_0u = t_f
    t_0l = np.maximum(t_i,np.minimum(t-t_0,t_f))

    t_1u = np.minimum(t_f,np.maximum(t-t_0,t_i))
    t_1l = np.maximum(t_i,np.minimum(t-t_1,t_f))

    t_2u = np.minimum(t_f,np.maximum(t-t_1,t_i))
    t_2l = t_i
    
    for i in range(s0.shape[0]):
        L_UV +=   s0[i] * N0[i] * (t_0u - t_0l) 
        L_UV +=  -s0[i] * N0[i] * (1/(exp1[i]+1)) * ((t-t_1u) ** (exp1[i]+1) - (t - t_1l) ** (exp1[i] + 1)) * t_0 ** (-exp1[i])
        L_UV +=  -s0[i] * N2[i] * (1/(exp2[i]+1)) * ((t-t_2u) ** (exp2[i]+1) - (t - t_2l) ** (exp2[i] + 1)) * t_1 ** (-exp2[i])
    return L_UV


@njit()
def Nion_evolving_fmassive(t,t_i,t_f,fMassive,Z,SFR):
    t,t_i,t_f = t*1e6,t_i*1e6,t_f*1e6
    s0 = SFR/1e6
    exp  = np.zeros_like(Z)
    lt0   = np.zeros_like(Z)
    
    lN0                  = 3.4e-9 * (1 + 0.62 * Z) + 53.66 * (1-0.32 * Z) * (fMassive + 0.051 * (1 + 227.6 * Z)) ** (0.00766 * (1 + 50.01 * Z))
    mask                 = fMassive>= 0.7- 0.5 * Z
    exp[mask]            = -5.07 * (1 + 2.16 * Z[mask]) - 0.49 * (1+8.96 * Z[mask])
    exp[~mask]           = -5.07 * (1 + 2.16 * Z[~mask]) + 0.49 * (1+8.96 * Z[~mask]) * np.cos(-5.15 * fMassive[~mask]- 6.16 * (1+ 2048 * Z[~mask]))
    lt0[fMassive<0.209]  = 0.49 * (1-23.37 * Z[fMassive< 0.209])+2.64 * (fMassive[fMassive<0.209] - 0.209)**2
    lt0[fMassive>=0.209] = 0.47 * (1-17.15 * Z[fMassive>=0.209])+0.06 * (1 - np.exp(-7.5 * (fMassive[fMassive>=0.209]-0.209)))
    lt0[lt0<0.477]       = 0.477
    t_0                  = 10 ** (lt0+6)
    N0                   = 10 ** lN0
    Nion = np.zeros_like(t)
    t_0u = t_f
    t_1l = t_i
    for i in range(s0.shape[0]):
        t_0l = np.maximum(t_i,np.minimum(t-t_0[i],t_f))
        t_1u = np.minimum(t_f,np.maximum(t-t_0[i],t_i))
        Nion += s0[i] * N0[i] * ((t_0u - t_0l) - 1/(exp[i]+1) * ((t-t_1u)**(exp[i]+1)-(t-t_1l)**(exp[i]+1))*t_0[i] **(-exp[i]))
    return Nion    

def get_Mag_from_L(L):
    UVperFreq = 10.**L * pc.lambda_UV_cm * pc.lambda_UV_cm * 1e8 / pc.clight_cm
    UVmag = -2.5 * np.log10(UVperFreq / (4.*np.pi*100.*pc.pc_cm*pc.pc_cm)) - 48.6
    return UVmag
             

def get_LUV_and_Nion(Gal):
    simParam = Gal.simParam
    if Gal.calc_lum:
        timeTmp = Ti.perf_counter()
        if Gal.IMF_type == 'salpeter':
            Gal.LUVHistory[Gal.j+1:] += LUV_salpeter(simParam.substepTimeList[Gal.j+1:] ,simParam.substepTimeList[Gal.j] ,simParam.substepTimeList[Gal.j+1] ,simParam.substepRedshiftList[Gal.j] ,Gal.metallicity, Gal.SFRHistory[Gal.j]/(Gal.dt*1e6))
            if Gal.calc_lum == True:
                Gal.NionHistory[Gal.j+1:] += Nion_salpeter(simParam.substepTimeList[Gal.j+1:], simParam.substepTimeList[Gal.j], simParam.substepTimeList[Gal.j+1], simParam.substepRedshiftList[Gal.j], Gal.metallicity, Gal.SFRHistory[Gal.j]/(Gal.dt*1e6))
                
        elif Gal.IMF_type == 'salpeter_cloud':
            Gal.LUVHistory[Gal.j+1:] += LUV_salpeter_cloud(simParam.substepTimeList[Gal.j+1:] ,simParam.substepTimeList[Gal.j] ,simParam.substepTimeList[Gal.j+1] ,Gal.maxStarMassVals, Gal.starGrid[Gal.j].sum(axis = 0)/(Gal.dt))
            if Gal.calc_lum == True:
                Gal.NionHistory[Gal.j+1:] += Nion_salpeter_cloud(simParam.substepTimeList[Gal.j+1:], simParam.substepTimeList[Gal.j], simParam.substepTimeList[Gal.j+1],Gal.maxStarMassVals, Gal.starGrid[Gal.j].sum(axis = 0)/(Gal.dt))
                
        elif Gal.IMF_type == 'evolving_cloud':
            mask = Gal.metalMask[Gal.j]
            fMassive = simParam.fmassiveList[Gal.j][mask]
            if len(fMassive):
                Gal.LUVHistory[Gal.j+1:] += LUV_evolving_cloud(simParam.substepTimeList[Gal.j+1:] ,simParam.substepTimeList[Gal.j] ,simParam.substepTimeList[Gal.j+1] ,Gal.maxStarMassVals, fMassive, Gal.starGrid[Gal.j][mask]/(Gal.dt))
                if Gal.calc_lum == True:
                    Gal.NionHistory[Gal.j+1:] += Nion_evolving_cloud(simParam.substepTimeList[Gal.j+1:], simParam.substepTimeList[Gal.j], simParam.substepTimeList[Gal.j+1],Gal.maxStarMassVals, fMassive, Gal.starGrid[Gal.j][mask]/(Gal.dt))
                    
        elif Gal.IMF_type == 'evolving':
            mask = Gal.metalMask[Gal.j]
            Gal.LUVHistory[Gal.j+1:] += LUV_evolving(simParam.substepTimeList[Gal.j+1:], simParam.substepTimeList[Gal.j], simParam.substepTimeList[Gal.j+1], simParam.substepRedshiftList[Gal.j], simParam.ZZ[mask], Gal.starGrid[Gal.j][mask]/(Gal.dt*1e6))
            if Gal.calc_lum == True:
                Gal.NionHistory[Gal.j+1:] += Nion_evolving(simParam.substepTimeList[Gal.j+1:], simParam.substepTimeList[Gal.j], simParam.substepTimeList[Gal.j+1], simParam.substepRedshiftList[Gal.j], simParam.ZZ[mask] ,Gal.starGrid[Gal.j][mask]/(Gal.dt*1e6))
                
        elif Gal.IMF_type == 'evolving2':
            mask = Gal.metalMask[Gal.j]
            fMassive = Gal.fmassiveHistory[Gal.j] * np.ones(sum(mask))
            Gal.LUVHistory[Gal.j+1:] += LUV_evolving_fmassive(simParam.substepTimeList[Gal.j+1:], simParam.substepTimeList[Gal.j], simParam.substepTimeList[Gal.j+1], fMassive, simParam.ZZ[mask], Gal.starGrid[Gal.j][mask]/(Gal.dt*1e6))
            if Gal.calc_lum == True:
                Gal.NionHistory[Gal.j+1:] += Nion_evolving_fmassive(simParam.substepTimeList[Gal.j+1:], simParam.substepTimeList[Gal.j], simParam.substepTimeList[Gal.j+1], fMassive, simParam.ZZ[mask], Gal.starGrid[Gal.j][mask]/(Gal.dt*1e6))  
                
        if (Gal.LUVHistory[Gal.j]>0):
            Gal.MUVHistory[Gal.j] = get_Mag_from_L(np.log10(Gal.LUVHistory[Gal.j]))
        Gal.timeLum += Ti.perf_counter()-timeTmp
