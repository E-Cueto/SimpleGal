import numpy as np
from numba import jit, njit, vectorize, prange

@njit()
def IMF_Salpeter(M,z = 0,Z=0):
    return 0.35/(0.1**-0.35-100**-0.35)*M**-2.35
@njit()
def IMF_Salpeter_cloud(M,Mmax):
    #if np.issalar(Mmax):
    #    return 0.35 / (0.1 ** -0.35 - Mmax ** -0.35) * M ** -2.35
    #else:
    imf = np.zeros((Mmax.shape[0],M.shape[0]))
    M0 =  M ** -2.35 
    for i in prange(Mmax.shape[0]):
        imf[i] = 0.35 / (0.1 ** -0.35 - Mmax[i] ** -0.35) * M0
    for i in prange(Mmax.shape[0]):
        imf[i][M>Mmax[i]] = 0
    return imf
@njit()
def fmassive(z,Z):
    f = np.zeros(Z.shape[0])
    f[Z==0] = 1
    X = 1 + np.log10(Z[Z>0]/0.0134)
    f[Z>0] = 1.07 * (1-2**X) + 0.04 * z * 2.67 ** X
    f[f>1]=1
    f[f<0]=0
    return f
@njit()
def fmassive_single(z,Z):
    if Z == 0:
        return 1
    X = 1 + np.log10(Z/0.0134)
    f = 1.07 * (1-2**X) + 0.04 * z * 2.67 ** X
    if f>1:
        f=1
    if f<0:
        f=0
    return f
@njit(parallel = True)
def get_fmassive_from_Mc(M,Mup = 100,Mlow = 0.1):
    return (Mup-M)*M**-1.35 / ((Mup-M)*M**-1.35-(M**-0.35-Mlow**-0.35)/(0.35))

@njit()
def get_fmassive_from_TBD_time_and_gasdensity(z,gasDensity,OB0 = 0.048,OM0 = 0.307115):
    factor           = np.sqrt((1+z)**3)
    fracBaryon       = OB0/OM0
    thisTBD          = gasDensity * factor
    thisTBDmax       = fracBaryon * 96.2341
    thisTBDthreshold = fracBaryon * (9+1)**1.5
    slope            = 10
    maximum          = 0.999

    return min(max(slope * (thisTBD-thisTBDthreshold) / (thisTBDmax - thisTBDthreshold),0),maximum)
@njit()
def get_fmassive_from_TBD_time_and_gasdensity_multiple(z,gasDensity,OB0 = 0.048,OM0 = 0.307115):
    factor           = np.sqrt((1+z)**3)
    fracBaryon       = OB0/OM0
    thisTBD          = gasDensity * factor
    thisTBDmax       = fracBaryon * 96.3241
    thisTBDthreshold = fracBaryon * (6+1)**1.5
    slope            = 7
    maximum          = 0.999
    f = slope * (thisTBD-thisTBDthreshold) / (thisTBDmax - thisTBDthreshold)
    f[f>maximum] = maximum
    f[f<0.]       = 0.
    return f
@njit()
def get_cutoff_mass_single(f_massive,MC):
    return MC[int(1000 * max(f_massive,0.001) -1)]
@njit(parallel = True, fastmath = True)
def get_cutoff_mass(f_massive, MC):
    f = f_massive
    f[f<0.001]=0.001
    index = 1000 * f -1
    M = np.zeros(f.shape[0])
    for i in prange(f.shape[0]):
        M[i] = MC[int(index[i])] 
    return M

@njit()
def IMF_evolving(M,z,Z,MC):
    f_massive = fmassive(z,Z)
    MC = get_cutoff_mass(f_massive,MC)
    norm = 1/(MC**(-1.35)*(100-MC)+ (0.1**(-0.35)-MC**-0.35)/(0.35))
    imf = np.zeros((len(M),len(Z)))
    for i in range(len(Z)):
        if MC[i]>M[0]:
            imf[M<MC[i],i] = M[M<MC[i]]**-2.35 * norm[i]
        imf[M>=MC[i],i] = 1/M[M>=MC[i]] * MC[i]**(-1.35) * norm[i]
    return imf
@njit(parallel = True)
def IMF_evolving_cloud(M,f_massive,Mmax,MC):
    MC = get_cutoff_mass(f_massive,MC)
    imf = np.zeros((len(f_massive),len(M),len(Mmax)))
    for i in prange(len(Mmax)):
        imfTmp = np.zeros((len(f_massive),len(M)))
        norm = 1/(MC**(-1.35)*(Mmax[i]-MC)+ (0.1**(-0.35)-MC**-0.35)/(0.35))
        for j in prange(len(f_massive)):
            if Mmax[i] <= MC[j]:
                imfTmp[j] = IMF_Salpeter_cloud(M,np.array([Mmax[i]]))[0]
            else:
                if MC[j]>M[0]:
                    imfTmp[j,M<MC[j]] = M[M<MC[j]]**-2.35 * norm[j]
                imfTmp[j,(M>=MC[j])*(M<Mmax[i])] = 1/M[(M>=MC[j])*(M<Mmax[i])] * MC[j]**(-1.35) * norm[j]
        imf[:,:,i] = imfTmp
    return imf

@njit()
def IMF_evolving_fmassive(M, z, gasDensity ,MC ,OB0 = 0.048,OM0 = 0.307115):
    f_massive = get_fmassive_from_TBD_time_and_gasdensity(z,gasDensity,OB0,OM0)
    MC = get_cutoff_mass_single(f_massive,MC)
    norm = 1/(MC**(-1.35)*(100-MC)+ (0.1**(-0.35)-MC**-0.35)/(0.35))
    imf = np.zeros((len(M)))
    if MC>M[0]:
        imf[M<MC] = M[M<MC]**-2.35 * norm
    imf[M>=MC] = 1/M[M>=MC] * MC**(-1.35) * norm
    return imf

@njit()
def get_fmassive_from_cloud_mass(M,logMmin,logMmax,fmin = 0,fmax = 1):
    logM = np.log10(M)
    if logM<logMmin: 
        return fmin
    elif logM>logMmax:
        return fmax
    else:
        return (logM-logMmin)/(logMmax-logMmin) * (fmax - fmin) + fmin
@njit()
def get_fmassive_from_cloud_mass_multiple(M,logMmin,logMmax,fmin = 0,fmax = 1):
    logM = np.log10(M)
    f = np.zeros_like(M)
    f[logM<logMmin] = fmin
    f[logM>logMmax] = fmax
    f[(logM>=logMmin)*(logM<=logMmax)] = (logM[(logM>=logMmin)*(logM<=logMmax)]-logMmin)/(logMmax-logMmin) * (fmax - fmin) + fmin
    return f
@njit()
def get_IMF_from_fmassive(M,f,MC):
    cutoffMass = get_cutoff_mass(f,MC)
    norm = 1/(cutoffMass**(-1.35)*(100-cutoffMass)+ (0.1**(-0.35)-cutoffMass**-0.35)/(0.35))
    imf = np.zeros((len(M),len(f)))
    for i in range(len(f)):
        if cutoffMass[i]>M[0]:
            imf[M<cutoffMass[i],i] = M[M<cutoffMass[i]]**-2.35 * norm[i]
        imf[M>=cutoffMass[i],i] = 1/M[M>=cutoffMass[i]] * cutoffMass[i]**(-1.35) * norm[i]
    return imf

def make_Xi_of_Z_and_fmassive_table(ZZ,fmassive,path):
    Z = ZZ[:,None]
    f = fmassive[None,:]
    Xi = 10 ** ((3.4e-9 * (1 + 0.62 * Z) + 53.66 * (1-0.32 * Z) * (f + 0.051 * (1 + 227.6 * Z)) ** (0.00766 * (1 + 50.01 * Z)))-6)
    np.savetxt(path + 'Xi_table.dat',Xi)