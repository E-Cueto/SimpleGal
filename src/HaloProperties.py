import numpy as np
import phys_const as pc
from scipy.integrate import simpson as simps
import astropy.units as u
import astropy.constants as ac


### Concentration parameter calculation:
def concentration_parameter(M, z, OL0 = 0.692885, OM0 = 0.307115,h0 = 0.67770):
    x = (OL0/OM0)**(1/3)/(0.5+z) #0.5 + z fits so much better to the values they get in the paper than doing 1+z. very very unclear why this is.
    return calc_B0(x,OL0,OM0)*calc_C(calc_sigma_prime(x, M, z,OL0,OM0,h0))

def calc_sigma_prime(x, M, z, OL0 = 0.692885, OM0 = 0.307115, h0 = 0.67770):
    return calc_B1(x,OL0,OM0) * calc_sigma(M, x,OL0,OM0,h0)

def calc_C(sigma_prime):
    A = 2.881; b = 1.257; c = 1.022; d = 0.060 # Based on fits with OL0 = 0.73, OM0 = 0.27, OB0 = 0.0469, ns = 0.95, h = 0.70, sigma8 = 0.82. WMAP-5 parameters. From Prada et al. 2012. 
    return A * ((sigma_prime / b) ** c + 1) * np.exp( d / (sigma_prime ** 2))

def calc_B0(x, OL0 = 0.692885, OM0 = 0.307115):
    return calc_cmin(x) / calc_cmin((OL0/OM0)**(1/3)) # 1.312 = (OL0/OM0)**(1/3) 
    
def calc_B1(x, OL0 = 0.692885, OM0 = 0.307115):
    return calc_inv_sigmamin(x) / calc_inv_sigmamin((OL0/OM0)**(1/3)) # 1.312 = (OL0/OM0)**(1/3) 
    
def calc_cmin(x):
    c0 = 3.681; c1 = 5.033; alpha = 6.948; x0 = 0.424 # Based on fits with OL0 = 0.73, OM0 = 0.27, OB0 = 0.0469, ns = 0.95, h = 0.70, sigma8 = 0.82. WMAP-5 parameters. From Prada et al. 2012. 
    return c0 + (c1 - c0) * (np.arctan(alpha * (x - x0)) / np.pi + 0.5)

def calc_inv_sigmamin(x):
    inv_sigma0 = 1.047; inv_sigma1 = 1.646; beta = 7.386; x1 = 0.526 # Based on fits with OL0 = 0.73, OM0 = 0.27, OB0 = 0.0469, ns = 0.95, h = 0.70, sigma8 = 0.82. WMAP-5 parameters. From Prada et al. 2012. 
    return inv_sigma0 + (inv_sigma1 - inv_sigma0) * (np.arctan(beta * (x - x1)) / np.pi + 0.5)

def calc_sigma(M,x, OL0 = 0.692885, OM0 = 0.307115, h0 = 0.67770):
    y = calc_y(M,h0)
    return calc_D(x,OL0,OM0) * (16.9 * (y ** 0.41)) / (1 + 1.102 * (y ** 0.2) + 6.22 * (y ** 0.333)) # Based on WMAP-5 parameters. From Klypin et al 2010.
    
def calc_D(x, OL0 = 0.692885, OM0 = 0.307115):
    if np.isscalar(x):
        xx = np.linspace(0,x,10000)
        return 2.5 * (OM0 / OL0 ) ** (1/3) * np.sqrt(1 + x ** 3) / (x ** 1.5) * simps(xx ** (3/2) / (1+xx**3)**(3/2),x=xx)
    else:
        D = np.zeros_like(x)
        for i,X in enumerate(x):
            xx = np.linspace(0,X,10000)
            D[i] = 2.5 * (OM0 / OL0 ) ** (1/3) * np.sqrt(1 + X ** 3) / (X ** 1.5) * simps(xx ** (3/2) / (1+xx**3)**(3/2),x=xx)
        return D
            
def calc_y(M, h0 = 0.67770):
    return 1e12 / (M*h0) 

### 

### Calculate Halo Mass with correa 2015 model https://arxiv.org/pdf/1501.04382
### This one would be good to use, but doesn't work well currently
def halo_mass_history(M0,z):
    c = 6.67 * ((M0)/(2e12*invh0))**(-0.092) #concentration-mass relation
    A = 900
    z_2 = (200/A * c**3 * calc_Y0(1) / (OM0 * calc_Y0(c)) - OL0/OM0)**(1/3)-1 #Formation redshift
    beta = -3/(1+z_2)
    alpha = (np.log(calc_Y0(1)/calc_Y0(c))-beta * z_2)/np.log(1+z_2)
    print('c=  ',c)
    print('z_2=',z_2)
    print('a=  ',alpha)
    print('b=  ',beta)
    Mz = M0 * (1+z)**alpha * np.exp(beta*z)
    return Mz
def calc_Y0(u):
    return np.log(1+u)-u/(1+u)

def dMdt_ferrera(M,z):
    alpha = 0.25
    beta = -0.75
    return 102.2 * h0 * (-alpha-beta*(1+z))*np.sqrt(OM0*(1+z)**3+OL0)*M/1e12
## Halo mass from ferrera 2024b https://arxiv.org/pdf/2405.20370
def calc_Mhalo(M0,z,beta = -0.75):
    alpha = 0.25
    M = M0 *  (1+z)**alpha * np.exp(beta*z)
    return M
    
### Calculate virial radius in cm
def calc_Rvir(Mhalo, OM0 = 0.307115): 
    #return ((2*ac.G*Mhalo*u.Msun/(Deltac*P18.H(redshift)**2))**(1/3)).to(u.cm).value
    return 0.0169 * (Mhalo/OM0)**(1/3)
def calc_Rvir2(Mhalo,z, OL0 = 0.692885, OM0 = 0.307115,h0 = 0.67770):
    H0 = Hubble_H(z,OL0,OM0,h0)
    return (Mhalo * pc.Msun_g * pc.G / (100 * (H0)**2))**(1/3)
    
def calc_vc(Mhalo,Rvir,redshift):
    #return np.sqrt( ac.G * Mhalo * u.Msun / (calc_Rvir(Mhalo,redshift)*u.cm * (1/redshift +1))).to(u.km/u.s).value
    return (np.sqrt(ac.G * Mhalo * u.Msun / (Rvir * u.kpc ))).to(u.km/u.s).value
### Hubble parameter at redshift z
def Hubble_H(z, OL0 = 0.692885, OM0 = 0.307115,h0 = 0.67770):
    H0 = h0 * 1e7/pc.Mpc_cm
    return H0 * (OM0 * (1+z)**3 + OL0)**0.5


### Radiative frredback from reionisation limits the amount of gas a galaxy can hold
def calc_maxMgas_radFeedback(Mhalo, redshift, OM0, OB0, zion = 8, zreion = 7):
    MgasMax = calc_fg_radFeedback(Mhalo, redshift, OM0, zion, zreion) * OB0 / OM0 * Mhalo
    return MgasMax

def calc_fg_radFeedback(Mhalo,redshift, OM0, zion = 8, zreion = 7):
    T = 1e4; mu = 0.59
    aheat = 0; alpha = 6
    #zion = 8; zreion = 7
    fa = calc_fa_radFeedback(redshift, zion, zreion, aheat, alpha)
    Mjeans = calc_jeans_mass_radFeedback(0., T, mu, OM0)
    Mcr = 8 * Mjeans * fa ** 1.5
    tmp = Mcr / Mhalo
    fg = np.minimum(( 1 + 0.26 * tmp) ** -3, 1)
    return fg

def calc_jeans_mass_radFeedback(z,T,mu,OM0):
    fact = (T * 1e-4) / (mu * (1+z))
    MJ = 0.125 * 2.5e11 * fact ** 1.5 / np.sqrt(OM0)
    return MJ

def calc_fa_radFeedback(z, zion, zreion, aheat, alpha):
    a      = 1 / (1 + z)
    areion = 1 / (1 + zreion)
    aion   = 1 / (1 + zion)
    fa = np.ones_like(z)
    maskh = a>=areion
    fa[maskh] = calc_fheat(a[maskh], aion,     aion, aheat, alpha) + calc_fion(a[maskh], areion  , aion) + calc_fcool(a[maskh], areion)
    maski = (a<areion)*(a>=aion)
    fa[maski] = calc_fheat(a[maski], aion,     aion, aheat, alpha) + calc_fion(a[maski], a[maski], aion)
    maskl = a<aion
    fa[maskl] = calc_fheat(a[maskl], a[maskl], aion, aheat, alpha)
    return fa

def calc_fheat(a, ap, aion, aheat, alpha):
    t1 = 1 / (alpha + 2) - np.sqrt(ap    / a) * (2 / (2 * alpha + 5))
    t2 = 1 / (alpha + 2) - np.sqrt(aheat / a) * (2 / (2 * alpha + 5))
    fheat = 3 / a * aion ** -alpha * (ap ** (alpha + 2) * t1 - aheat ** (alpha + 2) * t2)
    return fheat

def calc_fion(a, ap, aion):
    t1 = 0.1 * ap   * ap   * (5 - 4 * np.sqrt(ap   / a))
    t2 = 0.1 * aion * aion * (5 - 4 * np.sqrt(aion / a))
    fion = 3 / a * (t1 - t2)
    return fion

def calc_fcool(a, areion):
    fcool = areion * (1 - areion / a * (3 - 2 * np.sqrt(areion / a)))
    return fcool
