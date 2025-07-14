import numpy as np


### Statistical tools
import scipy.stats as stats
import scipy.interpolate as si
from scipy.integrate import simpson as simps
### Gotta Go FAST
from numba import jit, njit, vectorize, prange
import time as Ti
### System manipulation
import sys
sys.path.append('/home/vdm981_alumni_ku_dk/modi_mount/Astraeus/astraeus/dynamic_IMF/yield_table_FINAL/output_data_original')
sys.path.append('/home/vdm981_alumni_ku_dk/modi_mount/ToyModel/ToyModel/')

from utils import *
from StarFormation import *
from fescClouds import *
from fescISM import *
from HaloProperties import *
from SNfeedback import *
from IMF import *
from Luminosity import *
from simparam import *





class Galaxy:
    def __init__(self,simParam,IMF_type = 'salpeter',ISM_type = 'NFW',verbose = False,print_times=True,
                 cloud_seed = None,simple_clouds=False, calc_lum = True, calc_fesc=False, SNfeedback = 'delayed',
                 cloudMassScaleFactor = 1e8, constant_cloud_density = True, SFE_turnover = True):
        ### Timers
        self.timeTot = Ti.perf_counter()
        self.timefej  = 0; self.timeDraw   = 0; self.timeForm = 0; self.timeKill   = 0;
        self.timeLum  = 0; self.timefescMC = 0; self.timefesc = 0; self.timeTest   = 0;
        
        self.Mgas = simParam.Mgas0
        self.Mmetal = 0
        self.Mdust  = 0
        self.Mhalo = simParam.haloMasses[simParam.startSnap]
        self.snap = simParam.startSnap
        self.j = simParam.startStep
        
        self.IMF_type = IMF_type
        self.ISM_type = ISM_type
        self.Clump_type = simParam.Clump_type #Not used, but I haven't checked that it isn't still in the code somewhere so it is not commented out
        self.SNfeedback = SNfeedback

        #Boolean input parameters
        self.constantCloudDensity = constant_cloud_density
        self.calc_lum = calc_lum
        self.calc_fesc = calc_fesc # Do not turn this on, i do not think it will work.
        self.simple_clouds = simple_clouds
        self.SFE_turnover  = SFE_turnover
        self.print_times = print_times
        self.verbose = verbose
        
        #Arrays for storing info 
        self.gasMassHistory       = np.zeros(simParam.NtimestepsTot); self.gasMassIniHistory   = np.zeros(simParam.NtimestepsTot)
        self.stellarMassHistory   = np.zeros(simParam.NtimestepsTot); self.SFRHistory          = np.zeros(simParam.NtimestepsTot)
        self.haloMassHistory      = np.zeros(simParam.NtimestepsTot); self.clumpNumberHistory  = np.zeros(simParam.NtimestepsTot)
        self.clumpMassHistory     = np.zeros(simParam.NtimestepsTot); self.meanClumpHistory    = np.zeros(simParam.NtimestepsTot)
        self.medianClumpHistory   = np.zeros(simParam.NtimestepsTot); self.maxClumpHistory     = np.zeros(simParam.NtimestepsTot)
        self.metalMassHistory     = np.zeros(simParam.NtimestepsTot); self.metallicityHistory  = np.zeros(simParam.NtimestepsTot)
        self.LUVHistory           = np.zeros(simParam.NtimestepsTot); self.MUVHistory          = np.zeros(simParam.NtimestepsTot)
        self.NionHistory          = np.zeros(simParam.NtimestepsTot); self.smoothStarFormation = np.zeros(simParam.NtimestepsTot)
        self.fmassiveIndexHistory = np.zeros(simParam.NtimestepsTot); self.fmassiveHistory     = np.zeros(simParam.NtimestepsTot)
        self.fescHistory          = np.zeros(simParam.NtimestepsTot); self.riHistory           = np.zeros(simParam.NtimestepsTot)
        self.dustMassHistory      = np.zeros(simParam.NtimestepsTot); self.fescMCHistory       = np.zeros(simParam.NtimestepsTot)
        self.fescDustHistory      = np.zeros(simParam.NtimestepsTot); self.fescISMHistory      = np.zeros(simParam.NtimestepsTot)
        self.fejHistory           = np.zeros(simParam.NtimestepsTot); self.feffHistory         = np.zeros(simParam.NtimestepsTot)
        self.SNenergyHistory      = np.zeros(simParam.NtimestepsTot); self.MgasEjectHistory    = np.zeros(simParam.NtimestepsTot)
        self.metalMask            = np.zeros((simParam.NtimestepsTot,len(simParam.ZZ)),bool)
        self.cloudDensityHistory  = np.zeros(simParam.NtimestepsTot);
        self.MsmoothHistory       = np.zeros(simParam.NtimestepsTot);
        self.cloudGasHistory      = np.zeros(simParam.NtimestepsTot);
        self.cloudDustHistory     = np.zeros(simParam.NtimestepsTot);
        self.cloudMetalHistory    = np.zeros(simParam.NtimestepsTot);
        
        def finv(N):
            return (N * (simParam.maxMcl**(simParam.ClMFslope+1)-simParam.minMcl**(simParam.ClMFslope+1))+simParam.minMcl**(simParam.ClMFslope+1))**(1/(simParam.ClMFslope+1))
        self.cloudSeed = cloud_seed
        self.clumpDrawList = numbers_trans(finv,1_000_000,cloud_seed)
        self.clumpListIndex = 0
        self.clumpListIndexInit = 0
        self.allClumps = np.array([[0.,0,2, 0,simParam.startStep,0, 0,0,0, 0,0,0, 0,0]])
        self.starsFormed = 0
        self.cloudMassScaleFactor = cloudMassScaleFactor
        if self.cloudMassScaleFactor>0:
            self.cloudMassLimit = np.maximum(calc_max_cloud_mass(simParam.haloMassSubstep,simParam.substepRedshiftList,self.cloudMassScaleFactor),2 * simParam.minMcl)
        else:
            self.cloudMassLimit = np.zeros(simParam.NtimestepsTot) + simParam.maxMcl
        
        if self.constantCloudDensity:
            self.cloudDensityHistory[:] += simParam.clumpMeanDensityPercm3
        
        if simParam.smoothMassThreshold != 1:
            self.MsmoothHistory += calc_Mgas_for_threshold(simParam.smoothMassThreshold, simParam.minMcl, simParam.maxMcl, simParam.ClMFslope)
        self.MstarMax                 = 0
        self.fej                      = 0
        self.feff                     = 0
        self.MstarSnap                = 0
        self.dt                       = 0
        self.metallicity              = 0
        self.dustToGasRatio           = 0
        self.simParam = simParam
        self.maxStarMassBins = 10
        self.maxStarMassVals = np.arange(10,101,10)
        self.fs           = simParam.fs
        self.fw           = simParam.fw
        self.returns      = simParam.returns[:,:500].copy()
        self.yields       = simParam.yields[:,:500].copy()
        self.dustYields   = simParam.dustYields[:,:500].copy()
        self.fmassiveClList  = np.loadtxt('input_data/fmassive_cloud_list.dat')
        self.hasSmoothGrid = True
        self.cloudLimit   = 25_000
        iIMF = 0 if self.IMF_type == 'salpeter' else (1 if (self.IMF_type == 'evolving')+(self.IMF_type == 'evolving2') else (2 if self.IMF_type == 'salpeter_cloud' else (3 if self.IMF_type == 'evolving_cloud' else 4)))
        if self.constantCloudDensity:
            if simParam.clumpMeanDensityPercm3 in [1e2,1e3,1e4,1e5]:
                iN = int(np.log10(simParam.clumpMeanDensityPercm3)-2)
                self.smoothSFgrid = simParam.smoothSFGrids[iIMF][iN]
            else:
                self.smoothSFgrid = np.array(simParam.smoothSFGrids[iIMF])
        else:
            self.smoothSFgrid = np.array(simParam.smoothSFGrids[iIMF])
        
        ### Define the correct SFE function
        if self.SFE_turnover:
            self.SFE_function = calc_SFE
        else:
            self.SFE_function = calc_SFE_no_turn
            
        if IMF_type[-5:] == 'cloud':
            self.MclList         = np.loadtxt('input_data/cloudMassList.dat')
            self.MclListMin      = np.log10(self.MclList[0])
            self.MclListMax      = np.log10(self.MclList[-1])
            self.MclListBins     = len(self.MclList)
            
        if IMF_type == 'salpeter':
            self.starGrid     = np.zeros((simParam.NtimestepsTot,len(simParam.ZZ)))
            self.SNenergy     = simParam.SNenergy[0]
            self.IMF_list     = simParam.IMF_list[0]
            self.returns      = self.IMF_list * self.returns
            self.yields       = self.IMF_list * self.yields
            self.dustYields   = self.IMF_list * self.dustYields
        
        elif IMF_type == 'salpeter_cloud':
            self.maxStarMassGrid = simParam.cloudMstarLimit[0]
            self.SFEgrid         = simParam.cloudSFE[0]
            self.MclListFactor   = self.MclListBins / (self.MclListMax - self.MclListMin)
            self.starGrid        = np.zeros((simParam.NtimestepsTot,len(simParam.ZZ),self.maxStarMassBins))
            self.IMF_list        = (IMF_Salpeter_cloud(simParam.MM[:500],self.maxStarMassVals) * (simParam.MM[1]-simParam.MM[0])).T
            self.SNenergy        = simParam.SNenergy[2]  
            if simple_clouds:
                self.Nstar = get_numPerMass_SalpeterIMF(0.1,100.,2.35)
                self.smoothMstarFraction = np.zeros((1,self.maxStarMassBins))
                self.smoothMaxCloudMass  = np.zeros((1,self.maxStarMassBins))
                self.smoothMeanSFEVals   = np.zeros((1,self.maxStarMassBins))
                self.smoothMstarFractionExists = np.zeros(1,dtype = 'bool')
            if (self.hasSmoothGrid) * (self.constantCloudDensity) * (simParam.clumpMeanDensityPercm3 in [1e2,1e3,1e4,1e5]):
                self.smoothSFgrid = np.reshape(self.smoothSFgrid,(4,self.simParam.ZZ.shape[0], self.maxStarMassVals.shape[0]))
        
        elif IMF_type == 'evolving_cloud':
            self.maxStarMassGrid = simParam.cloudMstarLimit[1]
            self.SFEgrid         = simParam.cloudSFE[1]
            self.MclListFactor   = self.MclListBins / (self.MclListMax - self.MclListMin)
            self.McutoffClList   = np.loadtxt('input_data/Mc_cloud_list.dat')
            self.fmassiveIndexes = np.array([[find_nearest(simParam.fmassiveList[i,j],self.fmassiveClList) for j in range(200)] for i in range(simParam.NtimestepsTot)])
            self.starGrid     = np.zeros((simParam.NtimestepsTot,len(simParam.ZZ),self.maxStarMassBins))
            self.IMF_list     = IMF_evolving_cloud(simParam.MM[:500],self.fmassiveClList,self.maxStarMassVals,simParam.cutoffMass) * (simParam.MM[1]-simParam.MM[0])
            self.SNenergy     = simParam.SNenergy[3]
            if simple_clouds:
                self.Nstar = get_numPerMass_EvolvingIMF(0.1,100.,self.McutoffClList,self.fmassiveClList,2.35)
                self.smoothMstarFraction = np.zeros((self.fmassiveClList.shape[0],self.maxStarMassBins))
                self.smoothMaxCloudMass  = np.zeros((self.fmassiveClList.shape[0],self.maxStarMassBins))
                self.smoothMeanSFEVals   = np.zeros((self.fmassiveClList.shape[0],self.maxStarMassBins))
                self.smoothMstarFractionExists = np.zeros(self.fmassiveClList.shape[0],dtype = 'bool')
            if (self.hasSmoothGrid) * (self.constantCloudDensity) * (simParam.clumpMeanDensityPercm3 in [1e2,1e3,1e4,1e5]):
                self.smoothSFgrid = np.reshape(self.smoothSFgrid,(4,self.simParam.ZZ.shape[0], self.fmassiveClList.shape[0], self.maxStarMassVals.shape[0]))
        
        elif IMF_type == 'evolving':
            self.starGrid     = np.zeros((simParam.NtimestepsTot,len(simParam.ZZ)))
            self.SNenergy     = simParam.SNenergy[1]
            self.IMF_list     = simParam.IMF_list[1]
            if (self.hasSmoothGrid) * (self.constantCloudDensity) * (simParam.clumpMeanDensityPercm3 in [1e2,1e3,1e4,1e5]):
                self.smoothSFgrid = np.reshape(self.smoothSFgrid,(self.fmassiveClList.shape[0],self.simParam.ZZ.shape[0]))
        
        elif IMF_type == 'evolving2':
            self.starGrid     = np.zeros((simParam.NtimestepsTot,len(simParam.ZZ)))
            self.SNenergy     = simParam.SNenergy[1]
            self.IMF_list     = np.zeros((simParam.NtimestepsTot,500))
            if (self.hasSmoothGrid) * (self.constantCloudDensity) * (simParam.clumpMeanDensityPercm3 in [1e2,1e3,1e4,1e5]):
                self.smoothSFgrid = np.reshape(self.smoothSFgrid,(self.fmassiveClList.shape[0],self.simParam.ZZ.shape[0]))
                    
        else:
            print(f'{IMF_type} IS NOT A VALID IMF TYPE.')
        self.timeTot = Ti.perf_counter()- self.timeTot
    def reset_Galaxy(self):
        self.timeTot = Ti.perf_counter()
        self.timefej = 0; self.timeDraw   = 0; self.timeForm = 0; self.timeKill = 0; 
        self.timeLum = 0; self.timefescMC = 0; self.timefesc = 0; self.timeTest = 0;
        
        self.Mgas = self.simParam.Mgas0
        self.Mmetal = 0; self.metallicity    = 0
        self.Mdust  = 0; self.dustToGasRatio = 0
        self.Mhalo = self.simParam.haloMasses[self.simParam.startSnap]
        self.snap = self.simParam.startSnap
        self.j = self.simParam.startStep
        self.allClumps = np.array([[0.,0,2, 0,self.simParam.startStep,0, 0,0,0, 0,0,0, 0,0]])
        self.starGrid = np.zeros_like(self.starGrid)

        if self.cloudSeed is not None:
            self.clumpListIndex = 0
        else:
            if (self.clumpListIndex < self.clumpListIndexInit) + (self.clumpNumberHistory.sum() > 1.e6):
                def finv(N):
                    return (N * (self.simParam.maxMcl**(self.simParam.ClMFslope+1)-self.simParam.minMcl**(self.simParam.ClMFslope+1))+self.simParam.minMcl**(self.simParam.ClMFslope+1))**(1/(self.simParam.ClMFslope+1))
                self.clumpDrawList = numbers_trans(finv,1_000_000)
                self.clumpListIndex = 0
                self.clumpListIndexInit = 0
            else:
                self.clumpListIndexInit = self.clumpListIndex

        if not self.constantCloudDensity:
            self.cloudDensityHistory[:]  = 0.
            self.MsmoothHistory[:]       = 0.
            
        self.smoothStarFormation[:] = 0.; self.gasMassHistory[:]     = 0.; self.gasMassIniHistory[:]  = 0.; self.stellarMassHistory[:]   = 0.
        self.SFRHistory[:]          = 0.; self.haloMassHistory[:]    = 0.; self.clumpNumberHistory[:] = 0.; self.clumpMassHistory[:]     = 0.
        self.metalMassHistory[:]    = 0.; self.metallicityHistory[:] = 0.; self.LUVHistory[:]         = 0.; self.MUVHistory[:]           = 0.
        self.NionHistory[:]         = 0.; self.MgasEjectHistory[:]   = 0.; self.fmassiveHistory[:]    = 0.; self.fescHistory[:]          = 0.
        self.riHistory[:]           = 0.; self.dustMassHistory[:]    = 0.; self.fescMCHistory[:]      = 0.; self.fescDustHistory[:]      = 0.
        self.fescISMHistory[:]      = 0.; self.fejHistory[:]         = 0.; self.feffHistory[:]        = 0.; self.SNenergyHistory[:]      = 0.
        self.meanClumpHistory[:]    = 0.; self.medianClumpHistory[:] = 0.; self.maxClumpHistory[:]    = 0.; self.fmassiveIndexHistory[:] = 0.
        self.metalMask[:,:] = False
        self.cloudGasHistory[:]     = 0.;
        self.cloudDustHistory[:]    = 0.;
        self.cloudMetalHistory[:]   = 0.;
        if ((self.IMF_type == 'evolving_cloud') + (self.IMF_type == 'salpeter_cloud'))*(self.simple_clouds):
            self.smoothMstarFraction[:,:] = 0.
            self.smoothMaxCloudMass[:,:]  = 0.
            self.smoothMeanSFEVals[:,:]   = 0.
            self.smoothMstarFractionExists[:] = False
        self.timeTot = Ti.perf_counter()- self.timeTot
        if not self.constantCloudDensity:
            self.cloudDensityHistory[:] = 0.;
            self.MsmoothHistory[:]      = 0.;
   
    def advance_snap(self,Nsnaps = 0):
        totTime = Ti.perf_counter()
        simParam = self.simParam
        if not Nsnaps:
            Nsnaps = simParam.Nsnaps
        for adv in range(Nsnaps-1):
            if self.verbose:
                print(f'Snap {self.snap}. z = {simParam.redshift_list[self.snap]}')
                print(f'Metallicity and Metal Mass: {self.metallicity:5.3e}, {self.Mmetal:5.3e}')
            ### Get Halo Mass In this Snapshot 
            f_massive = 0
            cutoffMass = 0
            ### Amount of gas to be accreted from IGM during snap
            gasAccretionInSnap = (simParam.haloMasses[self.snap+1]-simParam.haloMasses[self.snap])*simParam.OB0/simParam.OM0
            ### Loop over subtimesteps
            for i in range(simParam.timestepList[self.snap]):
                self.dt = simParam.dtList[self.j]
                self.Mhalo = simParam.haloMassSubstep[self.j]
                self.Mgas += gasAccretionInSnap * self.dt / (simParam.time_list[self.snap+1]-simParam.time_list[self.snap])#/simParam.timestepList[self.snap]
                if self.Mgas + self.cloudGasHistory[self.j-1] > simParam.maxMgas[self.j]:
                    self.Mgas += simParam.maxMgas[self.j] - (self.Mgas + self.cloudGasHistory[self.j-1])
                self.gasMassIniHistory[self.j] = self.Mgas
                
                if (self.IMF_type == 'evolving') + (self.IMF_type == 'evolving_cloud'):
                    f_massive = fmassive_single(simParam.substepRedshiftList[self.j],self.metallicity)
                    cutoffMass = get_cutoff_mass_single(f_massive,simParam.cutoffMass)
                    self.fmassiveHistory[self.j] = f_massive
                elif self.IMF_type == 'evolving2':
                    f_massive = get_fmassive_from_TBD_time_and_gasdensity(simParam.substepRedshiftList[self.j],self.Mgas/self.Mhalo)
                    cutoffMass = get_cutoff_mass_single(f_massive,simParam.cutoffMass)
                    self.fmassiveHistory[self.j] = f_massive
                    
                timeTmp = Ti.perf_counter()
                self.fej = calc_SNejection_efficiency(self)
                self.fejHistory[self.j] = self.fej
                fstar = self.dt/20 * self.fs * (0.03703737 / (1/(simParam.substepRedshiftList[self.j]+1))**(3/2))
                self.feff = np.min([fstar,self.fej])
                self.feffHistory[self.j] = self.feff
                self.MstarMax = self.feff * (self.Mgas) #Max star formation in step
                MgasIni   = self.Mgas
                MmetalIni = self.Mmetal
                MdustIni  = self.Mdust
                
                remainingGasFraction = calc_remaining_gas_fraction(self.fej, self.feff)
                if self.verbose:
                    print(f' fej, feff, fremain: {self.fej:6.4f}, {self.feff:6.4f}, {remainingGasFraction:6.4f}')
                self.timefej += Ti.perf_counter()-timeTmp
                #print(f'substep: {self.j}, dt = {self.dt}, accreted = {gasAccretionInSnap * self.dt / (simParam.time_list[self.snap+1]-simParam.time_list[self.snap]):5.3e}')
                get_cloud_density(self)
                
                self.advance_substep(i)
                if self.fej>0:
                    self.MgasEjectHistory[self.j] += MgasIni * (1 - remainingGasFraction)
                    self.Mgas = max(MgasIni * (remainingGasFraction - 1) + self.Mgas,0.)
                    self.Mdust = max(MdustIni * (remainingGasFraction - 1) + self.Mdust,0.)
                    self.Mmetal = max(MmetalIni * (remainingGasFraction - 1) + self.Mmetal,0.)
                    
                    
                else:
                    self.MgasEjectHistory[self.j] += MgasIni
                    self.Mgas = 0
                    self.Mmetal = 0
                    self.Mdust = 0
                    
                    
                self.gasMassHistory[self.j] = self.Mgas
                self.metalMassHistory[self.j] = self.Mmetal
                self.dustMassHistory[self.j] = self.Mdust
                if self.Mgas>0:
                    self.metallicityHistory[self.j] = self.Mmetal / self.Mgas

                self.j+=1
            self.snap += 1
        self.timeTot += Ti.perf_counter() - totTime 
        
        if(self.print_times):
            print(f'Total Time: {self.timeTot:5.3e} s ')
            print(f'fej   Time: {self.timefej:5.3e} s ')
            print(f'Draw  Time: {self.timeDraw:5.3e} s')
            print(f'Form  Time: {self.timeForm:5.3e} s')
            print(f'Kill  Time: {self.timeKill:5.3e} s')
            print(f'fescC Time: {self.timefescMC:5.3e} s')
            print(f'fesc  Time: {self.timefesc:5.3e} s')
            
            print(f'Lum   Time: {self.timeLum:5.3e} s \n')
    def get_Xi_from_fmassive(self,M=0):
        if self.IMF_type == 'salpeter':
            return 10 ** 46.706
        elif self.IMF_type == 'evolving':
            lZ = np.log10(max(self.metallicity,1.5e-4))
            return 10 **( 45.93 - (lZ+2) * 1.17 + 0.064 * (lZ + 3.46) * self.simParam.substepRedshiftList[self.j])
        elif self.IMF_type == 'evolving2':
            fMassive = get_fmassive_from_TBD_time_and_gasdensity(self.simParam.substepRedshiftList[self.j],self.Mgas/self.Mhalo,self.simParam.OB0,self.simParam.OM0)
            return 10 ** ((3.4e-9 * (1 + 0.62 * self.metallicity) + 53.66 * (1-0.32 * self.metallicity) * (fMassive + 0.051 * (1 + 227.6 * self.metallicity)) ** (0.00766 * (1 + 50.01 * self.metallicity)))-6)
    
    def get_cloud_lifetime(self,M):
        if self.IMF_type == 'salpeter':
            return 10 ** 0.6
        elif self.IMF_type == 'salpeter_cloud':
            lN1 = 0.77682 * np.log10(M-6.647) + 31.291
            t0  = -0.2761 * np.log10(M-9.827) + 6.49
            slope0 = 0.2878 * np.log10(M+8.66) - 0.11
            lifetime =  10 ** ((lN1 - 32.403 -1.25 * 7 - slope0 * t0) / (-1.25 - slope0) - 6)
            if lifetime.shape[0] == 1:
                return lifetime[0]
            else:
                return lifetime
        elif self.IMF_type == 'evolving_cloud':
            lifetime = (1013 * M**(-1.8) + 2.257 )
            if lifetime.shape[0] == 1:
                return lifetime[0]
            else:
                return lifetime
        elif self.IMF_type == 'evolving':
            return 10 ** 0.425
        elif self.IMF_type == 'evolving2':
            fMassive = get_fmassive_from_TBD_time_and_gasdensity(self.simParam.substepRedshiftList[self.j],self.Mgas/self.Mhalo,self.simParam.OB0,self.simParam.OM0)
            return 10**max(0.49 * (1-23.37 * self.metallicity)+2.64 * (fMassive - 0.209)**2 if (fMassive<0.209) else 0.47 * (1-17.15 * self.metallicity)+0.06 * (1 - np.exp(-7.5 * (fMassive-0.209))),0.477)
        
    def advance_substep(self,i):
        simParam = self.simParam
        ### Find length of time step
        ### variables for this subtimestep
        MstarMaxTmp              = self.MstarMax# * (i+1) / (self.time_list[self.snap+1]-self.time_list[self.snap]) #Max star formation in this snapshot up to this timestep
        MgasTmp                  = self.Mgas#/(self.timestepList[self.snap-self.startSnap]-i)
        self.starsFormed         = 0. #Stars to be formed in this subsnap
        gasReturnedFromClumps    = 0 #Gas to be returned from clumps that finish star formation in this substep
        gasReturnedFromStars     = 0 #Gas to be returned from dying stars
        metalsReturnedFromClumps = 0 #Metals returned from clumps that finish star formation now
        metalsReturnedFromStars  = 0 #Metals returned from dying stars
        dustReturnedFromClumps   = 0 #Dust returned from clumps that finish star formation now
        dustReturnedFromStars    = 0 #Dust returned from dying stars
        
        smooth = False
        if self.Mgas > 0:
            self.metallicity = self.Mmetal/self.Mgas
            self.dustToGasRatio = self.Mdust/self.Mgas
            
        else:
            self.metallicity = 0
            self.dustToGasRatio = 0
        if self.verbose:
            print(f'substep {self.j:2}. z: {simParam.substepRedshiftList[self.j]:4.2f}. ISM metallicity: {self.metallicity:5.3e}. dt = {self.dt}. MaxMstar = {self.MstarMax:5.3e}. Mgas = {self.Mgas:5.3e}')
        ### Draw clumps from clump mass distribution
        #t1 = Ti.monotonic()
        #print(f'substep {self.j}, MgasTmp = {MgasTmp}, Msmooth = {self.MsmoothHistory[self.j]}, Tsmooth = {simParam.smoothMassThreshold}')
        if MgasTmp < self.MsmoothHistory[self.j]:
            self.smoothStarFormation[self.j] = 0
            timeTmp = Ti.perf_counter()
            clumpMasses,clumpStellarMasses,clumpStarFormationTimes,clumpStarFormationStartTimes,clumpMaxMstar, MgasTmpNew = draw_clump_mass(self,MgasTmp,MstarMaxTmp)
            ### Put the clumps and relevant info about them into array and add to the total list of clumps
            
            clumps = np.zeros((len(clumpMasses),14))
            clumps[:,0] = clumpMasses #Clump Gas masses
            clumps[:,1] = clumpStellarMasses #Clump stellar masses to be formed
            clumps[:,2] = clumpStarFormationTimes #Time to form the stars
            clumps[:,3] = clumpStarFormationTimes + clumpStarFormationStartTimes #Time left to form the stars
            clumps[:,4] = self.j #Time of formation
            clumps[:,5] = self.metallicity #metallicity of the clumps
            clumps[:,6] = where(simParam.ZZ,self.metallicity) #Metallicity index for adding to grid
            #if self.Clump_type == 'uniform':
            #    clumps[:,7] = (clumpMasses / simParam.clumpMeanDensityMsunPerpc3 * 3/(4*np.pi))**(1/3) #Mean Clump Radius
            #else:
            #    clumps[:,7] = (clumpMasses / (simParam.clumpMeanDensityMsunPerpc3 * 4 * np.pi * (3 - np.arctan(3))))**(1/3)
            #clumps[:,8] = 0 #Ionized front radius
            if self.Mgas>0:
                clumps[:,9] = self.dustToGasRatio # Dust fraction
                clumps[:,10]= self.dustToGasRatio * clumpMasses # Dust Mass
            
            clumps[:,11] = np.maximum(self.get_cloud_lifetime(self.maxStarMassVals[clumpMaxMstar.astype(int)]),clumpStarFormationTimes) + clumpStarFormationStartTimes # time cloud has existed
            clumps[:,12] = clumpStarFormationStartTimes # Time at which star formation starts
            clumps[:,13] = clumpMaxMstar
            self.timeDraw += Ti.perf_counter()-timeTmp
        else:
            self.smoothStarFormation[self.j] = 1
            smooth = True
            if MstarMaxTmp>0:
                timeTmp = Ti.perf_counter()
                if self.simple_clouds:
                    clumps = get_smooth_clump_list_simple(self,MstarMaxTmp)
                else:
                    clumps = get_smooth_clump_list(self,MstarMaxTmp)
                MgasTmpNew = MgasTmp-np.sum(clumps[:,0])
                self.Mmetal = self.Mmetal - np.sum(clumps[:,0]) * self.metallicity
                self.Mdust  = self.Mdust  - np.sum(clumps[:,0] * clumps[:,9])
                self.timeDraw += Ti.perf_counter()-timeTmp
            else:
                clumps = np.zeros((1,14))
                MgasTmpNew = MgasTmp
        if np.any(clumps[:,0]>0):
            self.allClumps = np.concatenate((self.allClumps,clumps),axis = 0)
            
        #Form Stars
        timeTmp = Ti.perf_counter()
        if np.sum(self.allClumps[:,0])>0:
            gasReturnedFromClumps, metalsReturnedFromClumps,dustReturnedFromClumps = form_stars_from_clumps(self)
        self.timeForm += Ti.perf_counter()-timeTmp
        if self.calc_fesc * (self.calc_lum==True):
            timeTmp = Ti.perf_counter()
            Qclouds,Nclouds = calc_Q_and_N_Clouds(self) #Compute fesc
            self.timefescMC += Ti.perf_counter()-timeTmp
        else:
            self.allClumps[:,11]-=self.dt
        
        self.allClumps = self.allClumps[self.allClumps[:,11]>=0] #Remove clumps which are done forming stars
        # Kill stars
        timeTmp = Ti.perf_counter()
        gasReturnedFromStars, metalsReturnedFromStars,dustReturnedFromStars = kill_stars(self)
        self.timeKill += Ti.perf_counter()-timeTmp
        if self.verbose:
            print(f'Gas and metal returned from stars:  {gasReturnedFromStars :5.3e}, {metalsReturnedFromStars:5.3e}')
            print(f'Gas and metal returned from clumps: {gasReturnedFromClumps:5.3e}, {metalsReturnedFromClumps:5.3e}')
        #print(f' Clump Gas: {gasReturnedFromClumps:6.3e}, Star Gas: {gasReturnedFromStars:6.3e},  Clump Metal: {metalsReturnedFromClumps:6.3e}, Star Metal: {metalsReturnedFromStars:6.3e}, Stars Formed {starsFormed:6.3e}')
        #Update gas mass
        self.Mgas = MgasTmpNew + gasReturnedFromClumps + gasReturnedFromStars

        #Update Metal Mass
        self.Mmetal  += metalsReturnedFromStars + metalsReturnedFromClumps
        self.Mdust += dustReturnedFromStars + dustReturnedFromClumps
        
        self.stellarMassHistory[self.j:] += self.starsFormed - gasReturnedFromStars
        self.SFRHistory[self.j] = self.starsFormed
        self.haloMassHistory[self.j] = self.Mhalo
        self.clumpMassHistory[self.j] = np.sum(clumps[:,0])
        if len(self.allClumps)>0:
            self.cloudGasHistory[self.j]   = np.sum(self.allClumps[:,0]) #np.sum(self.allClumps[:,0] - self.allClumps[:,1] * (1 - (self.allClumps[:,2]) / (self.allClumps[:,3] - self.allClumps[:,12])))
            self.cloudDustHistory[self.j]  = np.sum(self.allClumps[:,0] * self.allClumps[:,9]) #np.sum(self.allClumps[:,9] * (self.allClumps[:,0] - self.allClumps[:,1] * (1 - (self.allClumps[:,2]) / (self.allClumps[:,3] - self.allClumps[:,12]))))
            self.cloudMetalHistory[self.j] = np.sum(self.allClumps[:,0] * self.allClumps[:,5]) #np.sum(self.allClumps[:,5] * (self.allClumps[:,0] - self.allClumps[:,1] * (1 - (self.allClumps[:,2]) / (self.allClumps[:,3] - self.allClumps[:,12]))))
        if smooth:
            self.clumpNumberHistory[self.j] = int(np.sum(clumps[:,0])/simParam.meanClumpMass)
            self.meanClumpHistory[self.j] = simParam.meanClumpMass
            self.medianClumpHistory[self.j] = simParam.meanClumpMass
            self.maxClumpHistory[self.j] = simParam.maxMcl
        else:
            self.clumpNumberHistory[self.j] = len(clumps)
            if len(clumps):
                self.meanClumpHistory[self.j] = np.mean(clumps[:,0])
                self.medianClumpHistory[self.j] = np.median(clumps[:,0])
                self.maxClumpHistory[self.j] = np.max(clumps[:,0])
            else:
                self.meanClumpHistory[self.j] = 0
                self.medianClumpHistory[self.j] = 0
                self.maxClumpHistory[self.j] = 0
        get_LUV_and_Nion(self)
        if self.calc_fesc * (self.calc_lum==True):
            timeTmp = Ti.perf_counter()
            NgalStars = self.NionHistory[self.j+1] -Nclouds
            Qtotal = (NgalStars  + Qclouds)
            if self.NionHistory[self.j+1]>0:
                fescMC = Qtotal/self.NionHistory[self.j+1]
            else:
                fescMC=0
            self.fescMCHistory[self.j+1] = fescMC
            self.fescHistory[self.j+1] = calc_fesc(self,Qtotal) * fescMC
            self.timefesc += Ti.perf_counter()-timeTmp
            
        if self.verbose:
            print(f'Stars Formed: {self.starsFormed}, {np.sum(self.starGrid[self.j])}. Clumps formed: {len(clumps)}. Next Clump Mass: {self.clumpDrawList[self.clumpListIndex]}')
            print(f'allClumps shape: {self.allClumps.shape}')
