import numpy as np
import scipy.interpolate as si
import astropy.cosmology as co
from HaloProperties import *
from StarFormation import *
from fescClouds import *
from fescISM import *
from SNfeedback import *

class simParam:
    __slots__ = ['OM0','OB0','OL0','h0','invh0','P18',
                 'fs','fcl','minMcl','maxMcl','ClMFslope','meanClumpMass','smoothMassThreshold','tff',
                 'clumpMeanDensityPercm3', 'cm3Perkpc3','clumpMeanDensityPerpc3','MH_per_Msun','clumpMeanDensityMsunPerpc3',
                 'rmin', 'rmax', 'rminGal', 'alphaB', 'ci', 'omega', 'sigmaHI', 'psi', 'theta', 'Clump_type', 'kRstall', 'tStall', 'S0Salpeter', 'S0uEvolving', 'S0lEvolving', 'cloudMassFuncNorm', 'meanRiSmoothStarFormation', 'meanRstallSmoothStarFormation', 'cloudRadiusFromMass',
                 'MSN','fw','SN_IMFbins','Mhalo0','MHmin','MhaloBeta','endSnap','redshift_list','time_list','NtimestepsTot','timestepList','substepTimeList','dtList','substepRedshiftList',
                 'haloMasses','startSnap','Rvir','vc','startStep','Ntimesteps','Nsnaps','Mgas0','lifetimeIndexes','IMF_list',
                 'fmassiveList','cutoffMassList','SNenergy','numPrevStepsInStep','firstPrevStepInStep','prevStepIndex','totalNumIndexes',
                 'dustGrowthTimescale','MSNblastwave','fracColdGas','dustProductionRate','dustDestrEfficiency','dustDestructionRatePerDustFraction','RvirSubstep','haloMassSubstep',
                 'concentration_c','n0_factor_NFW','Myr_s','Msun_g','mp_g','kpc_cm','pc_cm', 'MM','ZZ','lifetime','returns','remnants','yields',
                 'dustYields','cutoffMass','redshift_list','time_list','smoothSFGrids','smoothSFInterps' , 'maxMgas', 'cloudSFE', 'cloudMstarLimit','turn', 'turnover']
    def __init__(self,MH0, Mgas0 = 0,MHmin = 10**7.4, maxMcl = 1e6, minMcl = 1e3,  fstar = 0.025,fwind = 0.2,fcloud = 0.05, smoothnessThreshold = 0.01, MhaloBeta = -0.75,Clump_type = 'uniform',omega = 0.6,clumpn0 = 10000,ClMFslope = -2,endSnap = 70,zion = 8, zreion = 7, turnover = True):
        ### Cosmology
        self.OM0 = 0.307115 #Cosmic mass fraction
        self.OB0 = 0.048    #Cosmic baryon fraction
        self.OL0 = 0.692885 #Cosmic lambda fraction
        self.h0 = 0.67770   #Hubble constant
        self.invh0 = 1/self.h0   #Inverse Hubble constant
        self.P18 = co.Planck18 #flat lambdaCDM cosmology
        
        
        ### Loading Files
        path = '/home/vdm981_alumni_ku_dk/modi_mount/Astraeus/astraeus/dynamic_IMF/yield_table_FINAL/output_data_original/'
        self.MM = np.loadtxt(path+'M.dat') # List of stellar masses
        self.ZZ = np.loadtxt(path+'Z.dat') # List of metallicities
        self.lifetime = np.loadtxt(path+'lifetimes.dat') *1000# List of stellar lifetimes by mass
        self.returns = np.loadtxt(path+'return_kobayashi2020.dat')#.T
        self.remnants = np.loadtxt(path+'remnant_kobayashi2020.dat')#.T
        self.yields = np.loadtxt(path+'yields_kobayashi2020.dat')#.T
        self.dustYields =np.loadtxt(path+'yields_dust_agb.dat')#.T
        self.cutoffMass = np.loadtxt(path+'mass_lookup_table_Salpeter.dat')
        

        
        ### Constants
        self.cm3Perkpc3                 = 2.938e64
        self.MH_per_Msun                = 1.1894799760908547e57
        self.alphaB                     = 2.6e-13# Hydrogen recombination coefficient
        self.sigmaHI                    = 3.2e-18 #in cm^2
        self.kRstall                    = 9.014585e+14 #9.014585e+14 =( X/m_H * 3/(8 pi G))^0.5
        self.Myr_s                      = 1e6*365.*24.*60.*60.
        self.Msun_g                     = 1.99e33
        self.mp_g                       = 1.673e-24
        self.kpc_cm                     = 3.0857e21
        self.pc_cm                      = 3.0857e18
        ### Star formation
        self.fs = fstar # Star formation efficiency (galaxy wide)
        self.fcl = fcloud # Star formation efficiency (galaxy wide, only when simple clouds is enabled)
        self.meanClumpMass = calc_mean_clump_mass(minMcl,maxMcl,ClMFslope)
        self.smoothMassThreshold = smoothnessThreshold
        self.maxMcl                     = maxMcl
        self.minMcl                     = minMcl
        self.ClMFslope                  = ClMFslope
        self.clumpMeanDensityPercm3     = clumpn0
        self.clumpMeanDensityPerpc3     = self.clumpMeanDensityPercm3 * self.pc_cm ** 3
        self.clumpMeanDensityMsunPerpc3 = self.clumpMeanDensityPerpc3 / self.MH_per_Msun
        rhoCloud = clumpn0 * 2.3416e-24#(clumpn0 * ac.m_p * 1.4 / u.cm**3).to(u.g/u.cm**3).value #Initial density of clouds (not to remain constant)
        self.tff = 6.6566e-11/np.sqrt(rhoCloud) #(np.sqrt(3*np.pi/(32*ac.G * rhoCloud * u.g/u.cm**3))).to(u.Myr).value
        
        self.rmin                       = 0.5
        self.rmax                       = 1.5
        self.rminGal                    = 0.1
        self.ci                         = 12.5 * 1e5 #cm/s. sound speed at ionised front
        
        ### Mean escape fraction calculations for smoothed star formation
        self.Clump_type                     = Clump_type
        #self.omega                          = 0.
        #if self.Clump_type == 'isothermal':
        #    self.omega                      = omega
        #self.psi                            = calc_psi(self.omega)
        #self.theta                          = 4/(7-4 * self.omega)
        #self.tStall                         = self.kRstall*self.psi/np.sqrt(self.clumpMeanDensityPercm3)/self.Myr_s
        #self.S0Salpeter                     = 10**46.56 #Nion per solar mass before 10^6.6 yr
        #self.S0uEvolving                    = 10**47.75 #Upper and lower bounds on Nion
        #self.S0lEvolving                    = 10 ** 43.2#for evolving IMF
        #self.cloudMassFuncNorm              = 1/(minMcl**-1-maxMcl**-1)
        #if self.Clump_type == 'uniform':
        #    self.meanRiSmoothStarFormation      = ((3 * self.fcl /(4 * np.pi * self.alphaB))**0.25 * (self.ci**2/self.clumpMeanDensityPercm3)**0.5 * 4/5 /self.psi)**self.psi #Multiply by t**psi * S0 ** psi/4 to get ri
        #    self.meanRstallSmoothStarFormation  = (((3*self.fcl) / (4*np.pi*self.alphaB))**0.25 * self.ci/self.clumpMeanDensityPercm3 * (4/5) * self.kRstall)**self.theta       #Multiply by S0**psi/4 to get rstall
        #    self.cloudRadiusFromMass            = (4 * np.pi * (self.clumpMeanDensityPercm3/self.MH_per_Msun)/3)**(1/3)
        #else:
        #    self.cloudRadiusFromMass            = (4 * np.pi * (self.clumpMeanDensityPercm3*self.mp_g/self.Msun_g)/3)**(1/3)
        #    self.meanRiSmoothStarFormation      = ((3 * self.fcl /(4 * np.pi * self.alphaB))**0.25 * (self.ci**2*self.cloudRadiusFromMass**(self.omega)/self.clumpMeanDensityPercm3)**0.5 * 4/5 /self.psi)**self.psi #Multiply by t**psi * S0 ** psi/4 to get ri
        #    self.meanRstallSmoothStarFormation  = (((3*self.fcl) / (4*np.pi*self.alphaB))**0.25 * self.ci/self.clumpMeanDensityPercm3 * self.cloudRadiusFromMass**(self.omega) * (4/5) * self.kRstall)**self.theta       #Multiply by S0**psi/4 to get rstall
            
        ### Dust
        self.dustGrowthTimescale  = 3e7
        self.MSNblastwave         = 6.8e3 # material that is accelerated to 100 km/s by SN blast wave */
        self.fracColdGas          = 0.5
        self.dustProductionRate   = 0.45
        self.dustDestrEfficiency  = 0.03
        self.dustYields[:,79:500]      = self.dustProductionRate
        self.dustDestructionRatePerDustFraction = np.zeros_like(self.dustYields)
        self.dustDestructionRatePerDustFraction[:,79:500] = (1-self.fracColdGas) * self.MSNblastwave * self.dustDestrEfficiency
        ### Supernova feedback
        self.MSN = 8 #Mass where stars go SN
        self.fw = fwind #SN wind coupling efficiency
        self.SN_IMFbins = 100 #number of IMF bins for SNenergy
        
        ### Galaxy initial parameters
        #self.Mgas0 = Mgas0 # Initial Gas mass at first snap
        self.Mhalo0 = MH0 # Halo mass at z = 0
        self.endSnap = endSnap
        self.MHmin   = MHmin
        
        #Time and redshift info
        self.redshift_list = np.loadtxt('/home/vdm981_alumni_ku_dk/modi_mount/Astraeus/astraeus/dynamic_IMF/redshift_list.txt').T[1] #List of snapshot redshifts
        self.time_list = self.P18.age(self.redshift_list).value * 1000 #List of snapshot ages in Myr
        self.timestepList        = np.loadtxt('input_data/timestepList.dat')[:self.endSnap+1].astype(int)
        self.NtimestepsTot       = np.sum(self.timestepList)
        self.substepTimeList     = np.loadtxt('input_data/substepTimeList.dat')[:self.NtimestepsTot]
        self.substepRedshiftList = np.loadtxt('input_data/substepRedshiftList.dat')[:self.NtimestepsTot]
        self.dtList              = np.loadtxt('input_data/dtList.dat')[:self.NtimestepsTot]
        
        ### Halo mass and properties
        self.haloMasses     = calc_Mhalo(MH0,self.redshift_list,MhaloBeta) #Halo Masses at snapshots (Not at timesteps, to emulate Astraeus)
        self.MhaloBeta      = MhaloBeta
        self.startSnap      = np.argwhere(self.haloMasses>MHmin)[0,0] #First snapshot where halo is massive enough 
        self.Rvir           = calc_Rvir2(self.haloMasses,self.redshift_list)/self.kpc_cm # calc_Rvir(self.haloMasses)
        self.vc             = calc_vc(self.haloMasses,self.Rvir,self.redshift_list)
        
        interp              = si.interp1d(self.redshift_list,self.haloMasses)
        self.haloMassSubstep     = np.zeros(self.NtimestepsTot)
        self.haloMassSubstep[1:] = interp(self.substepRedshiftList[1:])
        self.haloMassSubstep[0]  = self.haloMasses[0]
        self.RvirSubstep    = calc_Rvir2(self.haloMassSubstep,self.substepRedshiftList)
        #self.concentration_c = concentration_parameter(self.haloMassSubstep,self.substepRedshiftList,self.OL0,self.OM0,self.h0)
        #self.n0_factor_NFW   = n0_factor_NFW(self)

        self.maxMgas = calc_maxMgas_radFeedback(self.haloMassSubstep, self.substepRedshiftList, self.OM0, self.OB0,zion,zreion)

        
        self.startStep      = np.sum(self.timestepList[:self.startSnap]) #first timestep in startSnap
        self.Ntimesteps     = self.NtimestepsTot-self.startStep
        self.Nsnaps         = self.endSnap - self.startSnap + 1
        self.Mgas0          = 0 #self.haloMasses[self.startSnap] * self.OB0/OM0
        ### Indexes of stars from previous timesteps to die in current timestep
        self.lifetimeIndexes = np.zeros((self.NtimestepsTot,self.NtimestepsTot,2),int)
        self.lifetimeIndexes[:,:,0] = np.reshape(np.fromfile('input_data/lifetimeIndexesSalpeterLow.dat', dtype = 'int32'), (self.NtimestepsTot, self.NtimestepsTot))
        self.lifetimeIndexes[:,:,1] = np.reshape(np.fromfile('input_data/lifetimeIndexesSalpeterHigh.dat', dtype = 'int32'), (self.NtimestepsTot, self.NtimestepsTot))

        
        ### IMF values at each timestep for all IMFs
        self.IMF_list = [IMF_Salpeter(self.MM[:500]) * (self.MM[1]-self.MM[0]), np.zeros((self.substepTimeList.shape[0],self.ZZ.shape[0],500)),get_IMF_from_fmassive(self.MM[:500],np.linspace(0,1,101),self.cutoffMass).T * (self.MM[1]-self.MM[0]) ]
        self.fmassiveList = np.zeros((self.NtimestepsTot,self.ZZ.shape[0]))
        self.cutoffMassList = np.zeros_like(self.fmassiveList)
        for i in range(self.substepTimeList.shape[0]):
            self.IMF_list[1][i] = IMF_evolving(self.MM[:500],self.substepRedshiftList[i],self.ZZ,self.cutoffMass).T * (self.MM[1]-self.MM[0]) 
            self.fmassiveList[i] = fmassive(self.substepRedshiftList[i],self.ZZ)
            self.cutoffMassList[i] = get_cutoff_mass(self.fmassiveList[i],self.cutoffMass)
        self.cutoffMassList = self.cutoffMassList.astype(int)
        self.cutoffMassList[self.cutoffMassList<1]=1

        
        ### Load up grids for smooth star formation
        if turnover:
            turn = ''
        else:
            turn = '_no_turn'
        self.turn = turn
        self.turnover=turnover
        self.smoothSFGrids = [[],[],[],[]]
        if self.minMcl in np.array([1e3,1e4,1e5]):
            maxMcls = np.array([1e6,1e7,1e8,1e9]) if self.minMcl == 1e3 else np.array([1e7,1e8,1e9])
            if self.maxMcl in maxMcls:
                for i,imf in enumerate(['salpeter','evolving','salpeter_cloud','evolving_cloud']):
                    for j,n in enumerate(['1e2','1e3','1e4','1e5']):
                        name = f'SmoothSF_{imf}_{int(np.log10(self.minMcl))}_{int(np.log10(self.maxMcl))}_{n}.dat'
                        self.smoothSFGrids[i].append(np.fromfile(f'input_data/smooth_star_formation{turn}/'+name))
        
        interp0  = si.RegularGridInterpolator((np.array([2,3,4,5]),self.ZZ),np.array([self.smoothSFGrids[0][0],self.smoothSFGrids[0][1],self.smoothSFGrids[0][2],self.smoothSFGrids[0][3]]),'linear')
        data     = np.array([np.reshape(self.smoothSFGrids[1][0],(11,200)),np.reshape(self.smoothSFGrids[1][1],(11,200)),np.reshape(self.smoothSFGrids[1][2],(11,200)),np.reshape(self.smoothSFGrids[1][3],(11,200))])
        interp1  = si.RegularGridInterpolator((np.array([2,3,4,5]),np.linspace(0,1,11),self.ZZ),data,'linear')
        interp2 =  si.RegularGridInterpolator((np.array([2,3,4,5]),np.array([0,1,2,3]),self.ZZ,np.arange(10,101,10)),(np.array(self.smoothSFGrids[2]).reshape(4,4,200,10)))
        interp3 =  si.RegularGridInterpolator((np.array([2,3,4,5]),np.array([0,1,2,3]),self.ZZ,np.linspace(0,1,11),np.arange(10,101,10)),(np.array(self.smoothSFGrids[3]).reshape(4,4,200,11,10)))
        self.smoothSFInterps = [interp0,interp1,interp2,interp3]

        ### Load SFE and MstarLimit for cloud mass limited IMFs
        self.cloudSFE = [np.array([np.fromfile(f'input_data/cloud_sfe{turn}/SFE_salpeter_cloud_1e{x:d}.dat') for x in [2,3,4,5]]),np.array([np.fromfile(f'input_data/cloud_sfe{turn}/SFE_evolving_cloud_1e{x:d}.dat') for x in [2,3,4,5]])]
        self.cloudMstarLimit = [np.array([((np.fromfile(f'input_data/cloud_sfe{turn}/Mlimit_salpeter_cloud_1e{x:d}.dat')+5)//10).astype(int)-1 for x in [2,3,4,5]]),np.array([np.fromfile(f'input_data/cloud_sfe{turn}/Mlimit_evolving_cloud_1e{x:d}.dat',dtype = 'int32') for x in [2,3,4,5]])]
                         
        
        ### Supernova energy table
        self.SNenergy = [np.fromfile('input_data/SNfeedbackSalpeter0_70.dat'),np.fromfile('input_data/SNfeedbackEvolving0_70.dat'),np.fromfile('input_data/SNfeedbackSalpetercloud0_70.dat'), np.fromfile('input_data/SNfeedbackEvolvingcloud0_70.dat') ]
        numStepsToSave = int(get_SNtime(self.MSN))+1 
        self.numPrevStepsInStep  = np.zeros(self.NtimestepsTot+1,int)
        self.firstPrevStepInStep = np.zeros(self.NtimestepsTot+1,int)
        self.prevStepIndex       = np.zeros(self.NtimestepsTot+1,int)
        for step in range(1,self.NtimestepsTot):
            self.numPrevStepsInStep[step]  = np.min([numStepsToSave,step])
            self.firstPrevStepInStep[step] = step-self.numPrevStepsInStep[step]
            self.prevStepIndex[step]       = np.sum(self.numPrevStepsInStep[:step])
        self.totalNumIndexes = np.sum(self.numPrevStepsInStep)
        for step in range(self.startStep,self.NtimestepsTot):
            self.numPrevStepsInStep[step]  = np.min([numStepsToSave,step-self.startStep])
            self.firstPrevStepInStep[step] = step-self.numPrevStepsInStep[step]


    
    def mean_fesc_ri(self,S0,t):
        if self.Clump_type == 'uniform':
            return self.cloudMassFuncNorm*(self.meanRiSmoothStarFormation*S0**(self.psi/4)*t**(self.psi)/(self.rmax-self.rmin)*self.cloudRadiusFromMass*(12/(3*self.psi-16))*((self.maxMcl)**(self.psi/4-4/3)-(self.minMcl)**(self.psi/4-4/3))) + self.rmin/(self.rmax-self.rmin)
        else:
            return self.cloudMassFuncNorm*self.meanRiSmoothStarFormation*S0**(self.theta/4)*t**(self.theta)/(self.rmax-self.rmin)*self.cloudRadiusFromMass*(-self.theta/6*(9-5*self.omega))*((self.maxMcl)**(-self.theta/6*(9-5*self.omega))-(self.minMcl)**(-self.theta/6*(9-5*self.omega))) + self.rmin/(self.rmax-self.rmin)
    def mean_fesc_rstall(self,S0):
        if self.Clump_type == 'uniform':
            RstallUpperBoundMass = min((self.meanRstallSmoothStarFormation*S0**(self.psi/4)*self.cloudRadiusFromMass/self.rmin)**(12/(4-3*self.psi)),self.maxMcl)
            #print(f'Mean positive combined escape fraction with rstall',(R * e / (F - f) * (12 / (3*psi - 4)) * ((g)**(psi/4-1/3)-(minMcl)**(psi/4-1/3)) - f/(F-f)*np.log(g/minMcl) )/np.log(maxMcl/minMcl))
            return (self.meanRstallSmoothStarFormation*S0**(self.psi/4)/(self.rmax-self.rmin)*self.cloudRadiusFromMass*(12/(3*self.psi-4))*((RstallUpperBoundMass)**(self.psi/4-1/3)-(self.minMcl)**(self.psi/4-1/3)) - self.rmin/(self.rmax-self.rmin)*np.log(RstallUpperBoundMass/self.minMcl))/np.log(self.maxMcl/self.minMcl)
        else:
            RstallUpperBoundMass = min((self.meanRstallSmoothStarFormation*S0**(self.theta/4)*self.cloudRadiusFromMass/self.rmin)**(3/self.theta),self.maxMcl)
            return self.cloudMassFuncNorm*(self.meanRstallSmoothStarFormation*S0**(self.theta/4)/(self.rmax-self.rmin)*self.cloudRadiusFromMass*(-(2+self.theta)/3)*((RstallUpperBoundMass)**(-(2+self.theta)/3)-(self.minMcl)**(-(2+self.theta)/3)) + self.rmin/(self.rmax-self.rmin)*((RstallUpperBoundMass)**(-1)-(self.minMcl)**(-1)))
        
    def reload_simParam(self,MH0, Mgas0 = 0,MHmin = 10**7.4,endSnap = 70, maxMcl = 1e6, minMcl = 1e3, ClMFslope = -2, fstar = 0.025, fwind = 0.2,fcloud = 0.05, clumpn0 = 10000,smoothnessThreshold = 0.01,Clump_type = 'uniform',omega = 0.6,MhaloBeta = -0.75,zion = 8, zreion = 7, turnover = True):
        self.endSnap = endSnap
        if (MH0 != self.Mhalo0) + (MHmin != self.MHmin) + (MhaloBeta != self.MhaloBeta):
            self.Mhalo0 = MH0
            self.MHmin = MHmin
            ### Halo mass and properties
            self.haloMasses          = calc_Mhalo(MH0,self.redshift_list,MhaloBeta) #Halo Masses at snapshots (Not at timesteps, to emulate Astraeus)
            self.MhaloBeta           = MhaloBeta
            self.startSnap           = np.argwhere(self.haloMasses>MHmin)[0,0] #First snapshot where halo is massive enough 
            self.Rvir                = calc_Rvir2(self.haloMasses,self.redshift_list)/self.kpc_cm # calc_Rvir(self.haloMasses)
            self.vc                  = calc_vc(self.haloMasses,self.Rvir,self.redshift_list)
        
            interp                   = si.interp1d(self.redshift_list,self.haloMasses)
            self.haloMassSubstep     = np.zeros(self.NtimestepsTot)
            self.haloMassSubstep[1:] = interp(self.substepRedshiftList[1:])
            self.haloMassSubstep[0]  = self.haloMasses[0]
            self.RvirSubstep         = calc_Rvir2(self.haloMassSubstep,self.substepRedshiftList)
            #self.concentration_c     = concentration_parameter(self.haloMassSubstep,self.substepRedshiftList,self.OL0,self.OM0,self.h0)
            #self.n0_factor_NFW       = n0_factor_NFW(self)

            self.maxMgas = calc_maxMgas_radFeedback(self.haloMassSubstep, self.substepRedshiftList, self.OM0, self.OB0,zion,zreion)
            
            self.startStep           = np.sum(self.timestepList[:self.startSnap]) #first timestep in startSnap
            self.Ntimesteps          = self.NtimestepsTot-self.startStep
            self.Nsnaps              = self.endSnap - self.startSnap + 1
        
        self.fs = fstar # Star formation efficiency (galaxy wide)
        self.fcl = fcloud # Star formation efficiency (galaxy wide, only when simple clouds is enabled)
        self.fw = fwind #SN wind coupling efficiency

        if (maxMcl != self.maxMcl) + (minMcl != self.minMcl) + (self.turnover != turnover) + (self.smoothMassThreshold != smoothnessThreshold):
        
            self.meanClumpMass = calc_mean_clump_mass(minMcl,maxMcl,ClMFslope)
            self.smoothMassThreshold = smoothnessThreshold
            self.maxMcl                     = maxMcl
            self.minMcl                     = minMcl

            if turnover:
                turn = ''
            else:
                turn = '_no_turn'
            self.turn = turn
            self.turnover=turnover
            self.smoothSFGrids = [[],[],[],[]]
            if self.minMcl in np.array([1e3,1e4,1e5]):
                maxMcls = np.array([1e6,1e7,1e8,1e9]) if self.minMcl == 1e3 else np.array([1e7,1e8,1e9])
                if self.maxMcl in maxMcls:
                    for i,imf in enumerate(['salpeter','evolving','salpeter_cloud','evolving_cloud']):
                        for j,n in enumerate(['1e2','1e3','1e4','1e5']):
                            name = f'SmoothSF_{imf}_{int(np.log10(self.minMcl))}_{int(np.log10(self.maxMcl))}_{n}.dat'
                            self.smoothSFGrids[i].append(np.fromfile(f'input_data/smooth_star_formation{self.turn}/'+name))
        
            interp0  = si.RegularGridInterpolator((np.array([2,3,4,5]),self.ZZ),np.array([self.smoothSFGrids[0][0],self.smoothSFGrids[0][1],self.smoothSFGrids[0][2],self.smoothSFGrids[0][3]]),'linear')
            data     = np.array([np.reshape(self.smoothSFGrids[1][0],(11,200)),np.reshape(self.smoothSFGrids[1][1],(11,200)),np.reshape(self.smoothSFGrids[1][2],(11,200)),np.reshape(self.smoothSFGrids[1][3],(11,200))])
            interp1  = si.RegularGridInterpolator((np.array([2,3,4,5]),np.linspace(0,1,11),self.ZZ),data,'linear')
            interp2 =  si.RegularGridInterpolator((np.array([2,3,4,5]),np.array([0,1,2,3]),self.ZZ,np.arange(10,101,10)),(np.array(self.smoothSFGrids[2]).reshape(4,4,200,10)))
            interp3 =  si.RegularGridInterpolator((np.array([2,3,4,5]),np.array([0,1,2,3]),self.ZZ,np.linspace(0,1,11),np.arange(10,101,10)),(np.array(self.smoothSFGrids[3]).reshape(4,4,200,11,10)))
            self.smoothSFInterps = [interp0,interp1,interp2,interp3]

            ### Load SFE and MstarLimit for cloud mass limited IMFs
            self.cloudSFE = [np.array([np.fromfile(f'input_data/cloud_sfe{turn}/SFE_salpeter_cloud_1e{x:d}.dat') for x in [2,3,4,5]]),np.array([np.fromfile(f'input_data/cloud_sfe{turn}/SFE_evolving_cloud_1e{x:d}.dat') for x in [2,3,4,5]])]
            self.cloudMstarLimit = [np.array([((np.fromfile(f'input_data/cloud_sfe{turn}/Mlimit_salpeter_cloud_1e{x:d}.dat')+5)//10).astype(int)-1 for x in [2,3,4,5]]),np.array([np.fromfile(f'input_data/cloud_sfe{turn}/Mlimit_evolving_cloud_1e{x:d}.dat',dtype = 'int32') for x in [2,3,4,5]])]
        
        self.clumpMeanDensityPercm3     = clumpn0
        self.clumpMeanDensityPerpc3     = self.clumpMeanDensityPercm3 * self.pc_cm ** 3
        self.clumpMeanDensityMsunPerpc3 = self.clumpMeanDensityPerpc3 / self.MH_per_Msun
        rhoCloud = clumpn0 * 2.3416e-24#(clumpn0 * ac.m_p * 1.4 / u.cm**3).to(u.g/u.cm**3).value #Initial density of clouds (not to remain constant) (Not used so it remains constant for now even though n is not constant anymore)
        self.tff = 6.6566e-11/np.sqrt(rhoCloud) #(np.sqrt(3*np.pi/(32*ac.G * rhoCloud * u.g/u.cm**3))).to(u.Myr).value
        
        ### Mean escape fraction calculations for smoothed star formation
        
        self.Clump_type                     = Clump_type
        #self.omega                          = 0.
        #if self.Clump_type == 'isothermal':
        #    self.omega                      = omega
        #self.psi                            = calc_psi(self.omega)
        #self.theta                          = 4/(7-4 * self.omega)
        #self.tStall                         = self.kRstall*self.psi/np.sqrt(self.clumpMeanDensityPercm3)/self.Myr_s
        #self.cloudMassFuncNorm              = 1/(minMcl**-1-maxMcl**-1)
        #if self.Clump_type == 'uniform':
        #    self.meanRiSmoothStarFormation      = ((3 * self.fcl /(4 * np.pi * self.alphaB))**0.25 * (self.ci**2/self.clumpMeanDensityPercm3)**0.5 * 4/5 /self.psi)**self.psi #Multiply by t**psi * S0 ** psi/4 to get ri
        #    self.meanRstallSmoothStarFormation  = (((3*self.fcl) / (4*np.pi*self.alphaB))**0.25 * self.ci/self.clumpMeanDensityPercm3 * (4/5) * self.kRstall)**self.theta       #Multiply by S0**psi/4 to get rstall
        #    self.cloudRadiusFromMass            = (4 * np.pi * (self.clumpMeanDensityPercm3/self.MH_per_Msun)/3)**(1/3)
        #else:
        #    self.cloudRadiusFromMass            = (4 * np.pi * (self.clumpMeanDensityPercm3*self.mp_g/self.Msun_g)/3)**(1/3)
        #    self.meanRiSmoothStarFormation      = ((3 * self.fcl /(4 * np.pi * self.alphaB))**0.25 * (self.ci**2*self.cloudRadiusFromMass**(self.omega)/self.clumpMeanDensityPercm3)**0.5 * 4/5 /self.psi)**self.psi #Multiply by t**psi * S0 ** psi/4 to get ri
        #    self.meanRstallSmoothStarFormation  = (((3*self.fcl) / (4*np.pi*self.alphaB))**0.25 * self.ci/self.clumpMeanDensityPercm3 * self.cloudRadiusFromMass**(self.omega) * (4/5) * self.kRstall)**self.theta       #Multiply by S0**psi/4 to get rstall
            