import numpy as np
from simparam import *
import time as Ti
from Galaxy import *
import os
import gc
from analysis import *
###-------------------------------------------------------------------------------------------------------------###
###                                  SUITE CLASS                                                                ###
###-------------------------------------------------------------------------------------------------------------###
class Suite:
    def __init__(self):
        self.gasMassHistory     = []
        self.gasMassIniHistory  = []
        self.stellarMassHistory = []
        self.SFRHistory         = []
        self.metalMassHistory   = []
        self.metallicityHistory = []
        self.LUVHistory         = []
        self.NionHistory        = []
        self.MgasEjectHistory   = []
        self.dustMassHistory    = []
        self.SNenergyHistory    = []
        self.cloudGasHistory    = []
        self.cloudDustHistory   = []
        self.cloudMetalHistory  = []
        self.numRuns = 0
        self.simMhalo           = []
        self.simMmin            = []
        self.simMmax            = []
        self.simfstar           = []
        self.simfwind           = []
        self.simstartStep       = []
        self.simj               = []
        self.simBeta            = []
    def add_gal_to_suite(self, Gal,Mhalo,Mmin,Mmax,fstar,fwind,startStep,j,beta):
        self.gasMassHistory.append(Gal.gasMassHistory.copy())
        self.gasMassIniHistory.append(Gal.gasMassIniHistory.copy())
        self.stellarMassHistory.append(Gal.stellarMassHistory.copy())
        self.SFRHistory.append(Gal.SFRHistory.copy())
        self.metalMassHistory.append(Gal.metalMassHistory.copy())
        self.metallicityHistory.append(Gal.metallicityHistory.copy())
        self.LUVHistory.append(Gal.LUVHistory.copy())
        self.NionHistory.append(Gal.NionHistory.copy())
        self.MgasEjectHistory.append(Gal.MgasEjectHistory.copy())
        self.dustMassHistory.append(Gal.dustMassHistory.copy())
        self.SNenergyHistory.append(Gal.SNenergyHistory.copy())
        self.cloudDustHistory.append(Gal.cloudDustHistory.copy())
        self.cloudGasHistory.append(Gal.cloudGasHistory.copy())
        self.cloudMetalHistory.append(Gal.cloudMetalHistory.copy())

        self.numRuns += 1
        self.simMhalo.append(Mhalo)
        self.simMmin.append(Mmin)
        self.simMmax.append(Mmax)
        self.simfstar.append(fstar)
        self.simfwind.append(fwind)
        self.simstartStep.append(startStep)
        self.simj.append(j)
        self.simBeta.append(beta)
    def output(self,outDir, fname,append = False):
        if append:
            np.append(np.fromfile(outDir + 'Mgas'        + fname + '.dat'),np.array(self.gasMassHistory    ).flatten()).tofile(outDir + 'Mgas'        + fname + '.dat')
            np.append(np.fromfile(outDir + 'MgasIni'     + fname + '.dat'),np.array(self.gasMassIniHistory ).flatten()).tofile(outDir + 'MgasIni'     + fname + '.dat')
            np.append(np.fromfile(outDir + 'Mstar'       + fname + '.dat'),np.array(self.stellarMassHistory).flatten()).tofile(outDir + 'Mstar'       + fname + '.dat')
            np.append(np.fromfile(outDir + 'SFR'         + fname + '.dat'),np.array(self.SFRHistory        ).flatten()).tofile(outDir + 'SFR'         + fname + '.dat')
            np.append(np.fromfile(outDir + 'Mmetal'      + fname + '.dat'),np.array(self.metalMassHistory  ).flatten()).tofile(outDir + 'Mmetal'      + fname + '.dat')
            np.append(np.fromfile(outDir + 'metallicity' + fname + '.dat'),np.array(self.metallicityHistory).flatten()).tofile(outDir + 'metallicity' + fname + '.dat')
            np.append(np.fromfile(outDir + 'LUV'         + fname + '.dat'),np.array(self.LUVHistory        ).flatten()).tofile(outDir + 'LUV'         + fname + '.dat')
            np.append(np.fromfile(outDir + 'Nion'        + fname + '.dat'),np.array(self.NionHistory       ).flatten()).tofile(outDir + 'Nion'        + fname + '.dat')
            np.append(np.fromfile(outDir + 'MgasEject'   + fname + '.dat'),np.array(self.MgasEjectHistory  ).flatten()).tofile(outDir + 'MgasEject'   + fname + '.dat')
            np.append(np.fromfile(outDir + 'Mdust'       + fname + '.dat'),np.array(self.dustMassHistory   ).flatten()).tofile(outDir + 'Mdust'       + fname + '.dat')
            np.append(np.fromfile(outDir + 'SNenergy'    + fname + '.dat'),np.array(self.SNenergyHistory   ).flatten()).tofile(outDir + 'SNenergy'    + fname + '.dat')
            np.append(np.fromfile(outDir + 'MdustCl'     + fname + '.dat'),np.array(self.cloudDustHistory  ).flatten()).tofile(outDir + 'MdustCl'     + fname + '.dat')
            np.append(np.fromfile(outDir + 'MgasCl'      + fname + '.dat'),np.array(self.cloudGasHistory   ).flatten()).tofile(outDir + 'MgasCl'      + fname + '.dat')
            np.append(np.fromfile(outDir + 'MmetalCl'    + fname + '.dat'),np.array(self.cloudMetalHistory ).flatten()).tofile(outDir + 'MmetalCl'    + fname + '.dat')

            np.append(np.fromfile(outDir + 'catalogue'   + fname + '.dat'),np.array([self.simMhalo, self.simMmin, self.simMmax, self.simfstar, self.simfwind, self.simstartStep, self.simj, self.simBeta ]).T.flatten()).tofile(outDir + 'catalogue' + fname + '.dat')
        else:    
            x = [len(h) for h in self.gasMassHistory]
            if len(list(set(x)))>1:
                print('Mgas ragged. set is',set(x))
            np.array(self.gasMassHistory    ).tofile(outDir + 'Mgas'        + fname + '.dat')
            print('Saved Mgas',flush=True)
            x = [len(h) for h in self.gasMassIniHistory]
            if len(list(set(x)))>1:
                print('MgasIni ragged. set is',set(x))
            np.array(self.gasMassIniHistory ).tofile(outDir + 'MgasIni'     + fname + '.dat')
            print('Saved MgasIni',flush=True)
            x = [len(h) for h in self.stellarMassHistory]
            if len(list(set(x)))>1:
                print('Mstar ragged. set is',set(x))
            np.array(self.stellarMassHistory).tofile(outDir + 'Mstar'       + fname + '.dat')
            print('Saved Mstar',flush=True)
            x = [len(h) for h in self.SFRHistory]
            if len(list(set(x)))>1:
                print('SFR ragged. set is',set(x))
            np.array(self.SFRHistory        ).tofile(outDir + 'SFR'         + fname + '.dat')
            print('Saved SFR',flush=True)
            x = [len(h) for h in self.metalMassHistory]
            if len(list(set(x)))>1:
                print('Mmetal ragged. set is',set(x))
            np.array(self.metalMassHistory  ).tofile(outDir + 'Mmetal'      + fname + '.dat')
            print('Saved Mmetal',flush=True)
            x = [len(h) for h in self.metallicityHistory]
            if len(list(set(x)))>1:
                print('metallicity ragged. set is',set(x))
            np.array(self.metallicityHistory).tofile(outDir + 'metallicity' + fname + '.dat')
            print('Saved metallicity',flush=True)
            x = [len(h) for h in self.LUVHistory]
            if len(list(set(x)))>1:
                print('LUV ragged. set is',set(x))
            np.array(self.LUVHistory        ).tofile(outDir + 'LUV'         + fname + '.dat')
            print('Saved LUV',flush=True)
            x = [len(h) for h in self.NionHistory]
            if len(list(set(x)))>1:
                print('Nion ragged. set is',set(x))
            np.array(self.NionHistory       ).tofile(outDir + 'Nion'        + fname + '.dat')
            print('Saved Nion',flush=True)
            x = [len(h) for h in self.MgasEjectHistory]
            if len(list(set(x)))>1:
                print('MgasEject ragged. set is',set(x))
            np.array(self.MgasEjectHistory  ).tofile(outDir + 'MgasEject'   + fname + '.dat')
            print('Saved MgasEject',flush=True)
            x = [len(h) for h in self.dustMassHistory]
            if len(list(set(x)))>1:
                print('Mdust ragged. set is',set(x))
            np.array(self.dustMassHistory   ).tofile(outDir + 'Mdust'       + fname + '.dat')
            print('Saved Mdust',flush=True)
            x = [len(h) for h in self.SNenergyHistory]
            if len(list(set(x)))>1:
                print('SNenergy ragged. set is',set(x))
            np.array(self.SNenergyHistory   ).tofile(outDir + 'SNenergy'    + fname + '.dat')
            print('Saved SNenergy',flush=True)
            x = [len(h) for h in self.cloudDustHistory]
            if len(list(set(x)))>1:
                print('MdustCl ragged. set is',set(x))
            np.array(self.cloudDustHistory   ).tofile(outDir + 'MdustCl'    + fname + '.dat')
            print('Saved MdustCl',flush=True)
            x = [len(h) for h in self.cloudGasHistory]
            if len(list(set(x)))>1:
                print('MgasCl ragged. set is',set(x))
            np.array(self.cloudGasHistory   ).tofile(outDir + 'MgasCl'      + fname + '.dat')
            print('Saved MgasCl',flush=True)
            x = [len(h) for h in self.cloudMetalHistory]
            if len(list(set(x)))>1:
                print('MmetalCl ragged. set is',set(x))
            np.array(self.cloudMetalHistory   ).tofile(outDir + 'MmetalCl'  + fname + '.dat')
            print('Saved MmetalCl',flush=True)

            
            x = [len(self.simMhalo),len(self.simMmin),len(self.simMmax),len(self.simfstar),len(self.simfwind),len(self.simstartStep),len(self.simj),len(self.simBeta)]
            if len(list(set(x)))>1:
                print('catalogue ragged. set is',set(x))
            np.array([self.simMhalo,self.simMmin,self.simMmax,self.simfstar,self.simfwind,self.simstartStep,self.simj,self.simBeta]).T.tofile(outDir + 'catalogue' + fname + '.dat')  
            print('Saved catalogue',flush=True)
            





def run(IMFs, Halo_Masses, minMs, maxMs, fstar, fwind, SNfeedback, MhaloIni, SFEcloud, cloudMassLimit, cloudDensity, turnover, MhaloBeta, Nexp, outDir, append, extra):
    smoothnessThreshold = 0.006
    #smoothnessThreshold = 1
    simparam = simParam(Halo_Masses[0],smoothnessThreshold=smoothnessThreshold,maxMcl = maxMs[0],minMcl = minMs[0],fstar = fstar[0], fwind = fwind[0],MHmin = 10**MhaloIni[0])
    SFEcloudName = 0
    for ipar in range(len(IMFs)):
        NEXP = Nexp
        if SFEcloud[ipar] == None:
            SFEcloud[ipar] = 0.05
            SFEcloudName = 'dynamic'
            simpleClouds = False
        else:
            SFEcloudName = str(SFEcloud[ipar])
            simpleClouds = True
        output = Suite()
        do_This = True
        if cloudDensity[ipar] == False:
            cloudDensityName = 'Dynamic'
            n0 = 1e4
            constantCloudDensity = False
        else:
            cloudDensityName = f'1e{int(np.log10(cloudDensity[ipar]))}'
            n0 = cloudDensity[ipar]
            constantCloudDensity = True
        if cloudMassLimit[ipar] == 0:
                cloudName = '0.0e0'
        else:
            cloudName = f'{(cloudMassLimit[ipar]/10**(int(np.log10(cloudMassLimit[ipar])))):3.1f}e{int(np.log10(cloudMassLimit[ipar]))}'
        if turnover[ipar] == True:
            turnName = 'SFE_turnover_yes'
        else:
            turnName = 'SFE_turnover_off'
        typeName = f'_{IMFs[ipar]}_Mcl{int(np.log10(minMs[ipar]))}-{int(np.log10(maxMs[ipar]))}_fs{fstar[ipar]}_fw{fwind[ipar]}_SNfeedback_{SNfeedback[ipar]}_MhaloIni_{int(10*(MhaloIni[ipar]))}_SFE_{SFEcloudName}_maxMcl_{cloudName}_Z0_0000_n_{cloudDensityName}_{turnName}{extra[ipar]}'
        if not append:
            files = os.listdir(outDir)
            if f'catalogue{typeName}.dat' in files:
                do_This = False
        if do_This:
            print(typeName,flush=True)
            if maxMs[ipar]>1e8:
                Tsmooth = 0.005
            if '_smooth' in extra[ipar]:
                Tsmooth = 1
                NEXP = 1
            else:
                Tsmooth = smoothnessThreshold
            
            for iMhalo in range(len(Halo_Masses)):
                for iBeta in range(len(MhaloBeta)):
                    tt = Ti.perf_counter()
                    simparam.reload_simParam(MH0                 = Halo_Masses[iMhalo], 
                                             minMcl              = minMs[ipar], 
                                             maxMcl              = maxMs[ipar],
                                             smoothnessThreshold = Tsmooth,
                                             fstar               = fstar[ipar],
                                             fwind               = fwind[ipar],
                                             MHmin               = 10**MhaloIni[ipar],
                                             fcloud              = SFEcloud[ipar],
                                             MhaloBeta           = MhaloBeta[iBeta],
                                             clumpn0             = n0, 
                                             turnover            = turnover[ipar],
                                             )
                    Gal = Galaxy(simparam, IMF_type=IMFs[ipar], verbose = False, print_times = False, SNfeedback = SNfeedback[ipar],simple_clouds = simpleClouds, cloudMassScaleFactor = cloudMassLimit[ipar], constant_cloud_density = constantCloudDensity, SFE_turnover = turnover[ipar])
                    for iexp in range(NEXP):
                        Gal.advance_snap()
                        output.add_gal_to_suite(Gal,Halo_Masses[iMhalo],minMs[ipar],maxMs[ipar],fstar[ipar],fwind[ipar],simparam.startStep,Gal.j,MhaloBeta[iBeta])
                        Gal.reset_Galaxy()
                        if iexp == NEXP-1:
                            print(f'IMF: {IMFs[ipar]}, Mcl: {np.log10(minMs[ipar]):1.0f}-{np.log10(maxMs[ipar]):1.0f}, fs: {fstar[ipar]:5.3f}, fw: {fwind[ipar]:4.2f}, Mhalo: {Halo_Masses[iMhalo]:8.2e}, Beta: {MhaloBeta[iBeta]:5.2f}, MHmin: {MhaloIni[ipar]:3.1f}, SNfeedback: {SNfeedback[ipar]}, Mcl factor: {cloudMassLimit[ipar]:7.1e}, n0 = {cloudDensityName}, turn = {turnover[ipar]}, iexp: {iexp}, Tsmooth = {simparam.smoothMassThreshold:5.3f}, t = {Ti.perf_counter()- tt:5.2f}', flush = True)

            output.output(outDir,typeName,append)   
            del output
            gc.collect()




def run_smooth(IMFs, Halo_Masses, minMs, maxMs, fstar, fwind, SNfeedback, MhaloIni, SFEcloud,cloudMassLimit,MhaloBeta, outDir, append):
    smoothnessThreshold = 1
    smoothLUV = np.zeros((len(IMFs),len(Halo_Masses)*len(MhaloBeta),1114))
    smoothSFR = np.zeros((len(IMFs),len(Halo_Masses)*len(MhaloBeta),1114))
    
    simparam = simParam(Halo_Masses[0],smoothnessThreshold=smoothnessThreshold,maxMcl = maxMs[0],minMcl = minMs[0],fstar = fstar[0], fwind = fwind[0],MHmin = 10**MhaloIni[0])
    SFEcloudName = []
    simpleClouds = np.zeros(len(IMFs),dtype = bool)
    for ipar in range(len(IMFs)):
        if SFEcloud[ipar] is not None:
            SFEcloudName.append(str(SFEcloud[ipar]))
            simpleClouds[ipar] = True  
            print(f'Cloud Name {ipar} is not None')
        else:
            SFEcloud[ipar] = 0.05
            SFEcloudName.append('dynamic')
            simpleClouds[ipar] = False
    print(SFEcloud)
    print(SFEcloudName)
    print(simpleClouds)
    for iBeta in range(len(MhaloBeta)):
        for iMhalo in range(len(Halo_Masses)):
        
            simparam.reload_simParam(MH0                 = Halo_Masses[iMhalo], 
                                     minMcl              = minMs[ipar], 
                                     maxMcl              = maxMs[ipar],
                                     smoothnessThreshold = smoothnessThreshold,
                                     fstar               = fstar[ipar],
                                     fwind               = fwind[ipar],
                                     MHmin               = 10**MhaloIni[ipar],
                                     fcloud              = SFEcloud[ipar],
                                     MhaloBeta           = MhaloBeta[iBeta])
            print(f'iMhalo, {Halo_Masses[iMhalo]:4.2e}, {MhaloBeta[iBeta]}',flush = True)
            for ipar in range(len(IMFs)):
                Gal = Galaxy(simparam, IMF_type=IMFs[ipar], verbose = False, print_times = False, SNfeedback = SNfeedback[ipar], simple_clouds = simpleClouds[ipar],cloudMassScaleFactor = cloudMassLimit[ipar])
                Gal.advance_snap()
                print(np.sum(Gal.LUVHistory))
                print(np.sum(Gal.SFRHistory))
                smoothLUV[ipar,iMhalo + len(Halo_Masses) * iBeta] = Gal.LUVHistory.copy()
                smoothSFR[ipar,iMhalo + len(Halo_Masses) * iBeta] = Gal.SFRHistory.copy()
    for ipar in range(len(IMFs)):
        if cloudMassLimit[ipar] == 0:
            cloudName = '0.0e0'
        else:
            cloudName = f'{(cloudMassLimit[ipar]/10**(int(np.log10(cloudMassLimit[ipar])))):3.1f}e{int(np.log10(cloudMassLimit[ipar]))}'
        name = f'_{IMFs[ipar]}_Mcl{int(np.log10(minMs[ipar]))}-{int(np.log10(maxMs[ipar]))}_fs{fstar[ipar]}_fw{fwind[ipar]}_SNfeedback_{SNfeedback[ipar]}_MhaloIni_{int(10*MhaloIni[ipar])}_SFE_{SFEcloudName[ipar]}_maxMcl_{cloudName}_Z0_0000.dat'
        if not append:
            smoothLUV[ipar].tofile(outDir + f'smoothLUV'+name)
            smoothSFR[ipar].tofile(outDir + f'smoothSFR'+name)
        else:
            np.append(np.fromfile(outDir + f'smoothLUV'+name),smoothLUV[ipar]).tofile(outDir + f'smoothLUV'+name)
            np.append(np.fromfile(outDir + f'smoothSFR'+name),smoothSFR[ipar]).tofile(outDir + f'smoothSFR'+name)   





class SFRstochasticity:
    def __init__(self):
        self.sbTime     = np.array([])
        self.sbRedshift = np.array([])
        self.sbMstar    = np.array([])
        self.sbMhalo    = np.array([])
        self.qTime      = np.array([])
        self.qRedshift  = np.array([])
        self.qMstar     = np.array([])
        self.qMhalo     = np.array([])
        self.endRedshift= np.array([])
        self.endMstar   = np.array([])
        self.endMhalo   = np.array([])
        self.endMgas    = np.array([])
        self.sbStep     = np.array([])
        self.qStep      = np.array([])
        self.endStep    = np.array([])
    def add_sim(self,Gal):
        SFR = Gal.SFRHistory[:Gal.j]
        a = np.argwhere((SFR[1:]>0)*(SFR[:-1]==0)).T[0]
        b = np.argwhere((SFR[1:]==0)*(SFR[:-1]>0)).T[0]
        
        self.qMstar     = np.concatenate((self.qMstar ,    Gal.stellarMassHistory[b]))
        self.qMhalo     = np.concatenate((self.qMhalo ,    Gal.haloMassHistory[b]))
        self.qStep      = np.concatenate((self.qStep,b))
        if SFR[-1]>0:
            self.sbTime     = np.concatenate((self.sbTime,     Gal.simParam.substepTimeList[b]-Gal.simParam.substepTimeList[a[:-1]]))
            self.sbRedshift = np.concatenate((self.sbRedshift, Gal.simParam.substepRedshiftList[a[:-1]]))
            self.qTime      = np.concatenate((self.qTime,      Gal.simParam.substepTimeList[a[1:]]-Gal.simParam.substepTimeList[b]))
            self.qRedshift  = np.concatenate((self.qRedshift,  Gal.simParam.substepRedshiftList[b]))
            self.sbMstar    = np.concatenate((self.sbMstar,    Gal.stellarMassHistory[a[:-1]]))
            self.sbMhalo    = np.concatenate((self.sbMhalo,    Gal.haloMassHistory[a[:-1]]))
            self.endMhalo   = np.concatenate((self.endMhalo,   [Gal.haloMassHistory[a[-1]]]))
            self.endMstar   = np.concatenate((self.endMstar,   [Gal.stellarMassHistory[a[-1]]]))
            self.endRedshift= np.concatenate((self.endRedshift,[Gal.simParam.substepRedshiftList[a[-1]]]))
            self.endMgas    = np.concatenate((self.endMgas,    [Gal.gasMassHistory[a[-1]]]))
            self.sbStep     = np.concatenate((self.sbStep,a[:-1]))
            self.endStep    = np.concatenate((self.endStep,[a[-1]]))
        else:
            self.sbTime     = np.concatenate((self.sbTime,     Gal.simParam.substepTimeList[b]-Gal.simParam.substepTimeList[a]))
            self.sbRedshift = np.concatenate((self.sbRedshift, Gal.simParam.substepRedshiftList[a]))
            self.qTime      = np.concatenate((self.qTime,      Gal.simParam.substepTimeList[a[1:]]-Gal.simParam.substepTimeList[b[:-1]]))
            self.qRedshift  = np.concatenate((self.qRedshift,  Gal.simParam.substepRedshiftList[b[:-1]]))
            self.sbMstar    = np.concatenate((self.sbMstar,    Gal.stellarMassHistory[a]))
            self.sbMhalo    = np.concatenate((self.sbMhalo,    Gal.haloMassHistory[a]))
            self.endMhalo   = np.concatenate((self.endMhalo,   [-1]))
            self.endMstar   = np.concatenate((self.endMstar,   [-1]))
            self.endRedshift= np.concatenate((self.endRedshift,[-1]))
            self.endMgas    = np.concatenate((self.endMgas,    [-1]))
            self.sbStep     = np.concatenate((self.sbStep,a))
            self.endStep    = np.concatenate((self.endStep,[-1]))




