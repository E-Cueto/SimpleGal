import numpy as np
from IMF import *
from utils import *


def calc_SNenergy_past(Gal,simParam):
    result = 0
    f_massive = 0
    
    if Gal.IMF_type == 'salpeter':
        for prevStepNumber,prevStep in enumerate(range(simParam.firstPrevStepInStep[Gal.j],simParam.firstPrevStepInStep[Gal.j]+simParam.numPrevStepsInStep[Gal.j])):
            
            result += Gal.SNenergy[simParam.prevStepIndex[Gal.j] + prevStepNumber] * Gal.SFRHistory[prevStep]
    elif Gal.IMF_type == 'salpeter_cloud':
        for prevStepNumber,prevStep in enumerate(range(simParam.firstPrevStepInStep[Gal.j],simParam.firstPrevStepInStep[Gal.j]+simParam.numPrevStepsInStep[Gal.j])):
            mass = Gal.starGrid[prevStep].sum(axis = 0)
            for i in range(Gal.maxStarMassBins):
                result += Gal.SNenergy[simParam.prevStepIndex[Gal.j] + prevStepNumber + i * simParam.totalNumIndexes] * mass[i]
    elif Gal.IMF_type == 'evolving_cloud':
        stepIndex = slice(simParam.firstPrevStepInStep[Gal.j], simParam.firstPrevStepInStep[Gal.j]+simParam.numPrevStepsInStep[Gal.j])
        starsFormed = Gal.starGrid[stepIndex]
        fIndex = Gal.fmassiveIndexes[stepIndex]
        f_massive = Gal.fmassiveClList[fIndex]
        CMass     = fIndex
        index = (simParam.prevStepIndex[Gal.j] + np.arange(simParam.numPrevStepsInStep[Gal.j])[:,None,None] + np.add( CMass[:,:,None] * Gal.maxStarMassBins * simParam.totalNumIndexes,np.arange(Gal.maxStarMassBins) * simParam.totalNumIndexes))
        result += ((Gal.SNenergy[index] * starsFormed  ).sum(axis = 2) * (1-f_massive)).sum()
        result += ((Gal.SNenergy[index + Gal.maxStarMassBins * Gal.fmassiveClList.shape[0] * simParam.totalNumIndexes] * starsFormed ).sum(axis = 2) * f_massive).sum()
    elif Gal.IMF_type == 'evolving':
        for prevStepNumber,prevStep in enumerate(range(simParam.firstPrevStepInStep[Gal.j],simParam.firstPrevStepInStep[Gal.j]+simParam.numPrevStepsInStep[Gal.j])):
            f_massive =simParam.fmassiveList[prevStep]# fmassive(substepRedshiftList[prevStep],ZZ)
            CMass =simParam.cutoffMassList[prevStep]#get_cutoff_mass(f_massive)
            CMass[CMass<1]=1
#            CMass = np.array([np.max([int(m),1]) for m in cutoffMass])
            result += np.sum(Gal.SNenergy[simParam.prevStepIndex[Gal.j] + prevStepNumber + (CMass-1)*simParam.totalNumIndexes]*Gal.starGrid[prevStep] * (1-f_massive))
            result += np.sum(Gal.SNenergy[simParam.prevStepIndex[Gal.j] + prevStepNumber + (CMass-1)*simParam.totalNumIndexes + simParam.SN_IMFbins*simParam.totalNumIndexes] * Gal.starGrid[prevStep] * f_massive)
    elif Gal.IMF_type == 'evolving2':
        for prevStepNumber,prevStep in enumerate(range(simParam.firstPrevStepInStep[Gal.j],simParam.firstPrevStepInStep[Gal.j]+simParam.numPrevStepsInStep[Gal.j])):
            f_massive = get_fmassive_from_TBD_time_and_gasdensity(simParam.substepRedshiftList[prevStep],Gal.gasMassHistory[prevStep]/Gal.haloMassHistory[prevStep],simParam.OB0,simParam.OM0)
            CMass = max(int(get_cutoff_mass_single(f_massive,simParam.cutoffMass)),1)
#            CMass = np.array([np.max([int(m),1]) for m in cutoffMass])
            result += np.sum(Gal.SNenergy[simParam.prevStepIndex[Gal.j] + prevStepNumber + (CMass-1)*simParam.totalNumIndexes]*Gal.SFRHistory[prevStep] * (1-f_massive))
            result += np.sum(Gal.SNenergy[simParam.prevStepIndex[Gal.j] + prevStepNumber + (CMass-1)*simParam.totalNumIndexes + simParam.SN_IMFbins*simParam.totalNumIndexes] * Gal.SFRHistory[prevStep] * f_massive)
    return result
   
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
    #print(indexes,Gal.j)
    if len(indexes):
        if Gal.IMF_type == 'salpeter':
            imf = Gal.IMF_list
            for i in indexes:
                index = simParam.lifetimeIndexes[Gal.j,i] 
                metals += np.multiply(Gal.starGrid[i],Gal.yields[index[0]:index[1]]).sum()
                returnedGas += np.multiply(Gal.starGrid[i],Gal.returns[index[0]:index[1]]).sum()
                if index[1]>=79:
                    dust += Gal.SFRHistory[i] * np.sum(imf[max(index[0],79):index[1]]) * (simParam.dustProductionRate-dustDestructionRate)
                if index[0]<79:
                    dust += np.multiply(Gal.starGrid[i],Gal.dustYields[index[0]:min(index[1],79)]).sum()
        elif Gal.IMF_type == 'salpeter_cloud':
            imf = Gal.IMF_list

            for i in indexes:
                index = simParam.lifetimeIndexes[Gal.j,i] 
                metals      += (Gal.yields[:,index[0]:index[1]]@imf[index[0]:index[1]] * Gal.starGrid[i]).sum()
                returnedGas += (Gal.returns[:,index[0]:index[1]]@imf[index[0]:index[1]] * Gal.starGrid[i]).sum()
                if index[1]>=79:
                    dust += (Gal.starGrid[i].sum(axis = 0) * imf[max(index[0],79):index[1]].sum(axis = 0)).sum() * (simParam.dustProductionRate-dustDestructionRate)
                if index[0]<79:
                    dust += (Gal.dustYields[:,index[0]:min(index[1],79)]@imf[index[0]:min(index[1],79)] * Gal.starGrid[i]).sum()

        elif Gal.IMF_type == 'evolving_cloud':
            imf = Gal.IMF_list
            Gal.metalMask[Gal.j] = Gal.starGrid[Gal.j].sum(axis = 1)>0
            for i in indexes:
                index = simParam.lifetimeIndexes[Gal.j,i]
                mask = Gal.metalMask[i]
                starsFormed = imf[index[0]:index[1]+1,Gal.fmassiveIndexes[i][mask]] * Gal.starGrid[i][mask]
                
                metals += (Gal.yields[index[0]:index[1]+1,mask][:,:,None] * starsFormed).sum()
                returnedGas += (Gal.returns[index[0]:index[1]+1,mask][:,:,None] * starsFormed).sum()
                if index[1]>=79:
                    dust += (starsFormed[max(index[0],79)-index[0]:index[1]+1-index[0]]).sum() * (simParam.dustProductionRate-dustDestructionRate)
                if index[0]<79:
                    dust += (Gal.dustYields[index[0]:min(index[1],79)+1,mask][:,:,None] * starsFormed[:min(index[1],79)+1-index[0]]).sum()
        elif Gal.IMF_type == 'evolving': 
            for i in indexes: 
                index = simParam.lifetimeIndexes[Gal.j,i]
                returns = np.multiply(Gal.IMF_list[i,index[0]:index[1]], Gal.returns[index[0]:index[1]])
                returnedGas += np.multiply(Gal.starGrid[i],returns).sum()
                yields = np.multiply(Gal.IMF_list[i,index[0]:index[1]], Gal.yields[index[0]:index[1]])
                metals += np.multiply(Gal.starGrid[i],yields).sum()

                if index[0]<79:
                    dustYields = np.multiply(Gal.IMF_list[i,index[0]:min(index[1],79)], Gal.dustYields[index[0]:min(index[1],79)])
                    dust += np.multiply(Gal.starGrid[i],yields).sum()
                if index[1]>=79:
                    dust += np.multiply(Gal.starGrid[i], Gal.IMF_list[i,max(index[0],79):index[1]]).sum() * (simParam.dustProductionRate - dustDestructionRate)

        elif Gal.IMF_type == 'evolving2':
            Gal.IMF_list[Gal.j] = IMF_evolving_fmassive(simParam.MM[:500],simParam.substepRedshiftList[Gal.j],Gal.Mgas/Gal.Mhalo,simParam.cutoffMass,simParam.OB0,simParam.OM0) * dm 
            for i in indexes: 
                index = simParam.lifetimeIndexes[Gal.j,i]

                stellarMassList = np.outer(Gal.starGrid[i], Gal.IMF_list[i,index[0]:index[1]])
                returnedGas += np.multiply(stellarMassList,Gal.returns[:,index[0]:index[1]]).sum()
                metals += np.multiply(stellarMassList,Gal.yields[:,index[0]:index[1]]).sum()
                if index[0]<79:
                    
                    dust += np.multiply(stellarMassList[:,:min(index[1],79)-index[0]],Gal.dustYields[:,index[0]:min(index[1],79)]).sum()
                if index[1]>=79:
                    dust += Gal.SFRHistory[i] * np.sum(Gal.IMF_list[i,max(index[0],79):index[1]] * simParam.MM[max(index[0],79):index[1]]) * (simParam.dustProductionRate-dustDestructionRate)
                
    return returnedGas,metals,dust

def form_stars_from_clumps(Gal):
    n = Gal.allClumps.shape[0] # Number of star forming clumps
    metalsReturned=0. # Metal mass to be returned to ISM from finished clumps
    dustReturned = 0.
    gasReturned = 0.
    
    ### Loop over clumps
    dead_cloud_mask = Gal.allClumps[:,11]<Gal.dt
    dead_mask = (Gal.allClumps[:,3]<Gal.dt) * (Gal.allClumps[:,3]>0)
    new_mask  = Gal.allClumps[:,12] > 0
    new_dead_mask = dead_mask * new_mask
    new_or_dead_mask = dead_mask + new_mask
    new_not_dead_mask = (~dead_mask) * new_mask
    not_new_dead_mask = dead_mask * (~new_mask)
    if n:
        Gal.starsFormed += np.sum(Gal.allClumps[~new_or_dead_mask,1] * Gal.dt/Gal.allClumps[~new_or_dead_mask,2]) + np.sum(Gal.allClumps[not_new_dead_mask,1] * Gal.allClumps[not_new_dead_mask,3] / Gal.allClumps[not_new_dead_mask,2]) + np.sum(Gal.allClumps[new_dead_mask,1]) + np.sum(Gal.allClumps[new_not_dead_mask,1] * np.maximum(np.minimum(Gal.allClumps[new_not_dead_mask,3],Gal.dt) - Gal.allClumps[new_not_dead_mask,12],0)/Gal.allClumps[new_not_dead_mask,2])
        
        gasReturned += np.sum(Gal.allClumps[dead_cloud_mask,0]-Gal.allClumps[dead_cloud_mask,1])
        metalsReturned += np.sum((Gal.allClumps[dead_cloud_mask,0]-Gal.allClumps[dead_cloud_mask,1])*Gal.allClumps[dead_cloud_mask,5])
        dustReturned += np.sum((Gal.allClumps[dead_cloud_mask,0]-Gal.allClumps[dead_cloud_mask,1])*Gal.allClumps[dead_cloud_mask,9])
        deltaT = Gal.allClumps[:,3].copy()
        deltaT[~new_or_dead_mask] = Gal.dt
        deltaT[new_not_dead_mask] = Gal.dt - Gal.allClumps[new_not_dead_mask,12]
        deltaT[new_dead_mask] = Gal.allClumps[new_dead_mask,2]
        deltaT*= 1/Gal.allClumps[:,2]
        
        if (Gal.IMF_type == 'salpeter') + (Gal.IMF_type == 'evolving') + (Gal.IMF_type == 'evolving2'):
            for i in range(n):
                Gal.starGrid[Gal.j,int(Gal.allClumps[i,6])] +=  Gal.allClumps[i,1]*deltaT[i] #also added to the grid for later 
        elif (Gal.IMF_type == 'salpeter_cloud') + (Gal.IMF_type == 'evolving_cloud'):
            for i in range(n):
                Gal.starGrid[Gal.j,int(Gal.allClumps[i,6]),int(Gal.allClumps[i,13])] += Gal.allClumps[i,1] * deltaT[i]
        Gal.allClumps[~dead_mask,3] -= Gal.dt
        Gal.allClumps[:,12] -= Gal.dt#Gal.allClumps[~dead_mask,13]
        Gal.allClumps[dead_mask,3] = 0
        
    return gasReturned, metalsReturned,dustReturned



def draw_clump_mass(Gal,Mgas, MaxMstar):
    
    if Mgas>Gal.simParam.minMcl:
        M = Mgas
        
        ### Initial draw of clouds. Draws number of clouds assuming mean mass will equal theoretical mean
        # Adjust for star formation efficiency to not draw too many clouds
        clumpMasses   = []
        #XiDependency = []
        stellarMasses = []
        MstarLimits    = []
        clumpListMaxIndex = Gal.clumpDrawList.shape[0]-1
        stellarMassToForm = 0
        i=0
        if Gal.simple_clouds:
            while True:
                clumpMass = Gal.clumpDrawList[Gal.clumpListIndex]
                stellarMass = clumpMass * Gal.simParam.fcl
                
                if stellarMassToForm + stellarMass>MaxMstar:
                    break
                stellarMassToForm += stellarMass
                clumpMasses.append(clumpMass)
                stellarMasses.append(stellarMass)
                if Gal.clumpListIndex < clumpListMaxIndex:
                    Gal.clumpListIndex += 1
                else:
                    Gal.clumpListIndex = 0
        else:
            
            if (Gal.IMF_type == 'salpeter_cloud') + (Gal.IMF_type == 'evolving_cloud'):
                metalBin = where(Gal.simParam.ZZ,Gal.metallicity)
                while True:
                    clumpMass = Gal.clumpDrawList[Gal.clumpListIndex]
                    SFE,MstarLimit = Gal.get_MstarLimit_and_SFE(Gal,clumpMass,metalBin)
                    stellarMass = clumpMass * SFE
                    if stellarMassToForm + stellarMass>MaxMstar:
                        break
                    stellarMassToForm += stellarMass
                    clumpMasses.append(clumpMass)
                    stellarMasses.append(stellarMass)
                    MstarLimits.append(MstarLimit)
                    Gal.clumpListIndex = (Gal.clumpListIndex + 1) % clumpListMaxIndex
                if len(clumpMasses):
                    clumpMasses = np.array(clumpMasses)
                    stellarMasses = np.array(stellarMasses)
                    MstarLimits = np.array(MstarLimits,int)
            else:
                Xi = Gal.get_Xi_from_fmassive()
                T =calc_T_from_Z_Shaver1983(Gal.metallicity)
                vesc0 = calc_cs(T)
                stellarMasses, clumpMasses,Gal.clumpListIndex = draw_clouds_global_imf(Gal.clumpDrawList, 
                                                                                       Gal.clumpListIndex, 
                                                                                       clumpListMaxIndex, 
                                                                                       Gal.simParam.clumpMeanDensityPercm3, 
                                                                                       T, 
                                                                                       Xi, 
                                                                                       vesc0, 
                                                                                       MaxMstar)

        if len(clumpMasses):
            
            
            
            freeFallTimes = np.zeros_like(clumpMasses)+Gal.simParam.tff #This is not to remain constant like this
            if Gal.simple_clouds:
                starFormationTimes = freeFallTimes * 2.55
                starFormationStartTimes =  0
                MstarLimits = np.zeros_like(clumpMasses)
            else:
                starFormationTimes = freeFallTimes * calc_tsf(Gal.simParam.clumpMeanDensityPercm3,clumpMasses)# * np.array(XiDependency)
                starFormationStartTimes = freeFallTimes * calc_tsf0(Gal.simParam.clumpMeanDensityPercm3,clumpMasses) + np.linspace(0,Gal.dt,len(clumpMasses)+1)[:-1]
                if (Gal.IMF_type != 'evolving_cloud') * (Gal.IMF_type != 'salpeter_cloud'):
                    MstarLimits = np.zeros_like(clumpMasses)

            Gal.Mmetal = Gal.Mmetal - Gal.metallicity *  np.sum(clumpMasses)
            Gal.Mdust = Gal.Mdust - Gal.dustToGasRatio * np.sum(clumpMasses)
            return clumpMasses, stellarMasses, starFormationTimes,starFormationStartTimes, MstarLimits,Mgas-np.sum(clumpMasses)
        else:
            return np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0],int),Mgas

    else:
        return np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0],int),Mgas
