import numpy as np
from numba import njit, prange
from IMF import *
from utils import *

### Calculate the ejection efficiency, the fraction of the galaxys current gas mass that is not expelled by SN feedback in this timestep.
def calc_SNejection_efficiency(Gal):
    simParam = Gal.simParam
    SNejectionEfficiency = 0
    
    # Get the energy released by SN in this timestep from stars formed in previous timesteps
    if Gal.SNfeedback == 'delayed':
        SNenergyPast = calc_SNenergy_past(Gal,simParam)
    
    elif Gal.SNfeedback == '10myr':
        SNenergyPast = calc_SNenergy_10(Gal,simParam)
    #SNenergyPast = calc_SNenergy_instant(Gal,simParam)
    Gal.SNenergyHistory[Gal.j] = SNenergyPast
    factorSNenergyPast = 0
    # calculate the fraction of the galaxy's gas that will remain after the SN feedback.
    if Gal.Mgas>0:
        factorSNenergyPast = 1 - Gal.fw * SNenergyPast / ((Gal.Mgas) * simParam.vc[Gal.snap] * simParam.vc[Gal.snap])
    SNejectionEfficiency = np.nanmax([0,factorSNenergyPast])
    
    return SNejectionEfficiency

### Calculate the energy released from SN in this timestep, assuming all SN feedback occurs after 10 Myr
def calc_SNenergy_10(Gal,simParam):
    result = 0
    f_massive = 0
    
    if Gal.IMF_type == 'salpeter':
        result = Gal.IMF_list[79:500].sum() * 0.1 * Gal.SFRHistory[Gal.j-10] * 5e7 * simParam.invh0
    elif Gal.IMF_type == 'evolving':
        NSNperMsun = Gal.IMF_list[Gal.j-10,79:500].sum(axis = 0)
        result = np.sum(NSNperMsun * Gal.starGrid[Gal.j-10]) * 5e7 * simParam.invh0
    elif Gal.IMF_type == 'evolving2':
        result = Gal.IMF_list[Gal.j-10,79:500].sum() * Gal.SFRHistory[Gal.j-10] * 5e7 * simParam.invh0
    elif Gal.IMF_type == 'salpeter_cloud':
        result = (Gal.IMF_list[79:500].sum(axis = 0) * Gal.starGrid[Gal.j-10].sum(axis = 0)).sum() * 5e7 * simParam.invh0
    elif Gal.IMF_type == 'evolving_cloud':
        metalMask = Gal.metalMask[Gal.j-10]
        starsFormed = Gal.starGrid[Gal.j-10][metalMask]
        fIndex    = Gal.fmassiveIndexes[Gal.j-10][metalMask]  
        for i in range(np.sum(metalMask)):
            result += (Gal.IMF_list[fIndex[i],79:500].sum(axis = 0) * starsFormed[i]).sum() * 5e7 * simParam.invh0
    return result
    
### Calculate fraction of gas not expelled from galaxy by Sn feedback in this timestep
def calc_remaining_gas_fraction(fej,feff):
    return fej


### Calculate the energy released from SN in this timestep, from stars formed in previous time steps
def calc_SNenergy_past(Gal,simParam):
    # Find the proper IMF and run the relevant function. These are all parallelised with numba, which gets finicky when dealing with classes, so they cannot just be fed simParam and Gal.
    if Gal.IMF_type == 'salpeter':
        result = get_SNenergy_past_salpeter(simParam.firstPrevStepInStep[Gal.j], 
                                            simParam.numPrevStepsInStep[Gal.j],
                                            simParam.prevStepIndex[Gal.j],
                                            Gal.SFRHistory, 
                                            Gal.SNenergy, 
                                            Gal.j)
    
    elif Gal.IMF_type == 'salpeter_cloud':
        result = get_SNenergy_past_salpeter_cloud(simParam.firstPrevStepInStep[Gal.j], 
                                                  simParam.numPrevStepsInStep[Gal.j],
                                                  simParam.prevStepIndex[Gal.j],
                                                  Gal.maxStarMassBins, 
                                                  simParam.totalNumIndexes, 
                                                  Gal.starGrid, 
                                                  Gal.SNenergy, 
                                                  Gal.j)
    
    elif Gal.IMF_type == 'evolving_cloud':
        result = get_SNenergy_past_evolving_cloud(simParam.firstPrevStepInStep[Gal.j], 
                                                  simParam.numPrevStepsInStep[Gal.j],
                                                  simParam.prevStepIndex[Gal.j],
                                                  Gal.maxStarMassBins, 
                                                  simParam.totalNumIndexes, 
                                                  Gal.starGrid, 
                                                  Gal.SNenergy, 
                                                  Gal.fmassiveClList, 
                                                  Gal.fmassiveIndexes, 
                                                  Gal.metalMask,
                                                  Gal.j)
          
    elif Gal.IMF_type == 'evolving':
        result = get_SNenergy_past_evolving(simParam.firstPrevStepInStep[Gal.j], 
                                            simParam.numPrevStepsInStep[Gal.j],
                                            simParam.prevStepIndex[Gal.j],
                                            simParam.totalNumIndexes,  
                                            simParam.SN_IMFbins, 
                                            simParam.fmassiveList, 
                                            simParam.cutoffMassList, 
                                            Gal.starGrid, 
                                            Gal.SNenergy, 
                                            Gal.j)
        
    elif Gal.IMF_type == 'evolving2':
        result = get_SNenergy_past_evolving2(simParam.firstPrevStepInStep[Gal.j], 
                                             simParam.numPrevStepsInStep[Gal.j],
                                             simParam.prevStepIndex[Gal.j],
                                             simParam.totalNumIndexes,  
                                             simParam.SN_IMFbins, 
                                             Gal.fmassiveHistory, 
                                             simParam.cutoffMass, 
                                             Gal.SFRHistory, 
                                             Gal.SNenergy, 
                                             Gal.j)
        
    return result

# The overhead gained by parallelising this made it not worth it as it was already not very expensive
def get_SNenergy_past_salpeter(firstPrevStepInStep, numPrevStepsInStep,prevStepIndex, SFRHistory, SNenergy, j):
    result = 0
    # Looping over previous ~30 Myr of timesteps, as no SN occur before this.
    for prevStep in prange(firstPrevStepInStep,firstPrevStepInStep+numPrevStepsInStep):
        prevStepNumber = prevStep - firstPrevStepInStep
        # Look up the Energy released from SN in prevStep per stellar mass formed, and multiply by stellar mass formed
        result += SNenergy[prevStepIndex + prevStepNumber] * SFRHistory[prevStep]
    return result

@njit(parallel = True, fastmath = True)
def get_SNenergy_past_salpeter_cloud(firstPrevStepInStep, numPrevStepsInStep,prevStepIndex, maxStarMassBins, totalNumIndexes, starGrid, SNenergy, j):
    result = 0
    # Looping over previous ~30 Myr of timesteps, as no SN occur before this.
    for prevStep in prange(firstPrevStepInStep,firstPrevStepInStep+numPrevStepsInStep):
        prevStepNumber = prevStep - firstPrevStepInStep
        mass = starGrid[prevStep].sum(axis = 0)
        # Loop over IMF upper mass limits
        for m in prange(maxStarMassBins):
            # Look up the Energy released from SN in prevStep per stellar mass formed, for IMF with the m'th upper mass limit, and multiply by stellar mass formed in prevstep with this IMF
            result += SNenergy[prevStepIndex + prevStepNumber + m * totalNumIndexes] * mass[m]
    return result

@njit(parallel = True, fastmath = True)
def get_SNenergy_past_evolving_cloud(firstPrevStepInStep, numPrevStepsInStep,prevStepIndex, maxStarMassBins, totalNumIndexes, starGrid, SNenergy, fmassiveClList, fmassiveIndexes, metalMask, j):
    result = 0
    NumZbins = starGrid.shape[1]
    highMassOffset   = fmassiveClList.shape[0] * maxStarMassBins * totalNumIndexes
    fmassiveOffset = maxStarMassBins * totalNumIndexes
    # Looping over previous ~30 Myr of timesteps, as no SN occur before this.
    for prevStep in prange(firstPrevStepInStep,firstPrevStepInStep+numPrevStepsInStep):
        prevStepNumber = prevStep - firstPrevStepInStep
        starsFormed = starGrid[prevStep][metalMask[prevStep]]
        fIndex    = fmassiveIndexes[prevStep][metalMask[prevStep]]
        f_massive = fmassiveClList[fIndex]
        # Loop over metallicities
        for i in prange(starsFormed.shape[0]):
            # Loop over IMF upper mass limits
            for k in prange(maxStarMassBins):
                index   = prevStepIndex + prevStepNumber + fIndex[i] * fmassiveOffset + k * totalNumIndexes
                # Look up the Energy released from SN in prevStep per stellar mass formed, for IMF with the k'th upper mass limit and the value of fmassive set by the i'th metallicity, and multiply by stellar mass formed in prevstep with this IMF
                result += starsFormed[i,k] * (SNenergy[index] * (1 - f_massive[i]) + SNenergy[index + highMassOffset] * f_massive[i])
        
    return result

@njit(parallel = True, fastmath = True)
def get_SNenergy_past_evolving(firstPrevStepInStep, numPrevStepsInStep,prevStepIndex, totalNumIndexes, SN_IMFbins, fmassiveList, cutoffMassList, starGrid, SNenergy, j):
    result = 0
    NumZbins = fmassiveList.shape[1]
    # Looping over previous ~30 Myr of timesteps, as no SN occur before this.
    for prevStep in prange(firstPrevStepInStep,firstPrevStepInStep+numPrevStepsInStep):
        prevStepNumber = prevStep - firstPrevStepInStep
        f_massive      = fmassiveList[prevStep]
        CMass          = cutoffMassList[prevStep]
        # Loop over metallicities
        for i in prange(NumZbins):
            # Look up the Energy released from SN in prevStep per stellar mass formed, for IMF with the value of fmassive set by the i'th metallicity, and multiply by stellar mass formed in prevstep with this IMF
            result += SNenergy[prevStepIndex + prevStepNumber + (CMass[i]-1) * totalNumIndexes] * starGrid[prevStep,i] * (1-f_massive[i]) + SNenergy[prevStepIndex + prevStepNumber + (CMass[i]-1) * totalNumIndexes + SN_IMFbins * totalNumIndexes] * starGrid[prevStep,i] * f_massive[i]
    return result

@njit(parallel = True, fastmath = True)
def get_SNenergy_past_evolving2(firstPrevStepInStep, numPrevStepsInStep,prevStepIndex, totalNumIndexes, SN_IMFbins, fmassiveHistory, cutoffMass, SFRHistory, SNenergy, j):
    result = 0
    CMasses = get_cutoff_mass(fmassiveHistory[firstPrevStepInStep:firstPrevStepInStep+numPrevStepsInStep],cutoffMass)
    # Looping over previous ~30 Myr of timesteps, as no SN occur before this.
    for prevStep in prange(firstPrevStepInStep,firstPrevStepInStep+numPrevStepsInStep):
        prevStepNumber = prevStep - firstPrevStepInStep
        f_massive      = fmassiveHistory[prevStep]
        CMass          = int(CMasses[prevStepNumber])
        # Look up the Energy released from SN in prevStep per stellar mass formed, for IMF with the value of fmassive in prevStep, and multiply by stellar mass formed in prevstep
        result        += SNenergy[prevStepIndex + prevStepNumber + (CMass - 1) * totalNumIndexes] * SFRHistory[prevStep] * (1-f_massive) + SNenergy[prevStepIndex + prevStepNumber + (CMass - 1) * totalNumIndexes + SN_IMFbins * totalNumIndexes] * SFRHistory[prevStep] * f_massive 
    return result
        
#def calc_SNenergy_instant(Gal,simParam):
#    result = 0
#    if Gal.IMF_type == 'evolving':
#        NSN = np.sum(0.1*IMF_evolving(MM[79:500],simParam.substepRedshiftList[Gal.j-1],ZZ).sum(axis = 0) * Gal.starGrid[Gal.j-1])
#        result = 5e7 * invh0 * NSN
#    if Gal.IMF_type == 'salpeter':
#        NSN = 0.1 * IMF_Salpeter(MM[79:500]).sum() * Gal.SFRHistory[Gal.j-1]
#        result = 5e7 * invh0 * NSN
#    return result
        
### Create lookup table for SNenergy to be used in the above functions
def calc_SNenergy_delayed(substepTimeList,MSN,invh0,SN_IMFbins,IMF_type = 'salpeter'):
    endStep = substepTimeList.shape[0] #Number of subtimesteps
    
    #Number of substeps to save information for. We need not save a bunch of zeros, 
    #so we just get the number of substeps that could reasonably contain a supernova (~30 Myr back)
    numStepsToSave = int(get_SNtime(MSN))+1 
    
    #Here we find the actual number of subtimesteps for each snapshot to save. 
    #the first ones dont have the 30 or so previous substeps so they get fewer.
    #We also find the earliest timestep to be saved for each timestep.
    numPrevStepsInStep = np.zeros(endStep+1,int)
    firstPrevStepInStep = np.zeros(endStep+1,int)
    for step in range(1,endStep):
        numPrevStepsInStep[step] = np.min([numStepsToSave,step])
        firstPrevStepInStep[step] = step-numPrevStepsInStep[step]
    totalNumIndexes = np.sum(numPrevStepsInStep)
    
    print(f'numPrevStepsInStep:  {numPrevStepsInStep}')
    print(f'firstPrevStepInStep: {firstPrevStepInStep}')
    print(f'totalNumIndexes:     {totalNumIndexes}')
    
    ### Depending on the IMF the table looks different, so check for IMF first.
    if IMF_type == 'salpeter':
        SNenergyDelayed = np.zeros(totalNumIndexes)
        for step in range(1,endStep):
            SNenergyDelayed[np.sum(numPrevStepsInStep[:step]):np.sum(numPrevStepsInStep[:step])+numPrevStepsInStep[step]] = 5e7 * invh0 * get_SNfraction_per_Msun(substepTimeList[step],substepTimeList[step-1],substepTimeList[firstPrevStepInStep[step]+1:firstPrevStepInStep[step]+numPrevStepsInStep[step]+1],substepTimeList[firstPrevStepInStep[step]:firstPrevStepInStep[step]+numPrevStepsInStep[step]],-2.35,0.1,100,MSN,50)
    
    elif IMF_type == 'salpeter_cloud':
        MaxStarMasses = np.array([10,20,30,40,50,60,70,80,90,100])
        numMaxStarMasses = MaxStarMasses.shape[0]
        SNenergyDelayed = np.zeros(totalNumIndexes * numMaxStarMasses)
        for i,mass in enumerate(MaxStarMasses):
            for step in range(1,endStep):
                SNenergyDelayed[np.sum(numPrevStepsInStep[:step]) + i * totalNumIndexes: np.sum(numPrevStepsInStep[:step]) + numPrevStepsInStep[step] + i * totalNumIndexes] = 5e7 * invh0 * get_SNfraction_per_Msun(substepTimeList[step], substepTimeList[step-1], substepTimeList[firstPrevStepInStep[step]+1: firstPrevStepInStep[step]+numPrevStepsInStep[step]+1], substepTimeList[firstPrevStepInStep[step]:firstPrevStepInStep[step]+numPrevStepsInStep[step]], -2.35, 0.1, mass, MSN, 50)
    
    elif IMF_type == 'evolving_cloud':
        MaxStarMasses = np.array([10,20,30,40,50,60,70,80,90,100])
        numMaxStarMasses = MaxStarMasses.shape[0]
        numMassBins = len(SN_IMFbins)
        SNenergyDelayed = np.zeros(totalNumIndexes * numMaxStarMasses * numMassBins * 2)
        
        lowMslope = -2.35
        highMslope = -1
        MSNlowIndex = np.argwhere(SN_IMFbins>8)[-1,0]
        MSNupIndex = np.argwhere(SN_IMFbins>50)[0,0]
        MSNup = 50
        MSNlow = 8
        for CMass in range(numMassBins):
            for iMax,Mmax in enumerate(MaxStarMasses):
                CMassInMsun = SN_IMFbins[CMass];
                if (CMassInMsun >= MSNlowIndex): ### Low Mass Part
                    MaxSNmass = np.min([MSNup,CMassInMsun,Mmax])
                    for step in range(1,endStep):
                        indexMin = np.sum(numPrevStepsInStep[:step]) + iMax * totalNumIndexes + CMass * numMaxStarMasses * totalNumIndexes
                        indexMax = indexMin +numPrevStepsInStep[step]
                        SNenergyDelayed[indexMin:indexMax] = 5e7 * invh0 * get_SNfraction_per_Msun(substepTimeList[step], substepTimeList[step-1], substepTimeList[firstPrevStepInStep[step]+1:firstPrevStepInStep[step]+numPrevStepsInStep[step]+1], substepTimeList[firstPrevStepInStep[step]:firstPrevStepInStep[step]+numPrevStepsInStep[step]], lowMslope, 0.1, CMassInMsun, MSN, MaxSNmass);
                if( CMassInMsun <= MSNupIndex) * (CMassInMsun <=Mmax): ###  High Mass Part
                    MinSNmass = np.max([MSN, CMassInMsun]);
                    MaxSNmass = np.min([MSNup, Mmax]);
                    for step in range(1,endStep):
                        indexMin = np.sum(numPrevStepsInStep[:step]) + iMax * totalNumIndexes + CMass * numMaxStarMasses * totalNumIndexes + numMassBins * numMaxStarMasses * totalNumIndexes
                        indexMax = indexMin + numPrevStepsInStep[step]
                        SNenergyDelayed[indexMin:indexMax] = 5e7 * invh0 * get_SNfraction_per_Msun(substepTimeList[step], substepTimeList[step-1], substepTimeList[firstPrevStepInStep[step]+1:firstPrevStepInStep[step]+numPrevStepsInStep[step]+1], substepTimeList[firstPrevStepInStep[step]:firstPrevStepInStep[step]+numPrevStepsInStep[step]], highMslope, CMassInMsun, Mmax, MinSNmass, MaxSNmass); 
    
    else: #For the evolving and evolving2 IMFs
        numMassBins = SN_IMFbins
        SNenergyDelayed = np.zeros(totalNumIndexes*numMassBins*2)
        lowMslope = -2.35
        highMslope = -1
        MSNlowIndex = int(MSN)
        MSNupIndex = 50
        MSNup = 50        
        for CMass in range(1,numMassBins):
            CMassInMsun = CMass;
            if(CMass >= MSNlowIndex): # Low Mass Part
                MaxSNmass = np.min([MSNup,CMassInMsun])
                for step in range(1,endStep):
                    indexMin = np.sum(numPrevStepsInStep[:step]) + (CMass-1) * totalNumIndexes
                    indexMax = indexMin +numPrevStepsInStep[step]
                    SNenergyDelayed[indexMin:indexMax] = 5e7 * invh0 * get_SNfraction_per_Msun(substepTimeList[step], substepTimeList[step-1], substepTimeList[firstPrevStepInStep[step]+1:firstPrevStepInStep[step]+numPrevStepsInStep[step]+1] ,substepTimeList[firstPrevStepInStep[step]:firstPrevStepInStep[step]+numPrevStepsInStep[step]], lowMslope, 0.1, CMassInMsun, MSN, MaxSNmass);  
            if(CMass != numMassBins)*( CMass <= MSNupIndex): #High Mass Part
                MinSNmass = np.max([MSN, CMassInMsun]);
                for step in range(1,endStep):
                    indexMin = np.sum(numPrevStepsInStep[:step]) + (CMass-1) * totalNumIndexes + numMassBins*totalNumIndexes
                    indexMax = indexMin + numPrevStepsInStep[step]
                    SNenergyDelayed[indexMin:indexMax] = 5e7 * invh0 * get_SNfraction_per_Msun(substepTimeList[step], substepTimeList[step-1], substepTimeList[firstPrevStepInStep[step]+1:firstPrevStepInStep[step]+numPrevStepsInStep[step]+1], substepTimeList[firstPrevStepInStep[step]:firstPrevStepInStep[step]+numPrevStepsInStep[step]], highMslope, CMassInMsun, 100, MinSNmass, MSNup); 

    return SNenergyDelayed, numPrevStepsInStep, firstPrevStepInStep, totalNumIndexes
        
#This function is mostly an artifact of a previous version, there used to be more going on in here
def get_SNfraction_per_Msun(tsnap,tsnapPrev,tprevSnap,tprevprevSnap,IMFslope,minMass,maxMass,SNminMass,SNmaxMass):
    return calc_SNfraction_per_Msun(tsnap,tsnapPrev,tprevSnap,tprevprevSnap,SNminMass,SNmaxMass,minMass,maxMass,IMFslope)

### Get the Mass of a star that goes SN after a time timeInMyr
def get_SNmass(timeInMyr):
    return (0.8333e-3*(timeInMyr-3))**-0.540541

### get the lifetime of a star of mass M
def get_SNtime(M):
    return 1.2e3 * M**(-1.85) + 3

### Calculate The number of SN per stellar mass formed in a given stellar mass interval, Mlow-Mup, for a given IMF slope
### The method is described in appendix B of Hutter+2023 ( https://doi.org/10.1093/mnras/stad2230 ) ((Astraeus VIII))
def calc_SNfraction_per_Msun(tsnap,tsnapPrev,tprevSnap,tprevprevSnap,MSNlow,MSNup,Mlow,Mup,slope):
    tSNlow = get_SNtime(MSNlow)
    tup    = get_SNtime(MSNup)
    result = 0
    
    resultTminLow = np.zeros_like(tprevSnap);
    tminLowLow = np.zeros_like(tprevSnap) + tsnapPrev
    tminLowUp  = np.zeros_like(tprevSnap) + tsnap
    tminLowLow[tsnapPrev < tprevprevSnap + tup] = tprevprevSnap[tsnapPrev < tprevprevSnap + tup]+tup
    tminLowUp[tsnap > tprevprevSnap + tSNlow] = tprevprevSnap[tsnap > tprevprevSnap + tSNlow] + tSNlow
    
    resultTminUp = np.zeros_like(tprevSnap);
    tminUpLow = np.zeros_like(tprevSnap) + tsnapPrev
    tminUpUp  = np.zeros_like(tprevSnap) + tsnap
    tminUpLow[tsnapPrev < tprevprevSnap + tSNlow] = tprevprevSnap[tsnapPrev < tprevprevSnap + tSNlow]+tSNlow
    tminUpUp[tsnap > tprevSnap + tSNlow] = tprevSnap[tsnap > tprevSnap + tSNlow] + tSNlow
    
    resultTmaxLow = np.zeros_like(tprevSnap);
    tmaxLowLow = np.zeros_like(tprevSnap) + tsnapPrev
    tmaxLowUp  = np.zeros_like(tprevSnap) + tsnap
    tmaxLowLow[tsnapPrev < tprevprevSnap + tup] = tprevprevSnap[tsnapPrev < tprevprevSnap + tup]+tup
    tmaxLowUp[tsnap > tprevSnap + tup] = tprevSnap[tsnap > tprevSnap + tup] + tup
    
    resultTmaxUp = np.zeros_like(tprevSnap);
    tmaxUpLow = np.zeros_like(tprevSnap) + tsnapPrev
    tmaxUpUp  = np.zeros_like(tprevSnap) + tsnap
    tmaxUpLow[tsnapPrev < tprevSnap + tup] = tprevSnap[tsnapPrev < tprevSnap + tup]+tup
    tmaxUpUp[tsnap > tprevSnap + tSNlow] = tprevSnap[tsnap > tprevSnap + tSNlow] + tSNlow
    
    SFR = 1. / (tprevSnap - tprevprevSnap); # in Msun/sec
    preFactor = 0.;
    
    if slope!=-1:
        exponent = 0.540541 * (-slope-1);
        preFactor= (2+slope) / (1+slope) * (0.833333e-3)**exponent / (Mlow**(2+slope)-Mup **(2+slope))
        
        #/* tmin part of the integration */
        mask = tminLowLow < tminLowUp
        resultTminLow[mask] = (np.power(tminLowUp[mask] - tprevprevSnap[mask] - 3., exponent + 1.) - np.power(tminLowLow[mask] - tprevprevSnap[mask] - 3., exponent + 1.)) / (exponent + 1.);
        
        mask = tminUpLow < tminUpUp
        resultTminUp[mask] = np.power(tSNlow - 3., exponent) * (tminUpUp[mask] - tminUpLow[mask]);
        
        #/* tmax part of the integration */
        mask = tmaxLowLow < tmaxLowUp
        resultTmaxLow[mask] = np.power(tup - 3., exponent) * (tmaxLowUp[mask] - tmaxLowLow[mask]);
        
        mask = tmaxUpLow < tmaxUpUp
        resultTmaxUp[mask] = (np.power(tmaxUpUp[mask] - tprevSnap[mask] - 3., exponent + 1.) - np.power(tmaxUpLow[mask] - tprevSnap[mask] - 3., exponent + 1.)) / (exponent + 1.);
        
        result = SFR * preFactor * (resultTminLow + resultTminUp - resultTmaxLow - resultTmaxUp);
    else:
        preFactor = -0.540541/(Mlow-Mup);
        
        #/* tmin part of the integration */
        mask = tminLowLow < tminLowUp
        resultTminLow[mask] = (tminLowUp[mask] - tprevprevSnap[mask] - 3.)*np.log(tminLowUp[mask] - tprevprevSnap[mask] - 3.)-tminLowUp[mask] - (tminLowLow[mask] - tprevprevSnap[mask] - 3.)*np.log(tminLowLow[mask] - tprevprevSnap[mask] - 3.) + tminLowLow[mask];
        
        mask = tminUpLow < tminUpUp
        resultTminUp[mask] = np.log(tSNlow - 3.) * (tminUpUp[mask] - tminUpLow[mask]);    
        
        #/* tmax part of the integration */
        mask = tmaxLowLow < tmaxLowUp
        resultTmaxLow[mask] = np.log(tup - 3.) * (tmaxLowUp[mask] - tmaxLowLow[mask]);
        
        mask = tmaxUpLow < tmaxUpUp
        resultTmaxUp[mask] = (tmaxUpUp[mask] - tprevSnap[mask] - 3.)*np.log(tmaxUpUp[mask] - tprevSnap[mask] - 3.) - tmaxUpUp[mask] - (tmaxUpLow[mask] - tprevSnap[mask] - 3.)*np.log(tmaxUpLow[mask] - tprevSnap[mask] - 3.) + tmaxUpLow[mask];
        
        result = SFR * preFactor * (resultTminLow + resultTminUp - resultTmaxLow - resultTmaxUp); 
    return result