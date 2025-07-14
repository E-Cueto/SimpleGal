import numpy as np
import scipy.interpolate as si
import os
import phys_const as pc
substepRedshiftList = np.loadtxt('input_data/substepRedshiftList.dat')[:1114]
substepTimeList     = np.loadtxt('input_data/substepTimeList.dat')[:1114]

class SFRstochasticity:
    def __init__(self):
        self.sbTime     = np.array([])
        self.sbRedshift = np.array([])
        self.sbMstar    = np.array([])
        self.sbMhalo    = np.array([])
        self.sbLUV      = np.array([])
        self.qTime      = np.array([])
        self.qRedshift  = np.array([])
        self.qMstar     = np.array([])
        self.qMhalo     = np.array([])
        self.endRedshift= np.array([])
        self.endMstar   = np.array([])
        self.endMhalo   = np.array([])
        self.endMgas    = np.array([])
        self.endMgasIni = np.array([])
        self.endLUV     = np.array([])
        self.sbStep     = np.array([])
        self.qStep      = np.array([])
        self.endStep    = np.array([])
        self.fduty      = np.array([])
        self.sbMdust    = np.array([])
        self.endMdust   = np.array([])
    def add_sim(self,Dict,MhaloList,substepRedshiftList,substepTimeList,betaList):
        MH0_list = np.sort(np.array(list(set(Dict['catalogue'][:,0]))))
        for i in range(Dict['SFR'].shape[0]):
            SFR = Dict['SFR'][i,:int(Dict['catalogue'][i,6])]
            Mstar = Dict['Mstar'][i,:int(Dict['catalogue'][i,6])]
            Mgas  = Dict['Mgas'][i,:int(Dict['catalogue'][i,6])]
            MgasIni  = Dict['MgasIni'][i,:int(Dict['catalogue'][i,6])]
            LUV   = Dict['LUV'][i,:int(Dict['catalogue'][i,6])+2]
            Mdust = Dict['Mdust'][i,:int(Dict['catalogue'][i,6])+2]

            if Dict['catalogue'][0].shape == 7:
                betaIndex = np.argwhere(np.isclose(-0.75,betaList))[0,0]
                Mhalo = MhaloList[betaIndex][MH0_list == Dict['catalogue'][i,0]][0]
            else:
                betaIndex = np.argwhere(np.isclose(Dict['catalogue'][i,7],betaList))[0,0]
                Mhalo = MhaloList[betaIndex][MH0_list == Dict['catalogue'][i,0]][0]
            a = np.argwhere((SFR[1:]>0)*(SFR[:-1]==0)).T[0]+1
            b = np.argwhere((SFR[1:]==0)*(SFR[:-1]>0)).T[0]+1
            if np.min(SFR) == 0:
                last = np.where(SFR==0.)[0][-1]
                self.fduty      = np.concatenate((self.fduty,[(np.sum(SFR[:last]>0))/(substepTimeList[int(last)]-substepTimeList[int(Dict['catalogue'][i,5])])]))
            else:
                self.fduty      = np.concatenate((self.fduty,np.array([1.])))
            self.qMstar     = np.concatenate((self.qMstar ,    Mstar[b]))
            self.qMhalo     = np.concatenate((self.qMhalo ,    Mhalo[b]))
            self.qStep      = np.concatenate((self.qStep,b))
            
            if (SFR[-1]>0):
                if SFR[0]>0:
                    a = np.concatenate((np.array([0]),a))
                self.sbTime     = np.concatenate((self.sbTime,     substepTimeList[b]-substepTimeList[a[:-1]]))
                self.sbRedshift = np.concatenate((self.sbRedshift, substepRedshiftList[a[:-1]]))
                self.qTime      = np.concatenate((self.qTime,      substepTimeList[a[1:]]-substepTimeList[b]))
                self.qRedshift  = np.concatenate((self.qRedshift,  substepRedshiftList[b]))
                self.sbMstar    = np.concatenate((self.sbMstar,    Mstar[a[:-1]]))
                self.sbMhalo    = np.concatenate((self.sbMhalo,    Mhalo[a[:-1]]))
                self.sbLUV      = np.concatenate((self.sbLUV,      LUV[a[:-1]+2]))
                self.sbMdust    = np.concatenate((self.sbMdust,    Mdust[a[:-1]+2]))
                self.endMdust   = np.concatenate((self.endMdust,   [Mdust[a[-1]+2]]))
                self.endMhalo   = np.concatenate((self.endMhalo,   [Mhalo[a[-1]]]))
                self.endMstar   = np.concatenate((self.endMstar,   [Mstar[a[-1]]]))
                self.endLUV     = np.concatenate((self.endLUV  ,   [LUV[a[-1]+2]]))
                self.endRedshift= np.concatenate((self.endRedshift,[substepRedshiftList[a[-1]]]))
                self.endMgas    = np.concatenate((self.endMgas,    [Mgas[a[-1]]]))
                self.endMgasIni = np.concatenate((self.endMgasIni, [MgasIni[a[-1]]]))
                self.sbStep     = np.concatenate((self.sbStep,a[:-1]))
                self.endStep    = np.concatenate((self.endStep,[a[-1]]))
            else:
                if SFR[0]>0:
                    a = np.concatenate((np.array([0]),a))
                self.sbTime     = np.concatenate((self.sbTime,     substepTimeList[b]-substepTimeList[a]))
                self.sbRedshift = np.concatenate((self.sbRedshift, substepRedshiftList[a]))
                self.qTime      = np.concatenate((self.qTime,      substepTimeList[a[1:]]-substepTimeList[b[:-1]]))
                self.qRedshift  = np.concatenate((self.qRedshift,  substepRedshiftList[b[:-1]]))
                self.sbMstar    = np.concatenate((self.sbMstar,    Mstar[a]))
                self.sbMhalo    = np.concatenate((self.sbMhalo,    Mhalo[a]))
                self.sbLUV      = np.concatenate((self.sbLUV,      LUV[a+2]))
                self.endMhalo   = np.concatenate((self.endMhalo,   [-1]))
                self.endMstar   = np.concatenate((self.endMstar,   [-1]))
                self.endRedshift= np.concatenate((self.endRedshift,[-1]))
                self.endMgas    = np.concatenate((self.endMgas,    [-1]))
                self.endMgasIni = np.concatenate((self.endMgasIni, [-1]))
                self.sbMdust    = np.concatenate((self.sbMdust,    Mdust[a+2]))
                self.endMdust   = np.concatenate((self.endMdust,  [-1]))
                self.endLUV     = np.concatenate((self.endLUV,    [-1]))
                self.sbStep     = np.concatenate((self.sbStep,a))
                self.endStep    = np.concatenate((self.endStep,[-1]))
            

def load_file_from_type_and_name(Dir,Type,name):
    Arr = np.fromfile(Dir+Type+name+'.dat')
    Arr = np.reshape(Arr,(len(Arr)//1114,1114))
    return Arr
def load_file_from_file_name(Dir,name):
    Arr = np.fromfile(Dir+name)
    Arr = np.reshape(Arr,(len(Arr)//1114,1114))
    return Arr
def load_catalogue(Dir,name):
    Arr = np.fromfile(Dir+'catalogue_'+name)
    if len(Arr)%7==0:
        mod = 7
    else:
        mod = 8
    Arr = np.reshape(Arr,(len(Arr)//mod,mod))
    return Arr
def load_type(Dir,Type):
    allFiles = os.listdir(Dir)
    theseFiles = []
    for i in range(len(allFiles)):
        if allFiles[i][:len(Type)] == Type:
            theseFiles.append(allFiles[i])
    theseData = []
    for f in theseFiles:
        theseData.append(load_file_from_file_name(Dir,f))
    return np.array(theseData)
    
def load_name(Dir,MH0_list,name,Mhalo,betaList):
    allFiles = os.listdir(Dir)
    theseFiles = []
    for i in range(len(allFiles)):
        if (allFiles[i][-len(name):] == name)*(allFiles[i][:9]!='catalogue'):
            theseFiles.append(allFiles[i])
    theseData = {}
    for f in theseFiles:
        theseData[f[:len(f)-len(name)-1]] = load_file_from_file_name(Dir,f)
    theseData['catalogue'] = load_catalogue(Dir,name)
    theseData['name'] = name
    theseData['hasSmooth'] = False
    print(name)
    theseData['stochasticity'] = SFRstochasticity()
    theseData['stochasticity'].add_sim(theseData,Mhalo,substepRedshiftList,substepTimeList,betaList)
    theseData['MH0'] = MH0_list
    return theseData
    
def make_neat_data(name,inDir,Dir,MH0_list ,betaList, do_delta = False,UVLFzvals = np.array([5,6,7,8,9,10,11,12,13,14,15]),UVLFdeltazVals = np.array([0.1,0.2,0.5,1]),UVLFdeltaMUV = 0.2,zvals2d=np.array([5,7,9,11,13,15]),fdutyMhaloLimits = np.array([1e8,10**8.5,1e9,10**9.5,1e10,10**10.5,1e11])):
    Mhalo = get_mhalo_list(MH0_list,substepRedshiftList,betaList)
    print(name,flush=True)
    Dict = load_name(inDir, MH0_list, name, Mhalo, betaList)
    print('loaded dict',flush = True)
    make_end_properties(Dict, Dir, MH0_list, betaList, Mhalo)
    print('make end properties',flush = True)
    make_uvlf(Dict, Dir, MH0_list, betaList, Mhalo,UVLFzvals, UVLFdeltazVals, UVLFdeltaMUV)
    print('made uvlf',flush = True)
    make_2d_hist(Dict, Dir, MH0_list, betaList, Mhalo)
    print('made 2d hist',flush = True)
    make_2d_hist_z(Dict, Dir, MH0_list, betaList, Mhalo, zvals2d)
    print('made 2d hist z',flush = True)
    make_fduty(Dict, Dir, MH0_list ,betaList, Mhalo, fdutyMhaloLimits)
    print('made fduty',flush = True)
    #if 'smoothSFR' in Dict.keys():
    #    make_delta(Dict, Dir, MH0_list, betaList, Mhalo)
    #    print('made delta',flush = True)
        
def make_delta(Dict, Dir, MH0_list, betaList, Mhalo):
    deltaLUV = np.zeros(1114)
    deltaSFR = np.zeros(1114)
    deltaLUVtmp = np.zeros((len(Dict['SFR']),1114))
    deltaSFRtmp = np.zeros((len(Dict['SFR']),1114))
    NumHaloBins = len(Dict['MH0'])
    for m in range(len(Dict['SFR'])):
        haloIndex = np.argwhere(np.isclose(Dict['catalogue'][m,0],Dict['MH0']))[0,0]
        betaIndex = np.argwhere(np.isclose(Dict['catalogue'][m,7],betaList))[0,0]
        J = int(Dict['catalogue'][m,6])
        SFR = Dict['SFR'][m]
        LUV = Dict['LUV'][m]
        LUVsmooth = recurse_roll_mean(Dict['smoothLUV'][haloIndex + NumHaloBins * betaIndex],500)
        SFRsmooth = recurse_roll_mean(Dict['smoothSFR'][haloIndex + NumHaloBins * betaIndex],500)
        if np.min(SFR[:J]) == 0:
            last = np.where(SFR[:J]==0.)[0][-1]
        else:
            last = 1114
        mask = (SFR > 0)*(np.arange(1114)<=last)
        deltaSFRtmp[m][mask] = SFR[mask]/SFRsmooth[mask]
        deltaSFRtmp[m][~mask] = np.nan
        deltaLUVtmp[m][mask] = LUV[mask]/LUVsmooth[mask]
        deltaLUVtmp[m][~mask] = np.nan
    deltaSFR = np.nanmean(deltaSFRtmp,axis = 0)
    deltaLUV = np.nanmean(deltaLUVtmp,axis = 0)
    
    name = Dict['name'][:-4]
    if not np.isin(name,os.listdir(Dir)):
        os.mkdir(Dir + name)
    deltaSFR.tofile(Dir+name+'/deltaSFR.dat')
    deltaLUV.tofile(Dir+name+'/deltaLUV.dat')
    
def make_end_properties(Dict,Dir,MH0_list ,betaList,Mhalo):
    stoch = Dict['stochasticity']
    List = np.zeros((9,len(stoch.endRedshift)))
    
    name = Dict['name'][:-4]
    if not np.isin(name,os.listdir(Dir)):
        os.mkdir(Dir + name)
    List[0] = stoch.endRedshift
    List[1] = stoch.endStep
    List[2] = stoch.endMstar
    List[3] = stoch.endMhalo
    List[4] = stoch.endMgas
    List[5] = stoch.endMgasIni
    List[6] = stoch.endMdust
    List[7] = stoch.endLUV
    List[8] = stoch.fduty
    List.tofile(Dir+name+'/end_properties.dat')
    Dict['catalogue'].tofile(Dir+name+'/catalogue.dat')
    
def make_uvlf(Dict,Dir,MH0_list ,betaList,Mhalo,zvals = np.array([5,6,7,8,9,10,11,12,13,14,15]),deltazVals = np.array([0.1,0.2,0.5,1]),deltaMUV = 0.2):
    MUVmin,MUVmax = -30,-6
    MUVlist = np.arange(MUVmin,MUVmax,deltaMUV)
    UVLFlist  = np.zeros((len(zvals),len(deltazVals)+2,len(MUVlist)))
    UVLFalist = np.zeros((len(zvals),len(deltazVals)+2,len(MUVlist)))
    
    name = Dict['name'][:-4]
    if not np.isin(name,os.listdir(Dir)):
        os.mkdir(Dir + name)
    
    haloIndexes = np.zeros(Dict['LUV'].shape[0],int)
    betaIndexes = np.zeros(Dict['LUV'].shape[0],int)
    for run in range(Dict['LUV'].shape[0]):
        haloIndexes[run] = np.argwhere(np.isclose(Dict['catalogue'][run,0],Dict['MH0']))[0,0]
        betaIndexes[run] = np.argwhere(np.isclose(Dict['catalogue'][run,7],betaList))[0,0]
    LUV = Dict['LUV']
    Mh   = Mhalo[betaIndexes,haloIndexes]
    MUV  = get_Mag_from_L(np.log10(Dict['LUV']))
    MUVa = np.zeros_like(MUV)
    for run in range(MUV.shape[0]):
        MUVa[run] = get_Mag_from_L_attenuated(np.log10(Dict['LUV'][run]),substepRedshiftList,Dict['Mdust'][run],Mh[run])
    HMF = np.loadtxt('input_data/mVector_VSDMPL_z4.5.txt',skiprows=12)
    titles = [ r"$m$", r"$\sigma$", r"$ln(1/sigma)$", r"$n_{eff}$", r"$f(\sigma)$", r"$dn/dm$",r"$dn/dlnm$",r"$dn/dlog10m$",
              r"$n(>m)$",r"$\rho(>m)$",r"$\rho(<m)$",r"$Lbox(N=1)$"]

    ### Calculate halo mass function at relevant masses
    logM = np.log10(HMF[:,0])                  ### HMF[:,0] is halo masses
    dndlogM= HMF[:,7]                          ### HMF[:,7] is dn/dlog10m
    dlogndlogM=np.log10(HMF[:,7])              ### Get that baby log-scaled
    interp = si.CubicSpline(logM,dlogndlogM)   ### Interpolate because we dont have right ones here
    MH4_5s = np.log10(calc_Mhalo(Dict['catalogue'][:,0],4.5,Dict['catalogue'][:,7]))
    binsizeMhalo = np.zeros_like(MH4_5s)
    MH4_5_list=np.sort(list(set(MH4_5s)))
    deltaMH=0.5*(np.roll(MH4_5_list,-1)-(np.roll(MH4_5_list,1)))
    deltaMH[0] = (MH4_5_list[1])-(MH4_5_list[0])
    deltaMH[-1] = (MH4_5_list[-1])-(MH4_5_list[-2])

    for run in range(MH4_5s.shape[0]):
        MhaloIndex = np.argwhere(np.isclose(MH4_5s[run],MH4_5_list))[0,0]
        binsizeMhalo[run] = deltaMH[MhaloIndex]
    NH4_5s = 10 ** (interp(MH4_5s) + np.log10(binsizeMhalo))
    
    for i,dz in enumerate(deltazVals):
        zindexes = []
        for j in range(len(zvals)):
            #zindexes.append(np.array([1114-where(substepRedshiftList[::-1],zvals[i])]))
            zindexes.append(np.argwhere((zvals[j]+0.5 * dz >substepRedshiftList)*(zvals[j]-0.5 * dz <=substepRedshiftList)*(zvals[j]>=5)).T[0])
        for zval in range(len(zvals)):
                ### Loop over runs
                for run in range(len(MUV)):
                    ### Loop over redshift bins
                    for zbin in zindexes[zval]:
                        if LUV[run,zbin]>0:
                            index = where(MUVlist - 0.5 * deltaMUV,MUV[run][zbin])
                            UVLFlist[zval,i+2,index] += NH4_5s[run] / zindexes[zval].shape[0]
                        
                            indexa = where(MUVlist - 0.5 * deltaMUV,MUVa[run][zbin])
                            UVLFalist[zval,i+2,indexa] += NH4_5s[run] / zindexes[zval].shape[0]
    for zval in range(len(zvals)):
        UVLFlist[zval,0] = MUVlist - 0.5 * deltaMUV
        UVLFlist[zval,1] = MUVlist + 0.5 * deltaMUV
        UVLFalist[zval,0] = MUVlist - 0.5 * deltaMUV
        UVLFalist[zval,1] = MUVlist + 0.5 * deltaMUV
        UVLFlist[zval].tofile(Dir+name+f'/UVLF_z_{zvals[zval]}.dat')
        UVLFalist[zval].tofile(Dir+name+f'/UVLFa_z_{zvals[zval]}.dat')
        
def make_2d_hist(Dict,Dir,MH0_list ,betaList,Mhalo):
    zmin,zmax,dz = 4.5,25,0.1
    zlist = np.arange(zmin,zmax+0.5*dz,dz)
    fz = 1 / (zmax - zmin) * len(zlist)
    tmin,tmax,dt = -0.5,3,0.025
    tlist = np.arange(tmin,tmax+0.5*dt,dt)
    ft = 1 / (tmax - tmin) * len(tlist)
    Mmin,Mmax, dM = -30,0,0.1
    Mlist = np.arange(Mmin,Mmax+0.5*dM,dM)
    fM = 1 / (Mmax-Mmin) * len(Mlist)
    Mhalomin,Mhalomax, dMhalo = 6,14,0.05
    Mhalolist = np.arange(Mhalomin,Mhalomax+0.5*dMhalo,dMhalo)
    fMhalo = 1 / (Mhalomax-Mhalomin) * len(Mhalolist)

    sbTimeGrid  = np.zeros((zlist.shape[0],tlist.shape[0]))
    qTimeGrid   = np.zeros((zlist.shape[0],tlist.shape[0]))
    sbMUVgrid   = np.zeros((zlist.shape[0],Mlist.shape[0]))
    sbMUVaGrid  = np.zeros((zlist.shape[0],Mlist.shape[0]))
    sbMhaloGrid = np.zeros((Mhalolist.shape[0],tlist.shape[0]))
    qMhaloGrid  = np.zeros((Mhalolist.shape[0],tlist.shape[0]))
    
    sbTimes = np.log10(Dict['stochasticity'].sbTime)
    qTimes  = np.log10(Dict['stochasticity'].qTime)
    qredshift=Dict['stochasticity'].qRedshift
    redshift  = Dict['stochasticity'].sbRedshift
    LUV = Dict['stochasticity'].sbLUV
    Mhalo = Dict['stochasticity'].sbMhalo
    Mdust = Dict['stochasticity'].sbMdust
    qMhalo = Dict['stochasticity'].qMhalo
    Lmask = LUV>0
    MUV = get_Mag_from_L(np.log10(LUV[Lmask]))
    MUVa = get_Mag_from_L_attenuated(np.log10(LUV[Lmask]),redshift[Lmask],Mdust[Lmask],Mhalo[Lmask])
    for (z,m),ma in zip(zip(redshift[Lmask],MUV),MUVa):
        iz  = min(int((z  - zmin) * fz),len(zlist)-1)
        im  = min(int((m  - Mmin) * fM),len(Mlist)-1)
        ima = min(int((ma - Mmin) * fM),len(Mlist)-1)
        sbMUVgrid[iz,im]   += 1.
        sbMUVaGrid[iz,ima] += 1.
    for z,sb in zip(redshift,sbTimes):
        iz  = min(int((z  - zmin) * fz),len(zlist)-1)
        it  = min(int((sb - tmin) * ft),len(tlist)-1)
        sbTimeGrid[iz,it] += 1
    for z,q in zip(redshift,qTimes):
        iz  = min(int((z  - zmin) * fz),len(zlist)-1)
        it  = min(int((q - tmin) * ft),len(tlist)-1)
        qTimeGrid[iz,it] += 1
    for h,sb in zip(np.log10(Mhalo),sbTimes):
        it  = min(int((sb - tmin) * ft),len(tlist)-1)
        ih  = min(int((h  - Mhalomin) * fMhalo),len(Mhalolist) - 1)
        sbMhaloGrid[ih,it] += 1
    for h,q in zip(np.log10(qMhalo),qTimes):
        it  = min(int((q - tmin) * ft),len(tlist)-1)
        ih  = min(int((h  - Mhalomin) * fMhalo),len(Mhalolist) - 1)
        qMhaloGrid[ih,it] += 1
    
    name = Dict['name'][:-4]
    if not np.isin(name,os.listdir(Dir)):
        os.mkdir(Dir + name)
    zlist.tofile(Dir+name+'/zlist.dat')
    tlist.tofile(Dir+name+'/tlist.dat')
    Mlist.tofile(Dir+name+'/Mlist.dat')
    sbTimeGrid.tofile(Dir+name+'/sbTimeGrid.dat')
    qTimeGrid.tofile(Dir+name+'/qTimeGrid.dat')
    sbMUVgrid.tofile(Dir+name+'/sbMUVgrid.dat')
    sbMUVaGrid.tofile(Dir+name+'/sbMUVaGrid.dat')
    sbMhaloGrid.tofile(Dir + name + '/sbMhaloGrid.dat')
    qMhaloGrid.tofile(Dir + name + '/qMhaloGrid.dat')
    
def make_2d_hist_z(Dict,Dir,MH0_list ,betaList,Mhalo,zvals=np.array([5,7,9,11,13,15])):    
    MUVmin,MUVmax, dMUV = -30,0,0.1
    MUVlist = np.arange(MUVmin,MUVmax+0.5*dMUV,dMUV)
    fMUV = 1 / (MUVmax-MUVmin) * len(MUVlist)
    Mhalomin,Mhalomax, dMhalo = 7.4,14,0.05
    Mhalolist = np.arange(Mhalomin,Mhalomax+0.5*dMhalo,dMhalo)
    fMhalo = 1 / (Mhalomax-Mhalomin) * len(Mhalolist)

    name = Dict['name'][:-4]
    if not np.isin(name,os.listdir(Dir)):
        os.mkdir(Dir + name)
    Mhalolist.tofile(Dir + name + f'/Mhalolist.dat')
    zindexes = []
    for i in range(len(zvals)):
        zindexes.append(np.array([1114-where(substepRedshiftList[::-1],zvals[i])])[0])
    zindexes = np.array(zindexes)
    
    haloIndexes = np.zeros(Dict['LUV'].shape[0],int)
    betaIndexes = np.zeros(Dict['LUV'].shape[0],int)
    for run in range(Dict['LUV'].shape[0]):
        haloIndexes[run] = np.argwhere(np.isclose(Dict['catalogue'][run,0],Dict['MH0']))[0,0]
        betaIndexes[run] = np.argwhere(np.isclose(Dict['catalogue'][run,7],betaList))[0,0]
    
    for k in range(len(zvals)):
        MUVgrid  = np.zeros((len(Mhalolist),len(MUVlist)))
        MUVaGrid = np.zeros((len(Mhalolist),len(MUVlist)))
        #mask = (Dict['LUV'][:,zindexes[k]-1:zindexes[k]+1]>0)
        mask = (Dict['LUV'][:,zindexes[k]]>0)
        
        Mh   = Mhalo[betaIndexes,haloIndexes,zindexes[k]][mask]
        MUV  = get_Mag_from_L(np.log10(Dict['LUV'][:,zindexes[k]][mask]))
        MUVa = get_Mag_from_L_attenuated(np.log10(Dict['LUV'][:,zindexes[k]][mask]),zvals[k],Dict['Mdust'][:,zindexes[k]][mask],Mh)
        Mh = np.log10(Mh)
        for (h,uv),ua in zip(zip(Mh,MUV),MUVa):
            ih  = min(int((h  - Mhalomin) * fMhalo),len(Mhalolist) - 1)
            iu  = min(int((uv - MUVmin)   * fMUV),  len(MUVlist)   - 1)
            ia  = min(int((ua - MUVmin)   * fMUV),  len(MUVlist)   - 1)
            MUVgrid[ih,iu] += 1
            MUVaGrid[ih,ia] += 1
        MUVgrid.tofile(Dir + name + f'/z_{zvals[k]}_Mhalo_MUV.dat')
        MUVaGrid.tofile(Dir + name + f'/z_{zvals[k]}_Mhalo_MUVa.dat')
        
        for run in range(Dict['LUV'].shape[0]):
            if not np.any(Dict['SFR'][run,zindexes[k]:1073]==0):
                mask[run] = False
        MUVgrid  = np.zeros((len(Mhalolist),len(MUVlist)))
        MUVaGrid = np.zeros((len(Mhalolist),len(MUVlist)))
        Mh   = Mhalo[betaIndexes,haloIndexes,zindexes[k]][mask]
        MUV  = get_Mag_from_L(np.log10(Dict['LUV'][:,zindexes[k]][mask]))
        MUVa = get_Mag_from_L_attenuated(np.log10(Dict['LUV'][:,zindexes[k]][mask]),zvals[k],Dict['Mdust'][:,zindexes[k]][mask],Mh)
        Mh = np.log10(Mh)
        for (h,uv),ua in zip(zip(Mh,MUV),MUVa):
            ih  = min(int((h  - Mhalomin) * fMhalo),len(Mhalolist)-1)
            iu  = min(int((uv - MUVmin) * fMUV),len(MUVlist)-1)
            ia  = min(int((ua - MUVmin) * fMUV),len(MUVlist)-1)
            MUVgrid[ih,iu] += 1
            MUVaGrid[ih,ia] += 1
        MUVgrid.tofile(Dir + name + f'/z_{zvals[k]}_Mhalo_MUV_burst.dat')
        MUVaGrid.tofile(Dir + name + f'/z_{zvals[k]}_Mhalo_MUVa_burst.dat')
        
def make_fduty(Dict,Dir,MH0_list ,betaList,Mhalo,MhaloLimits = np.array([1e8,10**8.5,1e9,10**9.5,1e10,10**10.5,1e11])):
    name = Dict['name'][:-4]
    if not np.isin(name,os.listdir(Dir)):
        os.mkdir(Dir + name)
    fdutylist = np.zeros((len(MhaloLimits),1115))
    fdutylist[:,0] = MhaloLimits
    
    haloIndexes = np.zeros(Dict['LUV'].shape[0],int)
    betaIndexes = np.zeros(Dict['LUV'].shape[0],int)
    for run in range(Dict['LUV'].shape[0]):
        haloIndexes[run] = np.argwhere(np.isclose(Dict['catalogue'][run,0],Dict['MH0']))[0,0]
        betaIndexes[run] = np.argwhere(np.isclose(Dict['catalogue'][run,7],betaList))[0,0]
    Mh   = Mhalo[betaIndexes,haloIndexes]
    
    HMF = np.loadtxt('input_data/mVector_VSDMPL_z4.5.txt',skiprows=12)
    titles = [ r"$m$", r"$\sigma$", r"$ln(1/sigma)$", r"$n_{eff}$", r"$f(\sigma)$", r"$dn/dm$",r"$dn/dlnm$",r"$dn/dlog10m$",
              r"$n(>m)$",r"$\rho(>m)$",r"$\rho(<m)$",r"$Lbox(N=1)$"]

    ### Calculate halo mass function at relevant masses
    logM = np.log10(HMF[:,0])                  ### HMF[:,0] is halo masses
    dndlogM= HMF[:,7]                          ### HMF[:,7] is dn/dlog10m
    dlogndlogM=np.log10(HMF[:,7])              ### Get that baby log-scaled
    interp = si.CubicSpline(logM,dlogndlogM)   ### Interpolate because we dont have right ones here
    
    MH4_5s = np.log10(calc_Mhalo(Dict['catalogue'][:,0],4.5,Dict['catalogue'][:,7]))
    binsizeMhalo = np.zeros_like(MH4_5s)
    MH4_5_list=np.sort(list(set(MH4_5s)))
    deltaMH=0.5*(np.roll(MH4_5_list,-1)-(np.roll(MH4_5_list,1)))
    deltaMH[0] = (MH4_5_list[1])-(MH4_5_list[0])
    deltaMH[-1] = (MH4_5_list[-1])-(MH4_5_list[-2])
    for run in range(MH4_5s.shape[0]):
        MhaloIndex = np.argwhere(np.isclose(MH4_5s[run],MH4_5_list))[0,0]
        binsizeMhalo[run] = deltaMH[MhaloIndex]
    NH4_5s = 10 ** (interp(MH4_5s) + np.log10(binsizeMhalo))
    
    for k,M in enumerate(MhaloLimits):
        fduty = np.zeros(1114)
        Nduty = np.zeros(1114)
        for run,SFR in enumerate(Dict['SFR']):
            
            mask = Mh[run]>=M
            if np.any(mask):
                Nduty[mask] += 10**NH4_5s[run]
                fduty[mask] += (SFR[mask]>0).astype(int) * 10**NH4_5s[run]
        Fduty = np.zeros(1114)
        Fduty[Nduty>0] = fduty[Nduty>0]/Nduty[Nduty>0]
        fdutylist[k,1:] = Fduty
    fdutylist.tofile(Dir+name+'/fduty.dat')    
    
def density_estimation(m1, m2):
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]                                                     
    positions = np.vstack([X.ravel(), Y.ravel()])                                                       
    values = np.vstack([m1, m2])                                                                        
    kernel = stats.gaussian_kde(values)                                                                 
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z
def roll_mean(x,m = 2):
    return 0.25 * np.nansum( np.array([np.roll(x,1)*0.5 * (4-m) , m * x , np.roll(x,-1)*0.5*(4-m)]),axis = 0)
def recurse_roll_mean(x,n,m=2):
    for i in range(n):
        x = roll_mean(x,m)
    return x

def get_mhalo_list(MH0_list,substepRedshiftList,betaList):
    redshift_list = np.loadtxt('/home/vdm981_alumni_ku_dk/modi_mount/Astraeus/astraeus/dynamic_IMF/redshift_list.txt').T[1]
    
    Mhalo = np.zeros((len(betaList),len(MH0_list),1114))
    for b in range(len(betaList)):
        for i in range(len(MH0_list)):
            haloMasses     = calc_Mhalo(MH0_list[i],redshift_list,betaList[b]) #Halo Masses at snapshots (Not at timesteps, to emulate Astraeus)
            interp              = si.interp1d(redshift_list,haloMasses)
            Mhalo[b][i][1:] = interp(substepRedshiftList[1:])
            Mhalo[b][i][0]  = haloMasses[0]
    return Mhalo

radius_dust_grains = 5e-6
density_dust_grains = 2.25
grams_per_solar_mass = 1.99e33
G  = 6.67408e-8 # Gravitational constant (g/(cm^3 * s) or whatever
h = 0.6777 #Reduced hubble constant (km/s)/Mpc
cm_per_Mpc = 3.0857e24
ergs_per_Joule = 1e7 # I think thats what this is
H0 = h * ergs_per_Joule / cm_per_Mpc
omega_m = 0.307115
omega_l = 0.692885

def calc_Mhalo(M0,z,beta=-0.75):
    alpha = 0.25
    M = M0 *  (1+z)**alpha * np.exp(beta*z)
    return M

def hubble(z):
    return H0 * np.sqrt(omega_m * (1+z)**3 + omega_l)

def calc_Rvir(halomass,z):
    return (halomass * grams_per_solar_mass * G / (100 * hubble(z)**2)) ** (1/3)

def calc_dust_tau(dust_radius,dust_mass,radius_dust_grains = radius_dust_grains,density_dust_grains = density_dust_grains):
    sigma = dust_mass * grams_per_solar_mass / ( np.pi * dust_radius **2)
    
    tau   = 3/4 * sigma/(radius_dust_grains * density_dust_grains)
    return tau

def calc_dust_radius(Rvir,redshift,spin_param=0.04,dust_to_gas_radius_ratio = 1):
    return dust_to_gas_radius_ratio * Rvir * 4.5 * spin_param * ((1+redshift)/6)**1.8

def calc_extinction(redshift,Mdust,Mvir,radius_dust_grains = radius_dust_grains,density_dust_grains = density_dust_grains,model = 0):
    tau = calc_dust_tau(calc_dust_radius(calc_Rvir(Mvir,redshift),redshift),Mdust,radius_dust_grains,density_dust_grains)
    result = np.zeros(tau.shape[0])
    for i in range(tau.shape[0]):
        if tau[i] < 1e-8:
            result[i] = 1
        elif model == 1:
            result[i] = np.exp(-tau[i])
        else: 
            result[i] = (1-np.exp(-tau[i]))/tau[i]
    return result
def get_Mag_from_L(L, fc = 1):
    UVperFreq = 10.**L * pc.lambda_UV_cm * pc.lambda_UV_cm * 1e8 / pc.clight_cm * fc
    UVmag = -2.5 * np.log10(UVperFreq / (4.*np.pi*100.*pc.pc_cm*pc.pc_cm)) - 48.6
    return UVmag
def get_Mag_from_L_attenuated(L,z,Mdust,Mvir):
    return get_Mag_from_L(L,calc_extinction(z,Mdust,Mvir))
def profile_matrix_median_2(H, x_center,y_center,Hmin = 0):
    """
    Takes 2D array, H, and a set of bin center values.
    """
    xmin,xmax,ymin,ymax = np.min(x_center),np.max(x_center),np.min(y_center),np.max(y_center)
    shape = H.shape
    #xwidth = (xmax-xmin)/shape[0]
   # ywidth = (ymax-ymin)/shape[1]
    #x_center = np.arange(shape[0])*((xmax-xmin)/shape[0])+xmin+0.5*xwidth
    #y_center = np.arange(shape[1])*((ymax-ymin)/shape[1])+ymin+0.5*ywidth
    
    for i in range(len(x_center)):
        for j in range(len(y_center)):
            if H[i,j]<Hmin:
                H[i,j]=0
    wsums = H.sum(1)

    mask = wsums != 0
    median = np.zeros(len(x_center))

    for i in range(len(median)):
        if wsums[i]>0:
            median[i] = y_center[np.where(np.array([np.sum(H[i,:j]) for j in range(len(y_center))])<wsums[i]/2)[0][-1]]
    

    return x_center[mask], median[mask]
def get_median(Xpoints,Ypoints,n):
    medianX = np.linspace(np.min(Xpoints),np.max(Xpoints),n)
    deltaX = medianX[1]-medianX[0]
    medianY = np.zeros_like(medianX)
    for im in range(medianX.shape[0]):
        mask = (Xpoints > medianX[im] - 0.5 * deltaX) * (Xpoints <= medianX[im] + 0.5 * deltaX)
        if np.any(mask):
            medianY[im] = np.median(Ypoints[mask])
    return medianX,medianY    
def where(arr,value):
    up = len(arr)
    low = 0
    test = up//2
    while up-low>1:
        if arr[test]<value:
            low = test
        else:
            up = test
        test = (up + low)//2
    return test