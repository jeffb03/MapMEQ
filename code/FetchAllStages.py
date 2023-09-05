''' Main driver to run FracPlane.py test protocol
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
    x Reads in well survey file and AllStages.xlsx from DataPath
    x Reads / Writes AllStages_WorkingFile.xlsx in SavePath
    x Returns working MEQ data, survey file, perf mid-point, and perf interval
'''

import numpy as np
import pandas as pd


#---------------------------------------------------------------------------------------------------------------
def Step1(DataPath,SavePath):

    # in Step 1, must process AllStages sheet from Forge directory and then return working sheet below
    # NB: only do this once, Step1 only processed once per run

    ev = pd.read_excel(DataPath + 'AllStages.xlsx', sheet_name='AllStages')

    #ev = pd.read_excel(DataPath + 'StageThree.xlsx', sheet_name='AllStages')
    #ev = pd.read_excel(DataPath + 'AllStagesTrain.xlsx', sheet_name='AllStages')
    

    # read wellbore survey, should be in Finnila survey file coord system, with X, Y @ 16A
    wb = GetSurvey(DataPath)
    #wb = pd.read_excel(DataPath + '16A(78)-32.xlsx', sheet_name='data')
    
    perf, perfint = GetPerfs()
 
    
    # now clean up MEQ data below >

    # remove blanks in headings
    for old in ev.columns:
        ev.rename(columns={old: old.strip()}, inplace=True)

    # filter out empty rows that have zero depth
    ev = ev[ev['Depth']>0]          # looks redundant, but these are blank events
    ev = ev.reset_index()
    ev = ev.drop('index',axis=1)
    
    # convert MEQ data from feet to meters - only one time!
    ev['E'] = ev['E'] / 3.2808398950131
    ev['N'] = ev['N'] / 3.2808398950131
    ev['Depth'] = ev['Depth'] / 3.2808398950131

    # now convert Depth to Finnila convention: -ve into ground, wrt MSL
    ev['Depth'] = -ev['Depth'] + 1650.187

    # calculate elapsed seconds
    evDT = pd.to_datetime(ev['OriginDate'])
    x1 = pd.to_timedelta(ev['OriginTime'])
    evDT = evDT + x1
    ev.insert(2,'DT', evDT)

    # drop some columns for efficiency  
    ev.drop('OriginDate', axis=1, inplace=True)
    ev.drop('OriginTime', axis=1, inplace=True)
    # NB: dropped original "Stage" in Excel sheet so that we could use it here

    # these were manually removed from AllStages.xlsx, included here for documentation
    #ev.drop('TrigDate', axis=1, inplace=True)      
    #ev.drop('TrigTime', axis=1, inplace=True)
    #ev.drop('Profile', axis=1, inplace=True)

    # insert RootTime column
    ev.insert(2,'RootTime',np.zeros(ev.shape[0]))
    
    # calculate elapsed root-seconds for each stage, start with first event (not the perf record)
    for stage in range(1,4):
        ix = ev[ev['Stage']==stage].index
        Elapsed = pd.to_timedelta(ev['DT'][ix] - ev['DT'][ix[0]])
        tsec = pd.Series(Elapsed.dt.total_seconds())
        ev.loc[ix,'RootTime'] = tsec

    # now take the square-root; verified that root preferred to linear
    ev['RootTime'] = np.sqrt(ev['RootTime'])      

    # insert columns for timegroup and two other Groups
    # set all values to (-1), will be considered outlier unless reset
    ev.insert(3,'TimeGroup',(-1)*np.ones(ev.shape[0]))
    ev.insert(4,'SpaceGroup',(-1)*np.ones(ev.shape[0]))
    ev.insert(5,'ExtGroup',(-1)*np.ones(ev.shape[0]))           # extended group for reducing outliers
    ev.insert(6,'Outliers',(-9)*np.ones(ev.shape[0]))           # residuals (outliers of the outliers) = (-1)
    ev.insert(7,'Merged',(-1)*np.ones(ev.shape[0]))
    ev.insert(8,'FusedGroup',(-1)*np.ones(ev.shape[0]))
    ev.insert(9,'Work',(-1)*np.ones(ev.shape[0]))

    # save the working file
    ev.to_excel(SavePath + 'AllStages_WorkingFile.xlsx', sheet_name='Work', index=False)

    minvec = np.zeros(4)
    maxvec = np.zeros(4)
    minvec[0] = min(ev['E'])
    maxvec[0] = max(ev['E'])
    minvec[1] = min(ev['N'])
    maxvec[1] = max(ev['N'])
    minvec[2] = min(ev['Depth'])
    maxvec[2] = max(ev['Depth'])   
    minvec[3] = min(ev['RootTime'])
    maxvec[3] = max(ev['RootTime'])  

    return ev, wb, perf, perfint, minvec, maxvec


#---------------------------------------------------------------------------------------------------------------
def Step2(DataPath, SavePath):

    # Get a nice section of data to prove up the algorithm
    # !!! Using working file that has events with Depth=0 removed and some columns deleted
    # also, this file has depth reference correction from Step 1

    # load working MEQ data sheet
    ev = pd.read_excel(SavePath + 'AllStages_WorkingFile.xlsx', sheet_name='Work')

    # read wellbore survey, should be in Finila survey file coord system, with X, Y @ 16A
    wb = GetSurvey(DataPath)
    #wb = pd.read_excel(DataPath + '16A(78)-32.xlsx', sheet_name='data')
    
    # perforations
    perf, perfint = GetPerfs() 
    
    minvec = np.zeros(4)
    maxvec = np.zeros(4)
    minvec[0] = min(ev['E'])
    maxvec[0] = max(ev['E'])
    minvec[1] = min(ev['N'])
    maxvec[1] = max(ev['N'])
    minvec[2] = min(ev['Depth'])
    maxvec[2] = max(ev['Depth'])   
    minvec[3] = min(ev['RootTime'])
    maxvec[3] = max(ev['RootTime'])  

    return ev, wb, perf, perfint, minvec, maxvec



#---------------------------------------------------------------------------------------------------------------
def GetSurvey(DataPath):

# read wellbore survey, should be in Finnila survey file coord system
    wb = pd.read_excel(DataPath + '16A(78)-32.xlsx', sheet_name='data')

    wb['X'] -= 334638.97
    wb['Y'] -= 4263434.117

    return wb


#---------------------------------------------------------------------------------------------------------------
def GetPerfs():

# original perf list is in metric wrt 16A wellhead, +ve depth into ground, zero elevation = wellhead
# will convert to global coords wrt MSL for depth, leave E and N wrt 16A to avoid large digits

    perf = [[1156, -312, -932], [1092, -300, -908], [974, -266, -851]]
    perfint = [ [[1154, -312, -932], [1208, -321, -953]],
                [[1090, -300, -907], [1094, -301, -909]],
                [[972, -265, -850], [977, -267, -852]] ]     

    """ perf = [[1154, -312, 2582], [1092, -300, 2558], [974, -266, 2501]]
    perfint = [ [[1154, -312, 2582], [1208, -321, 2603]],
                [[1090, -300, 2557], [1094, -301, 2559]],
                [[972, -265, 2500], [977, -267, 2502]] ]     
    
    for p in perf:
        p[2] = -p[2] + 1650.187
    for pi in perfint:
        pi[0][2] = -pi[0][2] + 1650.187
        pi[1][2] = -pi[1][2] + 1650.187 """

    return perf, perfint

