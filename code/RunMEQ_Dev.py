''' Calculation Loop for MapMEQ, with Dev code
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, datetime
from mpl_toolkits.mplot3d import Axes3D

import FetchAllStages
import FracPlane, PlotFrac
import DataSegments
import ContiguousCount
import PlotENDT
import Frac2DFN
import DFN2StereoNet
import StagePlots

#---------------------------------------------------------------------------------------------------------------
class RunMEQ():
    ''' Receives pct outliers and min cluster points from MapMEQ
        and completes the time & space clustering.  It then runs
        the FracPlane analysis and returns total area and SSE
    '''

    def __init__(self, DataPath, SavePathBase):  
        super().__init__()

        self.DataPath = DataPath
        self.SavePathBase = SavePathBase
        self.SavePath = []

        self.MinTimePoints = []
        self.MinSpacePoints = []
        self.MaxTimeOutliers = []
        self.MaxSpaceOutliers = []
        self.TimeID = []
        self.SpaceID = []
        self.Fracs = []
        self.fuse_dist = 10.0
        self.FinalFracs = []

        self.diags = []                 # enable diagnostic plots
        self.diags2 = False             # second-level diags if needed

        now = datetime.datetime.now()
        self.datetime_string = now.strftime("%Y-%m-%d_%H.%M.%S")        # unique for this run
    
    
    #---------------------------------------------------------------------------------------------------------------
    def Calc(self, XMinTimePoints, XMinSpacePoints, XMaxTimeOutliers, XMaxSpaceOutliers):   

        All_MinTimePoints = []
        All_MinSpacePoints = []
        All_MaxTimeOutliers = []
        All_MaxSpaceOutliers = []
        All_nfracs = []
        All_noutliers = []
        All_tot_area = []
        All_tot_SSE = []

        batch = False
        plt.ioff()          # turn off interactive mode

        if not batch:

            self.diags = True
            self.diags2 = True
            self.MinTimePoints = XMinTimePoints
            self.MinSpacePoints = XMinSpacePoints
            self.MaxTimeOutliers = XMaxTimeOutliers
            self.MaxSpaceOutliers = XMaxSpaceOutliers

            self.TimeID = f'{XMinTimePoints[0]}-{XMaxTimeOutliers[0]}_{XMinTimePoints[1]}-{XMaxTimeOutliers[1]}_{XMinTimePoints[2]}-{XMaxTimeOutliers[2]}'
            self.SpaceID = f'{XMinSpacePoints[0]}-{XMaxSpaceOutliers[0]}_{XMinSpacePoints[1]}-{XMaxSpaceOutliers[1]}_{XMinSpacePoints[2]}-{XMaxSpaceOutliers[2]}'

            now = datetime.datetime.now()
            date_string = now.strftime('%Y-%m-%d')   
            self.SavePath = self.SavePathBase + f'MEQ_{date_string}/'      # a new directory for each day
            try:
                os.makedirs(self.SavePath)
            except:
                pass
            
            # identify the current directory run
            print(f'\n\n{self.SavePath}\n')

            # plot all data
            self.PlotAllData()

            # cluster data in time 
            # - can be skipped if repeatedly working the same set of files since this reads the archive and stores working version
            self.TimeClusters()

            # cluster data in space - this also is a one-shot since deterministic for a given set of params
            self.SpaceClusters()

            # extend fracs with outliers
            self.Extend()

            # do something with the remaining outliers, those not grouped by the above process
            self.Outliers()
            
            # merge the ExtGroup with Outlier frac group & make sequential frac numbers
            self.Merge()

            # fuse ExtGroup and Outliers in iterative fusion algorithm until no changes
            self.Fusion()

            self.FracCompare('Merged', 'FusedGroup')

            # collect frac planes  
            self.PlotFracPlanes('Outliers', circles=1)

            # collect frac planes 
            self.TimeLapse_1('Merged', circles=1)

            self.TimeLapse_1('FusedGroup', circles=1)

            self.TimeLapse_3('FusedGroup', circles=1)

            # connect to perfs
            
            self.Connections()

            # StageDataPlots has been internally augmented with the Connections for Stage 2
            #StagePlots.StageDataPlots(self,'FusedGroup')

            # write the DFN file
            # NB: Fusion and Connections must be run in the current exec before Frac2DFN
            Frac2DFN.Frac2DFN(self)

            DFN2StereoNet.DFN2StereoNet(self)


            # now self.fracs contains all the fracs we want to send to GeoDT
            # plot by stage - stereonet & rose plots
            #StagePlots.StageFracPlots(self)
            #StagePlots.StageDataPlots(self,'FusedGroup')


            # now, calculate/plot FracPlane geometries for all stages
            #nfracs, noutliers, tot_area, tot_SSE = self.AllStagesFracPlaneDiags()
            #print(f'Num Fracs {nfracs}, Num Outliers {noutliers}, Tot Area {tot_area:.0f}, Tot SSE {tot_SSE:.0f}')
            
            # show each plane
            #self.EachFracPlane()


            plt.show()

            input('Done.  Press enter to quit ...')


        else:

            self.diags = True
            for MinTimePoints in XMinTimePoints:
                for MinSpacePoints in XMinSpacePoints:
                    for MaxTimeOutliers in XMaxTimeOutliers:
                        for MaxSpaceOutliers in XMaxSpaceOutliers:

                            self.MinTimePoints = MinTimePoints
                            self.MinSpacePoints = MinSpacePoints
                            self.MaxTimeOutliers = MaxTimeOutliers
                            self.MaxSpaceOutliers = MaxSpaceOutliers

                            self.SavePath = self.SavePathBase + f'MEQ_{date_string}/'      # a new directory for each day

                            try:
                                os.makedirs(self.SavePath)
                            except:
                                pass
                            
                            # identify the current directory run
                            print(f'\n\n{self.SavePath}\n')

                            # cluster data in time
                            self.TimeClusters()
                            #continue

                            # cluster data in space
                            self.SpaceClusters()

                            # collect frac planes
                            #self.SpaceFracPlanes()

                            # evaluate differences in fracs & fuse as needed
                            self.FracFusion()
                        
                            # now, calculate FracPlane geometries for FusedGroup
                            nfracs, noutliers, tot_area, tot_SSE = self.AllStagesFracPlaneDiags()
                            print(f'Num Fracs {nfracs}, Num Outliers {noutliers}, Tot Area {tot_area:.0f}, Tot SSE {tot_SSE:.0f}')

                            All_MinTimePoints.append(MinTimePoints)
                            All_MinSpacePoints.append(MinSpacePoints)
                            All_MaxTimeOutliers.append(MaxTimeOutliers)
                            All_MaxSpaceOutliers.append(MaxSpaceOutliers)
                            All_nfracs.append(nfracs)
                            All_noutliers.append(noutliers)
                            All_tot_area.append(tot_area)
                            All_tot_SSE.append(tot_SSE)

            df = pd.DataFrame([All_MinTimePoints, All_MaxTimeOutliers, All_MinSpacePoints, All_MaxSpaceOutliers,
                            All_nfracs, All_noutliers, All_tot_area, All_tot_SSE])
            df = df.T
            df.columns = ['MinTimePoints', 'MaxTimeOutliers', 'MinSpacePoints', 'MaxSpaceOutliers',
                            'nfracs', 'noutliers', 'tot_area', 'tot_SSE']
            
            # use datetime for filename since crosses many run folders
            #dstr = datetime.now.strftime("%Y-%m-%d_%H.%M.%S") + '/'
            pfn = self.SavePathBase + (f'MEQ_' + self.datetime_string + '.xlsx')
            df.to_excel(pfn)


        return
    
    #---------------------------------------------------------------------------------------------------------------
    def S2_Connections(self): 

        # NB: Fusions and Connections must be run in the current exec before Frac2DFN
        # after Fusion, self.Fracs has most recent fracture maps
        # append prescribed fracs below to enable communication between fractures

        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath) 

        stage = [2, 2]
        strike = [20, 190]
        dip = [90, 0]
        radius = [50, 50]
        location = [np.array(perf[1])+[0, 0, 25], np.array(perf[1])+[0, 0, 50]]

        # add vertical frac and low-dip frac to Stage 2 for interconnection with perfs
        f1 = FracPlane.FracPlane()
        f2 = FracPlane.FracPlane()
        f1.MakeCircularFrac(stage[0], location[0], radius[0], dip[0], strike[0])          
        f2.MakeCircularFrac(stage[1], location[1], radius[1], dip[1], strike[1])          
        
        return f1, f2

    #---------------------------------------------------------------------------------------------------------------
    def param_string(self, stage):   

        # param string is now stage dependent, so must use func to set as needed
        if stage == 0:
            txt = ''
        if stage == 1:
            txt = f'T{self.MinTimePoints[0]}p-{self.MaxTimeOutliers[0]}o_S{self.MinSpacePoints[0]}p-{self.MaxSpaceOutliers[0]}o'
        elif stage == 2:
            txt = f'T{self.MinTimePoints[1]}p-{self.MaxTimeOutliers[1]}o_S{self.MinSpacePoints[1]}p-{self.MaxSpaceOutliers[1]}o'
        else:    
            txt = f'T{self.MinTimePoints[2]}p-{self.MaxTimeOutliers[2]}o_S{self.MinSpacePoints[2]}p-{self.MaxSpaceOutliers[2]}o'

        return txt


    #---------------------------------------------------------------------------------------------------------------
    def TimeClusters(self):

        # this section is the first step to read the raw data and apply DBSCAN to obtain TimeGroup bins

        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step1(self.DataPath, self.SavePath)        
        print(ev.columns)

        # let's do time segmentation by stage, searching for epsilon that meets outlier criteria MAX_PCT_OUTLIERS
        # initial results with 4-5 points and ~10% outlier settings
        # epsTrial = [0.0075,      # Stage 1 pretty good w 0.0075: N=10, 10% outlier
        #             0.0015,      # Stage 2 only 2 groups w 0.0075, 0.0020: N=6, 9% outlier, 0.0015: N=12, 10% outlier
        #             0.0007]      # Stage 3 only 3 groups w 0.0075, 0.0010: N=20, 5% outlier, 0.0007: N=36, 10% outlier
        

        epsTrial = [0.0005, 0.0007, 0.0010, 0.0015, 0.0020, 0.0025, 0.0050, 0.0075, 0.010, 0.0125, 0.015, 0.020, 0.025]

        # this should start as null, but we'll fill below then save in WorkingFile
        TimeGroupCol = ev['TimeGroup'].copy()

        fig = plt.figure(figsize=(16,10))
        nlabels = np.zeros(len(epsTrial))
        outliers = np.zeros(len(epsTrial))
        stage_pctoutlier = np.zeros(3)
        stage_epsidx = np.zeros(3)
        stage_label = []
        GroupCount = 0
        for stage in range(1,4):
            thisTimePoints = self.MinTimePoints[stage-1]
            thisTimeOutliers = self.MaxTimeOutliers[stage-1]
            nlabels = []
            outliers = []
            labels = []
            evs = ev[ev['Stage']==stage]
            epsidx = []
            for eps in epsTrial:
                nlabel, pctoutlier, label = DataSegments.time_segments(evs, eps, thisTimePoints) 
                nlabels.append(nlabel)
                outliers.append(pctoutlier)
                labels.append(label)

                # reached condition, can drop out now
                if pctoutlier <= thisTimeOutliers:
                    epsidx = eps
                    stage_pctoutlier[stage-1] = pctoutlier
                    stage_epsidx[stage-1] = epsidx
                    stage_label.append(label)
                    break
            
            pass

            # if needed, now find results for lowest epsilon that preserves min pct of data, else the boundary condition
            if epsidx == []:
                pct = np.array(outliers)
                ept = np.array(epsTrial)
                try:
                    epsidx = ept[pct<thisTimeOutliers][0]      # find first epsilon with less than MAX_PCT_OUTLIERS_TIME
                except:
                    epsidx = epsTrial[-1]                   # choose last one if all the way to end
                nlabel, pctoutlier, label = DataSegments.time_segments(evs, epsidx, thisTimePoints)
                stage_pctoutlier[stage-1] = pctoutlier
                stage_epsidx[stage-1] = epsidx
                stage_label.append(label)

          # only diags from here to end of func, plot results
            details = True
            if details:

                ix = epsTrial.index(epsidx)
                epsPlot = epsTrial[0:ix+1]

                fig = plt.figure(figsize=(16,10))
                plt.rcParams.update({'font.size': 12})
                ax1 = fig.add_subplot(221)
                ax1.plot(epsPlot,nlabels,'*-',linewidth=3)
                ax1.grid(True)
                ax1.set_xlabel('epsilon')
                ax1.set_ylabel('nlabels')
                ax1.set_ylim(0,max(nlabels))
                plt.title(f'Stage = {stage}\n')
                plt.xticks(epsPlot, rotation='vertical')

                ax2 = fig.add_subplot(223)
                ax2.plot(epsPlot,outliers,'*-',linewidth=3, color='r')
                ax2.grid(True)
                ax2.set_xlabel('epsilon')
                ax2.set_ylabel('pct outliers', color='r')
                ax2.set_ylim(0,max(outliers))
                ax2.tick_params(axis='y', labelcolor='r')
                plt.xticks(epsPlot, rotation='vertical')
                    
                ax3 = fig.add_subplot(224)
                colorset = sns.color_palette("bright", int(nlabel))
                sns.scatterplot(x=evs['E'],y=evs['Depth'], hue=label, palette=colorset)
                ax3.set_xlabel('Easting')
                ax3.set_ylabel('Depth')
                ax3.plot(wb['X'], wb['D'])
                for p in perfint:
                    ax3.plot([p[0][0], p[1][0]], [p[0][2], p[1][2]], color='k', linewidth=6)
                ax3.set_xlim(minvec[0],maxvec[0])
                ax3.set_ylim(minvec[2],maxvec[2])
                plt.title(f'eps = {epsidx}, outl = {pctoutlier:.1f}%, ncpt = {thisTimePoints}\n')

                ax4 = fig.add_subplot(222)
                colorset = sns.color_palette("bright", int(nlabel))
                sns.scatterplot(x=evs['E'],y=evs['N'], hue=label, palette=colorset)
                ax4.set_xlabel('Easting')
                ax4.set_ylabel('Northing')
                ax4.plot(wb['X'], wb['Y'])
                for p in perfint:
                    ax4.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color='k', linewidth=6)
                ax4.set_xlim(minvec[0],maxvec[0])
                ax4.set_ylim(minvec[1],maxvec[1])

                fig.tight_layout(w_pad=2)

                pfn = f'{self.SavePath}Stage_{stage}_TimeClusters_Eps_{epsidx}.png'
                plt.savefig(pfn)

            # now update TimeGroupCol with label values, offset by cumulative GroupCount
            # NB: label = -1 indicates null and should remain null, must be reset after adding GroupCount
            ix = ev[ev['Stage']==stage].index
            values = label + GroupCount
            values[label<0] = -1
            TimeGroupCol[ix] = values

            # update GroupCount for next stage, starting value for next cycle
            GroupCount = max(TimeGroupCol) + 1

            # save figure to show time clustering results

            if not details:
                evs = ev[ev['Stage']==stage]
                if stage == 1:
                    first = 231
                    second = 234
                elif stage == 2:
                    first = 232
                    second = 235
                elif stage == 3:
                    first = 233
                    second = 236

                ax = fig.add_subplot(first)
                colorset = sns.color_palette("bright", int(nlabel))
                sns.scatterplot(x=evs['E'],y=evs['Depth'], hue=label, palette=colorset)
                ax.set_xlabel('Easting')
                ax.set_ylabel('Depth')
                plt.title(f'S {stage}, eps {epsidx}, N {int(nlabel)}, outl {pctoutlier:.1f}% / {thisTimeOutliers}%, ncpt {thisTimePoints}\n',
                        fontsize=12)
                ax.plot(wb['X'], wb['D'])
                for p in perfint:
                    ax.plot([p[0][0], p[1][0]], [p[0][2], p[1][2]], color='k', linewidth=6)
                ax.set_xlim(minvec[0], maxvec[0])
                ax.set_ylim(minvec[2], maxvec[2])

                ax = fig.add_subplot(second)
                sns.scatterplot(x=evs['RootTime'], y=evs['Depth'], hue=label, palette=colorset)
                ax.set_xlabel('RootTime')
                ax.set_ylabel('Depth')

        # now update the xlsx file for the time groupings calculated above
        pfn = self.SavePath + 'AllStages_WorkingFile.xlsx'
        sheet_name='Work'
        df = pd.read_excel(pfn, sheet_name=sheet_name)
        df['TimeGroup'] = TimeGroupCol
        df.to_excel(pfn, sheet_name=sheet_name, index=False)
            
        # NB: if PlotENDT is enabled above, there is interference with this plot, and it resembles Stage 3 results
        # and not the full Stage 1, 2, and 3 results that should be expected.  Don't know why, should not happen.
        if not details:
            pfn = f'{self.SavePath}TimeClustering_AllStages.png'
            plt.savefig(pfn)
        #plt.show()

        # individual plots for each stage
        # NB had to put down here because when interspersed above, the combined plot was messed up
        # - apparently, cannot build two plots at the same time!

        if self.diags:
            for stage in range(1,4):
                evs = ev[ev['Stage']==stage]
                title = (f'Stage = {stage}, eps = {stage_epsidx[stage-1]}, outlier = {stage_pctoutlier[stage-1]:.1f}% / {thisTimeOutliers}%, pts = {thisTimePoints}')
                spfn = f'{self.SavePath}TimeClustering_Stage{stage}_{self.TimeID}.png'
                PlotENDT.PlotENDT(wb, perfint, evs['E'], evs['N'], evs['Depth'], evs['RootTime'], stage_label[stage-1], minvec, maxvec, title, spfn)



    #---------------------------------------------------------------------------------------------------------------
    def SpaceClusters(self):

    # this section receives the AllStages_WorkingFile data 
    # TimeGroup has been set, and this routine applies spatial clustering and calculates clusters SpaceGroup
    # results are updated to WorkingFile

        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath) 

        # let's start with a search for epsilon, with cutoff criteria of missing labels

        FracCount = 0
        #epsTrial = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]  #, 100, 120, 140, 160, 180, 200]

        # this should start as null, but we'll fill below then save in WorkingFile
        SpaceGroupCol = ev['SpaceGroup'].copy()

        nn = max(ev['TimeGroup']) + 1
        for k in range(0,nn):
            evs = ev[ev['TimeGroup']==k]
            stage = evs['Stage'].iat[0]
            print(f'>number of events for TimeGroup {k} is {evs.shape[0]}')

            thisSpacePoints = self.MinSpacePoints[stage-1]
            thisSpaceOutliers = self.MaxSpaceOutliers[stage-1]

            # TimeGroup = 7 requires custom params to get best results!
            if evs['TimeGroup'].iloc[0] == 7:
                epsTrial = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                thisSpacePoints = 4
                thisSpaceOutliers = 25
            elif evs['TimeGroup'].iloc[0] == 8:
                epsTrial = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                thisSpacePoints = 6
                thisSpaceOutliers = 25
            else:
                epsTrial = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]

            # will need to loop over eps here to select best value
            nlabels = []
            outliers = []
            labels = []
            epsPlot = []
            epsidx = []
            for eps in epsTrial:
                nlabel, pctoutlier, label = DataSegments.space_segments(evs, eps, thisSpacePoints)  
                nlabels.append(nlabel)
                outliers.append(pctoutlier)
                labels.append([label])

                # reached condition, can drop out now
                if pctoutlier <= thisSpaceOutliers:
                    epsidx = eps
                    break

            # if needed, now find results for lowest epsilon that preserves min pct of data, else the boundary condition
            if epsidx == []:
                pct = np.array(outliers)
                ept = np.array(epsTrial)
                try:
                    epsidx = ept[pct<thisSpaceOutliers][0]      # find first epsilon with less than MAX_PCT_OUTLIERS
                except:
                    epsidx = epsTrial[-1]                   # choose last one if all the way to end
                nlabel, pctoutlier, label = DataSegments.space_segments(evs, epsidx, thisSpacePoints)

            if self.diags:
                title = (f'Stage = {stage}, TimeGroup = {k}, eps = {epsidx}, outlier = {pctoutlier:.1f}% / {thisSpaceOutliers}%, ncpt = {thisSpacePoints}\n')
                spfn = f'{self.SavePath}SpaceClustering_TimeGroup{k}_{self.SpaceID}.png'
                PlotENDT.PlotENDT(wb, perfint, evs['E'], evs['N'], evs['Depth'], evs['RootTime'], label, minvec, maxvec, title, spfn)

            # now update FracCount and insert cumulative results in SpaceGroup
            # also need to update the xlsx file for the time groupings calculated above, for each stage in turn
            # NB: use 'xf' here instead of 'df' to avoid conflicts with plot references

            # now update TimeGroupCol with label values, offset by cumulative GroupCount
            # NB: label = -1 indicates null and should remain null, must be reset after adding GroupCount
            ix = ev[ev['TimeGroup']==k].index
            values = label + FracCount
            values[label<0] = -1
            SpaceGroupCol[ix] = values
            if len(values) == 3:
                input('stop at 432')
              

            # update GroupCount for next stage, starting value for next cycle
            FracCount += (max(label) + 1)       # update for next cycle

            # only diags from here to end of func, plot results
            if self.diags2:

                ix = epsTrial.index(epsidx)
                epsPlot = epsTrial[0:ix+1]

                fig = plt.figure(figsize=(16,10))
                plt.rcParams.update({'font.size': 12})
                ax1 = fig.add_subplot(221)
                ax1.plot(epsPlot,nlabels,'*-',linewidth=3)
                ax1.grid(True)
                ax1.set_xlabel('epsilon')
                ax1.set_ylabel('nlabels')
                ax1.set_ylim(0,max(nlabels))
                plt.title(f'Stage = {stage}, TimeGroup = {k}\n')

                ax2 = fig.add_subplot(223)
                ax2.plot(epsPlot,outliers,'*-',linewidth=3, color='r')
                ax2.grid(True)
                ax2.set_xlabel('epsilon')
                ax2.set_ylabel('pct outliers', color='r')
                ax2.set_ylim(0,max(outliers))
                ax2.tick_params(axis='y', labelcolor='r')
                    
                ax3 = fig.add_subplot(224)
                colorset = sns.color_palette("bright", int(nlabel))
                sns.scatterplot(x=evs['E'],y=evs['Depth'], hue=label, palette=colorset)
                ax3.set_xlabel('Easting')
                ax3.set_ylabel('Depth')
                ax3.set_xlim(minvec[0],maxvec[0])
                ax3.set_ylim(minvec[2],maxvec[2])
                ax3.plot(wb['X'], wb['D'])
                for p in perfint:
                    ax3.plot([p[0][0], p[1][0]], [p[0][2], p[1][2]], color='k', linewidth=6)
                

                ax4 = fig.add_subplot(222)
                colorset = sns.color_palette("bright", int(nlabel))
                sns.scatterplot(x=evs['E'],y=evs['N'], hue=label, palette=colorset)
                ax4.set_xlabel('Easting')
                ax4.set_ylabel('Northing')
                ax4.plot(wb['X'], wb['Y'])
                for p in perfint:
                    ax4.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color='k', linewidth=6)
                ax4.set_xlim(minvec[0],maxvec[0])
                ax4.set_ylim(minvec[1],maxvec[1])
                plt.title(f'eps = {epsidx}, outl = {pctoutlier:.1f}%, ncpt = {thisSpacePoints}\n')

                fig.tight_layout(w_pad=2)

                strng = f'Stage_{stage}_TimeGroup_{k}_FracCount_{FracCount}_Eps_{epsidx}' 
                #plt.title(strng.replace('_',' '))   # not suptitle
                pfn = f'{self.SavePath}{strng}.png'
                plt.savefig(pfn)
                plt.show()

                #if ev['TimeGroup'].iloc[0] == 7:
                #    pass

        # now update the xlsx file for the groupings calculated above
        # also copy SpaceGroup results to Work column to start fusion iterations
        pfn = self.SavePath + 'AllStages_WorkingFile.xlsx'
        sheet_name='Work'
        df = pd.read_excel(pfn, sheet_name=sheet_name)
        df['SpaceGroup'] = SpaceGroupCol
        df.to_excel(pfn, sheet_name=sheet_name, index=False)

        nn = max(df['SpaceGroup']) + 1
        for k in range(0,nn):
            dfs = df[df['SpaceGroup']==k]
            print(f'>number of events for SpaceGroup {k} is {dfs.shape[0]}')



#---------------------------------------------------------------------------------------------------------------
    def Extend(self): 

        # plan is to spin through each outlier (SpaceGroup = -1) and evaluate frac planes
        # for inclusion if there is a decrease in RMSerror
        # calc individually for each stage
        # ev['ExtGroup'] will end with values of corresponding StageGroup and -1 if still outlier 

        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath) 
        ev['ExtGroup'] = ev['SpaceGroup']

        knt = 0
        for stage in range(1,4):
            # map all fracs for this stage
            evs = ev[ev['Stage']==stage]
            SpaceGroup = np.unique(evs['SpaceGroup'])[1:]
            nFrac = len(SpaceGroup) 
            hF_All = []
            for k in range(0,nFrac):
                hF = FracPlane.FracPlane()
                #thisgroup = SpaceGroup[k]
                #print(f'SpaceGroup: {thisgroup}')
                evk = evs[evs['SpaceGroup']==SpaceGroup[k]]
                hF.RootTime = evk['RootTime']
                hF.Stage = int(evk['Stage'].iloc[0])                    
                hF.TimeGroup = int(evk['TimeGroup'].iloc[0])
                hF.SpaceGroup = int(evk['SpaceGroup'].iloc[0])          # at this point, frac -> SpaceGroup is 1-to-1 (before fusion)
                hF.Calc(evk['E'], evk['N'], evk['Depth'], evk['Quality'])
                if len(hF.eigval) == 3:                     # may get here with null hF
                    hF_All.append(hF)
                else:
                    pass

            # test each outlier
            out = evs[evs['SpaceGroup']==-1]        # this selects initial outliers
            ix = out.index
            for k in range(0,len(ix)):              # loop over outliers
                #if k < 137:
                #    continue
                #if out["Stage"].loc[ix[k]] == 3:
                #    print(f'ExtGroup {out["ExtGroup"].loc[ix[137]]}, Source {out["Source"].loc[ix[137]]}')
                #if out["ExtGroup"].loc[ix[137]] > -1:
                #    pass
                rms_decr = 1e4                     
                for hF in hF_All:
                    T = FracPlane.FracPlane()
                    xp = out['E'].loc[ix[k]]
                    yp = out['N'].loc[ix[k]]
                    dp = out['Depth'].loc[ix[k]]
                    # reject if distance > 1.1*eff_rad
                    dist = (hF.Xmean[0]-xp)**2 + (hF.Xmean[1]-yp)**2 + (hF.Xmean[2]-dp)**2
                    if np.sqrt(dist) > 1.1*hF.effective_radius:
                        continue
                    xx = np.append(hF.X[:,0],xp)
                    yy = np.append(hF.X[:,1],yp)
                    dd = np.append(hF.X[:,2],dp)
                    T.Calc(xx, yy, dd)
                    #print(f'hF rad: {hF.mean_radius:.2f}, T rad: {T.mean_radius:.2f}')
                    rms_chg = T.RMSerror - hF.RMSerror 
                    rad_chg = T.effective_radius - hF.effective_radius                
                    if (rms_chg < rms_decr) and (rad_chg < 0.10*hF.effective_radius     # CRITERIA for EXTEND !!!
                                and np.dot(T.normal,hF.normal) > 0.98):
                        thisgrp = hF.SpaceGroup
                        rms_decr = rms_chg  
                        Trad = T.effective_radius             
                        hFrad = hF.effective_radius
                # add this outlier to selected SpaceGroup/frac if negative rms_chg
                # - the frac with largest decrease in RMS was selected
                if rms_decr < 0:
                    knt += 1
                    ev['ExtGroup'].loc[ix[k]] = thisgrp
#                    print(f'hF rad: {hFrad:.2f}, T rad: {Trad:.2f}')

        print(f'Extend reassigned {knt} outliers')

        # post results to the WorkingFile
        pfn = self.SavePath + 'AllStages_WorkingFile.xlsx'
        sheet_name='Work'
        df = pd.read_excel(pfn, sheet_name=sheet_name)
        df['ExtGroup'] = ev['ExtGroup']
        df.to_excel(pfn, sheet_name=sheet_name, index=False)

        # what do they look like?
        if self.diags:
            self.PlotFracPlanes2('SpaceGroup', 'ExtGroup', circles=1)



#---------------------------------------------------------------------------------------------------------------
    def Outliers(self): 

        """ # now update TimeGroupCol with label values, offset by cumulative GroupCount
            # NB: label = -1 indicates null and should remain null, must be reset after adding GroupCount
            ix = ev[ev['TimeGroup']==k].index
            values = label + FracCount
            values[label<0] = -1
            SpaceGroupCol[ix] = values """


        # start with plotting all outliers by stage and magnitude
        # work w outlier data remaining from ExtGroup

        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath)        
        print(ev.columns)

        for stage in range(1,4):

            evs = ev[ev['Stage']==stage]
            evs = evs[evs['ExtGroup']==-1]
            thing = f'Outliers_By_Mag_Stage_{stage}'
            pfn = f'{self.SavePath}{thing}.png'
            mag = np.round(2*evs['MomMag'] + 5)
            PlotENDT.PlotENDT(wb, perfint, evs['E'], evs['N'], evs['Depth'], evs['RootTime'], 
                     mag, minvec, maxvec, thing, pfn)
   
        # now try to map outliers the into spatial clusters (borrom from SpaceGroup above)

        #ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath) 
        outl = ev[ev['ExtGroup']==-1]

        # initiate FracCount from ExtGroup values
        FracCount = np.max(ev['ExtGroup']) + 1
        epsTrial = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]

        # this should start as null, but we'll fill below then save in WorkingFile
        OutlierCol = ev['Outliers'].copy()
        
        for stage in range(1,4): 
            
            out = outl[outl['Stage']==stage].copy()
            label = np.ones(len(out))

            title = (f'Stage = {stage}, N_outliers = {len(out):.0f}\n')
            spfn = f'{self.SavePath}Outliers_{stage}.png'
            PlotENDT.PlotENDT(wb, perfint, out['E'], out['N'], out['Depth'], out['RootTime'], label, minvec, maxvec, title, spfn)

            thisSpacePoints = 5            # override self.MinSpacePoints
            thisSpaceOutliers = 35         # override self.MaxSpaceOutliers

            nlabels = []
            outliers = []
            labels = []
            epsPlot = []
            epsidx = []
            for eps in epsTrial:
                nlabel, pctoutlier, label = DataSegments.space_segments(out, eps, thisSpacePoints)  
                nlabels.append(nlabel)
                outliers.append(pctoutlier)
                labels.append([label])

                # reached condition, can drop out now
                if pctoutlier <= thisSpaceOutliers:
                    epsidx = eps
                    break

            # if needed, now find results for lowest epsilon that preserves min pct of data, else the boundary condition
            if epsidx == []:
                pct = np.array(outliers)
                ept = np.array(epsTrial)
                try:
                    epsidx = ept[pct<thisSpaceOutliers][0]      # find first epsilon with less than MAX_PCT_OUTLIERS
                except:
                    epsidx = epsTrial[-1]                   # choose last one if all the way to end
                nlabel, pctoutlier, label = DataSegments.space_segments(out, epsidx, thisSpacePoints)

            if self.diags:
                title = (f'Outliers: Stage = {stage}, eps = {epsidx}, outlier = {pctoutlier:.1f}% / {thisSpaceOutliers}%, ncpt = {thisSpacePoints}\n')
                spfn = f'{self.SavePath}Outliers{stage}.png'
                PlotENDT.PlotENDT(wb, perfint, out['E'], out['N'], out['Depth'], out['RootTime'], label, minvec, maxvec, title, spfn)


            # now update FracCount and insert cumulative results in ExtGroup
            # also need to update the xlsx file for the time groupings calculated above, for each stage in turn
            # NB: use 'xf' here instead of 'df' to avoid conflicts with plot references

            # now update TimeGroupCol with label values, offset by cumulative GroupCount
            # NB: label = -1 indicates null and should remain null, must be reset after adding GroupCount
            ix = out[out['ExtGroup']==-1].index
            values = label + FracCount
            values[label<0] = -1
            OutlierCol[ix] = values

            # update GroupCount for next stage, starting value for next cycle
            FracCount += (max(label) + 1)       # update for next cycle

            # only diags from here to end of func, plot results
            if True:

                ix = epsTrial.index(epsidx)
                epsPlot = epsTrial[0:ix+1]

                fig = plt.figure(figsize=(16,10))
                plt.rcParams.update({'font.size': 12})
                ax1 = fig.add_subplot(221)
                ax1.plot(epsPlot,nlabels,'*-',linewidth=3)
                ax1.grid(True)
                ax1.set_xlabel('epsilon')
                ax1.set_ylabel('nlabels')
                ax1.set_ylim(0,max(nlabels))
                plt.title(f'Stage = {stage}\n')

                ax2 = fig.add_subplot(223)
                ax2.plot(epsPlot,outliers,'*-',linewidth=3, color='r')
                ax2.grid(True)
                ax2.set_xlabel('epsilon')
                ax2.set_ylabel('pct outliers', color='r')
                ax2.set_ylim(0,max(outliers))
                ax2.tick_params(axis='y', labelcolor='r')
                    
                ax3 = fig.add_subplot(224)
                colorset = sns.color_palette("bright", int(nlabel))
                sns.scatterplot(x=out['E'],y=out['Depth'], hue=label, palette=colorset)
                ax3.set_xlabel('Easting')
                ax3.set_ylabel('Depth')
                ax3.plot(wb['X'], wb['D'])
                for p in perfint:
                    ax3.plot([p[0][0], p[1][0]], [p[0][2], p[1][2]], color='k', linewidth=6)
                ax3.set_xlim(minvec[0],maxvec[0])
                ax3.set_ylim(minvec[2],maxvec[2])
                plt.title(f'eps = {epsidx}, outl = {pctoutlier:.1f}%, ncpt = {thisSpacePoints}\n')

                ax4 = fig.add_subplot(222)
                colorset = sns.color_palette("bright", int(nlabel))
                sns.scatterplot(x=out['E'],y=out['N'], hue=label, palette=colorset)
                ax4.set_xlabel('Easting')
                ax4.set_ylabel('Northing')
                ax4.plot(wb['X'], wb['Y'])
                for p in perfint:
                    ax4.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color='k', linewidth=6)
                ax4.set_xlim(minvec[0],maxvec[0])
                ax4.set_ylim(minvec[1],maxvec[1])

                fig.tight_layout(w_pad=2)

                pfn = f'{self.SavePath}Stage_{stage}_Outliers_Eps_{epsidx}.png'
                plt.savefig(pfn)
                #plt.show()

        # now update the xlsx file for the groupings calculated above

        #pfn = self.SavePath + 'Outliers.xlsx'
        #sheet_name='Work'
        #out.to_excel(pfn, sheet_name=sheet_name, index=False)

        # now update the xlsx file for the groupings calculated above
        pfn = self.SavePath + 'AllStages_WorkingFile.xlsx'
        sheet_name='Work'
        df = pd.read_excel(pfn, sheet_name=sheet_name)
        df['Outliers'] = OutlierCol
        df.to_excel(pfn, sheet_name=sheet_name, index=False)



#---------------------------------------------------------------------------------------------------------------
    def Merge(self): 

        # start w getting data
        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath) 

        # merge ExtGroup and Outliers together
        # since the Outlier processing introduced new frac elements
        # ix addresses elements that are ExtGrp/outliers but were filled in Outliers

        merge = ev['ExtGroup'].copy() 
        ix = (merge==-1) & (ev['Outliers'] != -9)       # Outlier blanks are (-9) for ExtGrp > -1
        #valmax = np.max(merge)                         # SpaceGroup clusters have indices starting at 0
        merge.loc[ix] = ev['Outliers'].loc[ix]          # + valmax 

        # now need to make frac numbers sequential
        xx = self.Sequential(merge)
        #ev['Sequential'] = xx

        # set columns to merge and save
        ev['Merged'] = xx

        pfn = self.SavePath + 'AllStages_WorkingFile.xlsx'
        sheet_name='Work'
        ev.to_excel(pfn, sheet_name=sheet_name, index=False)

    

#---------------------------------------------------------------------------------------------------------------
    def Sequential(self, inp):

        # each time a new value is found, will update map with corresponding reordered value

        #vals = np.unique(inp)
        newval = -1                  # will have outliers
        map = []
        out = []
        for k in range(0,len(inp)):
            if inp[k] not in map:           # new value, must bump it
                map.append(inp[k])
                out.append(newval)
                newval += 1
            else:
                out.append(map.index(inp[k])-1)

        return out


#---------------------------------------------------------------------------------------------------------------
    def Fusion(self): 

        # start w getting data
        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath) 

        ev['Work'] = ev['Merged']       # fuse Merged w Outliers

    # iterate until Work array and Fused array have same values
    # Work starts with ExtGroup, results post to FusedGroup column
    # after one pass, of course there could now be other fracs to be combined
    # so repeat until no changes between iterations
     
    # start with calculating frac planes based on Work indices
    # calculate and plot frac plane geometries 

    # iterate until done

        knt = 0
        done = False
        while not done:

            hF_All = []
            nFrac = max(ev['Work']) + 1
            for k in range(0,nFrac):
                evs = ev[ev['Work'] == k]
                #if len(evs) < 4:
                #    continue
                hF = FracPlane.FracPlane()
                hF.Stage = evs['Stage']
                hF.TimeGroup = evs['TimeGroup']
                hF.SpaceGroup = evs['SpaceGroup']
                hF.ExtGroup = evs['ExtGroup']
                hF.Outliers = evs['Outliers']
                hF.FusedGroup = k
                hF.Calc(evs['E'], evs['N'], evs['Depth'], evs['Quality'])
                hF_All.append(hF)
            self.Fracs = hF_All         # will use in Corrections below 

            # evaluate distance between frac means and angles to determine if the fracs should be merged

            iGroup = []
            jGroup = []
            meandist = []
            anglediff = []
            meanqual = []

            # double-loop to calculate diffs
            fracs = hF_All
            nfracs = len(fracs)

            for i in range(0,nfracs):
                for j in range(i+1,nfracs):
                    try:
                        dist = np.sqrt( (fracs[i].Xmean[0]-fracs[j].Xmean[0])**2  
                                        + (fracs[i].Xmean[1]-fracs[j].Xmean[1])**2 
                                        + (fracs[i].Xmean[2]-fracs[j].Xmean[2])**2)
                        angle = 57.3* np.arccos( (fracs[i].normal[0]*fracs[j].normal[0])  
                                        + (fracs[i].normal[1]*fracs[j].normal[1])
                                        + (fracs[i].normal[2]*fracs[j].normal[2]) )
                        qual = np.min((fracs[i].mean_quality, fracs[j].mean_quality)) 
                    except:
                        pass
                    iGroup.append(i)
                    jGroup.append(j)
                    meandist.append(dist)
                    anglediff.append(angle)
                    meanqual.append(qual)


            df = pd.DataFrame((iGroup, jGroup, meandist, anglediff, meanqual))
            df = df.T

            df.columns = ['iGroup','jGroup','dist','angle','qual']
            df = df.sort_values(by='dist')

            strng = self.param_string(0)
            pfn = self.SavePath + (f'{strng}_FusedFracs{knt}.xlsx')
            df.to_excel(pfn)

            # now let's fuse fracs up to a certain distance
            # start by applying the lower value of SpaceGroup to the fused frac that has higher group count
            # at the end, will renumber for sequential values

            #ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath)  
            #ev['FusedGroup'] = ev['SpaceGroup']

            ev['FusedGroup'] = ev['Work']

            fx = df[df['dist'] <= self.fuse_dist]
            for k in range(0,len(fx)):
                ixs = fx['iGroup'].iloc[k]
                jxs = fx['jGroup'].iloc[k]

                if jxs < ixs:               # should not happen, just checking
                    isave = ixs
                    ixs = jxs
                    jxs = isave
                ev.loc[ev['FusedGroup']==jxs,'FusedGroup'] = ixs
            
            # got the fused set, now fill holes and make count contiguous

            val = ev['FusedGroup'].copy()
            cc = ContiguousCount.ContiguousCount(val)
            ev['FusedGroup'] = cc
            knt += 1

            # compare Work and FusedGroup to see if we're done yet
            if np.all(ev['Work'] == ev['FusedGroup']):
                done = True

            else:
                ev['Work'] = ev['FusedGroup']


        print(f'>> Fusion iterations = {knt}')


        # NB: self.Fracs comprises data for FusedGroup at this time, from last iteration with no changes
        # reset Work to void
        ev['Work'] = (-1)*np.ones(ev.shape[0])
        pfn = self.SavePath + 'AllStages_WorkingFile.xlsx'
        sheet_name='Work'
        ev.to_excel(pfn, sheet_name=sheet_name, index=False)

        self.Fracs = hF_All         # will use in Corrections & DFN below 

#---------------------------------------------------------------------------------------------------------------
    def Connections(self): 

        # NB: Fusions and Connections must be run in the current exec before Frac2DFN
        # after Fusion, self.Fracs has most recent fracture maps
        # append prescribed fracs below to enable communication between fractures

        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath) 

        stage = [2, 2]
        strike = [20, 190]
        dip = [90, 0]
        radius = [50, 50]
        location = [np.array(perf[1])+[0, 0, 25], np.array(perf[1])+[0, 0, 50]]

        # add vertical frac and low-dip frac to Stage 2 for interconnection with perfs
        for k in range(0, len(strike)):
            hF = FracPlane.FracPlane()
            hF.MakeCircularFrac(stage[k], location[k], radius[k], dip[k], strike[k])          
            self.Fracs.append(hF)

        # now self.fracs contains all the fracs we want to send to GeoDT

   


#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#               DISPLAY SECTION OF CODE BELOW
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
    def PlotAllData(self):

        # start with plotting all data by stage and magnitude

        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step1(self.DataPath, self.SavePath)        
        print(ev.columns)

        for stage in range(1,4):

            evs = ev[ev['Stage']==stage]
            thing = f'All_Data_By_Mag_Stage_{stage}'
            pfn = f'{self.SavePath}{thing}.png'
            mag = np.round(2*evs['MomMag'] + 5)
            PlotENDT.PlotENDT(wb, perfint, evs['E'], evs['N'], evs['Depth'], evs['RootTime'], 
                     mag, minvec, maxvec, thing, pfn)


   #---------------------------------------------------------------------------------------------------------------
    def PlotFracPlanes(self, select, circles): 

    # calculate and plot FracPlane geometries for SpaceGroup
    
        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath)     

        hF_All = []
        Stage = []
        nFrac = max(ev[select]) #+ 1
        for k in range(0,nFrac):
            evs = ev[ev[select] == k]
            if len(evs) < 4:
                continue
            hF = FracPlane.FracPlane()
            hF.Stage = int(evs['Stage'].iloc[0])
            hF.TimeGroup = int(evs['TimeGroup'].iloc[0])
            hF.SpaceGroup = int(evs['SpaceGroup'].iloc[0])
            hF.ExtGroup = int(evs['ExtGroup'].iloc[0])
            hF.Calc(evs['E'], evs['N'], evs['Depth'], evs['Quality'])
            if hF.X == []:
                input('RunMEQ(969): Stop')
            hF_All.append(hF)
            Stage.append(evs['Stage'].iloc[0])

        knt = -1
        for hF in hF_All:

            # setup the figure
            fig = plt.figure(figsize=(16,12))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot(xs=wb.X, ys=wb.Y, zs=wb.D)
            for k in range(0,3):
                ax.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                            ys = [perfint[k][0][1], perfint[k][1][1]],
                            zs = [perfint[k][0][2], perfint[k][1][2]],  
                            color='k', linewidth=6)

            ax.scatter(xs=perf[1][0], ys=perf[1][1], zs=perf[1][2], color='k', marker='D', s=8)
            ax.scatter(xs=perf[2][0], ys=perf[2][1], zs=perf[2][2], color='k', marker='D', s=8)

            ax.set_xlim(minvec[0],maxvec[0])
            ax.set_ylim(minvec[1],maxvec[1])
            ax.set_zlim(minvec[2],maxvec[2])

            ax.set_xlabel('\nEasting')
            ax.set_ylabel('\nNorthing')
            ax.set_zlabel('\nDepth')

            area = []
            SSE = []

            #nFrac = len(hF_All)
            colorset = sns.color_palette("bright", 1)  #nFrac)
        
            knt += 1
            # plot event points
            ax.scatter(xs=hF.X[:,0], ys=hF.X[:,1], zs=hF.X[:,2], color=colorset[0])

            # plot normal vector from mean point, normal should always be downward pointing
            vlen = 20
            vn = vlen * hF.normal.T + hF.Xmean
            nv = np.vstack((hF.Xmean, vn))
            ax.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=colorset[0])
            ax.scatter(xs=hF.Xmean[0], ys=hF.Xmean[1], zs=hF.Xmean[2], color=colorset[0])

            #ax.scatter(xs=hF.Xhat[:,0], ys=hF.Xhat[:,1], zs=hF.Xhat[:,2], color='k')
            #ax2 = fig.add_subplot(122)
            #ax2.scatter(hF.Yhat[:,0], hF.Yhat[:,1], color='k')

            area.append(hF.area)
            SSE.append(hF.SSE)

            if not circles:
                ax.plot_trisurf(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[0], alpha=0.5)
                ax.plot(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[0], alpha=0.5)

            else:
                ax.plot_trisurf(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[0], alpha=0.5)
                ax.plot(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[0], alpha=0.5)

            #poly = Poly3DCollection(verts=np.array(hF.perim), facecolors='g', alpha=0.5)
            #ax.add_collection3d(poly)

            stagegroup = ev[ev['Stage'] == hF.Stage]
            outliers = stagegroup[stagegroup['ExtGroup'] == -1]
            #for k in range(0,len(stagegroup)):
            #ax.scatter(stagegroup['E'], stagegroup['N'], stagegroup['Depth'], color='r', marker='x', s=2)
            #for k in range(0,len(outliers)):

            # plot available outliers
            #ax.scatter(outliers['E'], outliers['N'], outliers['Depth'], color='k', marker='x', s=2)
            
            textstr0 = f'Data: {select}, Stage: {Stage[knt]}, SpaceGroup: {knt} of {nFrac-1}'     # knt is zero-based
            textstr1 = f'area = {np.round(hF.area)} m2'
            textstr2 = f'RMS = {np.round(hF.RMSerror)}'
            #outlier = 100*len(outliers)/(len(ev)-3)
            #textstr3 = f'outlier = {outlier:.1f}%'
            plt.title( textstr0 + ',   ' + textstr1 + ',   ' + textstr2)

            pfn = f'{self.SavePath}Stage_{Stage[knt]}_Grp_{knt}_Frac3d.png'
            plt.savefig(pfn)
            #plt.show()
        
        



#---------------------------------------------------------------------------------------------------------------
    def PlotFracPlanes2(self, select, select2, circles): 

    # calculate and plot FracPlane geometries for SpaceGroup
    
        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath)     

        Stage = []
        nFrac = max(ev[select]) 

        for k in range(0,nFrac):
            evs = ev[ev[select] == k]
            evs2 = ev[ev[select2] == k]
            if len(evs) < 4:
                continue
            hF = FracPlane.FracPlane()
            hF.Stage = int(evs['Stage'].iloc[0])
            hF.TimeGroup = int(evs['TimeGroup'].iloc[0])
            hF.SpaceGroup = int(evs['SpaceGroup'].iloc[0])
            hF.ExtGroup = int(evs['ExtGroup'].iloc[0])
            hF.Calc(evs['E'], evs['N'], evs['Depth'], evs['Quality'])
            if hF.X == []:
                input('RunMEQ(969): Stop')
            Stage.append(evs['Stage'].iloc[0])

            # setup the figure
            fig = plt.figure(figsize=(16,12))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot(xs=wb.X, ys=wb.Y, zs=wb.D)
            for k in range(0,3):
                ax.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                            ys = [perfint[k][0][1], perfint[k][1][1]],
                            zs = [perfint[k][0][2], perfint[k][1][2]],  
                            color='k', linewidth=6)

            ax.scatter(xs=perf[1][0], ys=perf[1][1], zs=perf[1][2], color='k', marker='D', s=8)
            ax.scatter(xs=perf[2][0], ys=perf[2][1], zs=perf[2][2], color='k', marker='D', s=8)

            ax.set_xlim(minvec[0],maxvec[0])
            ax.set_ylim(minvec[1],maxvec[1])
            ax.set_zlim(minvec[2],maxvec[2])

            ax.set_xlabel('\nEasting')
            ax.set_ylabel('\nNorthing')
            ax.set_zlabel('\nDepth')

            area = []
            SSE = []
            colorset = sns.color_palette("bright", 1)  #nFrac)
        
            # plot event points
            ax.scatter(xs=hF.X[:,0], ys=hF.X[:,1], zs=hF.X[:,2], color=colorset[0])
            ax.scatter(evs2['E'], evs2['N'], evs2['Depth'], color='r', marker='o', s=2)

            # plot normal (red) vector from mean point, normal should always be downward pointing
            vlen = 20
            vn = vlen * hF.normal.T + hF.Xmean
            nv = np.vstack((hF.Xmean, vn))
            ax.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=colorset[0])
            ax.scatter(xs=hF.Xmean[0], ys=hF.Xmean[1], zs=hF.Xmean[2], color=colorset[0])

            #ax.scatter(xs=hF.Xhat[:,0], ys=hF.Xhat[:,1], zs=hF.Xhat[:,2], color='k')
            #ax2 = fig.add_subplot(122)
            #ax2.scatter(hF.Yhat[:,0], hF.Yhat[:,1], color='k')

            area.append(hF.area)
            SSE.append(hF.SSE)

            if not circles:
                ax.plot_trisurf(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[0], alpha=0.5)
                ax.plot(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[0], alpha=0.5)

            else:
                ax.plot_trisurf(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[0], alpha=0.5)
                ax.plot(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[0], alpha=0.5)

            #poly = Poly3DCollection(verts=np.array(hF.perim), facecolors='g', alpha=0.5)
            #ax.add_collection3d(poly)

            stagegroup = ev[ev['Stage'] == hF.Stage]
            outliers = stagegroup[stagegroup['ExtGroup'] == -1]
            #for k in range(0,len(stagegroup)):
            #ax.scatter(stagegroup['E'], stagegroup['N'], stagegroup['Depth'], color='r', marker='x', s=2)
            #for k in range(0,len(outliers)):

            # plot available outliers
            #ax.scatter(outliers['E'], outliers['N'], outliers['Depth'], color='k', marker='x', s=2)
            
            textstr0 = f'Data: {select}, Stage: {hF.Stage}, SpaceGroup: {hF.SpaceGroup} of {nFrac-1}'     
            textstr1 = f'area = {np.round(hF.area)} m2'
            textstr2 = f'RMS = {np.round(hF.RMSerror)}'
            #outlier = 100*len(outliers)/(len(ev)-3)
            #textstr3 = f'outlier = {outlier:.1f}%'
            plt.title( textstr0 + ',   ' + textstr1 + ',   ' + textstr2)

            pfn = f'{self.SavePath}Stage_{hF.Stage}_Grp_{hF.SpaceGroup}_Frac3d.png'
            plt.savefig(pfn)
            #plt.show()
        
       



    #---------------------------------------------------------------------------------------------------------------
    def FracCompare(self, select1, select2): 

    # calculate, plot, and compare FracPlane geometries for two groups
    # first collect the two fracture sets
  
        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath)       

        hF_1 = []
        nFrac = max(ev[select1]) + 1
        for k in range(0,nFrac):
            ev1 = ev[ev[select1] == k]
            hF = FracPlane.FracPlane()
            hF.RootTime = ev1['RootTime']
            hF.Stage = ev1['Stage']
            hF.TimeGroup = ev1['TimeGroup']
            hF.SpaceGroup = ev1['SpaceGroup']
            hF.FusedGroup = ev1['FusedGroup']
            hF.Calc(ev1['E'], ev1['N'], ev1['Depth'], ev1['Quality'])
            if len(hF.eigval) == 3:                     # may get here with null hF
                hF_1.append(hF)

        hF_2 = []
        nFrac = max(ev[select2]) + 1
        for k in range(0,nFrac):
            ev2 = ev[ev[select2] == k]
            hF = FracPlane.FracPlane()
            hF.RootTime = ev2['RootTime']
            hF.Stage = ev2['Stage']
            hF.TimeGroup = ev2['TimeGroup']
            hF.SpaceGroup = ev2['SpaceGroup']
            hF.FusedGroup = ev2['FusedGroup']
            hF.Calc(ev2['E'], ev2['N'], ev2['Depth'], ev2['Quality'])
            if len(hF.eigval) == 3:                     # may get here with null hF
                hF_2.append(hF)


        

        while True:

            # get frac indices
            f1 = int(input(f'Select frac for {select1}: '))
            f2 = int(input(f'Select frac for {select2}: '))

            # setup the figure
            fig = plt.figure(figsize=(16,9))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot(xs=wb.X, ys=wb.Y, zs=wb.D)
            for k in range(0,3):
                ax1.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                            ys = [perfint[k][0][1], perfint[k][1][1]],
                            zs = [perfint[k][0][2], perfint[k][1][2]],  
                            color='k', linewidth=6)
            ax1.scatter(xs=perf[1][0], ys=perf[1][1], zs=perf[1][2], color='k', marker='D', s=8)
            ax1.scatter(xs=perf[2][0], ys=perf[2][1], zs=perf[2][2], color='k', marker='D', s=8)
            ax1.set_xlim(minvec[0],maxvec[0])
            ax1.set_ylim(minvec[1],maxvec[1])
            ax1.set_zlim(minvec[2],maxvec[2])
            ax1.set_xlabel('\nEasting')
            ax1.set_ylabel('\nNorthing')
            ax1.set_zlabel('\nDepth')
            ax1.set_title(f'{select1} frac {f1}')

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot(xs=wb.X, ys=wb.Y, zs=wb.D)
            for k in range(0,3):
                ax2.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                            ys = [perfint[k][0][1], perfint[k][1][1]],
                            zs = [perfint[k][0][2], perfint[k][1][2]],  
                            color='k', linewidth=6)
            ax2.scatter(xs=perf[1][0], ys=perf[1][1], zs=perf[1][2], color='k', marker='D', s=8)
            ax2.scatter(xs=perf[2][0], ys=perf[2][1], zs=perf[2][2], color='k', marker='D', s=8)
            ax2.set_xlim(minvec[0],maxvec[0])
            ax2.set_ylim(minvec[1],maxvec[1])
            ax2.set_zlim(minvec[2],maxvec[2])
            ax2.set_xlabel('\nEasting')
            ax2.set_ylabel('\nNorthing')
            ax2.set_zlabel('\nDepth')
            ax2.set_title(f'{select2} frac {f2}')

            # frac 1
            hF = hF_1[f1]
            ax1.scatter(xs=hF.X[:,0], ys=hF.X[:,1], zs=hF.X[:,2], color='b')
            # plot normal vector from mean point, normal should always be downward pointing
            vlen = 20
            vn = vlen * hF.normal.T + hF.Xmean
            nv = np.vstack((hF.Xmean, vn))
            ax1.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color='b')
            ax1.scatter(xs=hF.Xmean[0], ys=hF.Xmean[1], zs=hF.Xmean[2], color='b')
            ax1.plot_trisurf(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color='b', alpha=0.5)
            ax1.plot(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color='b', alpha=0.5)

            # frac 2
            hF = hF_2[f2]
            ax2.scatter(xs=hF.X[:,0], ys=hF.X[:,1], zs=hF.X[:,2], color='b')
            # plot normal vector from mean point, normal should always be downward pointing
            vlen = 20
            vn = vlen * hF.normal.T + hF.Xmean
            nv = np.vstack((hF.Xmean, vn))
            ax2.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color='b')
            ax2.scatter(xs=hF.Xmean[0], ys=hF.Xmean[1], zs=hF.Xmean[2], color='b')
            ax2.plot_trisurf(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color='b', alpha=0.5)
            ax2.plot(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color='b', alpha=0.5)

            plt.show(block=False)

            #input('Enter to continue')
               



    #---------------------------------------------------------------------------------------------------------------
    def AllStagesFracPlaneDiags(self): 

    # calculate and plot FracPlane geometries for FusedGroup
  
        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath)       

        hF_All = []
        nFrac = max(ev['FusedGroup']) + 1
        for k in range(0,nFrac):
            evs = ev[ev['FusedGroup'] == k]
            hF = FracPlane.FracPlane()
            hF.RootTime = evs['RootTime']
            hF.Stage = evs['Stage']
            hF.TimeGroup = evs['TimeGroup']
            hF.SpaceGroup = evs['SpaceGroup']
            hF.FusedGroup = evs['FusedGroup']
            hF.Calc(evs['E'], evs['N'], evs['Depth'], evs['Quality'])
            if len(hF.eigval) == 3:                     # may get here with null hF
                hF_All.append(hF)

            try: 
                stage = hF.Stage.iloc[0]
            except:
                stage = hF.Stage

            # plot each FusedGroup
            if self.diags:
                title = (f'FusedGroup = {k}, {self.param_string(stage)}\n')
                spfn = f'{self.SavePath}FusedGroup{k}.png'
                PlotFrac.PlotFrac(hF, minvec, maxvec, title, spfn)

        self.FusedFracs = hF_All

        # setup the figure
        fig = plt.figure(figsize=(16,12))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(xs=wb.X, ys=wb.Y, zs=wb.D)
        for k in range(0,3):
            ax.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                        ys = [perfint[k][0][1], perfint[k][1][1]],
                        zs = [perfint[k][0][2], perfint[k][1][2]],  
                        color='k', linewidth=6)


        ax.scatter(xs=perf[1][0], ys=perf[1][1], zs=perf[1][2], color='k', marker='D', s=8)
        ax.scatter(xs=perf[2][0], ys=perf[2][1], zs=perf[2][2], color='k', marker='D', s=8)

        ax.set_xlim(minvec[0],maxvec[0])
        ax.set_ylim(minvec[1],maxvec[1])
        ax.set_zlim(minvec[2],maxvec[2])

        ax.set_xlabel('\nEasting')
        ax.set_ylabel('\nNorthing')
        ax.set_zlabel('\nDepth')
    
        area = []
        SSE = []
        
        # calc full diags
        if True:

            nFrac = len(hF_All)
            colorset = sns.color_palette("bright", nFrac)         
            knt = -1
            for hF in hF_All:
                
                area.append(hF.area)
                SSE.append(hF.SSE)
            
                knt += 1
                thiscolor = colorset[knt]

                # plot event points
                ax.scatter(xs=hF.X[:,0], ys=hF.X[:,1], zs=hF.X[:,2], color=thiscolor)

                # plot normal vector from mean point, normal should always be downward pointing
                vlen = 20
                vn = vlen * hF.normal.T + hF.Xmean
                nv = np.vstack((hF.Xmean, vn))
                ax.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=thiscolor)
                ax.scatter(xs=hF.Xmean[0], ys=hF.Xmean[1], zs=hF.Xmean[2], color=thiscolor)

                if False:
                    ax.plot_trisurf(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=thiscolor, alpha=0.5)
                    ax.plot(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=thiscolor, alpha=0.5)
                else:
                    ax.plot_trisurf(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=thiscolor, alpha=0.25)
                    ax.plot(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=thiscolor, alpha=0.25)

            outliers = ev[ev['FusedGroup'] == -1]
            ax.scatter(outliers['E'], outliers['N'], outliers['Depth'], color='k', marker='x', s=2)
            
            textstr1 = f'area = {np.round(sum(area))} m2'
            textstr2 = f'SSE = {np.round(sum(SSE))}'
            len_outliers = len(outliers)
            outlier = 100*len_outliers/(len(ev) - 3)     # discount three perforating events
            textstr3 = f'outlier = {outlier:.1f}%'
            plt.title(self.param_string(0) + '   ' + textstr1 + ',   ' + textstr2 + ',   ' + textstr3)

            pfn = f'{self.SavePath}FusedFrac3d.png'
            plt.savefig(pfn)
            #plt.show()
            
        # calc vars to return if no diags
        else:

            for hF in hF_All:

                area.append(hF.area)
                SSE.append(hF.SSE)

                outliers = ev[ev['FusedGroup'] == -1]
                len_outliers = len(outliers)


        return len(hF_All), len_outliers, sum(area), sum(SSE)
 

#---------------------------------------------------------------------------------------------------------------
    def TimeLapse_1(self, select, circles): 

    # calculate and plot FracPlane geometries for SpaceGroup
    
        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath)     

        hF_All = []
        Stage = []
        area = 0
        nFrac = max(ev[select]) + 1
        for k in range(0,nFrac):
            evs = ev[ev[select] == k]
            hF = FracPlane.FracPlane()
            hF.Stage = int(evs['Stage'].iloc[0])
            hF.TimeGroup = int(evs['TimeGroup'].iloc[0])
            hF.SpaceGroup = int(evs['SpaceGroup'].iloc[0])
            hF.Calc(evs['E'], evs['N'], evs['Depth'], evs['Quality'])
            hF_All.append(hF)
            #if len(hF.eigval) == 3:     # may get here with null hF (should be taken care of in SpaceGroup)
            area += hF.area
            Stage.append(evs['Stage'].iloc[0])

        # first the setup
        evs = ev

        minvec = np.zeros(3)
        maxvec = np.zeros(3)
        minvec[0], maxvec[0] = 875, 1250
        minvec[1], maxvec[1] = -425, -50
        minvec[2], maxvec[2] = -1050, -650

        # setup the figure
        fig = plt.figure(figsize=(16,12))
        ax = fig.add_subplot(111, projection=Axes3D.name)

        plt.title(f'\n\n\n\n\n\nFracture Stimulation of FORGE 16A(78)-32, Stages 1-3\nArea = {area:.0f} m^3')

        ax.plot(xs=wb.X, ys=wb.Y, zs=wb.D)
        for k in range(0,3):
            ax.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                        ys = [perfint[k][0][1], perfint[k][1][1]],
                        zs = [perfint[k][0][2], perfint[k][1][2]],  
                        color='k', linewidth=6)

        ax.scatter(xs=perf[1][0], ys=perf[1][1], zs=perf[1][2], color='k', marker='D', s=8)
        ax.scatter(xs=perf[2][0], ys=perf[2][1], zs=perf[2][2], color='k', marker='D', s=8)

        ax.set_xlim(minvec[0],maxvec[0])
        ax.set_ylim(minvec[1],maxvec[1])
        ax.set_zlim(minvec[2],maxvec[2])

        ax.set_xlabel('\nEasting')
        ax.set_ylabel('\nNorthing')
        ax.set_zlabel('\nDepth')

        # animation ref: https://matplotlib.org/stable/gallery/animation/animation_demo.html#sphx-glr-gallery-animation-animation-demo-py

        # now we're ready - too slow if plot one point in turn so need to batch it
        # but would like to maintain points colored by frac

        fracs = evs[select] + 1               # bump for outliers = -1, which will have 0-index in fracs
        stage_fracs = np.unique(fracs)
        colorset = sns.color_palette("bright", np.max(stage_fracs)+1)       # bump for 0 
        npts = len(evs['E'])
        touch = np.zeros(np.max(stage_fracs)+1)
        dx = 100
        px = fracs.index[0] - dx
        nbatch = npts // dx
        if nbatch*dx < npts:
            nbatch += 1
        for k in range(0,nbatch):
            px += dx
            kx = np.arange(px,np.min((px+dx, fracs.index[0]+npts)))
            print(f'Now displaying points: {kx}')
            plt.pause(0.5)

            kfrac = np.unique(fracs[kx]) 
            kout = kx[fracs[kx]==0]
            keve = kx[fracs[kx]!=0]

            cvec = []
            for thisk in keve:
                cvec.append(colorset[fracs[thisk]])
                print(f'kx: {thisk}, fracs: {fracs[thisk]}, cvec: {colorset[fracs[thisk]]}')

            if ( len(keve) > 0):
                ax.scatter(xs=evs['E'].loc[keve], ys=evs['N'].loc[keve], zs=evs['Depth'].loc[keve], c=cvec, marker='o', s=8)
            if ( len(kout) > 0):
                ax.scatter(xs=evs['E'].loc[kout], ys=evs['N'].loc[kout], zs=evs['Depth'].loc[kout], color='k', marker='.', s=4)

            for kf in kfrac:
                if kf == 0:                 # don't try frac for outliers, remember +1 so kf=0 is outliers
                    continue
                try:
                    hF = hF_All[kf-1]           # adjust for outlier +1  !!!
                    if len(hF.eigval) == 3:     # may get here with null hF
                        if touch[kf] == 0:
                            touch[kf] = 1
                            if circles:
                                print(f'kf {kf} colorset: {colorset[kf]}')
                                ax.plot_trisurf(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[kf], alpha=0.25)
                                ax.plot(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[kf], alpha=0.25)
                            else:
                                ax.plot_trisurf(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[kf], alpha=0.5)
                                ax.plot(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[kf], alpha=0.5)
                            # plot normal vector from mean point, normal should always be downward pointing
                            vlen = 20
                            vn = vlen * hF.normal.T + hF.Xmean
                            nv = np.vstack((hF.Xmean, vn))
                            ax.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=colorset[kf])
                except:
                    pass

        return

#---------------------------------------------------------------------------------------------------------------
    def TimeLapse_3(self, select, circles): 

    # calculate and plot FracPlane geometries for SpaceGroup
    
        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath)     

        hF_All = []
        Stage = []
        nFrac = max(ev[select]) + 1
        for k in range(0,nFrac):
            evs = ev[ev[select] == k]
            hF = FracPlane.FracPlane()
            hF.Stage = int(evs['Stage'].iloc[0])
            hF.TimeGroup = int(evs['TimeGroup'].iloc[0])
            hF.SpaceGroup = int(evs['SpaceGroup'].iloc[0])
            hF.Calc(evs['E'], evs['N'], evs['Depth'], evs['Quality'])
            hF_All.append(hF)
            Stage.append(evs['Stage'].iloc[0])

        # plot 3d by stage w closer boundaries
        for stage in range(1,4):

            # first the setup
            evs = ev[ev['Stage']==stage]

            minvec = np.zeros(3)
            maxvec = np.zeros(3)
            if stage == 1:
                minvec[0], maxvec[0] = 1075, 1225
                minvec[1], maxvec[1] = -400, -200
                minvec[2], maxvec[2] = -1050, -775
                #minvec[0], maxvec[0] = 900, 1275
                #minvec[1], maxvec[1] = -400, -100
                #minvec[2], maxvec[2] = -1050, -650
            elif stage == 2:
                minvec[0], maxvec[0] = 1000, 1150
                minvec[1], maxvec[1] = -400, -200
                minvec[2], maxvec[2] = -1000, -650
            elif stage == 3:
                minvec[0], maxvec[0] = 900, 1100
                minvec[1], maxvec[1] = -400, -50
                minvec[2], maxvec[2] = -1000, -650

            # setup the figure
            fig = plt.figure(figsize=(16,12))
            ax = fig.add_subplot(111, projection=Axes3D.name)
            #ax = fig.add_subplot(111, projection='3d')
            #ax = Axes3D(fig)
            plt.suptitle = f'Stage {stage}'

            ax.plot(xs=wb.X, ys=wb.Y, zs=wb.D)
            for k in range(0,3):
                ax.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                            ys = [perfint[k][0][1], perfint[k][1][1]],
                            zs = [perfint[k][0][2], perfint[k][1][2]],  
                            color='k', linewidth=6)

            ax.scatter(xs=perf[1][0], ys=perf[1][1], zs=perf[1][2], color='k', marker='D', s=8)
            ax.scatter(xs=perf[2][0], ys=perf[2][1], zs=perf[2][2], color='k', marker='D', s=8)

            ax.set_xlim(minvec[0],maxvec[0])
            ax.set_ylim(minvec[1],maxvec[1])
            ax.set_zlim(minvec[2],maxvec[2])

            ax.set_xlabel('\nEasting')
            ax.set_ylabel('\nNorthing')
            ax.set_zlabel('\nDepth')

            # animation ref: https://matplotlib.org/stable/gallery/animation/animation_demo.html#sphx-glr-gallery-animation-animation-demo-py

            # now we're ready - too slow if plot one point in turn so need to batch it
            # but would like to maintain points colored by frac

            fracs = evs[select] + 1               # bump for outliers = -1, which will have 0-index in fracs
            stage_fracs = np.unique(fracs)
            colorset = sns.color_palette("bright", np.max(stage_fracs)+1)       # bump for 0 
            npts = len(evs['E'])
            touch = np.zeros(np.max(stage_fracs)+1)
            dx = 10
            px = fracs.index[0] - dx
            nbatch = npts // dx
            if nbatch*dx < npts:
                nbatch += 1
            for k in range(0,nbatch):
                px += dx
                kx = np.arange(px,np.min((px+dx, fracs.index[0]+npts)))
                print(f'Now displaying points: {kx}')
                plt.pause(0.5)

                kfrac = np.unique(fracs[kx]) 
                kout = kx[fracs[kx]==0]
                keve = kx[fracs[kx]!=0]

                cvec = []
                for thisk in keve:
                    cvec.append(colorset[fracs[thisk]])
                    print(f'kx: {thisk}, fracs: {fracs[thisk]}, cvec: {colorset[fracs[thisk]]}')

                if ( len(keve) > 0):
                    ax.scatter(xs=evs['E'].loc[keve], ys=evs['N'].loc[keve], zs=evs['Depth'].loc[keve], c=cvec, marker='o', s=8)
                if ( len(kout) > 0):
                    ax.scatter(xs=evs['E'].loc[kout], ys=evs['N'].loc[kout], zs=evs['Depth'].loc[kout], color='k', marker='.', s=4)


                #kz = (kx == 0)
                #print(kx[kz])

                # simple dot if outlier, but if a frac point then plot point and frac if not yet rendered
                #if kfrac == -1:
                #    ax.scatter(xs=evs['E'].iloc[kx], ys=evs['N'].iloc[kx], zs=evs['Depth'].iloc[kx], color='k', marker='x', s=3)
                #else:
                #    ax.scatter(xs=evs['E'].iloc[kx], ys=evs['N'].iloc[kx], zs=evs['Depth'].iloc[kx], color=colorset[0], marker='o', s=8)

                for kf in kfrac:
                    if kf == 0:                 # don't try frac for outliers, remember +1 so kf=0 is outliers
                        continue
                    hF = hF_All[kf-1]           # adjust for outlier +1  !!!
                    if len(hF.eigval) == 3:     # may get here with null hF
                        if touch[kf] == 0:
                            touch[kf] = 1
                            if circles:
                                ax.plot_trisurf(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[kf], alpha=0.25)
                                ax.plot(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[kf], alpha=0.25)
                            else:
                                ax.plot_trisurf(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[kf], alpha=0.5)
                                ax.plot(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[kf], alpha=0.5)
                            # plot normal vector from mean point, normal should always be downward pointing
                            vlen = 20
                            vn = vlen * hF.normal.T + hF.Xmean
                            nv = np.vstack((hF.Xmean, vn))
                            ax.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=colorset[kf])
         


#---------------------------------------------------------------------------------------------------------------
    def TimeLapse_Save(self, select, circles): 

    # calculate and plot FracPlane geometries for SpaceGroup
    
        ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath)     

        hF_All = []
        Stage = []
        nFrac = max(ev[select]) + 1
        for k in range(0,nFrac):
            evs = ev[ev[select] == k]
            hF = FracPlane.FracPlane()
            hF.Stage = int(evs['Stage'].iloc[0])
            hF.TimeGroup = int(evs['TimeGroup'].iloc[0])
            hF.SpaceGroup = int(evs['SpaceGroup'].iloc[0])
            hF.Calc(evs['E'], evs['N'], evs['Depth'], evs['Quality'])
            hF_All.append(hF)
            Stage.append(evs['Stage'].iloc[0])

        # plot 3d by stage w closer boundaries
        for stage in range(1,2):

            evs = ev[ev['Stage']==stage]

            minvec = np.zeros(3)
            maxvec = np.zeros(3)
            if stage == 1:
                minvec[0], maxvec[0] = 1075, 1225
                minvec[1], maxvec[1] = -400, -200
                minvec[2], maxvec[2] = -1050, -775
            elif stage == 2:
                minvec[0], maxvec[0] = 1000, 1150
                minvec[1], maxvec[1] = -400, -200
                minvec[2], maxvec[2] = -1000, -650
            elif stage == 3:
                minvec[0], maxvec[0] = 900, 1100
                minvec[1], maxvec[1] = -400, -50
                minvec[2], maxvec[2] = -1000, -650

            # setup the figure
            fig = plt.figure(figsize=(16,12))
            ax = fig.add_subplot(111, projection=Axes3D.name)
            #ax = fig.add_subplot(111, projection='3d')
            #ax = Axes3D(fig)
            plt.suptitle = f'Stage {stage}'

            ax.plot(xs=wb.X, ys=wb.Y, zs=wb.D)
            for k in range(0,3):
                ax.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                            ys = [perfint[k][0][1], perfint[k][1][1]],
                            zs = [perfint[k][0][2], perfint[k][1][2]],  
                            color='k', linewidth=6)

            ax.scatter(xs=perf[1][0], ys=perf[1][1], zs=perf[1][2], color='k', marker='D', s=8)
            ax.scatter(xs=perf[2][0], ys=perf[2][1], zs=perf[2][2], color='k', marker='D', s=8)

            ax.set_xlim(minvec[0],maxvec[0])
            ax.set_ylim(minvec[1],maxvec[1])
            ax.set_zlim(minvec[2],maxvec[2])

            ax.set_xlabel('\nEasting')
            ax.set_ylabel('\nNorthing')
            ax.set_zlabel('\nDepth')

            # animation ref: https://matplotlib.org/stable/gallery/animation/animation_demo.html#sphx-glr-gallery-animation-animation-demo-py

            # plot one point in turn, colored by frac stage
            stage_fracs = np.unique(evs['Merged'])
            colorset = sns.color_palette("bright", np.max(stage_fracs)+1)
            npts = len(evs['E'])
            touch = np.zeros(np.max(stage_fracs)+1)
            dx = 20
            px = -dx
            nbatch = npts // dx
            if nbatch*dx < npts:
                nbatch += 1
            for k in range(0,nbatch):
                px += dx
                kx = np.arange(px,np.min((px+dx,npts)))
                print(f'Now diplaying points: {kx}')
                plt.pause(0.25)
                kfrac = np.unique(evs['Merged'].iloc[kx])
                kfrac = np.delete(kfrac,kfrac==-1)

                ax.scatter(xs=evs['E'].iloc[kx], ys=evs['N'].iloc[kx], zs=evs['Depth'].iloc[kx], color='k', marker='x', s=3)

                # simple dot if outlier, but if a frac point then plot point and frac if not yet rendered
                #if kfrac == -1:
                #    ax.scatter(xs=evs['E'].iloc[kx], ys=evs['N'].iloc[kx], zs=evs['Depth'].iloc[kx], color='k', marker='x', s=3)
                #else:
                #    ax.scatter(xs=evs['E'].iloc[kx], ys=evs['N'].iloc[kx], zs=evs['Depth'].iloc[kx], color=colorset[0], marker='o', s=8)
                for kf in kfrac:
                    hF = hF_All[kf]
                    if len(hF.eigval) == 3:                     # may get here with null hF
                        if touch[kf] == 0:
                            touch[kf] = 1
                            if circles:
                                ax.plot_trisurf(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[kf], alpha=0.25)
                                ax.plot(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[kf], alpha=0.25)
                            else:
                                ax.plot_trisurf(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[kf], alpha=0.5)
                                ax.plot(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[kf], alpha=0.5)
                            # plot normal vector from mean point, normal should always be downward pointing
                            vlen = 20
                            vn = vlen * hF.normal.T + hF.Xmean
                            nv = np.vstack((hF.Xmean, vn))
                            ax.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=colorset[kf])



            currGroup = hF_All[1].TimeGroup
            knt = -1
            for hF in hF_All:

                knt += 1
                if hF.TimeGroup != currGroup:
                    plt.pause(0.25)
                    currGroup = hF.TimeGroup

                """ ax.plot(xs=wb.X, ys=wb.Y, zs=wb.D)
                for k in range(0,3):
                    ax.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                                ys = [perfint[k][0][1], perfint[k][1][1]],
                                zs = [perfint[k][0][2], perfint[k][1][2]],  
                                color='k', linewidth=6)

                ax.scatter(xs=perf[1][0], ys=perf[1][1], zs=perf[1][2], color='k', marker='D', s=8)
                ax.scatter(xs=perf[2][0], ys=perf[2][1], zs=perf[2][2], color='k', marker='D', s=8)

                ax.set_xlim(minvec[0],maxvec[0])
                ax.set_ylim(minvec[1],maxvec[1])
                ax.set_zlim(minvec[2],maxvec[2])

                ax.set_xlabel('\nEasting')
                ax.set_ylabel('\nNorthing')
                ax.set_zlabel('\nDepth') """
            
                # plot event points
                #ax.scatter(xs=hF.X[:,0], ys=hF.X[:,1], zs=hF.X[:,2], color=colorset[knt])


                #ax.scatter(xs=hF.Xhat[:,0], ys=hF.Xhat[:,1], zs=hF.Xhat[:,2], color='k')
                #ax2 = fig.add_subplot(122)
                #ax2.scatter(hF.Yhat[:,0], hF.Yhat[:,1], color='k')

                if circles:
                    ax.plot_trisurf(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[knt], alpha=0.25)
                    ax.plot(hF.circle[:,0], hF.circle[:,1], hF.circle[:,2], color=colorset[knt], alpha=0.25)
                else:
                    ax.plot_trisurf(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[knt], alpha=0.5)
                    ax.plot(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[knt], alpha=0.5)

                # plot normal vector from mean point, normal should always be downward pointing
                vlen = 20
                vn = vlen * hF.normal.T + hF.Xmean
                nv = np.vstack((hF.Xmean, vn))
                ax.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=colorset[knt])
                ax.scatter(xs=hF.Xmean[0], ys=hF.Xmean[1], zs=hF.Xmean[2], color=colorset[knt])

                #poly = Poly3DCollection(verts=np.array(hF.perim), facecolors='g', alpha=0.5)
                #ax.add_collection3d(poly)

                #spacegroup = ev[ev['Stage'] == hF.Stage]
                #outliers = spacegroup[spacegroup['SpaceGroup'] == -1]
                #for k in range(0,len(spacegroup)):
                
                ax.scatter(hF.X[:,0], hF.X[:,1], hF.X[:,2], color=colorset[knt], marker='.', s=3)
                #for k in range(0,len(outliers)):
                #ax.scatter(outliers['E'], outliers['N'], outliers['Depth'], color='k', marker='x', s=2)
                
                textstr0 = f'Stage: {Stage[knt]},  SpaceGroup: {knt} of {nFrac-1}'     # knt is zero-based
                textstr1 = f'area = {np.round(hF.area)} m2'
                textstr2 = f'SSE = {np.round(hF.SSE)}'
                #outlier = 100*len(outliers)/(len(ev)-3)
                #textstr3 = f'outlier = {outlier:.1f}%'
                plt.title( textstr0 + ',   ' + textstr1 + ',   ' + textstr2)

                pfn = f'{self.SavePath}Frac3d.png'
                plt.savefig(pfn)
                #plt.show()


