''' Main driver to run FracPlane.py test protocol
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplstereonet

import FetchAllStages
import FracPlane


def example_plot(ax, fontsize=12):
    ax.plot([1, 2])

    #ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

def example_plot3(ax, fontsize=12):
    ax.plot(xs=(0,0,0), ys=(1,1,1), zs=(2,2,2))

    #ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_zlabel('z-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)


#---------------------------------------------------------------------------------------------------------------
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


#---------------------------------------------------------------------------------------------------------------
def StageDataPlots(self, select):

# get the data
    ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath) 

    #ev = ev[ev[select]>-1]   # may want to plot outliers

    done_connection = True
    pointsplot = False
    for stage in range(1,4):
        evs = ev[ev['Stage']==stage]
        nstage = len(evs)
        outliers = evs[evs[select]==-1]     # pick outliers first, then remove
        evs = evs[evs[select]>=-1]          # exclude outliers here
        colm = evs[select]
        selectvals = np.unique(colm)
        selectvals = selectvals[selectvals>=0]
        nfracs = len(selectvals)

        # collect data to make a set of fracs
        emin = min(evs['E']) #- 10
        emax = max(evs['E']) #+ 10
        nmin = min(evs['N']) #- 10
        nmax = max(evs['N']) #+ 10
        dmin = min(evs['Depth']) #- 10
        dmax = max(evs['Depth']) #+ 10

        # Setup the display fields
        fig = plt.figure(figsize=(14,12))
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2, projection='3d')
        ax2 = plt.subplot2grid((3, 3), (2, 0), projection='stereonet')
        ax3 = plt.subplot2grid((3, 3), (2, 1), projection='polar')
        ax4 = plt.subplot2grid((3, 3), (0, 2))
        ax5 = plt.subplot2grid((3, 3), (1, 2))
        ax6 = plt.subplot2grid((3, 3), (2, 2))

        hF = []                         # not using hF_All here for brevity
        strikes = []
        dips = []
        area = []
        SSE = []
        colorset = sns.color_palette("bright", nfracs)
        for j in range(0,nfracs):
            #if j != 1:
            #    continue
            df = evs[evs[select]==selectvals[j]]
            #print(df)
            h = FracPlane.FracPlane
            # must include h as first arg below
            h.Calc(h, xev=df['E'], yev=df['N'], dev=df['Depth'], qual=df['Quality'])
            hF.append(h)
            #print(h.strike)
            strikes.append(h.strike)
            dips.append(h.dip)
            area.append(h.area)
            SSE.append(h.SSE)

            E = df['E']
            N = df['N']
            D = df['Depth']
            
            if pointsplot:

                # got the frac data, now build out the plot; start w 3d
                # plot event points
                ax1.scatter(xs=E, ys=N, zs=D, color=colorset[j])

                # plot normal vector from mean point
                vlen = 20           # +ve value since now depth is -ve
                vn = vlen * h.normal.T + h.Xmean
                nv = np.vstack((h.Xmean, vn))
                ax1.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=colorset[j])
                ax1.scatter(xs=h.Xmean[0], ys=h.Xmean[1], zs=h.Xmean[2], color=colorset[j])
                #ax.scatter(xs=h.Xhat[:,0], ys=h.Xhat[:,1], zs=h.Xhat[:,2], color='k')

                ax1.plot_trisurf(h.perim[:,0], h.perim[:,1], h.perim[:,2], color=colorset[j], alpha=0.5)
                ax1.plot(h.perim[:,0], h.perim[:,1], h.perim[:,2], color=colorset[j], alpha=0.5)

            else:
                ax1.plot_trisurf(h.circle[:,0], h.circle[:,1], h.circle[:,2], color=colorset[j], alpha=0.5)
                ax1.plot(h.circle[:,0], h.circle[:,1], h.circle[:,2], color=colorset[j], alpha=0.5)
      
                if stage == 2 and not done_connection:
                    done_connection = True
                    f1, f2 = self.S2_Connections()
                    ax1.plot_trisurf(f1.circle[:,0], f1.circle[:,1], f1.circle[:,2], color=colorset[j], alpha=0.5)
                    ax1.plot(f1.circle[:,0], f1.circle[:,1], f1.circle[:,2], color=colorset[j], alpha=0.5)
                    ax1.plot_trisurf(f2.circle[:,0], f2.circle[:,1], f2.circle[:,2], color=colorset[j], alpha=0.5)
                    ax1.plot(f2.circle[:,0], f2.circle[:,1], f2.circle[:,2], color=colorset[j], alpha=0.5)

            # plot outliers
            ax1.scatter(xs=outliers['E'], ys=outliers['N'], zs=outliers['Depth'], color='k', marker='x', s=2)


            set_axes_equal(ax1)
            ax1.set_xlabel('Easting')
            ax1.set_ylabel('Northing')           
            ax1.set_zlabel('Depth')
            
            textstr1 = f'\n\nStage {stage}: area = {np.round(sum(area))} m2'
            textstr2 = f'SSE = {np.round(sum(SSE))} m2'
            textstr3 = f'outliers = {len(outliers)} / {nstage}'
            ax1.set_title(textstr1 + ',  ' + textstr2 + ',  ' + textstr3)

            # now add the wellbore and perf intervals
            ax1.plot(xs=wb.X[-40:], ys=wb.Y[-40:], zs=wb.D[-40:], color='b')
            for k in range(0,3):
                ax1.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                        ys = [perfint[k][0][1], perfint[k][1][1]],
                        zs = [perfint[k][0][2], perfint[k][1][2]],  
                        color='k', linewidth=6)


            # second plot is pole stereonet plot + rose diagram for frac azimuth
            # http://geologyandpython.com/structural_geology.html
            # most of this is done after loop over fractures when all data is collected

            ax2.pole(h.strike, h.dip, color=colorset[j], markersize=18)
            #ax.rake(h.strike, h.dip, -25)
            ax2.grid()
            

            # E vs N
            sns.scatterplot(ax=ax4, x=E, y=N, color=colorset[j])
            ax4.set_xlim(emin, emax)
            ax4.set_ylim(nmin, nmax)
            ax4.set_xlabel('Easting')
            ax4.set_ylabel('Northing')
            ax4.plot(h.perim[:,0], h.perim[:,1], color=colorset[j], alpha=0.5)
            # add the wellbore and perf intervals
            ax4.plot(wb.X[-40:], wb.Y[-40:], color='b')
            for k in range(0,3):
                ax4.plot([perfint[k][0][0], perfint[k][1][0]], 
                        [perfint[k][0][1], perfint[k][1][1]],  
                        color='k', linewidth=6)
            ax4.axis('equal')

            # E vs D
            sns.scatterplot(ax=ax5, x=E, y=D, color=colorset[j])
            ax5.set_xlim(emin, emax)
            ax5.set_ylim(dmin, dmax)
            ax5.set_xlabel('Easting')
            ax5.set_ylabel('Depth')
            ax5.plot(h.perim[:,0], h.perim[:,2], color=colorset[j], alpha=0.5)
            # add the wellbore and perf intervals
            ax5.plot(wb.X[-40:], wb.D[-40:], color='b')
            for k in range(0,3):
                ax5.plot([perfint[k][0][0], perfint[k][1][0]], 
                        [perfint[k][0][2], perfint[k][1][2]],  
                        color='k', linewidth=6)
            ax5.axis('equal')

            # E vs D
            sns.scatterplot(ax=ax6, x=N, y=D, color=colorset[j])
            ax6.set_xlim(nmin, nmax)
            ax6.set_ylim(dmin, dmax)
            ax6.set_xlabel('Northing')
            ax6.set_ylabel('Depth')
            ax6.plot(h.perim[:,1], h.perim[:,2], color=colorset[j], alpha=0.5)
            # add the wellbore and perf intervals
            ax6.plot(wb.Y[-40:], wb.D[-40:], color='b')
            for k in range(0,3):
                ax6.plot([perfint[k][0][1], perfint[k][1][1]], 
                        [perfint[k][0][2], perfint[k][1][2]],  
                        color='k', linewidth=6)
            ax6.axis('equal')

        # put cumulative density chart on stereonet
        ax2.density_contourf(strikes, dips, measurement='poles', cmap='Greens')  #cmap='Reds')
        #ax2.rake(strikes, dips, -25)
        #ax2.plane(strikes, dips, 'g-', linewidth=2)
        ax2.set_title('Stereonet Pole Map', y=1.10, fontsize=12)

        # rose plot must be done at the end after all fracs processed
        bin_edges = np.arange(-5, 366, 10)
        number_of_strikes, bin_edges = np.histogram(strikes, bin_edges)
        number_of_strikes[0] += number_of_strikes[-1]
        half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
        two_halves = np.concatenate([half, half])

        ax3.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, 
            width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k')
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
        ax3.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight= 'black')
        ax3.set_title('Strike Distribution Rose Chart', y=1.10, fontsize=12)

        plt.tight_layout()
        #plt.show()

        #ax = fig.add_subplot(111, projection='stereonet')



        #example_plot3(ax1)
        #example_plot(ax2)
        #example_plot(ax3)
        #example_plot(ax4)
        #example_plot(ax5)
        #example_plot(ax6)

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
def StageFracPlots(self):

# in this function, the self.Fracs data will be plotted directly for each stage
# still need some external data
    ev, wb, perf, perfint, minvec, maxvec = FetchAllStages.Step2(self.DataPath, self.SavePath) 

    #all_outliers = ev[ev['FusedGroup'] == -1]   # may want to plot outliers

    pointsplot = False
    for stage in range(1,4):
        #if stage != 2:
        #    continue

        hF = []                         # not using hF_All here for brevity
        strikes = []
        dips = []
        area = []
        SSE = []
        for frac in self.Fracs:
            try:
                test = frac.Stage.iloc[0]
            except:
                test = frac.Stage
            if test != stage:
                continue
            if frac == []:
                input('enter to continue')
            hF.append(frac)
            strikes.append(frac.strike)
            dips.append(frac.dip)
            area.append(frac.area)
            SSE.append(frac.SSE)

        nfracs = len(hF) 
        evs = ev[ev['Stage']==stage] 
        outliers = evs[evs['FusedGroup'] == -1]  

        # collect data to make a set of fracs
        emin = min(evs['E']) #- 10
        emax = max(evs['E']) #+ 10
        nmin = min(evs['N']) #- 10
        nmax = max(evs['N']) #+ 10
        dmin = min(evs['Depth']) #- 10
        dmax = max(evs['Depth']) #+ 10

        # Setup the display fields
        fig = plt.figure(figsize=(14,12))
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2, projection='3d')
        ax2 = plt.subplot2grid((3, 3), (2, 0), projection='stereonet')
        ax3 = plt.subplot2grid((3, 3), (2, 1), projection='polar')
        ax4 = plt.subplot2grid((3, 3), (0, 2))
        ax5 = plt.subplot2grid((3, 3), (1, 2))
        ax6 = plt.subplot2grid((3, 3), (2, 2))

        colorset = sns.color_palette("bright", nfracs)
        for j in range(0,nfracs):
            #if j != 1:
            #    continue
            #df = evs[evs[select]==selectvals[j]]
            #print(df)
            #h = FracPlane.FracPlane
            # must include h as first arg below
            #h.Calc(h, xev=df['E'], yev=df['N'], dev=df['Depth'], qual=df['Quality'])
            #hF.append(h)

            #if hF[j].Source == 'data':
            #    continue

            #print(hF[j].strike)
            strikes.append(hF[j].strike)
            dips.append(hF[j].dip)
            area.append(hF[j].area)
            SSE.append(hF[j].SSE)

            E = hF[j].X[:,0]
            N = hF[j].X[:,1]
            D = hF[j].X[:,2]
            
            if pointsplot:

                # got the frac data, now build out the plot; start w 3d
                # plot event points
                ax1.scatter(xs=E, ys=N, zs=D, color=colorset[j])

                # plot normal vector from mean point
                vlen = 20           # +ve value since now depth is -ve
                vn = vlen * hF[j].normal.T + hF[j].Xmean
                nv = np.vstack((hF[j].Xmean, vn))
                ax1.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=colorset[j])
                ax1.scatter(xs=hF[j].Xmean[0], ys=hF[j].Xmean[1], zs=hF[j].Xmean[2], color=colorset[j])
                #ax.scatter(xs=hF[j].Xhat[:,0], ys=hF[j].Xhat[:,1], zs=hF[j].Xhat[:,2], color='k')

                ax1.plot_trisurf(hF[j].perim[:,0], hF[j].perim[:,1], hF[j].perim[:,2], color=colorset[j], alpha=0.5)
                ax1.plot(hF[j].perim[:,0], hF[j].perim[:,1], hF[j].perim[:,2], color=colorset[j], alpha=0.5)

            else:
                ax1.plot_trisurf(hF[j].circle[:,0], hF[j].circle[:,1], hF[j].circle[:,2], color=colorset[j], alpha=0.5)
                ax1.plot(hF[j].circle[:,0], hF[j].circle[:,1], hF[j].circle[:,2], color=colorset[j], alpha=0.5)
                # plot normal vector from mean point
                vlen = 20           # +ve value since now depth is -ve
                vn = vlen * hF[j].normal.T + hF[j].Xmean
                nv = np.vstack((hF[j].Xmean, vn))
                ax1.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=colorset[j])

            # plot outliers
            ax1.scatter(xs=outliers['E'], ys=outliers['N'], zs=outliers['Depth'], color='k', marker='x', s=2)


            set_axes_equal(ax1)
            ax1.set_xlabel('Easting')
            ax1.set_ylabel('Northing')           
            ax1.set_zlabel('Depth')
            
            textstr1 = f'\n\nStage {stage}: area = {np.round(sum(area))} m2'
            textstr2 = f'SSE = {np.round(sum(SSE))} m2'
            textstr3 = f'outliers = {len(outliers)} / {len(evs)}'
            ax1.set_title(textstr1 + ',  ' + textstr2 + ',  ' + textstr3)

            # now add the wellbore and perf intervals
            ax1.plot(xs=wb.X[-40:], ys=wb.Y[-40:], zs=wb.D[-40:], color='b')
            for k in range(0,3):
                ax1.plot(xs = [perfint[k][0][0], perfint[k][1][0]], 
                        ys = [perfint[k][0][1], perfint[k][1][1]],
                        zs = [perfint[k][0][2], perfint[k][1][2]],  
                        color='k', linewidth=6)


            # second plot is pole stereonet plot + rose diagram for frac azimuth
            # http://geologyandpython.com/structural_geology.html
            # most of this is done after loop over fractures when all data is collected

            ax2.pole(hF[j].strike, hF[j].dip, color=colorset[j], markersize=18)
            #ax.rake(hF[j].strike, hF[j].dip, -25)
            ax2.grid()
            

            # E vs N
            sns.scatterplot(ax=ax4, x=E, y=N, color=colorset[j])
            ax4.set_xlim(emin, emax)
            ax4.set_ylim(nmin, nmax)
            ax4.set_xlabel('Easting')
            ax4.set_ylabel('Northing')
            ax4.plot(hF[j].perim[:,0], hF[j].perim[:,1], color=colorset[j], alpha=0.5)
            # add the wellbore and perf intervals
            ax4.plot(wb.X[-40:], wb.Y[-40:], color='b')
            for k in range(0,3):
                ax4.plot([perfint[k][0][0], perfint[k][1][0]], 
                        [perfint[k][0][1], perfint[k][1][1]],  
                        color='k', linewidth=6)
            ax4.axis('equal')

            # E vs D
            sns.scatterplot(ax=ax5, x=E, y=D, color=colorset[j])
            ax5.set_xlim(emin, emax)
            ax5.set_ylim(dmin, dmax)
            ax5.set_xlabel('Easting')
            ax5.set_ylabel('Depth')
            ax5.plot(hF[j].perim[:,0], hF[j].perim[:,2], color=colorset[j], alpha=0.5)
            # add the wellbore and perf intervals
            ax5.plot(wb.X[-40:], wb.D[-40:], color='b')
            for k in range(0,3):
                ax5.plot([perfint[k][0][0], perfint[k][1][0]], 
                        [perfint[k][0][2], perfint[k][1][2]],  
                        color='k', linewidth=6)
            ax5.axis('equal')

            # E vs D
            sns.scatterplot(ax=ax6, x=N, y=D, color=colorset[j])
            ax6.set_xlim(nmin, nmax)
            ax6.set_ylim(dmin, dmax)
            ax6.set_xlabel('Northing')
            ax6.set_ylabel('Depth')
            ax6.plot(hF[j].perim[:,1], hF[j].perim[:,2], color=colorset[j], alpha=0.5)
            # add the wellbore and perf intervals
            ax6.plot(wb.Y[-40:], wb.D[-40:], color='b')
            for k in range(0,3):
                ax6.plot([perfint[k][0][1], perfint[k][1][1]], 
                        [perfint[k][0][2], perfint[k][1][2]],  
                        color='k', linewidth=6)
            ax6.axis('equal')

        # put cumulative density chart on stereonet
        ax2.density_contourf(strikes, dips, measurement='poles', cmap='Greens')  #cmap='Reds')
        #ax2.rake(strikes, dips, -25)
        #ax2.plane(strikes, dips, 'g-', linewidth=2)
        ax2.set_title('Stereonet Pole Map', y=1.10, fontsize=12)

        # rose plot must be done at the end after all fracs processed
        bin_edges = np.arange(-5, 366, 10)
        number_of_strikes, bin_edges = np.histogram(strikes, bin_edges)
        number_of_strikes[0] += number_of_strikes[-1]
        half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
        two_halves = np.concatenate([half, half])

        ax3.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, 
            width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k')
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
        ax3.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight= 'black')
        ax3.set_title('Strike Distribution Rose Chart', y=1.10, fontsize=12)

        plt.tight_layout()
        #plt.show()

        #ax = fig.add_subplot(111, projection='stereonet')



        #example_plot3(ax1)
        #example_plot(ax2)
        #example_plot(ax3)
        #example_plot(ax4)
        #example_plot(ax5)
        #example_plot(ax6)

    return

    # setup the figure
    #plt.ion()
    fig = plt.figure(figsize=(16,11))
    fig.suptitle(f'\n{Title}')

    #ax = fig.add_axes(rect=[0,0,1,1], projection='3d')
    #fig.set(facecolor=(1,1,1))

    label = hF.SpaceGroup
    nlabel = len(np.unique(label))
    #labelseries = pd.Series(label)
    colorset = sns.color_palette("bright", nlabel)

    E = hF.X[:,0]
    N = hF.X[:,1]
    D = hF.X[:,2]
    T = hF.RootTime

    ax = fig.add_subplot(231)
    sns.scatterplot(x=E, y=N, hue=label, palette=colorset)
    ax.set_xlim(minvec[0], maxvec[0])
    ax.set_ylim(minvec[1], maxvec[1])
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    
    ax = fig.add_subplot(232)
    sns.scatterplot(x=E, y=D, hue=label, palette=colorset)
    ax.set_xlim(minvec[0], maxvec[0])
    ax.set_ylim(minvec[2], maxvec[2])
    ax.set_xlabel('Easting')
    ax.set_ylabel('Depth')
    
    ax = fig.add_subplot(233)
    sns.scatterplot(x=N, y=D, hue=label, palette=colorset)
    ax.set_xlim(minvec[1], maxvec[1])
    ax.set_ylim(minvec[2], maxvec[2])
    ax.set_xlabel('Northing')
    ax.set_ylabel('Depth')
    
    ax = fig.add_subplot(234)
    sns.scatterplot(x=T, y=E, hue=label, palette=colorset)
    ax.set_xlim(minvec[3], maxvec[3])
    ax.set_ylim(minvec[0], maxvec[0])
    ax.set_xlabel('RootTime')
    ax.set_ylabel('Easting')
    
    ax = fig.add_subplot(235)
    sns.scatterplot(x=T, y=N, hue=label, palette=colorset)
    ax.set_xlim(minvec[3], maxvec[3])
    ax.set_ylim(minvec[1], maxvec[1])
    ax.set_xlabel('RootTime')
    ax.set_ylabel('Northing')

    ax = fig.add_subplot(236)
    sns.scatterplot(x=T, y=D, hue=label, palette=colorset)
    ax.set_xlim(minvec[3], maxvec[3])
    ax.set_ylim(minvec[2], maxvec[2])
    ax.set_xlabel('RootTime')
    ax.set_ylabel('Depth')

    fig.tight_layout(w_pad=2)
    
    # save to file
    plt.savefig(pfn)
    #plt.show()

    return

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.invert_zaxis()

    area = []
    SSE = []

    nFrac = len(AllhF)
    colorset = sns.color_palette("bright", nFrac)
    
    knt = -1
    for hF in AllhF:
    
        knt += 1
        # plot event points
        ax.scatter(xs=hF.X[:,0], ys=hF.X[:,1], zs=hF.X[:,2], color=colorset[knt])

        # plot normal (red) vector from mean point
        vlen = -20
        vn = vlen * hF.normal.T + hF.Xmean
        nv = np.vstack((hF.Xmean, vn))
        ax.plot(xs=nv[:,0], ys=nv[:,1], zs=nv[:,2], color=colorset[knt])
        ax.scatter(xs=hF.Xmean[0], ys=hF.Xmean[1], zs=hF.Xmean[2], color=colorset[knt])

        #ax.scatter(xs=hF.Xhat[:,0], ys=hF.Xhat[:,1], zs=hF.Xhat[:,2], color='k')

        #ax2 = fig.add_subplot(122)
        #ax2.scatter(hF.Yhat[:,0], hF.Yhat[:,1], color='k')
        #ax2.invert_yaxis()
    
        #ik = hF.edgepts
        #ax2.plot(hF.Yhat[ik,0], hF.Yhat[ik,1], color='r')
        #ax2.fill(hF.Yhat[ik,0], hF.Yhat[ik,1], color='r', alpha=0.5)

        area.append(hF.area)
        SSE.append(hF.SSE)

        ax.plot_trisurf(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[knt], alpha=0.5)
        ax.plot(hF.perim[:,0], hF.perim[:,1], hF.perim[:,2], color=colorset[knt], alpha=0.5)
        #poly = Poly3DCollection(verts=np.array(hF.perim), facecolors='g', alpha=0.5)
        #ax.add_collection3d(poly)

    for k in range(0,len(outliers)):
        ax.scatter(outliers[k][0], outliers[k][1], outliers[k][2], color='k', marker='x', s=2)
    
    textstr1 = f'area = {np.round(sum(area))} m2'
    textstr2 = f'SSE = {np.round(sum(SSE))} m2'
    textstr3 = f'N_missed = {len(outliers)}'
    plt.title(textstr1 + ',  ' + textstr2 + ',  ' + textstr3)

    plt.show()


    #fig.canvas.draw()
    # #for k in range(len(hF.X)):
    #   time.sleep(0.1)
    #   ax.scatter(xs=hF.X[k,0], ys=hF.X[k,1], zs=hF.X[k,2], color='r')
    #   fig.canvas.draw()

    #plt.show()

    return

