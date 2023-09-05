''' Main driver to run FracPlane.py test protocol
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import FetchAllStages
import FracPlane    


def PlotFracPlanes(hF_All, outliers, stage):

    # Get the data
    DataPath = 'W:/Geo/FORGE/'
    ev, wb, perf, perfint = FetchAllStages.FetchAllStages(DataPath)        
    

    # setup the figure
    plt.ion()
    fig = plt.figure(figsize=(12,12))
    #ax = fig.add_axes(rect=[0,0,1,1], projection='3d')
    #fig.set(facecolor=(1,1,1))

    ax = fig.add_subplot(111, projection='3d')
    #ax.set(facecolor='w')

    ax.plot(xs=wb.X, ys=wb.Y, zs=wb.D)

    ax.set_xlim(min(ev['X']),max(ev['X']))
    ax.set_ylim(min(ev['Y']),max(ev['Y']))
    ax.set_zlim(min(ev['Depth']),max(ev['Depth']))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.invert_zaxis()

    area = []
    SSE = []

    nFrac = len(hF_All)
    colorset = sns.color_palette("bright", nFrac)
    
    knt = -1
    for hF in hF_All:
    
        knt += 1
        # plot event points
        ax.scatter(xs=hF.X[:,0], ys=hF.X[:,1], zs=hF.X[:,2], color=colorset[knt])

        # plot normal (red) vector from mean point, normal should always be downward pointing
        vlen = 20
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

