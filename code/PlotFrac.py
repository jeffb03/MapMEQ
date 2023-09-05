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


#---------------------------------------------------------------------------------------------------------------
def PlotFrac(hF, minvec, maxvec, Title, pfn):

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

