''' Display routine
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection 


def PlotENDT(wb, perfint, E, N, D, T, label, minvec, maxvec, Title, pfn, msg):

    # try size = label
    size = label

    # setup the figure
    #plt.ion()
    newtitle = Title.replace('_',' ')
    fig = plt.figure(figsize=(18,11))
    fig.suptitle(f'\n{newtitle}\n{msg}')

    #ax = fig.add_axes(rect=[0,0,1,1], projection='3d')
    #fig.set(facecolor=(1,1,1))

    labelseries = pd.Series(label)
    colorset = sns.color_palette("bright", labelseries.nunique())

    ax = fig.add_subplot(231)
    sns.scatterplot(x=E, y=N, hue=size, palette=colorset, size=size)
    ax.legend(title='scale')
    ax.plot(wb['X'], wb['Y'])
    for p in perfint:
        ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color='k', linewidth=6)
    ax.set_xlim(minvec[0], maxvec[0])
    ax.set_ylim(minvec[1], maxvec[1])
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    
    ax = fig.add_subplot(232)
    sns.scatterplot(x=E, y=D, hue=label, palette=colorset, size=size)
    ax.legend(title='scale')
    ax.plot(wb['X'], wb['D'])
    for p in perfint:
        ax.plot([p[0][0], p[1][0]], [p[0][2], p[1][2]], color='k', linewidth=6)
    ax.set_xlim(minvec[0], maxvec[0])
    ax.set_ylim(minvec[2], maxvec[2])
    ax.set_xlabel('Easting')
    ax.set_ylabel('Depth')
    
    ax = fig.add_subplot(233)
    sns.scatterplot(x=N, y=D, hue=label, palette=colorset, size=size)
    ax.legend(title='scale')
    ax.plot(wb['Y'], wb['D'])
    for p in perfint:
        ax.plot([p[0][1], p[1][1]], [p[0][2], p[1][2]], color='k', linewidth=6)
    ax.set_xlim(minvec[1], maxvec[1])
    ax.set_ylim(minvec[2], maxvec[2])
    ax.set_xlabel('Northing')
    ax.set_ylabel('Depth')
    
    ax = fig.add_subplot(234)
    sns.scatterplot(x=T, y=E, hue=label, palette=colorset, size=size)
    ax.legend(title='scale')
    ax.set_xlim(minvec[3], maxvec[3])
    ax.set_ylim(minvec[0], maxvec[0])
    ax.set_xlabel('RootTime')
    ax.set_ylabel('Easting')
    
    ax = fig.add_subplot(235)
    sns.scatterplot(x=T, y=N, hue=label, palette=colorset, size=size)
    ax.legend(title='scale')
    ax.set_xlim(minvec[3], maxvec[3])
    ax.set_ylim(minvec[1], maxvec[1])
    ax.set_xlabel('RootTime')
    ax.set_ylabel('Northing')

    ax = fig.add_subplot(236)
    sns.scatterplot(x=T, y=D, hue=label, palette=colorset, size=size)
    ax.legend(title='scale')
    ax.set_xlim(minvec[3], maxvec[3])
    ax.set_ylim(minvec[2], maxvec[2])
    ax.set_xlabel('RootTime')
    ax.set_ylabel('Depth')

    fig.tight_layout(w_pad=1)
    
    # save to file
    plt.savefig(pfn)
    #plt.show()

   
