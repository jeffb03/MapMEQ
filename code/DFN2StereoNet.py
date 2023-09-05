'''  plot frac stereonet
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet

#---------------------------------------------------------------------------------------------------------------
def DFN2StereoNet(self): 

# read DFN file
# http://geologyandpython.com/structural_geology.html

    #df = pd.read_csv('w:\Geo\MS_DFN_Global_Coords.csv')
    pfn = self.SavePath + 'DFN_Global_Coords_MSL.csv'
    df = pd.read_csv(pfn)

# remove last two rows that were added

    df = df[0:-2]
    strike = df['Strike[deg]']
    dip = df['Dip_Angle[deg]']                 

    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(strike, bin_edges)

    number_of_strikes[0] += number_of_strikes[-1]

    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])

    fig = plt.figure(figsize=(16,9.5))

    ax = fig.add_subplot(121, projection='stereonet')
    ax.pole(strike, dip, c='k', label='Pole of the Planes')
    ax.density_contourf(strike, dip, measurement='poles', cmap='Reds')
    ax.set_title('Density coutour of the Poles', y=1.06, fontsize=15)
    ax.grid()

    ax = fig.add_subplot(122, projection='polar')
    ax.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, 
        width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
    ax.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight= 'black')
    ax.set_title('Rose Diagram of the "Fault System"', y=1.06, fontsize=15)    

    fig.tight_layout()

    pfn = self.SavePath + 'Stereonet.png'
    plt.savefig(pfn)
