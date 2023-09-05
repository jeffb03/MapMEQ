'''  build DFN file
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
'''

import numpy as np
import pandas as pd

#---------------------------------------------------------------------------------------------------------------
def Frac2DFN(self): 

 # for now, self.Fracs is the latest result

# output the FinalFrac to DFN file
# make a pandas DataFrame from the vars we need, add column headings, and save to excel file

# add back in global coord E, N offset at 16A wellhead
# depths will be referenced to MSL as expected in GeoDT input
# thus, add in 16A wh offsets:  f['X'] += 334638.97  &  f['Y'] += 4263434.117

    flen = len(self.Fracs)
    fracoutput = np.zeros((flen,12))
    for k in range(0,flen):
        frac = self.Fracs[k]
        if len(frac.Xmean) == 0:
            continue
        fracoutput[k][0] = np.round(frac.Xmean[0] + 334638.97)      # add in global coord offset for E, N
        fracoutput[k][1] = np.round(frac.Xmean[1] + 4263434.117)
        fracoutput[k][2] = np.round(frac.Xmean[2])                  # depth is referenced to MSL, -ve down

        # use effective radius and calculated convex hull area for now
        fracoutput[k][3] = np.round(frac.effective_radius)
        fracoutput[k][4] = np.round(frac.area)
                    
        fracoutput[k][5] = np.round(frac.trend, decimals=1)
        fracoutput[k][6] = np.round(frac.plunge, decimals=1)
        fracoutput[k][7] = np.round(frac.strike, decimals=1)
        fracoutput[k][8] = np.round(frac.dip, decimals=1)
        fracoutput[k][9] = 0.0002
        fracoutput[k][10] = 3.33E-09
        fracoutput[k][11] = 5.00E-06

    df = pd.DataFrame(fracoutput, columns=['FractureX[m]', 'FractureY[m]', 'FractureZ[m]', 'FractureRadius[m]', 
                                            'Area[m2]', 'Trend[deg]', 'Plunge[deg]', 'Strike[deg]',
                                            'Dip_Angle[deg]', 'Aperture[m]', 'Permeability[m2]', 'Compressibility[1/kPa]'])

    pfn = self.SavePath + 'DFN_Global_Coords_MSL.csv'
    df.to_csv(pfn, index=False)

 