''' Subroutine for MapMEQ
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
'''

import numpy as np
import pandas as pd
 
#----------------------------------------------------------------------------------------------------
def ContiguousCount(ain):

    k = 0
    for x in ain:
        ain[k] = int(x)
        k += 1
    aout = pd.Series(ain)
    
    values = np.unique(ain)
    #print(values)
    nvalues = len(values)
    
    current = min(values)
    for val in values:
        aout.iloc[ain==val] = current
        current += 1

    if ( len(np.unique(aout)) != nvalues ):
        print('error in ContiguousCount')

    return aout.to_list()

