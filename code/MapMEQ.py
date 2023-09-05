''' Main Driver for MapMEQ
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
'''

import datetime
import RunMEQ
import sys

MAX_PCT_OUTLIERS_TIME = 15
MAX_PCT_OUTLIERS_SPACE = 15
MIN_CLUSTER_POINTS_TIME = 6
MIN_CLUSTER_POINTS_SPACE = 8

print(sys.version)
DataPath = '../runs/'
SavePathBase = '../runs/'  

#---------------------------------------------------------------------------------------------------------------

now = datetime.datetime.now()

# 2023-05-24 converted code to use list of points & outliers to specify values for each stage

MEQ = RunMEQ.RunMEQ(DataPath, SavePathBase)

MIN_CLUSTER_POINTS_TIME = [8, 6, 6]
MAX_PCT_OUTLIERS_TIME = [25, 15, 15]

MIN_CLUSTER_POINTS_SPACE = [4, 5, 7]
MAX_PCT_OUTLIERS_SPACE = [22, 20, 10]

MEQ.Calc(MIN_CLUSTER_POINTS_TIME, MIN_CLUSTER_POINTS_SPACE, MAX_PCT_OUTLIERS_TIME, MAX_PCT_OUTLIERS_SPACE)


#s=f'T{N1}p-{O1}o_S{N2}p-{O2}o'

input("Press Enter to quit")

quit()

