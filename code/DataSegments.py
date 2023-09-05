''' Data retrieval routine for MapMEQ
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
'''

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import FracPlane
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


#----------------------------------------------------------------------------------------------------
def display_categories(model,data):
    
    labels = model.fit_predict(data)
    sns.scatterplot(data=data,x=data[:,0],y=data[:,1],hue=labels,palette='Set1')


#----------------------------------------------------------------------------------------------------
def time_segments(evs, eps, min_samples):

    # evs is subset of ev for specific Stage

    data = np.array([evs['RootTime']]).T              
    
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_data)

    missed = len(labels[labels==-1])
    total = len(labels)
    nlabels = max(labels) - min(labels) + 1
    #print(f'Time: Failed to classify: {missed} out of {total}')

    return nlabels, 100*missed/total, labels


#----------------------------------------------------------------------------------------------------
def space_segments(evs, eps, min_samples):

    # evs is subset of ev for specific TimeGroup
    # evaluate for spatial clustering & apply FracPlane algorithm on the clusters

    data = np.array([evs['E'], evs['N'], evs['Depth']]).T

    # NB: leave this here in case want to try scaler again
    #scaler = MinMaxScaler()
    #scaler = StandardScaler()
    #scaled_data = scaler.fit_transform(data)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    #labels = dbscan.fit_predict(scaled_data)
    labels = dbscan.fit_predict(data)

    missed = len(labels[labels==-1])
    total = len(labels)
    nlabels = max(labels) - min(labels) + 1
    #print(f'Space: Failed to classify: {missed} out of {total}')

    return nlabels, 100*missed/total, labels

