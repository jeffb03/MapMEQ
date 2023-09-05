''' FracPlane class receives list of event coords and calculates projection on a plane
    Includes TestFracPlane to verify results
    Jeffrey R Bailey, 2023
    Version 1.0, 2023-09-01
'''


import numpy as np
#import pandas as pd
from scipy.spatial import ConvexHull


#-----------------------------------------------------------------------------------------     
class FracPlane():
    ''' Receives event coordinates in pd.DataFrame format
        Calc function will fit plane to the points and return frac params
        Many frac params accessible from the class object
    '''

    def __init__(self):  
        super().__init__()

        self.Source = []
        self.X = []
        self.Xmean = []
        self.Y = []
        self.Yhat = []
        self.Xhat = []

        self.eigval = []
        self.eigvec = []
        self.inplane = []
        self.normal = []
        self.strike = []
        self.trend = []
        self.dip = []
        self.plunge = []

        self.SSE = []
        self.RMSerror = []
        self.hull = []
        self.area = []
        self.edgepts = []
        self.perim = []

        self.radii = []
        self.effective_radius = []
        self.mean_radius = []
        self.stddev_radius = []
        self.mean_quality = []
        self.circle = []

        self.RootTime = []
        self.Stage = []
        self.TimeGroup = []
        self.SpaceGroup = []
        self.ExtGroup = []
        self.FusedGroup = []


    #-----------------------------------------------------------------------------------------     
    def Calc(self, xev, yev, dev, qual=0):      # X ~ Easting, Y ~ Northing, D ~ Depth

        # calc PCA analysis and eigensolution

        self.X = np.array((xev, yev, dev)).T
        self.Xmean = np.mean(self.X, axis=0)
        self.Y = self.X-self.Xmean  # zero-mean points

        self.mean_quality = np.mean(qual)

        self.eigval, self.eigvec = PCA3d(self.Y)
        # if returned nothing (data too small), then clear class vars & return
        if self.eigval == []:
            self.Source = []
            self.X = []
            self.Xmean = []
            self.Y = []
            self.Yhat = []
            self.Xhat = []

            self.eigval = []
            self.eigvec = []
            self.inplane = []
            self.normal = []
            self.strike = []
            self.trend = []
            self.dip = []
            self.plunge = []

            self.SSE = []
            self.RMSerror = []
            self.hull = []
            self.area = []
            self.edgepts = []
            self.perim = []

            self.radii = []
            self.effective_radius = []
            self.mean_radius = []
            self.stddev_radius = []
            self.mean_quality = []
            self.circle = []

            self.RootTime = []
            self.Stage = []
            self.TimeGroup = []
            self.SpaceGroup = []
            self.ExtGroup = []
            self.FusedGroup = []
            return
         
        # coming this way, source is data
        self.Source = 'data'
        
        # first two eigenvectors lie in the fracture plane 
        self.inplane = self.eigvec[:, 0:2]

        # third eigenvector is perpendicular to fracture plane; choose downgoing vector direction
        # NB: NOW using depth -ve into the ground, have to invert if z comp is +ve
        norm = np.array(self.eigvec[:, 2])
        if norm[2] >= 0:
            self.normal = -norm
        else:
            self.normal = norm
        easting = self.normal[0]
        northing = self.normal[1]

        # following angles are in degrees for direct output to DFN model
        #print(f'easting, northing: {self.normal[0]}, {self.normal[1]}')
        self.trend = (np.arctan2(easting, northing) * (180 / np.pi)) % 360      # use mod to get +ve values
        self.strike = (self.trend + 90) % 360

        #print(f'normal vertical comp: {self.normal[2]}')
        self.plunge = ( (np.arcsin(-self.normal[2]) * (180 / np.pi)) ) % 90
        self.dip = 90 - self.plunge

        #print(f'plunge = {self.plunge:.2f} , dip = {self.dip:.2f}, normal2 = {self.normal[2]:.2f}, (normal vector length)^2 = {(self.normal[0]**2 + self.normal[1]**2 + self.normal[2]**2):.2f}')
        #print(f'strike = {self.strike:.2f} , trend = {self.trend:.2f}\n')


        # calc projection on the plane of first two eigenvectors
        A = self.inplane
        AT = A.T
        P = np.linalg.inv(AT.dot(A))
        P = P.dot(AT)

        b = self.Y.T
        yhat = P.dot(b)
        p = A.dot(yhat).T
        self.Xhat = np.array((p[:,0]+self.Xmean[0], p[:,1]+self.Xmean[1], p[:,2]+self.Xmean[2])).T

        # calculate total error
        e = self.X - self.Xhat
        self.SSE = 0
        for k in range(len(e)):
            self.SSE += e[k,0]**2 + e[k,1]**2 + e[k,2]**2
        self.RMSerror = np.sqrt(self.SSE/len(e))
        
        # now calculate convex hull of Yhat and its area
        self.Yhat = yhat.T
        self.hull = ConvexHull(self.Yhat)
        self.area = self.hull.volume                    # yup, use volume in 2D not area!
        
        ik = list(self.hull.vertices)
        ik.append(ik[0]) 
        self.edgepts = ik                               # close the loop

        # find 3D coords of frac projected surface
        self.perim = np.array((self.Xhat[ik,0], self.Xhat[ik,1], self.Xhat[ik,2])).T

        # calculate fracture radii and mean/std values
        # Yhat has zero mean and values lie in the fracture plane, go for it!
        r = self.Yhat
        rlen = len(r) 
        if rlen <= 0:
            print('FracPlane zero divide error at 139')
            input('Press ENTER to continue...')

        radii = np.zeros(rlen)
        radii_sq = np.zeros(rlen)
        for k in range(0,rlen):
            radii[k] = np.sqrt(r[k][0]**2 + r[k][1]**2)
            radii_sq[k] = radii[k]**2
        self.mean_radius = np.mean(radii)
        self.stddev_radius = np.sqrt(np.sum(radii_sq)/rlen - self.mean_radius**2)      # verified 5/10/23

        # first tests show that this area is large, so let's calculate "effective radius" and use it for now
        self.effective_radius = np.sqrt(self.area / np.pi)

        # now provide 3d circle of the effective radius
        theta = np.linspace(0, 2 * np.pi, 201)
        y = self.effective_radius*np.cos(theta)
        z = self.effective_radius*np.sin(theta)
        rtheta = np.array([y, z])
        self.circle = self.Xmean + self.inplane.dot(rtheta).T

        return
    

    #-----------------------------------------------------------------------------------------      
    def MakeCircularFrac(self, stage, location, radius, dip, strike):      # X ~ Easting, Y ~ Northing, D ~ Depth

            # coming this way, source is synthetic
            self.Source = 'synthetic'
            
            # initialization is done when class created, just set values we need here

            self.Stage = stage
            self.Xmean = location
            self.effective_radius = radius
            self.dip = dip
            self.strike = strike

            self.mean_radius = radius
            self.stddev_radius = 0.0
            self.area = np.pi * radius**2

            self.trend = (self.strike - 90) % 360
            self.plunge = 90 - self.dip

            # calculate eigenvalues, i.e. coordinate rotations for these angles
            cos_theta = np.cos(self.strike/57.3)               
            sin_theta = np.sin(self.strike/57.3)            # remember to convert from degrees to rads!
            cos_delta = np.cos(self.dip/57.3)            
            sin_delta = np.sin(self.dip/57.3)
            self.eigvec = np.zeros((3,3))
            #self.eigvec[:,0] = [cos_theta, -sin_theta, 0]
            #self.eigvec[:,1] = [sin_theta, cos_theta, 0]
            #self.eigvec[:,2] = [0, 0, 1]

            self.eigvec[:,0] = [cos_theta*cos_delta, sin_theta,  cos_theta*sin_delta]
            self.eigvec[:,1] = [-sin_theta*cos_delta, cos_theta, -sin_theta*sin_delta]
            self.eigvec[:,2] = [-sin_delta, 0, cos_delta]
            self.eigvec = self.eigvec.T

            # third eigenvector is perpendicular to fracture plane; choose downgoing vector direction
            # NB: NOW using depth -ve into the ground, have to invert if z comp is +ve
            norm = np.array(self.eigvec[:, 2])
            if norm[2] >= 0:
                self.normal = -norm
            else:
                self.normal = norm
            easting = self.normal[0]
            northing = self.normal[1]

            trend = (np.arctan2(easting, northing) * (180 / np.pi)) % 360
            #print(trend)

            # first two eigenvectors lie in the fracture plane 
            self.inplane = self.eigvec[:, 0:2]

            # now provide 3d circle of the effective radius
            theta = np.linspace(0, 2 * np.pi, 201)
            y = self.effective_radius*np.cos(theta)
            z = self.effective_radius*np.sin(theta)
            rtheta = np.array([y, z])
            self.circle = self.Xmean + self.inplane.dot(rtheta).T

            xx = self.circle[:,0]
            yy = self.circle[:,1]
            dd = self.circle[:,2]
            self.X = np.array((xx, yy, dd)).T
            self.perim = self.X
      

            return



#-----------------------------------------------------------------------------------------
def PCA3d(data, correlation=False, sort=True):
    """ Applies Principal Component Analysis to the data

    Parameters
    ----------
    data: array
        The array containing the data. The array must have NxM dimensions, where each
        of the N rows represents a different individual record and each of the M columns
        represents a different variable recorded for that individual record.
            array([
            [V11, ... , V1m],
            ...,
            [Vn1, ... , Vnm]])

    correlation(Optional) : bool
            Set the type of matrix to be computed (see Notes):
                If True compute the correlation matrix.
                If False(Default) compute the covariance matrix.

    sort(Optional) : bool
            Set the order that the eigenvalues/vectors will have
                If True(Default) they will be sorted (from higher value to less).
                If False they won't.
    Returns
    -------
    eigenvalues: (1,M) array
        The eigenvalues of the corresponding matrix.

    eigenvector: (M,M) array
        The eigenvectors of the corresponding matrix.

    Notes
    -----
    The correlation matrix is a better choice when there are different magnitudes
    representing the M variables. Use covariance matrix in other cases.

    """
    #: assume that data is zero-mean (do this only once!)

    #: ensure shape is 3 rows x n columns, and that there are at least 4 points
    q = data.shape
    if max(q) < 4:
        eigenvalues = []
        eigenvectors = []
        return eigenvalues, eigenvectors
    
    if q[0] != 3:
        data = data.T

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:

        matrix = np.corrcoef(data)

    else:
        
        matrix = np.cov(data)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def TestFracPlane():

    hF = FracPlane()

    case = 2

    # specifying quadrants, in order: 1, 4, 3, 2
    
    # plane up 45 deg to the east
    if case == 1:
        Easting = [1000, 0, 0, 1000]   
        Northing = [1000, 1000, -1000, -1000]
        Depth = [1000, 0, 0, 1000]
    
    # 30 degree dip, up to east
    elif case == 2:
        Easting = [1000, 0, 0, 1000]   
        Northing = [1000, 1000, -1000, -1000]
        Depth = [577.4, 0, 0, 577.4]
    
    # 30 degree dip, strike to SE (135)
    elif case == 3:
        Easting = [1000, -1000, -1000, 1000]   
        Northing = [1000, 1000, -1000, -1000]
        Depth = [816, 0, -816, 0] 

    # plane up 45 deg to the NW (315)
    elif case == 4:
        Easting = [1000, -1000, -1000, 1000]   
        Northing = [1000, 1000, -1000, -1000]
        Depth = [-816, 0, 816, 0]       

    hF.Calc(Easting, Northing, Depth, 0)

#TestFracPlane()
#quit()