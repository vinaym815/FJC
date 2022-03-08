#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sympy import Matrix
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# In[2]:


def TransfromationMatrix(screw_axis, theta, d):
    v = np.reshape(screw_axis[0:3], (-1,1))
    k_norm = np.linalg.norm(screw_axis[3:])
    if(k_norm >0):
        kx = screw_axis[3]/k_norm
        ky = screw_axis[4]/k_norm
        kz = screw_axis[5]/k_norm
    else:
        kx, ky, kz = 0, 0, 0

    K = np.array([[0, -kz, ky],[kz, 0, -kx],[-ky, kx, 0]])
    r = np.sin(theta)*K + (1-np.cos(theta))*np.dot(K,K)
    
    mat_rodriguez = r + np.identity(3)
    translation_offset = np.dot(-r,v)+ d*np.array([[kx], [ky], [kz]])

    transform_Mat = np.concatenate((mat_rodriguez, translation_offset), axis=1)
    transform_Mat =  np.concatenate(( transform_Mat, np.array([[0, 0, 0, 1]])) ,axis=0)
    
    return transform_Mat


# In[3]:


def ComputeScrew(bodyVecs, globalVecs):
    nVecs = bodyVecs.shape[0]
    displacementVecs = globalVecs - bodyVecs
    relativeDispalcmentVecs = displacementVecs - np.tile(displacementVecs[0,:], (nVecs,1))
    
    dirCos = Matrix(relativeDispalcmentVecs).nullspace()[0]
    dirCos = np.array(dirCos.tolist()).astype(np.float64)
    dirCos = dirCos.reshape(3,)/np.linalg.norm(dirCos)

    d = np.dot(dirCos, displacementVecs[0])
    translationVec = d*dirCos

    perpVecs = displacementVecs - np.tile(translationVec, (nVecs,1))
    perpVecs_midPoint = bodyVecs + perpVecs/2

    A = np.concatenate((perpVecs, [dirCos]), axis=0)

    b = np.concatenate((np.diagonal(np.dot(perpVecs, perpVecs_midPoint.T)).reshape(-1,1), 
                        np.array([[0.0],])), axis =0 )
    
    # point on the screw axis that is nearest to the origin
    rho = np.linalg.lstsq(A, b, rcond=None)[0].reshape(3,)
    
    # Will fail if the point is on the axis of rotation
    Base = np.linalg.norm(np.cross(dirCos,perpVecs_midPoint[0]-rho))
    Perp = np.linalg.norm(perpVecs[0]/2)
    theta = 2*np.arctan2(Perp, Base)
    
    return {"dirCosines":dirCos, "rho":rho, "rotation":theta, "translation":d}


# In[4]:


# Screw details
screw_axis = np.array([2, 2, 3, 1, 4, 1]) #x1, y1, z1, L, M, N
theta = np.pi/3
translation = 3
transform_Mat = TransfromationMatrix(screw_axis, theta, translation)


# In[5]:


bodyVecs = np.identity(3)

coodinates = np.concatenate((bodyVecs.T, np.ones((1,3))), axis=0)
transformed_coords =  np.dot(transform_Mat, coodinates)
globalVecs = transformed_coords[0:-1,:].T

# Computing the screw transformation matrix associated with it
vin = ComputeScrew(bodyVecs, globalVecs)
newTransformMat= TransfromationMatrix(np.concatenate((vin["rho"],vin["dirCosines"])), vin["rotation"],vin["translation"])


# In[6]:


print(newTransformMat)
print(transform_Mat)


# In[ ]:




