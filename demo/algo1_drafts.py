# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 20:44:20 2022

@author: Hélène
"""

import numpy as np

a = np.zeros((4, 5, 3, 2))
print(a)

b = np.ones((4, 5, 3, 2))
print(b)

a[0][0]

np.append(a, b, axis = 0)
np.append(a, b, axis = 1)
np.append(a, b, axis = 2)
np.append(a, b, axis = 3)

#%%
def binarize(x):
    return 1. if(x>=0.5) else 0.
print(binarize(0.7))

vbinarize = np.vectorize(binarize)

# Number of particles N
N = 15

#Number of timesteps T
T = 10
mask_shape= 512
    
masks = np.zeros((T+1,N,mask_shape))
np.shape(masks)

#mask_shape = tuple(mask_shape)
for i in range(0,N):    
    masks[0][i] = np.random.rand(mask_shape)
    masks[0][i] = vbinarize(masks[0][i])
    print(masks[0][i])
    
print(masks[0])   
#Initialise N random masks M0_i

#%%
def flip(x):
    return 0. if x==1. else 1.
vflip = np.vectorize(flip)

a = masks[0][1]
np.random.choice(a, size = int(0.4*len(a)), replace = False)
np.reshape(a, (3,3))

#%% Bit-flipping

def flip(x):
    return 0. if x==1. else 1.

a = masks[0][1]
def bitflip_dim_2(a, d=0.5):
    array = np.reshape(a,(np.shape(a)[0]*np.shape(a)[1],))
    index = np.random.choice(range(len(array)), size = int(d*len(array)), replace = False)
    for x in index:
        array[x] = flip(array[x])
    return np.reshape(array,(np.shape(a)[0],np.shape(a)[1]))

def bitflip(a, d=0.5):
    array = a
    index = np.random.choice(range(len(a)), size = int(d*len(a)), replace = False)
    for x in index:
        array[x] = flip(array[x])
    return array
    

print(a)
print(bitflip(a))
#%%
import scipy.stats as st
noise = 0.1
Sigma = np.identity(2)*noise
st.multivariate_normal.pdf([1,2], mean = [0,1], cov= Sigma)

#%%% Resampling

def resample(w, sample):
    sample = masks[t]
    return np.random.choice(sample, size = N, replace = True, p = w)
 

#%%
import scipy.stats as st

x = np.ones((T+1,N))
for t in range(1,T+1):
    w= []*N
    for i in range(0,N+1):
        # Sample Mt_i ~ p(Mt_i|M(t-1)_i) by bit-flipping
        masks[t][i] = bitflip(masks[t-1][i])
        # Sample xt_i=f_theta(x(t-1), u_t|Mt_i)
        ##x[t][i] = model.predict(x[t-1][i], u[t], masks[t][i])
        # Evaluate w_i = N(z_t|xt_i, Sigma)
        w[i] = st.multivariate_normal.pdf(z[t], mean = x[t][i], cov= Sigma)
    w = w/sum(w)
    masks[t] = np.random.choice(masks[t], size = N, replace = True, p = w)
    # Compute the nest mask estimate (mean)
    mask_estimate[t] = 1/N * sum(masks[t])
    mask_estimate[t] = vbinarize(mask_estimate[t])
return (mask_estimate)
