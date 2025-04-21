import matplotlib.pyplot as plt
import numpy as np


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Varies

def ParticipationRatio(vector):
	return np.sum(vector)**2 / (np.sum(vector**2))


def Normalize(vector):
	return vector/np.sqrt(np.sum(vector**2))


def OrthogonalBasis(M, N):

	z = np.random.normal(0, 1, (M,N))    # from Gaussian statistics
	z[0,:] = Normalize(z[0,:])

	for ii_vector in range(M-1):    # apply ortho-normalization
		for ii_past in range(ii_vector+1):

			z[ii_vector+1,:] -= np.dot(z[ii_vector+1,:],z[ii_vector-ii_past,:])*z[ii_vector-ii_past,:]
		
		z[ii_vector+1,:] = Normalize(z[ii_vector+1,:])

	return z


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Simulations

def SimulateActivity(t, X0, W, U):

	print (' ** Simulating... **')

	# Setup

	deltaT = t[1]-t[0]

	X = np.zeros(( len(t), len(X0) ))
	X[0,:] = X0

	# Integrate activity

	for ii_time in range(len(t[:len(t)-1])):

		X[ii_time+1,:] = X[ii_time,:] + deltaT*(-X[ii_time,:] + np.dot(W, X[ii_time,:])) \
							+ np.sqrt(deltaT)*np.dot(U, np.random.normal(0, 1, len(X0)))

	return X


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Analysis

def PCA(X, doCenter=False, doNormalize=False):

	if doCenter: X = X-np.mean(X,0)[None,:]
	if doNormalize: X = X/np.std(X,0)[None,:]

	eigvals, v = np.linalg.eig(np.dot(X.T, X)/X.shape[0])

	v = v[:,np.flipud(np.argsort(eigvals))]
	eigvals = eigvals[np.flipud(np.argsort(eigvals))]

	Xproj = (np.dot(v.T, X.T)).T

	return eigvals.real, v.real, Xproj.real

	