
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.io
import math
import scipy.optimize
from matplotlib import cm

sys.path.append('../')

import fct_facilities as fac
import fct_varies as var

path_plot = 'Plots_rhomn_low/'
path_data = 'Data_rhomn_low/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

# General

N = 50
k = 2

rho_mn = -0.5
u = 1

# Simulations

T = 20000
deltaT = 0.1
t = np.linspace(0, T, int(T/deltaT))

Tcut = 2000
Tcut_plot = 19850


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### FROM THEORY

eig_theory = np.zeros(( 3 ))
eig_lr_theory = np.zeros(( 2 ))
a_theory = np.zeros(( 2 ))
b_theory = np.zeros(( 2 ))

#

lambda_ = k*rho_mn
alpha = k/(1-lambda_)

#### From analytical expression

eig_lr_theory[0] = k/(2*(2-lambda_)) * ( 2*rho_mn+alpha + np.sqrt((2*rho_mn+alpha)**2 -4*(rho_mn**2-1)) )
eig_lr_theory[1] = k/(2*(2-lambda_)) * ( 2*rho_mn+alpha - np.sqrt((2*rho_mn+alpha)**2 -4*(rho_mn**2-1)) )

eig_theory[0] = u**2/2. * (1+eig_lr_theory[0])
eig_theory[1] = u**2/2. * (1+eig_lr_theory[1])
eig_theory[2] = u**2/2. * (1)

a_theory[0] = 1./2*( alpha + np.sqrt((2*rho_mn+alpha)**2 -4*(rho_mn**2-1)) )
a_theory[1] = 1./2*( alpha - np.sqrt((2*rho_mn+alpha)**2 -4*(rho_mn**2-1)) )

b_theory[0] = 1.
b_theory[1] = 1.


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### SIMULATIONS

doCompute = 1

if doCompute:

	# Initialize network

	z = var.OrthogonalBasis(2, N)
	m = z[0,:]
	n = rho_mn*z[0,:] + np.sqrt(1-rho_mn**2)*z[1,:]

	W = k*np.outer(m, n)
	U = np.eye(N)

	# Run simulation

	X0 = np.random.normal(0, 1, N)

	X = var.SimulateActivity(t, X0, W, U)
	Xcut = X[int(Tcut/deltaT):,:]
	Xcut_plot = X[int(Tcut_plot/deltaT):,:]

	eigvals, v, Xproj = var.PCA(Xcut)
	eigvals_plot, v_plot, Xproj_plot = var.PCA(Xcut_plot)

	# Build eigenvectors from theory

	v_theory = np.zeros(( 2, N ))

	v_theory[0,:] = var.Normalize(a_theory[0]*m + b_theory[0]*n)
	v_theory[1,:] = var.Normalize(a_theory[1]*m + b_theory[1]*n)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### Store

	fac.Store(Xcut, 'Xcut.p', path_data)
	fac.Store(Xproj_plot, 'Xproj_plot.p', path_data)

	fac.Store(eigvals, 'eigvals.p', path_data)
	fac.Store(v, 'v.p', path_data)

	fac.Store([ m, n ], 'network_vectors.p', path_data)
	fac.Store(v_theory, 'v_theory.p', path_data)

else:

	Xcut = fac.Retrieve('Xcut.p', path_data)
	Xproj_plot = fac.Retrieve('Xproj_plot.p', path_data)

	eigvals = fac.Retrieve('eigvals.p', path_data)
	v = fac.Retrieve('v.p', path_data)

	m, n = fac.Retrieve('network_vectors.p', path_data)
	v_theory = fac.Retrieve('v_theory.p', path_data)



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

Nsample = 5

for ii_neuron in range(Nsample):
	plt.plot(np.linspace(0, T-Tcut, Xcut.shape[0]), Xcut[:,ii_neuron])

plt.xlabel(r'Time')
plt.ylabel('Activity $x_i$')

plt.xlim(0, T-Tcut)

plt.savefig(path_plot+'sample.pdf')

plt.show()

#

fac.SetPlotDim(3., 1.5)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.bar(np.arange(N)+1, eigvals, color='0.7')

plt.plot(1, eig_theory[0], marker='X', markersize=3, color='0')
plt.plot(N, eig_theory[1], marker='X', markersize=3, color='0')

for ii_comp in range(N-2):
	plt.plot(ii_comp+2, eig_theory[2], marker='X', markersize=3, color='0')

plt.xlim(-1, N+1)
plt.xticks([0, 25, 50])

plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])

plt.xlabel('Component')
plt.ylabel('Var explained')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=5)

plt.savefig(path_plot+'PC_spectrum.pdf')

plt.show()

#

fac.SetPlotDim(2, 1.8)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(Xproj_plot[:,0], Xproj_plot[:,-1], 'gray')

plt.gca().set_aspect('equal')

plt.savefig(path_plot+'sample_2D.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot(Xproj_plot[:,0], Xproj_plot[:,1], 'gray')

plt.gca().set_aspect('equal')

plt.savefig(path_plot+'sample_2D_b.pdf')

plt.show()

#

fac.SetPlotDim(1., 0.9)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(np.fabs(np.dot(np.vstack([v[:,0],v[:,-1]]), v_theory.T)), interpolation = 'nearest', vmax = 1, vmin = -1, cmap='RdBu_r')

plt.gca().set_aspect('equal')
plt.axis('off')

plt.colorbar()

plt.savefig(path_plot+'overlap_theory.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(np.dot(np.vstack([v[:,0],v[:,-1]]), np.vstack([m, n]).T), interpolation = 'nearest', vmax = 1, vmin = -1, cmap='RdBu_r')

plt.gca().set_aspect('equal')
plt.axis('off')

plt.colorbar()

plt.savefig(path_plot+'overlap_mu.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(np.fabs(np.dot(np.vstack([v[:,0],v[:,-1]]), var.Normalize(np.random.normal(0, 1, (N,1))))), interpolation = 'nearest', vmax = 1, vmin = -1, cmap='RdBu_r')

plt.gca().set_aspect('equal')
plt.axis('off')

plt.colorbar()

plt.savefig(path_plot+'overlap_random.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
