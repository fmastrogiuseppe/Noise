
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

sys.path.append('../../')

import fct_facilities as fac
import fct_varies as var

path_plot = 'Plots_rhomn/'
path_data = 'Data_rhomn/'


# Simulates n=u

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

# General

N = 50
k = 2

rho_mn1 = 0.1
rho_mn2 = -0.3
u = 1

# Simulations

T = 200000
deltaT = 0.1
t = np.linspace(0, T, int(T/deltaT))

Tcut = 20000
Tcut_plot = 198500 #19950


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### FROM THEORY

eig_theory = np.zeros(( 5 ))
eig_lr_theory = np.zeros(( 5 ))
a_theory = np.zeros(( 4 ))
b_theory = np.zeros(( 4 ))

#

lambda1_ = k*rho_mn1
lambda2_ = k*rho_mn2

alpha1 = k/(1-lambda1_)
alpha2 = k/(1-lambda2_)

#### From analytical expression

eig_lr_theory[0] = k/(2*(2-lambda1_)) * ( 2*rho_mn1+alpha1 + np.sqrt((2*rho_mn1+alpha1)**2 -4*(rho_mn1**2-1)) )
eig_lr_theory[1] = k/(2*(2-lambda1_)) * ( 2*rho_mn1+alpha1 - np.sqrt((2*rho_mn1+alpha1)**2 -4*(rho_mn1**2-1)) )
eig_lr_theory[2] = k/(2*(2-lambda2_)) * ( 2*rho_mn2+alpha2 + np.sqrt((2*rho_mn1+alpha2)**2 -4*(rho_mn2**2-1)) )
eig_lr_theory[3] = k/(2*(2-lambda2_)) * ( 2*rho_mn2+alpha2 - np.sqrt((2*rho_mn1+alpha2)**2 -4*(rho_mn2**2-1)) )
eig_lr_theory[4] = 0

eig_theory = u**2/2. * (1+eig_lr_theory)

a_theory[0] = 1./2*( alpha1 + np.sqrt((2*rho_mn1+alpha1)**2 -4*(rho_mn1**2-1)) )
a_theory[1] = 1./2*( alpha1 - np.sqrt((2*rho_mn1+alpha1)**2 -4*(rho_mn1**2-1)) )
a_theory[2] = 1./2*( alpha2 + np.sqrt((2*rho_mn2+alpha2)**2 -4*(rho_mn2**2-1)) )
a_theory[3] = 1./2*( alpha2 - np.sqrt((2*rho_mn2+alpha2)**2 -4*(rho_mn2**2-1)) )

b_theory[0] = 1.
b_theory[1] = 1.
b_theory[2] = 1.
b_theory[3] = 1.


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### SIMULATIONS

doCompute = 1

if doCompute:

	# Initialize network

	z = var.OrthogonalBasis(4, N)
	m1 = z[0,:]
	n1 = rho_mn1*z[0,:] + np.sqrt(1-rho_mn1**2)*z[1,:]
	m2 = z[2,:]
	n2 = rho_mn2*z[2,:] + np.sqrt(1-rho_mn2**2)*z[3,:]

	W = k*(np.outer(m1, n1)+np.outer(m2, n2))
	U = np.eye(N)

	# Run simulation

	X0 = np.random.normal(0, 1, N)

	X = var.SimulateActivity(t, X0, W, U)
	Xcut = X[int(Tcut/deltaT):,:]
	Xcut_plot = X[int(Tcut_plot/deltaT):,:]

	eigvals, v, Xproj = var.PCA(Xcut)
	eigvals_plot, v_plot, Xproj_plot = var.PCA(Xcut_plot)

	# Build eigenvectors from theory

	v_theory = np.zeros(( 4, N ))

	v_theory[0,:] = var.Normalize(a_theory[0]*m1 + b_theory[0]*n1)
	v_theory[2,:] = var.Normalize(a_theory[1]*m1 + b_theory[1]*n1)
	v_theory[1,:] = var.Normalize(a_theory[2]*m2 + b_theory[2]*n2)
	v_theory[3,:] = var.Normalize(a_theory[3]*m2 + b_theory[3]*n2)


	#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
	#### Store

	fac.Store(Xcut, 'Xcut.p', path_data)
	fac.Store(Xproj_plot, 'Xproj_plot.p', path_data)

	fac.Store(eigvals, 'eigvals.p', path_data)
	fac.Store(v, 'v.p', path_data)

	fac.Store([ m1, n1, m2, n2 ], 'network_vectors.p', path_data)
	fac.Store(v_theory, 'v_theory.p', path_data)

else:

	Xcut = fac.Retrieve('Xcut.p', path_data)
	Xproj_plot = fac.Retrieve('Xproj_plot.p', path_data)

	eigvals = fac.Retrieve('eigvals.p', path_data)
	v = fac.Retrieve('v.p', path_data)

	m1, n1, m2, n2 = fac.Retrieve('network_vectors.p', path_data)
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
plt.plot(N-1, eig_theory[1], marker='X', markersize=3, color='0')
plt.plot(2, eig_theory[2], marker='X', markersize=3, color='0')
plt.plot(N, eig_theory[3], marker='X', markersize=3, color='0')

for ii_comp in range(N-4):
	plt.plot(ii_comp+3, eig_theory[4], marker='X', markersize=3, color='0')

plt.xlim(-1, N+1)
plt.xticks([0, 25, 50])

plt.ylim(0, 2.6)
plt.yticks([0, 1.3, 2.6])

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

fac.SetPlotDim(2., 1.9)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(np.fabs(np.dot(np.vstack([-v[:,0],-v[:,1],-v[:,-2],v[:,-1]]), v_theory.T)), interpolation = 'nearest', vmax = 1, vmin = -1, cmap='RdBu_r')

plt.gca().set_aspect('equal')
plt.axis('off')

plt.colorbar()

plt.savefig(path_plot+'overlap_theory.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(np.dot(np.vstack([-v[:,0],-v[:,1],-v[:,-2],-v[:,-1]]), np.vstack([m1, n1, m2, n2]).T), interpolation = 'nearest', vmax = 1, vmin = -1, cmap='RdBu_r')

plt.gca().set_aspect('equal')
plt.axis('off')

plt.colorbar()

plt.savefig(path_plot+'overlap_mu.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
