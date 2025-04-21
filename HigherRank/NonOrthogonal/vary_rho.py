
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

path_plot = 'Plots_vary_rho/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 50

k_values = [ 2 ] # np.linspace(1, 10, 4)
rho_mm_values = np.linspace(-1, 0.99, 200)
rho_nn_values = np.linspace(-1, 0.99, 200)
u = 1

rho_mn1 = 0
rho_mn2 = 0

#

eig = np.zeros(( 4, len(k_values), len(rho_mm_values), len(rho_nn_values) ))
eig_lr = np.zeros(( 4, len(k_values), len(rho_mm_values), len(rho_nn_values) ))
dimensionality = np.zeros(( len(k_values), len(rho_mm_values), len(rho_nn_values) ))

eig_theory = np.zeros(( 4, len(k_values), len(rho_mm_values), len(rho_nn_values) ))
eig_lr_theory = np.zeros(( 4, len(k_values), len(rho_mm_values), len(rho_nn_values) ))
dimensionality_theory = np.zeros(( len(k_values), len(rho_mm_values), len(rho_nn_values) ))


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### FROM THEORY

for ii_k, k in enumerate(k_values):

	for ii_rho_mm, rho_mm in enumerate(rho_mm_values):

		for ii_rho_nn, rho_nn in enumerate(rho_mm_values):

			lambda1_ = k*rho_mn1
			lambda2_ = k*rho_mn2

			alpha1 = k/(2-lambda1_)
			alpha2 = k/(2-lambda2_)

			beta11 = k**2/((1-lambda1_)*(2-lambda1_))
			beta22 = k**2/((1-lambda2_)*(2-lambda2_))
			beta12 = k**2*(4-lambda1_-lambda2_)/((2-lambda1_-lambda2_)*(2-lambda1_)*(2-lambda2_))

			#### From analytical expression

			# Compute blocks

			B11 = np.array( [ [rho_mn1*alpha1, 		alpha1 + rho_mn1*beta11] ,\
							  [alpha1, 				rho_mn1*alpha1 + beta11 + rho_mm*rho_nn*beta12]  ] )

			B12 = np.array( [ [0, 					rho_nn*alpha2 + rho_mn1*rho_nn*beta12] ,\
							  [rho_mm*alpha2, 		rho_nn*beta12 + rho_mm*beta22]  ] )

			B21 = np.array( [ [0, 					rho_nn*alpha1 + rho_mn2*rho_nn*beta12] ,\
							  [rho_mm*alpha1, 		rho_nn*beta12 + rho_mm*beta11]  ] )

			B22 = np.array( [ [rho_mn2*alpha2, 		alpha2 + rho_mn2*beta22] ,\
							  [alpha2, 				rho_mn2*alpha2 + beta22 + rho_mm*rho_nn*beta12]  ] )

			# Assemble blocks

			M = np.vstack( ( np.hstack((B11, B12)), np.hstack((B21, B22)) ) )


			# Compute eigenvalues

			eig_lr_theory[:,ii_k,ii_rho_mm,ii_rho_nn] = np.linalg.eigvals(M).real
			eig_theory[:,ii_k,ii_rho_mm,ii_rho_nn] = u**2/2. * (1+eig_lr_theory[:,ii_k,ii_rho_mm,ii_rho_nn])

			dimensionality_theory[ii_k,ii_rho_mm,ii_rho_nn] = ( N + np.sum(eig_lr_theory[:,ii_k,ii_rho_mm,ii_rho_nn]) )**2 / \
													( N + 2*np.sum(eig_lr_theory[:,ii_k,ii_rho_mm,ii_rho_nn]) +\
													 np.sum(eig_lr_theory[:,ii_k,ii_rho_mm,ii_rho_nn]**2)) 



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()

cm_subsection = np.linspace(0.2, 0.99, len(k_values)) 
palette = [ cm.Blues(x) for x in cm_subsection ]

#

fac.SetPlotDim(2.3, 2.)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.imshow(dimensionality_theory[0,:,:], origin='lower', \
	interpolation = 'nearest', cmap='pink_r', \
	extent = (min(rho_mm_values), max(rho_mm_values), min(rho_nn_values), max(rho_nn_values)), \
	vmin = 1, vmax = N )

plt.gca().set_aspect('equal')

plt.colorbar()
plt.grid(False)

plt.xlabel(r'Connectivity overlap $\rho_{m^1m^1}$')
plt.ylabel(r'Connectivity overlap $\rho_{n^2n^2}$')

plt.xlim(-1,1)
plt.xticks([-1, 0, 1])

plt.ylim(-1,1)
plt.yticks([-1, 0, 1])

plt.savefig(path_plot+'dimensionality.pdf')
ax.ticklabel_format(useOffset=False)

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
sys.exit(0)
