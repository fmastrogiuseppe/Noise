
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

path_plot = 'Plots_vary_rhomn/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 50

k_values = [ 2 ] # np.linspace(1, 10, 4)
rho_mn1_values = np.linspace(-1, 0.99, 200)
rho_mn2_values = np.linspace(-1, 0.99, 200)
u = 1

#

eig = np.zeros(( 4, len(k_values), len(rho_mn1_values), len(rho_mn2_values) ))
eig_lr = np.zeros(( 4, len(k_values), len(rho_mn1_values), len(rho_mn2_values) ))
dimensionality = np.zeros(( len(k_values), len(rho_mn1_values), len(rho_mn2_values) ))

eig_theory = np.zeros(( 4, len(k_values), len(rho_mn1_values), len(rho_mn2_values) ))
eig_lr_theory = np.zeros(( 4, len(k_values), len(rho_mn1_values), len(rho_mn2_values) ))
dimensionality_theory = np.zeros(( len(k_values), len(rho_mn1_values), len(rho_mn2_values) ))


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### FROM THEORY

for ii_k, k in enumerate(k_values):

	for ii_rho_mn1, rho_mn1 in enumerate(rho_mn1_values):

		for ii_rho_mn2, rho_mn2 in enumerate(rho_mn1_values):

			lambda1_ = k*rho_mn1
			lambda2_ = k*rho_mn2

			if lambda1_<1 and lambda2_<1:

				#### From analytical expression

				alpha1 = k/(1-lambda1_)
				alpha2 = k/(1-lambda2_)

				eig_lr_theory[0,ii_k,ii_rho_mn1,ii_rho_mn2] = k/(2*(2-lambda1_)) * ( 2*rho_mn1+alpha1 + np.sqrt((2*rho_mn1+alpha2)**2 -4*(rho_mn1**2-1)) )
				eig_lr_theory[1,ii_k,ii_rho_mn1,ii_rho_mn2] = k/(2*(2-lambda1_)) * ( 2*rho_mn1+alpha1 - np.sqrt((2*rho_mn1+alpha2)**2 -4*(rho_mn1**2-1)) )

				eig_lr_theory[2,ii_k,ii_rho_mn1,ii_rho_mn2] = k/(2*(2-lambda2_)) * ( 2*rho_mn2+alpha2 + np.sqrt((2*rho_mn1+alpha2)**2 -4*(rho_mn2**2-1)) )
				eig_lr_theory[3,ii_k,ii_rho_mn1,ii_rho_mn2] = k/(2*(2-lambda2_)) * ( 2*rho_mn2+alpha2 - np.sqrt((2*rho_mn1+alpha2)**2 -4*(rho_mn2**2-1)) )

				eig_theory[0,ii_k,ii_rho_mn1,ii_rho_mn2] = u**2/2. * (1+eig_lr_theory[0,ii_k,ii_rho_mn1,ii_rho_mn2])
				eig_theory[1,ii_k,ii_rho_mn1,ii_rho_mn2] = u**2/2. * (1+eig_lr_theory[1,ii_k,ii_rho_mn1,ii_rho_mn2])

				eig_theory[2,ii_k,ii_rho_mn1,ii_rho_mn2] = u**2/2. * (1+eig_lr_theory[2,ii_k,ii_rho_mn1,ii_rho_mn2])
				eig_theory[3,ii_k,ii_rho_mn1,ii_rho_mn2] = u**2/2. * (1+eig_lr_theory[3,ii_k,ii_rho_mn1,ii_rho_mn2])

				dimensionality_theory[ii_k,ii_rho_mn1,ii_rho_mn2] = ( N + np.sum(eig_lr_theory[:,ii_k,ii_rho_mn1,ii_rho_mn2]) )**2 / \
														( N + 2*np.sum(eig_lr_theory[:,ii_k,ii_rho_mn1,ii_rho_mn2]) +\
														 np.sum(eig_lr_theory[:,ii_k,ii_rho_mn1,ii_rho_mn2]**2)) 

			else:

				eig_theory[:,ii_k,ii_rho_mn1,ii_rho_mn2] = np.nan
				eig_lr_theory[:,ii_k,ii_rho_mn1,ii_rho_mn2] = np.nan
				dimensionality_theory[ii_k,ii_rho_mn1,ii_rho_mn2] = np.nan


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

plt.axvline(x=1./k, ls='--', color='0', linewidth=0.7)
plt.axhline(y=1./k, ls='--', color='0', linewidth=0.7)

plt.imshow(dimensionality_theory[0,:,:], origin='lower', \
	interpolation = 'nearest', cmap='pink_r', \
	extent = (min(rho_mn1_values), max(rho_mn1_values), min(rho_mn2_values), max(rho_mn2_values)), \
	vmin = 1, vmax = N )

plt.gca().set_aspect('equal')

plt.colorbar()
plt.grid(False)

plt.xlabel(r'Connectivity overlap $\rho_{m^1n^1}$')
plt.ylabel(r'Connectivity overlap $\rho_{m^2n^2}$')

plt.xlim(-1,1)
plt.xticks([-1, 0, 1])

plt.ylim(-1,1)
plt.yticks([-1, 0, 1])

plt.savefig(path_plot+'dimensionality.pdf')
ax.ticklabel_format(useOffset=False)

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
sys.exit(0)
