
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

path_plot = 'Plots_scan_theory/'
path_data = 'Data_scan_theory/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

# General

k = 2

rho_mn1 = 0.
rho_mn2 = 0.

# rho_mm = 0
# rho_nn = 0

u = 1

# Sampling

Nsamples = int(1e5)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PREDICTION FOR GENERAL CASE

eig_lr = np.zeros(( Nsamples, 4 ))
eig_ = np.zeros(( Nsamples, 4 ))


doCompute = 1

if doCompute:

	for ii_sample in range(Nsamples):

		# Initialize network

		rho_mm = np.random.uniform(-1,1)
		rho_nn = np.random.uniform(-1,1)

		# rho_mn1 = np.random.uniform(-1,1/k)
		# rho_mn2 = np.random.uniform(-1,1/k)

		lambda1_ = k*rho_mn1
		lambda2_ = k*rho_mn2

		alpha1 = k/(2-lambda1_)
		alpha2 = k/(2-lambda2_)

		beta11 = k**2/((1-lambda1_)*(2-lambda1_))
		beta22 = k**2/((1-lambda2_)*(2-lambda2_))
		beta12 = k**2*(4-lambda1_-lambda2_)/((2-lambda1_-lambda2_)*(2-lambda1_)*(2-lambda2_))


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

		eig_lr[ii_sample,:] = np.linalg.eigvals(M).real

		eig_[ii_sample,:] = u**2/2. * (1+eig_lr[ii_sample,:])

		eig_lr[ii_sample,:] = np.sort(eig_lr[ii_sample,:])
		eig_[ii_sample,:] = np.sort(eig_[ii_sample,:])

		if eig_lr[ii_sample,1]>0: print(rho_mm, rho_nn, rho_mn1, rho_mn2)


	fac.Store(eig_lr, 'eig_lr.p', path_data)
	fac.Store(eig_, 'eig_.p', path_data)

else:

	eig_lr = fac.Retrieve('eig_lr.p', path_data)
	eig_ = fac.Retrieve('eig_.p', path_data)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

parts = plt.violinplot( eig_, widths=0.6, showmeans=True, showextrema=True )
for pc in parts['bodies']:
    pc.set_color('0.6')
    pc.set_edgecolor('0.4')

parts['cmins'].set_colors('0.4')
parts['cmaxes'].set_colors('0.4')
parts['cbars'].set_colors('0.4')

parts['cmeans'].set_colors('0.4')

plt.axhline(y=u**2/2., color='0', linewidth=0.7)

plt.xlabel(r'Eigenvalue number')
plt.ylabel(r'Modified eigenvalues')

# plt.xlim(0, T-Tcut)

plt.savefig(path_plot+'distribution.pdf')

plt.show()

#
eig_[eig_>1e2] = 1e2

fg = plt.figure()
ax = plt.axes(frameon=True)

parts = plt.violinplot( eig_, widths=0.6, showmeans=True, showextrema=True )
for pc in parts['bodies']:
    pc.set_color('0.6')
    pc.set_edgecolor('0.4')

parts['cmins'].set_colors('0.4')
parts['cmaxes'].set_colors('0.4')
parts['cbars'].set_colors('0.4')

parts['cmeans'].set_colors('0.4')

plt.axhline(y=u**2/2., color='0', linewidth=0.7)

plt.xlabel(r'Eigenvalue number (sorted)')
plt.ylabel(r'Modified eigenvalues')

plt.ylim(0, 3.)
plt.yticks([0, 1, 2, 3])

# plt.xlim(0, T-Tcut)

plt.savefig(path_plot+'distribution_zoom.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
