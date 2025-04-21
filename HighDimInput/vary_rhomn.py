
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

path_plot = 'Plots_vary_rhomn/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

N = 50

k_values = [ 2 ]
rho_mn_values = np.linspace(-1, 0.99, 1000)
u = 1

#

eig_theory = np.zeros(( 2, len(k_values), len(rho_mn_values) ))
eig_lr_theory = np.zeros(( 2, len(k_values), len(rho_mn_values) ))
dimensionality_theory = np.zeros(( len(k_values), len(rho_mn_values) ))
a_theory = np.zeros(( 2, len(k_values), len(rho_mn_values) ))
b_theory = np.zeros(( 2, len(k_values), len(rho_mn_values) ))
a_overlap_theory = np.zeros(( 2, len(k_values), len(rho_mn_values) ))
b_overlap_theory = np.zeros(( 2, len(k_values), len(rho_mn_values) ))


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### FROM THEORY

for ii_k, k in enumerate(k_values):

	for ii_rho_mn, rho_mn in enumerate(rho_mn_values):

		lambda_ = k*rho_mn

		if lambda_<1:

			#### From reduced matrix

			alpha = k/(1-lambda_)
			cov_lr_reduced = k/(2-lambda_) * np.array([ [rho_mn,1,rho_mn], [1,rho_mn,1], [alpha, alpha*rho_mn, alpha] ])

			eig_lr_ = np.linalg.eigvals(cov_lr_reduced)
			eig_lr_ = eig_lr_[np.flipud(np.argsort((eig_lr_)**2))][0:2]

			eig_lr_theory[:,ii_k,ii_rho_mn] = eig_lr_
			eig_theory[:,ii_k,ii_rho_mn] = u**2/2. * (1+eig_lr_)

			#### From analytical expression

			alpha = k/(1-lambda_)

			eig_lr_theory[0,ii_k,ii_rho_mn] = k/(2*(2-lambda_)) * ( 2*rho_mn+alpha + np.sqrt((2*rho_mn+alpha)**2 -4*(rho_mn**2-1)) )
			eig_lr_theory[1,ii_k,ii_rho_mn] = k/(2*(2-lambda_)) * ( 2*rho_mn+alpha - np.sqrt((2*rho_mn+alpha)**2 -4*(rho_mn**2-1)) )

			eig_theory[0,ii_k,ii_rho_mn] = u**2/2. * (1+eig_lr_theory[0,ii_k,ii_rho_mn])
			eig_theory[1,ii_k,ii_rho_mn] = u**2/2. * (1+eig_lr_theory[1,ii_k,ii_rho_mn])

			dimensionality_theory[ii_k,ii_rho_mn] = ( N + eig_lr_theory[0,ii_k,ii_rho_mn] + eig_lr_theory[1,ii_k,ii_rho_mn] )**2 / \
													( N + 2*(eig_lr_theory[0,ii_k,ii_rho_mn] + eig_lr_theory[1,ii_k,ii_rho_mn]) +\
													 eig_lr_theory[0,ii_k,ii_rho_mn]**2 + eig_lr_theory[1,ii_k,ii_rho_mn]**2 ) 

			a_theory[0,ii_k,ii_rho_mn] = 1./2*( alpha + np.sqrt((2*rho_mn+alpha)**2 -4*(rho_mn**2-1)) )
			a_theory[1,ii_k,ii_rho_mn] = 1./2*( alpha - np.sqrt((2*rho_mn+alpha)**2 -4*(rho_mn**2-1)) )

			b_theory[0,ii_k,ii_rho_mn] = 1
			b_theory[1,ii_k,ii_rho_mn] = 1

			a_overlap_theory[0,ii_k,ii_rho_mn] = (a_theory[0,ii_k,ii_rho_mn]+rho_mn)/\
													np.sqrt(1.+2*rho_mn*a_theory[0,ii_k,ii_rho_mn]+a_theory[0,ii_k,ii_rho_mn]**2)
			a_overlap_theory[1,ii_k,ii_rho_mn] = (a_theory[1,ii_k,ii_rho_mn]+rho_mn)/\
													np.sqrt(1.+2*rho_mn*a_theory[1,ii_k,ii_rho_mn]+a_theory[1,ii_k,ii_rho_mn]**2)

			b_overlap_theory[0,ii_k,ii_rho_mn] = (a_theory[0,ii_k,ii_rho_mn]*rho_mn+1)/\
													np.sqrt(1.+2*rho_mn*a_theory[0,ii_k,ii_rho_mn]+a_theory[0,ii_k,ii_rho_mn]**2)
			b_overlap_theory[1,ii_k,ii_rho_mn] = (a_theory[1,ii_k,ii_rho_mn]*rho_mn+1)/\
													np.sqrt(1.+2*rho_mn*a_theory[1,ii_k,ii_rho_mn]+a_theory[1,ii_k,ii_rho_mn]**2)

		else:

			eig_theory[:,ii_k,ii_rho_mn] = np.nan
			eig_lr_theory[:,ii_k,ii_rho_mn] = np.nan
			dimensionality_theory[ii_k,ii_rho_mn] = np.nan

			a_theory[:,ii_k,ii_rho_mn] = np.nan
			b_theory[:,ii_k,ii_rho_mn] = np.nan

			a_overlap_theory[:,ii_k,ii_rho_mn] = np.nan
			b_overlap_theory[:,ii_k,ii_rho_mn] = np.nan


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()

cm_subsection = np.linspace(0.2, 0.99, len(k_values)) 
palette = [ cm.Blues(x) for x in cm_subsection ]

#

fac.SetPlotDim(2.15, 1.85)

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.plot([-1,1./k], [u**2/2.,u**2/2.], color='0.6', linewidth=1.2, ls='-')

for ii_k, k in enumerate(k_values):

	plt.axvline(x=1./k, ls='--', color='0', linewidth=0.7)

	plt.plot(rho_mn_values, eig_theory[0,ii_k,:], '-', color='#FF686E', linewidth=1.2, label='$k='+str(round(k,2))+'$')
	plt.plot(rho_mn_values, eig_theory[1,ii_k,:], '-', color='#FF686E', linewidth=1.2)

plt.xlabel(r'Connectivity overlap $\rho_{mn}$')
plt.ylabel('Covariance eigenvalues')

plt.xlim(-1,1)
plt.xticks([-1, 0, 1])

plt.ylim(0, 2.4)
plt.yticks([0, 1.2, 2.4])

# plt.legend(loc=2, frameon=False)

plt.savefig(path_plot+'eig.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=1, ls='-', color='k', linewidth='0.7')
plt.axhline(y=N, ls='-', color='k', linewidth='0.7')

for ii_k, k in enumerate(k_values):

	plt.axvline(x=1./k, ls='--', color='0', linewidth=0.7)

	plt.plot(rho_mn_values, dimensionality_theory[ii_k,:], '-', color='#FF686E', linewidth=1.2, label='$k='+str(round(k,2))+'$')

plt.xlabel(r'Connectivity overlap $\rho_{mn}$')
plt.ylabel('Dimensionality')

plt.xlim(-1,1)
plt.xticks([-1, 0, 1])

plt.ylim(-5, 55)
plt.yticks([0, 25, 50])

plt.savefig(path_plot+'dimensionality.pdf')
ax.ticklabel_format(useOffset=False)

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=0, ls='-', color='k', linewidth='0.7')

for ii_k, k in enumerate(k_values):

	plt.axvline(x=1./k, ls='--', color='0', linewidth=0.7)

	plt.plot(rho_mn_values, a_theory[0,ii_k,:], '-', color='#FF686E', linewidth=1.2, label='$k='+str(round(k,2))+'$')
	plt.plot(rho_mn_values, b_theory[0,ii_k,:], ':', color='#FF686E', linewidth=1.2)

plt.xlabel(r'Connectivity overlap $\rho_{mn}$')
plt.ylabel(r'$\bm{v^+}$ components')

plt.xlim(-1,1)
plt.xticks([-1, 0, 1])

plt.ylim(-1,3)
plt.yticks([-1, 1, 3])

# plt.legend(loc=2, frameon=False)

plt.savefig(path_plot+'v+.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=0, ls='-', color='k', linewidth='0.7')

for ii_k, k in enumerate(k_values):

	plt.axvline(x=1./k, ls='--', color='0', linewidth=0.7)

	plt.plot(rho_mn_values, a_theory[1,ii_k,:], '-', color='#FF686E', linewidth=1.2)
	plt.plot(rho_mn_values, b_theory[1,ii_k,:], ':', color='#FF686E', linewidth=1.2)

plt.xlabel(r'Connectivity overlap $\rho_{mn}$')
plt.ylabel(r'$\bm{v^-}$ components')

plt.xlim(-1,1)
plt.xticks([-1, 0, 1])

plt.ylim(-2,2)
plt.yticks([-2, 0, 2])

plt.savefig(path_plot+'v-.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=0, ls='-', color='k', linewidth='0.7')

for ii_k, k in enumerate(k_values):

	plt.axvline(x=1./k, ls='--', color='0', linewidth=0.7)

	plt.plot(rho_mn_values, a_overlap_theory[0,ii_k,:], '-', color='#FF686E', linewidth=1.2, label='$k='+str(round(k,2))+'$')
	plt.plot(rho_mn_values, b_overlap_theory[0,ii_k,:], ':', color='#FF686E', linewidth=1.2)

plt.xlabel(r'Connectivity overlap $\rho_{mn}$')
plt.ylabel(r'$\bm{v^+}$ overlap')

plt.xlim(-1,1)
plt.xticks([-1, 0, 1])

plt.ylim(-1,3)
plt.yticks([-1, 1, 3])

# plt.legend(loc=2, frameon=False)

plt.savefig(path_plot+'v+_overlap.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=0, ls='-', color='k', linewidth='0.7')

for ii_k, k in enumerate(k_values):

	plt.axvline(x=1./k, ls='--', color='0', linewidth=0.7)

	plt.plot(rho_mn_values, a_overlap_theory[1,ii_k,:], '-', color='#FF686E', linewidth=1.2)
	plt.plot(rho_mn_values, b_overlap_theory[1,ii_k,:], ':', color='#FF686E', linewidth=1.2)

plt.xlabel(r'Connectivity overlap $\rho_{mn}$')
plt.ylabel(r'$\bm{v^-}$ overlap')

plt.xlim(-1,1)
plt.xticks([-1, 0, 1])

plt.ylim(-2,2)
plt.yticks([-2, 0, 2])

plt.savefig(path_plot+'v-_overlap.pdf')

plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
