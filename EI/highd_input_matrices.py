
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

path_plot = 'Plots_highd_input/'



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

u = 1


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### FROM THEORY - vary w and g

w_values = np.linspace(1, 10, 351)
g_values = np.linspace(0, 2, 351)

eig_conn = np.zeros(( len(w_values), len(g_values) ))
eig_lr_theory = np.zeros(( 2, len(w_values), len(g_values) ))
eig_theory = np.zeros(( 2, len(w_values), len(g_values) ))
a_theory = np.zeros(( 2, len(w_values), len(g_values) ))
b_theory = np.zeros(( 2, len(w_values), len(g_values) ))


# General

for ii_w, w in enumerate(w_values):

	for ii_g, g in enumerate(g_values):

		k = w*np.sqrt(2*(1+g**2))
		rho_mn = (1-g)/np.sqrt(2*(1+g**2))
		lambda_ = k*rho_mn
		eig_conn[ii_w,ii_g] = lambda_

		alpha = k/(1-lambda_)

		m = np.array([1, 1])/np.sqrt(2)
		n = np.array([1, -g])/np.sqrt(g**2+1)

		if lambda_<1:

			# Eigenvalues

			#### From analytical expression

			eig_lr_theory[0,ii_w,ii_g] = k/(2*(2-lambda_)) * ( 2*rho_mn+alpha + np.sqrt((2*rho_mn+alpha)**2 -4*(rho_mn**2-1)) )
			eig_lr_theory[1,ii_w,ii_g] = k/(2*(2-lambda_)) * ( 2*rho_mn+alpha - np.sqrt((2*rho_mn+alpha)**2 -4*(rho_mn**2-1)) )

			eig_theory[0,ii_w,ii_g] = u**2/2. * (1+eig_lr_theory[0,ii_w,ii_g])
			eig_theory[1,ii_w,ii_g] = u**2/2. * (1+eig_lr_theory[1,ii_w,ii_g])

			#### From covariance matrix

			alpha = k/(1-lambda_)

			cov = u**2/2. * ( np.eye(2) + (k/(2-lambda_))*((np.outer(m,n)+np.outer(n,m)) + (k/(1-lambda_))*np.outer(m,m)) )

			eig_, eigvec_ = np.linalg.eig(cov)
			eigvec_ = eigvec_[:,np.flipud(np.argsort(eig_))]


			# Eigenvectors (overlap with sum and difference)

			#### From covariance matrix

			a_theory[:,ii_w,ii_g] = (eigvec_[0,:]+eigvec_[1,:])/np.sqrt(2)
			b_theory[:,ii_w,ii_g] = (eigvec_[0,:]-eigvec_[1,:])/np.sqrt(2)

			if a_theory[0,ii_w,ii_g]<0:
				eigvec_[:,0] *= -1
				a_theory[:,ii_w,ii_g] = (eigvec_[0,:]+eigvec_[1,:])/np.sqrt(2)
				b_theory[:,ii_w,ii_g] = (eigvec_[0,:]-eigvec_[1,:])/np.sqrt(2)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()
fac.SetPlotDim(1.78, 1.7)

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=1, ls=':', linewidth=0.7, color='k')
plt.plot(w_values, (w_values-1)/w_values, color='k', linewidth=0.7)

plt.imshow(eig_conn.T, origin='lower', extent = (np.min(w_values), np.max(w_values), np.min(g_values), np.max(g_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -np.max(np.fabs(eig_conn)), vmax = np.max(np.fabs(eig_conn)), cmap='RdBu_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(w_values))
plt.xticks([1, np.max(w_values)])

plt.ylim(0, np.max(g_values))
plt.yticks([0, np.max(g_values)])

plt.xlabel(r'Recurrent strength $w$')
plt.ylabel(r'Inh dominance $g$')

plt.savefig(path_plot+'lambda.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=1, ls='--', linewidth=0.7, color='k')
plt.plot(w_values, (w_values-1)/w_values, color='k', linewidth=1.4)

plt.imshow(eig_theory[0,:].T, origin='lower', extent = (np.min(w_values), np.max(w_values), np.min(g_values), np.max(g_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 30, cmap='pink_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(w_values))
plt.xticks([1, np.max(w_values)])

plt.ylim(0, np.max(g_values))
plt.yticks([0, np.max(g_values)])

plt.xlabel(r'Recurrent strength $w$')
plt.ylabel(r'Inh dominance $g$')

plt.savefig(path_plot+'var+.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=1, ls='--', linewidth=0.7, color='k')
plt.plot(w_values, (w_values-1)/w_values, color='k', linewidth=1.4)

plt.imshow(eig_theory[1,:].T, origin='lower', extent = (np.min(w_values), np.max(w_values), np.min(g_values), np.max(g_values)), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 0.5, cmap='pink_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(w_values))
plt.xticks([1, np.max(w_values)])

plt.ylim(0, np.max(g_values))
plt.yticks([0, np.max(g_values)])

plt.xlabel(r'Recurrent strength $w$')
plt.ylabel(r'Inh dominance $g$')

plt.savefig(path_plot+'var-.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=1, ls='--', linewidth=0.7, color='k')
plt.plot(w_values, (w_values-1)/w_values, color='k', linewidth=1.4)

plt.imshow(a_theory[0,:].T, origin='lower', extent = (np.min(w_values), np.max(w_values), np.min(g_values), np.max(g_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -1, vmax = 1, cmap='RdBu_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(w_values))
plt.xticks([1, np.max(w_values)])

plt.ylim(0, np.max(g_values))
plt.yticks([0, np.max(g_values)])

plt.xlabel(r'Recurrent strength $w$')
plt.ylabel(r'Inh dominance $g$')

plt.savefig(path_plot+'eigvec+.pdf')

plt.show()


#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=1, ls='--', linewidth=0.7, color='k')
plt.plot(w_values, (w_values-1)/w_values, color='k', linewidth=0.7)

plt.imshow(b_theory[0,:].T, origin='lower', extent = (np.min(w_values), np.max(w_values), np.min(g_values), np.max(g_values)), aspect='auto', \
	interpolation = 'nearest', vmin = -1, vmax = 1, cmap='RdBu_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(w_values))
plt.xticks([1, np.max(w_values)])

plt.ylim(0, np.max(g_values))
plt.yticks([0, np.max(g_values)])

plt.xlabel(r'Recurrent strength $w$')
plt.ylabel(r'Inh dominance $g$')

plt.savefig(path_plot+'eigvec-.pdf')

plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
