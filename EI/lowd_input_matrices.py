
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

path_plot = 'Plots_lowd_input/'


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PARAMETERS

theta_values = np.linspace(0, 2*np.pi, 151)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### FROM THEORY - vary w

g = 1.01

w_values = np.linspace(1, 10, 151)

eig_theory = np.zeros(( 2, len(w_values), len(theta_values) ))
a_theory = np.zeros(( 2, len(w_values), len(theta_values) ))
b_theory = np.zeros(( 2, len(w_values), len(theta_values) ))


# General

for ii_w, w in enumerate(w_values):

	k = w*np.sqrt(2*(1+g**2))
	rho_mn = (1-g)/np.sqrt(2*(1+g**2))
	lambda_ = k*rho_mn

	for ii_theta, theta in enumerate(theta_values):

		rho_nu = (np.cos(theta)-g*np.sin(theta))/np.sqrt(1+g**2)
		rho_mu = (np.cos(theta)+np.sin(theta))/np.sqrt(2)
		rho_uu = 1.

		alpha = (rho_nu*k)/(2-lambda_)
		beta = (rho_nu*k)**2/((2-lambda_)*(1-lambda_))

		m = np.array([1, 1])/np.sqrt(2)
		n = np.array([1, -g])/np.sqrt(g**2+1)

		# Eigenvalues

		#### From analytical expression

		eig_theory[0,ii_w,ii_theta] = 1/4. * ( rho_uu+2*alpha*rho_mu+beta + np.sqrt((rho_uu+2*alpha*rho_mu+beta)**2 -4*(rho_uu-rho_mu**2)*(beta-alpha**2)) )
		eig_theory[1,ii_w,ii_theta] = 1/4. * ( rho_uu+2*alpha*rho_mu+beta - np.sqrt((rho_uu+2*alpha*rho_mu+beta)**2 -4*(rho_uu-rho_mu**2)*(beta-alpha**2)) )

		#### From covariance matrix

		u = np.array([np.cos(theta), np.sin(theta)])

		cov = 0.5*( np.outer(u,u) + alpha*(np.outer(m,u)+np.outer(u,m)) + beta*np.outer(m,m) )
		eig_, eigvec_ = np.linalg.eig(cov)
		eigvec_ = eigvec_[:,np.flipud(np.argsort(eig_))]


		# Eigenvectors (overlap with sum and difference)

		#### From covariance matrix

		a_theory[:,ii_w,ii_theta] = (eigvec_[0,:]+eigvec_[1,:])/np.sqrt(2)
		b_theory[:,ii_w,ii_theta] = (eigvec_[0,:]-eigvec_[1,:])/np.sqrt(2)

		if a_theory[0,ii_w,ii_theta]<0:
			eigvec_[:,0] *= -1
			a_theory[:,ii_w,ii_theta] = (eigvec_[0,:]+eigvec_[1,:])/np.sqrt(2)
			b_theory[:,ii_w,ii_theta] = (eigvec_[0,:]-eigvec_[1,:])/np.sqrt(2)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fac.SetPlotParams()
fac.SetPlotDim(1.83, 1.7)

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=45, ls='--', color='0', linewidth=0.7)
plt.axhline(y=90, ls='-', color='0', linewidth=0.7)
plt.axhline(y=135, ls='--', color='0', linewidth=0.7)

plt.imshow(eig_theory[0,:].T, origin='lower', extent = (np.min(w_values), np.max(w_values), 0, 360), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = np.max(np.fabs(eig_theory)), cmap='pink_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(w_values))
plt.xticks([1, np.max(w_values)])

plt.ylim(0, 180)
plt.yticks([0, 90, 180])

plt.xlabel(r'Recurrent strength $w$')
plt.ylabel(r'Input direction')

plt.savefig(path_plot+'var+_w.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=45, ls='--', color='0', linewidth=0.7)
plt.axhline(y=90, ls='-', color='0', linewidth=0.7)
plt.axhline(y=135, ls='--', color='0', linewidth=0.7)

plt.imshow(eig_theory[1,:].T, origin='lower', extent = (np.min(w_values), np.max(w_values), 0, 360), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 0.5, cmap='pink_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(w_values))
plt.xticks([1, np.max(w_values)])

plt.ylim(0, 180)
plt.yticks([0, 90, 180])

plt.xlabel(r'Recurrent strength $w$')
plt.ylabel(r'Input direction')

plt.savefig(path_plot+'var-_w.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=45, ls='--', color='0', linewidth=0.7)
plt.axhline(y=90, ls='-', color='0', linewidth=0.7)
plt.axhline(y=135, ls='--', color='0', linewidth=0.7)

plt.imshow(np.sum(eig_theory,0).T, origin='lower', extent = (np.min(w_values), np.max(w_values), 0, 360), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = np.max(np.fabs(np.sum(eig_theory,0))), cmap='pink_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(w_values))
plt.xticks([1, np.max(w_values)])

plt.ylim(0, 180)
plt.yticks([0, 90, 180])

plt.xlabel(r'Recurrent strength $w$')
plt.ylabel(r'Input direction')

plt.savefig(path_plot+'var_w.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=45, ls='--', color='0', linewidth=0.7)
plt.axhline(y=90, ls='-', color='0', linewidth=0.7)
plt.axhline(y=135, ls='--', color='0', linewidth=0.7)

plt.imshow(a_theory[0,:].T, origin='lower', extent = (np.min(w_values), np.max(w_values), 0, 360), aspect='auto', \
	interpolation = 'nearest', vmin = -1, vmax = 1, cmap='RdBu_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(w_values))
plt.xticks([1, np.max(w_values)])

plt.ylim(0, 180)
plt.yticks([0, 90, 180])

plt.xlabel(r'Recurrent strength $w$')
plt.ylabel(r'Input direction')

plt.savefig(path_plot+'eigvec+_sum_w.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=45, ls='--', color='0', linewidth=0.7)
plt.axhline(y=90, ls='-', color='0', linewidth=0.7)
plt.axhline(y=135, ls='--', color='0', linewidth=0.7)

plt.imshow(b_theory[0,:].T, origin='lower', extent = (np.min(w_values), np.max(w_values), 0, 360), aspect='auto', \
	interpolation = 'nearest', vmin = -1, vmax = 1, cmap='RdBu_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(w_values))
plt.xticks([1, np.max(w_values)])

plt.ylim(0, 180)
plt.yticks([0, 90, 180])

plt.xlabel(r'Recurrent strength $w$')
plt.ylabel(r'Input direction')

plt.savefig(path_plot+'eigvec+_diff_w.pdf')

plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### FROM THEORY - vary g

w = 2.

g_values = np.linspace(1, 2, 151)

eig_theory = np.zeros(( 2, len(g_values), len(theta_values) ))
a_theory = np.zeros(( 2, len(g_values), len(theta_values) ))
b_theory = np.zeros(( 2, len(g_values), len(theta_values) ))


# General

for ii_g, g in enumerate(g_values):

	k = w*np.sqrt(2*(1+g**2))
	rho_mn = (1-g)/np.sqrt(2*(1+g**2))
	lambda_ = k*rho_mn

	for ii_theta, theta in enumerate(theta_values):

		rho_nu = (np.cos(theta)-g*np.sin(theta))/np.sqrt(1+g**2)
		rho_mu = (np.cos(theta)+np.sin(theta))/np.sqrt(2)
		rho_uu = 1.

		alpha = (rho_nu*k)/(2-lambda_)
		beta = (rho_nu*k)**2/((2-lambda_)*(1-lambda_))

		m = np.array([1, 1])/np.sqrt(2)
		n = np.array([1, -g])/np.sqrt(g**2+1)


		# Eigenvalues

		#### From analytical expression

		eig_theory[0,ii_g,ii_theta] = 1/4. * ( rho_uu+2*alpha*rho_mu+beta + np.sqrt((rho_uu+2*alpha*rho_mu+beta)**2 -4*(rho_uu-rho_mu**2)*(beta-alpha**2)) )
		eig_theory[1,ii_g,ii_theta] = 1/4. * ( rho_uu+2*alpha*rho_mu+beta - np.sqrt((rho_uu+2*alpha*rho_mu+beta)**2 -4*(rho_uu-rho_mu**2)*(beta-alpha**2)) )

		#### From covariance matrix

		u = np.array([np.cos(theta), np.sin(theta)])

		cov = 0.5*( np.outer(u,u) + alpha*(np.outer(m,u)+np.outer(u,m)) + beta*np.outer(m,m) )
		eig_, eigvec_ = np.linalg.eig(cov)
		eigvec_ = eigvec_[:,np.flipud(np.argsort(eig_))]


		# Eigenvectors (overlap with sum and difference)

		#### From covariance matrix

		a_theory[:,ii_g,ii_theta] = (eigvec_[0,:]+eigvec_[1,:])/np.sqrt(2)
		b_theory[:,ii_g,ii_theta] = (eigvec_[0,:]-eigvec_[1,:])/np.sqrt(2)

		if a_theory[0,ii_g,ii_theta]<0:
			eigvec_[:,0] *= -1
			a_theory[:,ii_g,ii_theta] = (eigvec_[0,:]+eigvec_[1,:])/np.sqrt(2)
			b_theory[:,ii_g,ii_theta] = (eigvec_[0,:]-eigvec_[1,:])/np.sqrt(2)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### PLOT

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=45, ls='--', color='0', linewidth=0.7)
plt.axhline(y=90, ls='-', color='0', linewidth=0.7)
plt.axhline(y=135, ls='--', color='0', linewidth=0.7)

plt.imshow(eig_theory[0,:].T, origin='lower', extent = (np.min(g_values), np.max(g_values), 0, 360), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = np.max(np.fabs(eig_theory)), cmap='pink_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(g_values))
plt.xticks([1, np.max(g_values)])

plt.ylim(0, 180)
plt.yticks([0, 90, 180])

plt.xlabel(r'Inh dominance $g$')
plt.ylabel(r'Input direction')

plt.savefig(path_plot+'var+_g.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=45, ls='--', color='0', linewidth=0.7)
plt.axhline(y=90, ls='-', color='0', linewidth=0.7)
plt.axhline(y=135, ls='--', color='0', linewidth=0.7)

plt.imshow(eig_theory[1,:].T, origin='lower', extent = (np.min(g_values), np.max(g_values), 0, 360), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = 0.5, cmap='pink_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(g_values))
plt.xticks([1, np.max(g_values)])

plt.ylim(0, 180)
plt.yticks([0, 90, 180])

plt.xlabel(r'Inh dominance $g$')
plt.ylabel(r'Input direction')

plt.savefig(path_plot+'var-_g.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=45, ls='--', color='0', linewidth=0.7)
plt.axhline(y=90, ls='-', color='0', linewidth=0.7)
plt.axhline(y=135, ls='--', color='0', linewidth=0.7)

plt.imshow(np.sum(eig_theory,0).T, origin='lower', extent = (np.min(g_values), np.max(g_values), 0, 360), aspect='auto', \
	interpolation = 'nearest', vmin = 0, vmax = np.max(np.fabs(np.sum(eig_theory,0))), cmap='pink_r')

# plt.axis('off')

# plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(g_values))
plt.xticks([1, np.max(g_values)])

plt.ylim(0, 180)
plt.yticks([0, 90, 180])

plt.xlabel(r'Inh dominance $g$')
plt.ylabel(r'Input direction')

plt.savefig(path_plot+'var_g.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=45, ls='--', color='0', linewidth=0.7)
plt.axhline(y=90, ls='-', color='0', linewidth=0.7)
plt.axhline(y=135, ls='--', color='0', linewidth=0.7)

plt.imshow(a_theory[0,:].T, origin='lower', extent = (np.min(g_values), np.max(g_values), 0, 360), aspect='auto', \
	interpolation = 'nearest', vmin = -1, vmax = 1, cmap='RdBu_r')

# plt.axis('off')

plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(g_values))
plt.xticks([1, np.max(g_values)])

plt.ylim(0, 180)
plt.yticks([0, 90, 180])

plt.xlabel(r'Inh dominance $g$')
plt.ylabel(r'Input direction')

plt.savefig(path_plot+'eigvec+_sum_g.pdf')

plt.show()

#

fg = plt.figure()
ax = plt.axes(frameon=True)

plt.axhline(y=45, ls='--', color='0', linewidth=0.7)
plt.axhline(y=90, ls='-', color='0', linewidth=0.7)
plt.axhline(y=135, ls='--', color='0', linewidth=0.7)

plt.imshow(b_theory[0,:].T, origin='lower', extent = (np.min(g_values), np.max(g_values), 0, 360), aspect='auto', \
	interpolation = 'nearest',  vmin = -1, vmax = 1, cmap='RdBu_r')

# plt.axis('off')

plt.colorbar()
plt.grid(False)

plt.xlim(1, np.max(g_values))
plt.xticks([1, np.max(g_values)])

plt.ylim(0, 180)
plt.yticks([0, 90, 180])

plt.xlabel(r'Inh dominance $g$')
plt.ylabel(r'Input direction')

plt.savefig(path_plot+'eigvec+_diff_g.pdf')

plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
