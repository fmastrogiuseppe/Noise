import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import rc
import matplotlib as mpl


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Plot

# Palette 1: colorful
pink = '#E38A8A'
light_green = '#A3F0A3'
green = '#7FB285'
pine_green = '#136F63'
light_pine_green = '#79DDD0'
violet = '#A888BF'
light_red_purp = '#FF616D'
red_purp = '#F73B5C' #'#D83A56'
light_blue = '#3C8DAD'
dark_blue = '#125D98'
queen_blue = '#456990'
orange = '#F5A962'
yellow = '#FFD966'
lavander = '#D5C6E0'
african_violet = '#B084CC'
dark_liver = '#56494C'
dark_purple = '#30011E'
grey_green = '#5B7B7A'
coral = '#ff9f85'
blue_green = '#77CFBF'
eton_blue = '#96c8a2'

palette_colorful = [ pink, light_green, eton_blue, orange, lavander, pine_green, coral ]


def SetPlotParams():

	plt.style.use('ggplot')
	plt.rc('text', usetex=True)
	plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

	fig_width = 2. # width in inches
	fig_height = 1.8  # height in inches
	fig_size =  [fig_width,fig_height]
	plt.rcParams['figure.figsize'] = fig_size
	plt.rcParams['figure.autolayout'] = True

	plt.rcParams['lines.linewidth'] = 1.
	plt.rcParams['lines.markeredgewidth'] = 0.3
	plt.rcParams['lines.markersize'] = 2.5
	plt.rcParams['font.size'] = 10
	plt.rcParams['legend.fontsize'] = 8
	plt.rcParams['axes.facecolor'] = '1'
	plt.rcParams['axes.edgecolor'] = '0'
	plt.rcParams['axes.linewidth'] = '0.7'

	plt.rcParams['axes.labelcolor'] = '0'
	plt.rcParams['axes.labelsize'] = 9.5
	plt.rcParams['axes.titlesize'] = 9.5
	plt.rcParams['xtick.labelsize'] = 8
	plt.rcParams['ytick.labelsize'] = 8
	plt.rcParams['xtick.color'] = '0'
	plt.rcParams['ytick.color'] = '0'
	plt.rcParams['xtick.major.size'] = 2
	plt.rcParams['ytick.major.size'] = 2

	plt.rcParams['font.sans-serif'] = 'Arial'

def SetPlotDim(x,y):

	fig_width = x # width in inches
	fig_height = y # height in inches
	fig_size =  [fig_width,fig_height]
	plt.rcParams['figure.figsize'] = fig_size
	plt.rcParams['figure.autolayout'] = True


def TruncateCmap(cmap, minval=0.0, maxval=1.0, n=100):
	'''
	https://stackoverflow.com/a/18926541
	'''
	if isinstance(cmap, str):
		cmap = plt.get_cmap(cmap)
		new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Data managing

def Store (obj, name, path):

	f = open ( path+name, 'wb')
	pickle.dump(obj,f, protocol=2)
	f.close()

def Retrieve (name, path):

	f = open( path+name, 'rb')
	obj = pickle.load(f)
	f.close()

	return obj
