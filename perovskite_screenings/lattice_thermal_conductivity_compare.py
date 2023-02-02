import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from matplotlib.patches import Patch

rc('text', usetex=True)

import matplotlib.pylab as pylab

params = {'legend.fontsize': '14',
          'figure.figsize': (6,5),
          'axes.labelsize': 30,
          'axes.titlesize': 28,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20}
pylab.rcParams.update(params)

data={'BaHfO':{'kl':10.9,'sigma_2':0.640,'sigma_3':0.481},
      #'SrTiO':{'kl':10.2,'sigma_2':0.800,'sigma_3':0.765},
      'KMgF':{'kl':7.6,'sigma_2':0.978,'sigma_3':0.623},
      'BaZrO':{'kl':6.75,'sigma_2':0.595,'sigma_3':0.455},
      'CsCaF':{'kl':3.05,'sigma_2':0.379,'sigma_3':0.320},
      'RbCaF':{'kl':2.5,'sigma_2':0.479,'sigma_3':0.390},
      'KZnF':{'kl':2.5,'sigma_2':0.563,'sigma_3':0.438},
      'CsHgF':{'kl':0.5,'sigma_2':0.569,'sigma_3':0.498},
      'CsSrF':{'kl':0.98,'sigma_2':0.530,'sigma_3':0.447},
      'RbCdF':{'kl':0.8,'sigma_2':0.409,'sigma_3':0.348},
      'RbMgF':{'kl':5.9,'sigma_2':1.057,'sigma_3':0.659},
      'CsCdF':{'kl':1.65,'sigma_2':0.404,'sigma_3':0.323}}

x=[]
y=[]
for k in data.keys():
    y.append(data[k]['kl'])
    x.append(data[k]['sigma_2']/data[k]['sigma_3'])
    plt.text(x[-1],y[-1],k+"$_{3}$")
plt.plot(x,y,'o',c='#cb0000',ms=15)
plt.xlabel('$\\sigma^{(2)}/\\sigma^{(3)}$')
plt.ylabel('$\\kappa_{\\mbox{\\large{3,4ph}}}^{\\mbox{\\large{SCP}}}$ (W/mK)')
plt.tight_layout()
plt.show()
#plt.savefig('/Users/z3079335/Desktop/kappa_sigma_ratio.pdf')