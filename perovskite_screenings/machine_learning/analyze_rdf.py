import matplotlib.pyplot as plt
import matplotlib
from core.external.vasp.rdf import *
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.vasp import Xdatcar
import warnings

warnings.filterwarnings("ignore")
import os

from matplotlib import rc
import matplotlib.pylab as pylab

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
params = {'legend.fontsize': '10',
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}
pylab.rcParams.update(params)

cmap = plt.cm.coolwarm
norm = matplotlib.colors.Normalize(vmin=1, vmax=100)

import glob

all_vasprun = glob.glob('vasprun_prod_*.xml')
for i in range(len(all_vasprun)):
    print("Loading the molecular dynamic trajectory... " + str(i + 1))
    trajectory = Trajectory.from_file(filename='vasprun_prod_' + str(i + 1) + '.xml')
    trajectory.write_Xdatcar('XDATCAR_temp')
    xd = Xdatcar('XDATCAR_temp')
    os.remove('XDATCAR_temp')
    K_distribution = RadialDistributionFunction.from_species_strings(structures=xd.structures, species_i='K',
                                                                     species_j='Br')
    plt.plot(K_distribution.r, K_distribution.smeared_rdf(sigma=0.1), label='K-K', c=cmap(norm(i)), alpha=0.6)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

plt.colorbar(sm)
plt.tight_layout()
plt.savefig('RDF.pdf')
