import os
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.pylab as pylab

rc('text', usetex=True)
params = {'legend.fontsize': '14',
          'figure.figsize': (6, 5),
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

from core.external.vasp.anharmonic_score import *

systems = ['CsPbI','CsSnI','CsPbBr','CsSnBr']
systems = ['SrTiO','SrSnO','BaTiO','BaSnO']
colors =  ['#085f63', '#49beb7','#fccf4d', '#ef255f']

current_working_dir = os.getcwd()
for i,system in enumerate(systems):
    os.chdir(system+"_Pm3m")
    print(system)
    scorer_short = AnharmonicScore(md_frames='./vasprun_md.xml',  ref_frame='./SPOSCAR',force_constants=None, potim=2,force_sets_filename='FORCE_SETS_222')
    scorer_long = AnharmonicScore(md_frames='./vasprun_md_long.xml',  ref_frame='./SPOSCAR',force_constants=None, potim=2,force_sets_filename='FORCE_SETS_222')

    structural_sigma_short, time_series_short = scorer_short.structural_sigma(return_trajectory=True)
    structural_sigma_long, time_series_long = scorer_long.structural_sigma(return_trajectory=True)

    plt.plot(time_series_short, structural_sigma_short, '-', c=colors[i], lw =2) #abel=systems[i] + '$_{3}$' )
    plt.plot(time_series_long, structural_sigma_long, '-', label=systems[i] + '$_{3}$', c=colors[i],alpha=0.5)

    os.chdir('..')

plt.legend(ncol=2)
plt.xlabel('Time (fs)')
plt.ylabel("$\\sigma^{(2)}(t)$")
plt.tight_layout()
plt.savefig('sigma_trajectory_long_ox_chalco.pdf')


