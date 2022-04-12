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
colors =  ['#085f63', '#49beb7','#fccf4d', '#ef255f']

from core.external.vasp.anharmonic_score import *

scorer_perfect_cubic = AnharmonicScore(md_frames='./vasprun_md.xml', ref_frame='./SPOSCAR', force_constants=None, potim=2,
                               force_sets_filename='FORCE_SETS_222')

scorer_averaged = AnharmonicScore(md_frames='./vasprun_md.xml', ref_frame='./CONTCAR_ion_opt', force_constants=None, potim=2,
                               force_sets_filename='FORCE_SETS_222_averaged')

structural_sigma_cubic, time_series_cubic = scorer_perfect_cubic.structural_sigma(return_trajectory=True)
structural_sigma_average, time_series_average = scorer_averaged.structural_sigma(return_trajectory=True)

plt.plot(time_series_cubic, structural_sigma_cubic, ':', c=colors[0], lw=2, label='cubic reference' )
plt.plot(time_series_average, structural_sigma_average, '-', c=colors[1], lw=2, label='MD averaged reference' )

from collect_data import get_temperature_dependent_second_order_fc
tdep_fc2 = get_temperature_dependent_second_order_fc()
scorer = AnharmonicScore(md_frames='./vasprun_md.xml', ref_frame='./SPOSCAR',
                         force_constants=tdep_fc2,
                         include_third_order=False, potim=2)
sigmas, t = scorer.structural_sigma(return_trajectory=True)
plt.plot(t, sigmas, '-', c=colors[2], lw=2, label='$\\tilde{\\sigma}^{(2,2)}$' )

plt.legend()
plt.xlabel('Time (fs)')
plt.ylabel("$\\sigma^{(2)}(t)$")
plt.tight_layout()

plt.savefig('perfect_average_compare.pdf')