import glob
import os
import matplotlib.pyplot as plt
from core.external.vasp.anharmonic_score import *
from core.dao.vasp import VaspReader
from matplotlib import rc
from ase.db import connect
import sqlite3
import json

from core.internal.builders.crystal import map_ase_atoms_to_crystal

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

db = connect('sigma.db')

_db = sqlite3.connect('sigma.db')
cur = _db.cursor()
cur.execute("SELECT * FROM systems")
rows = cur.fetchall()

all_uids = []
for row in rows:
    for i in row:
        if 'uid' in str(i):
            this_dict = json.loads(str(i))
            all_uids.append(this_dict['uid'])

zero_d_chloride_sigma = {}
zero_d_bromide_sigma = {}
two_d_chloride_sigma = {}
two_d_bromide_sigma = {}

for id in all_uids:
    row = db.get(selection=[('uid', '=', id)])
    structure = map_ase_atoms_to_crystal(row.toatoms())
    _dict = structure.all_atoms_count_dictionaries()

    try:
        n_cl = _dict['Cl']
    except KeyError:
        n_cl = 0

    try:
        n_br = _dict['Br']
    except KeyError:
        n_br = 0

    try:
        n_i = _dict['I']
    except KeyError:
        n_i = 0

    if ('Pnma' not in id):
        if (('0D' in id) and ('Cl' in id)):
            comp = n_cl / 2.0
            zero_d_chloride_sigma[comp] = row.data['sigmas']
        elif (('2D' in id) and ('Cl' in id)):
            comp = n_cl / 2.0
            two_d_chloride_sigma[comp] = row.data['sigmas']
        elif (('0D' in id) and ('Br' in id)):
            comp = n_br / 2.0
            zero_d_bromide_sigma[comp] = row.data['sigmas']
        elif (('2D' in id) and ('Br' in id)):
            comp = n_br / 2.0
            two_d_bromide_sigma[comp] = row.data['sigmas']

        if ('pure_Cs3Sb2I9-0D' in id):
            zero_d_bromide_sigma[0.0] = row.data['sigmas']
            zero_d_chloride_sigma[0.0] = row.data['sigmas']

        if ('pure_Cs3Sb2I9-2D' in id):
            two_d_bromide_sigma[0.0] = row.data['sigmas']
            two_d_chloride_sigma[0.0] = row.data['sigmas']

    if ('Pnma' in id) and ("PbI" in id):
        pb_i_pnma_sigma = row.data['sigmas']

zero_d_chloride_keys = list(sorted(zero_d_chloride_sigma.keys()))
zero_d_bromide_keys = list(sorted(zero_d_bromide_sigma.keys()))
two_d_chloride_keys = list(sorted(two_d_chloride_sigma.keys()))
two_d_bromide_keys = list(sorted(two_d_bromide_sigma.keys()))
fig, ax = plt.subplots(figsize=(6, 5))

c = '#36688D'
bp2 = plt.boxplot([two_d_bromide_sigma[k] for k in two_d_bromide_keys], positions=[_x for _x in two_d_bromide_keys],
                  widths=[2 / len(two_d_bromide_keys) for _ in two_d_bromide_keys], patch_artist=True,
                  boxprops=dict(facecolor=c, color=c, alpha=1),
                  capprops=dict(color=c),
                  whiskerprops=dict(color=c),
                  # flierprops=dict(color=c, markeredgecolor=c),
                  medianprops=dict(color=c),
                  showfliers=False)

c = '#F3Cd05'
bp1 = plt.boxplot([zero_d_bromide_sigma[k] for k in zero_d_bromide_keys],
                  positions=[_x for _x in zero_d_bromide_keys],
                  widths=[2 / len(zero_d_bromide_keys) + 0.1 for _ in zero_d_bromide_keys], patch_artist=True,
                  boxprops=dict(facecolor=c, color=c, alpha=0.7),
                  capprops=dict(color=c),
                  whiskerprops=dict(color=c, alpha=0.7),
                  # flierprops=dict(color=c, markeredgecolor=c,alpha=0.7),
                  medianprops=dict(color=c),
                  showfliers=False)

ax.plot([_x for _x in zero_d_bromide_keys],[np.average(pb_i_pnma_sigma) for _ in zero_d_bromide_keys],'r--',lw=2)

ax.legend([bp2["boxes"][0], bp1["boxes"][0]], ['0D', '2D'], loc='upper right')
ax.set_ylabel('$\\sigma(y,T)$', fontsize=20)
ax.set_xlabel('$y$ in Cs$_{3}$Sb$_2$Br$_{y}$I$_{9-y}$', fontsize=16)
ax.text(1.0,0.59, '$\sigma$(300 K) for $\gamma$-CsPbI$_3$', c='r',fontsize=12)
xticklabels = ['%.1f' % i for i in [_x for _x in two_d_bromide_keys]]
ax.set_ylim([0.18,0.65])
ax.set_xticklabels(xticklabels, rotation=45)

plt.tight_layout()
plt.savefig("sigma_box.pdf")
