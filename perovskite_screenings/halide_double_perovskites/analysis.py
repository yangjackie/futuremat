import logging

from ase.db import connect
import sqlite3
import json
import os
import math
import argparse

from matplotlib.lines import Line2D
from numpy import dot

from core.internal.builders.crystal import map_ase_atoms_to_crystal
from core.models import Crystal
from core.models.element import shannon_radii
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from ase.geometry import *
from ase.neighborlist import *
from matplotlib.patches import Patch

rc('text', usetex=True)

import matplotlib.pylab as pylab

params = {'legend.fontsize': '14',
          'figure.figsize': (7.5,6),
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

A_site = ['Li', 'Na', 'K', 'Rb', 'Cs']
X_site = ['F', 'Cl', 'Br', 'I']
M_site_mono = ['Pd', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Cu', 'Ag', 'Au', 'Hg', 'In', 'Tl']
M_site_tri = ['Pd', 'Ir', 'Pr', 'Rh', 'Ru', 'La', 'Mo', 'Nd', 'Ni', 'Nb', 'Lu', 'Ce', 'Mn', 'Co', 'Cr', 'Dy', 'Er',
              'Sc',
              'Ta', "Tb", 'Eu', 'Y', 'Al', 'Gd', 'Ga', 'In', 'As', 'Sb', 'Bi', 'Fe', "Sb", "Sc", "Sm", "Ti",
              "Tl", "Tm", "V", "Y", 'Au']

M_site_mono_exclusive = [x for x in M_site_mono if x not in M_site_tri]
M_site_tri_exclusive = [x for x in M_site_tri if x not in M_site_mono]
M_site_variable = [x for x in M_site_tri if x in M_site_mono]


def chemical_classifier(crystal: Crystal) -> dict:
    atom_dict = crystal.all_atoms_count_dictionaries()
    all_elements = list(atom_dict.keys())
    stochiometry = list(sorted([atom_dict[k] for k in all_elements]))
    chemical_dict = {'A_cation': None, 'M_cation_mono': None, 'M_cation_tri': None, 'X_anion': None}

    for e in all_elements:
        if e in X_site:
            chemical_dict['X_anion'] = e
            all_elements.remove(e)

    if (stochiometry == [1, 1, 3]) or (stochiometry == [2, 2, 6]):
        for e in all_elements:
            if (e in M_site_mono) and (e in M_site_tri):
                chemical_dict['M_cation_mono'] = e
                chemical_dict['M_cation_tri'] = e
                all_elements.remove(e)
        chemical_dict['A_cation'] = all_elements[-1]
        assert (None not in [chemical_dict[k] for k in chemical_dict.keys()])
    elif stochiometry == [1, 3, 6]:
        for e in all_elements:
            if (e in A_site) and (e in M_site_mono):
                chemical_dict['A_cation'] = e
                chemical_dict['M_cation_mono'] = e
                all_elements.remove(e)
        chemical_dict['M_cation_tri'] = all_elements[-1]
        assert (None not in [chemical_dict[k] for k in chemical_dict.keys()])
    elif stochiometry == [1, 1, 2, 6]:
        for e in all_elements:
            if (e in A_site) and (atom_dict[e] == 2):
                chemical_dict['A_cation'] = e
                all_elements.remove(e)

        M_site_elements = all_elements

        for e in M_site_elements:
            if e in M_site_mono_exclusive:
                chemical_dict['M_cation_mono'] = e
                M_site_elements.remove(e)
                if len(M_site_elements) == 1:
                    chemical_dict['M_cation_tri'] = M_site_elements[-1]
            elif e in M_site_tri_exclusive:
                chemical_dict['M_cation_tri'] = e
                M_site_elements.remove(e)
                if len(M_site_elements) == 1:
                    chemical_dict['M_cation_mono'] = M_site_elements[-1]
        if len(M_site_elements) == 2:
            if all([m in M_site_variable for m in M_site_elements]):
                # cannot really tell which one in which charge state, randomly assign one
                print('variable valence, randomly assigned')
                if 'Pd' not in M_site_elements:
                    chemical_dict['M_cation_mono'] = M_site_elements[0]
                    chemical_dict['M_cation_tri'] = M_site_elements[1]
                else:
                    chemical_dict['M_cation_tri'] = 'Pd'
                    M_site_elements.remove('Pd')
                    chemical_dict['M_cation_mono'] = M_site_elements[-1]
                M_site_elements = []
        assert (None not in [chemical_dict[k] for k in chemical_dict.keys()])

    return chemical_dict


def geometric_fingerprint(crystal: Crystal):
    chemistry = chemical_classifier(crystal)
    r_a = shannon_radii[chemistry['A_cation']]["1"]["VI"]['r_ionic']
    r_m = shannon_radii[chemistry['M_cation_mono']]["1"]["VI"]['r_ionic']
    r_mp = shannon_radii[chemistry['M_cation_tri']]["3"]["VI"]['r_ionic']
    r_x = shannon_radii[chemistry['X_anion']]["-1"]["VI"]['r_ionic']
    print(chemistry['X_anion'],r_x)
    return chemistry, octahedral_factor(r_m, r_mp, r_x), octahedral_mismatch(r_m, r_mp,r_x), generalised_tolerance_factor(r_a,r_m,r_mp,r_x)


def octahedral_factor(r_m, r_mp, r_x):
    return (r_m + r_mp) / (2.0 * r_x)


def octahedral_mismatch(r_m, r_mp, r_x):
    return abs(r_m - r_mp) / (2.0 * r_x)


def generalised_tolerance_factor(r_a, r_m, r_mp, r_x):
    nominator = r_a + r_x
    denominator = (r_m + r_mp) / 2.0 + r_x
    denominator = denominator ** 2 + (r_m - r_mp) ** 2 / 4
    denominator = math.sqrt(denominator) * math.sqrt(2)
    return nominator / denominator


def formation_energy_landscape(db, uids, switch='A-site'):
    data_dict = {'formation_energies': [], 'octahedral_factors': [], 'octahedral_mismatch': [], 'tolerance_factors': [], 'A_site_cation':[]}
    all_data_dict = {x: data_dict for x in X_site}

    if switch in  ['A-site']:
        from perovskite_screenings.analysis import halide_C,halide_B,halide_A,tolerance_factor
        from perovskite_screenings.analysis import octahedral_factor as pv_octahedral_factor

        pv_tolerance_f=[]
        pv_octahedral_f=[]
        for c in halide_C:
            for a in halide_A:
                for b in halide_B:
                    tolerance_f = tolerance_factor(a, b, c, type='goldschmidt')
                    octahedral_f = pv_octahedral_factor(b, c)
                    pv_tolerance_f.append(tolerance_f)
                    pv_octahedral_f.append(octahedral_f)

        plt.scatter(pv_octahedral_f,pv_tolerance_f,alpha=0.3,marker='+',s=20,label='ABX$_{3}$')

    min_energy=100000
    max_energy=-100000
    min_oct_mismatch = 100000
    max_oct_mismatch = -100000
    colors = []
    for uid in uids:
        row = None
        formation_energy = None
        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            pass
        if row is not None:
            atoms = row.toatoms()
            crystal = map_ase_atoms_to_crystal(atoms)

            try:
                formation_energy = row.key_value_pairs['formation_energy']
                print('system ' + uid + ' Formation Energy ' + str(formation_energy) + ' eV/atom')
            except KeyError:
                pass

        if formation_energy is not None:
            chemistry, octahedral_factor, octahedral_mismatch, generalised_tolerance_factor = geometric_fingerprint(
                crystal)
            print(octahedral_factor, octahedral_mismatch, generalised_tolerance_factor)
            all_data_dict[chemistry['X_anion']]['formation_energies'].append(formation_energy)
            all_data_dict[chemistry['X_anion']]['octahedral_factors'].append(octahedral_factor)
            if octahedral_factor>=octahedral_mismatch+1-math.sqrt(2):
                __octahedral_mismatch = octahedral_mismatch
            else:
                __octahedral_mismatch = -1

            all_data_dict[chemistry['X_anion']]['octahedral_mismatch'].append(__octahedral_mismatch)
            all_data_dict[chemistry['X_anion']]['tolerance_factors'].append(generalised_tolerance_factor)
            all_data_dict[chemistry['X_anion']]["A_site_cation"].append(chemistry['A_cation'])

            if formation_energy<min_energy:
                min_energy = formation_energy
            if formation_energy>max_energy:
                max_energy = formation_energy
            if octahedral_mismatch<min_oct_mismatch:
                min_oct_mismatch=octahedral_mismatch
            if octahedral_mismatch>max_oct_mismatch:
                max_oct_mismatch=octahedral_mismatch

    for i, x in enumerate(X_site):
        if i == 0:
            marker = '^'
        if i == 1:
            marker = 's'
        if i == 2:
            marker = 'd'
        if i == 3:
            marker = 'p'
        if switch == 'formation_energy':
            plt.scatter(all_data_dict[x]['octahedral_factors'], all_data_dict[x]['tolerance_factors'], marker=marker,
                        norm=mpl.colors.Normalize(vmin=min_energy * 1.1, vmax=max_energy * 1.1),
                        c=all_data_dict[x]['formation_energies'], edgecolor=None, alpha=0.45, s=25,
                        cmap=plt.get_cmap('RdYlGn'), label='X=' + x)
        elif switch == 'octahedral_mismatch':
            plt.scatter(all_data_dict[x]['octahedral_factors'], all_data_dict[x]['tolerance_factors'], marker=marker,
                        norm=mpl.colors.Normalize(vmin=min_oct_mismatch * 1.1, vmax=max_oct_mismatch * 1.1),
                        c=all_data_dict[x]['octahedral_mismatch'], edgecolor=None, alpha=0.45, s=25,
                        cmap=plt.get_cmap('RdYlGn'), label='X=' + x)
        elif switch == 'A-site':
            colors = []
            for a in all_data_dict[x]['A_site_cation']:
                if a=='Li': colors.append('#344d90')
                elif a=='Na': colors.append('#5cc5ef')
                elif a=='K': colors.append("#ffb745")
                elif a=='Rb': colors.append("#ffbebd")
                elif a=='Cs': colors.append("#CB0000")
            plt.scatter(all_data_dict[x]['octahedral_factors'], all_data_dict[x]['tolerance_factors'], marker='s',
                        c=colors, edgecolor=None, alpha=0.25, s=25)

    def f1(x): return  (x+1)-x #stretch limit
    def f2(x): return  (0.44*x+1.37)/(math.sqrt(2)*(x+1))
    def f3(x): return  (0.73*x+1.13) / (math.sqrt(2) * (x + 1))
    def f4(x): return 2.46/np.sqrt(2*(x+1)**2)

    t = np.arange(0.1, 1.3, 0.05)

    y1=f1(np.arange(math.sqrt(2)-1, 0.77, 0.01))
    y2=f2(np.arange(math.sqrt(2)-1, 0.8, 0.01))
    y3=f3(np.arange(0.8, 1.14, 0.01))
    y4=f4(np.arange(0.73, 1.14, 0.01))
    plt.plot(np.arange(math.sqrt(2)-1, 0.77, 0.01), y1, 'k--')
    plt.plot(np.arange(math.sqrt(2)-1, 0.8, 0.01), y2, 'k--')
    plt.plot(np.arange(0.8, 1.14, 0.01), y3, 'k--')
    plt.plot(np.arange(0.73 , 1.14, 0.01), y4, 'k--')
    plt.vlines(x=math.sqrt(2)-1,ymin=0.78,ymax=1,color='k',linestyles='--')
    plt.vlines(x=1.14, ymin=0.65, ymax=0.83,color='k',linestyles='--')

    plt.xlabel('Octahedral factors $(\\bar{\\mu})$')
    plt.ylabel('Tolerance factors $(t)$')

    if switch == 'formation_energy':
        plt.legend()
        plt.colorbar(label='$E_{f}$ (eV/atom)')
    elif switch == 'octahedral_mismatch':
        plt.legend()
        cbar=plt.colorbar(label='Octahedral mismatch $(\\Delta\\mu)$',extend='min')

    elif switch == 'A-site':
        legend_elements = [Patch(facecolor='#344d90', edgecolor='k', label='A=Li'),
                           Patch(facecolor='#5cc5ef', edgecolor='k', label='A=Na'),
                           Patch(facecolor="#ffb745", edgecolor='k', label='A=K'),
                           Patch(facecolor="#ffbebd", edgecolor='k', label='A=Rb'),
                           Patch(facecolor="#CB0000", edgecolor='k', label='A=Cs'),
                           Line2D([0], [0], marker='+', color='w', label='ABX$_3$',markersize=5,markeredgecolor='b',alpha=0.4)
                           ]
        plt.legend(handles=legend_elements, loc=1, fontsize=12, ncol=1)
    plt.tight_layout()
    if switch == 'A-site':
        name = "formation_energy_landscape_dpv_A.pdf"
    elif switch == 'formation_energy':
        name = "formation_energy_landscape_dpv.pdf"
    elif switch ==  'octahedral_mismatch':
        name = "formation_energy_landscape_dpv_delta_mu.pdf"
    plt.savefig(name)

def sigma_landscape(db,uids,x='formation_energies'):
    formation_energy_dict={'F':[],'Cl':[],'Br':[],'I':[]}
    sigma_dict={'F':[],'Cl':[],'Br':[],'I':[]}
    color_dict = {'F': '#061283', 'Cl': '#FD3C3C', 'Br': '#FFB74C', 'I': '#138D90'}

    for uid in uids:
        row = None
        formation_energy = None
        sigma = None
        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            pass
        if row is not None:
            atoms = row.toatoms()
            crystal = map_ase_atoms_to_crystal(atoms)
            chemistry = chemical_classifier(crystal)

            try:
                formation_energy = row.key_value_pairs['formation_energy']
            except KeyError:
                pass

            try:
                sigma = row.key_value_pairs['sigma_300K_single']
            except KeyError:
                pass

            print('system ' + uid + ' Formation Energy ' + str(formation_energy) + ' eV/atom; Sigma '+str(sigma))
            if (formation_energy is not None) and (sigma is not None) and (str(sigma)!='nan'):
                X = chemistry['X_anion']
                formation_energy_dict[X].append(formation_energy)
                sigma_dict[X].append(sigma)

    if x=='formation_energies':
        for k in formation_energy_dict.keys():
            plt.scatter(formation_energy_dict[k],sigma_dict[k],alpha=0.6,marker='o',s=25,edgecolor=None,c=color_dict[k])

        legend_elements = [Patch(facecolor=color_dict['F'], edgecolor='k', label='X=F'),
                           Patch(facecolor=color_dict['Cl'], edgecolor='k', label='X=Cl'),
                           Patch(facecolor=color_dict['Br'], edgecolor='k', label='X=Br'),
                           Patch(facecolor=color_dict['I'], edgecolor='k', label='X=I')]
        plt.legend(handles=legend_elements, loc=1, fontsize=12, ncol=1)
        plt.axhline(y=1, color='k', linestyle='--')
        plt.ylim([0,2])
        plt.xlabel('$\\Delta E_{f}$ (eV/atom)')
        plt.ylabel('$\\sigma^{(2)}$ (300 K)')
        plt.tight_layout()
        plt.savefig('sigma_Ef_landscape.pdf')

if __name__ == "__main__":
    dbname = os.path.join(os.getcwd(), 'double_halide_pv.db')

    # ====================================================================
    # this is a hack to get all the uids from the database
    all_uids = []
    _db = sqlite3.connect(dbname)
    cur = _db.cursor()
    cur.execute("SELECT * FROM systems")
    rows = cur.fetchall()

    for row in rows:
        for i in row:
            if 'uid' in str(i):
                this_dict = json.loads(str(i))
                this_uid = this_dict['uid']
                if 'dpv' in this_uid:
                    all_uids.append(this_uid)
    # ====================================================================

    # use the ASE db interface
    db = connect(dbname)

    """
    for uid in all_uids:
        row = None
        sigma = None
        try:
            row = db.get(selection=[('uid', '=', uid)])
        except:
            pass
        if row is not None:
            try:
                sigma = row.key_value_pairs['sigma_300K_single']
                print('system ' + uid + ' sigma ' + str(sigma))
            except KeyError:
                pass
    """

    #formation_energy_landscape(db, all_uids, switch='A-site')
    sigma_landscape(db, all_uids)
