import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sqlite3
import json
import math
import pickle

from ase.db import connect
from itertools import permutations
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy import stats

rc('text', usetex=True)

import matplotlib.pylab as pylab

params = {'legend.fontsize': '10',
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 28,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

from core.internal.builders.crystal import map_ase_atoms_to_crystal

reference_atomic_energies = {}


def get_reference_atomic_energies(db, all_keys=None):
    for k in all_keys:
        if ('element' in k):
            row = db.get(selection=[('uid', '=', k)])
            total_energy = row.key_value_pairs['total_energy']
            element = k.split('_')[-1]
            structure = map_ase_atoms_to_crystal(row.toatoms())
            total_energy = total_energy / structure.all_atoms_count_dictionaries()[element]
            reference_atomic_energies[element] = total_energy


def total_energies(db, X='Cl', dim='0D', all_keys=None, keep_ids=False):
    formation_energy_dict = {}
    for k in all_keys:
        if ((X in k) and (dim in k)) or (('pure_Cs3Sb2I9' in k) and (dim in k)):
            row = db.get(selection=[('uid', '=', k)])
            structure = map_ase_atoms_to_crystal(row.toatoms())
            _dict = structure.all_atoms_count_dictionaries()
            total_energy = row.key_value_pairs['total_energy'] / structure.total_num_atoms()
            try:
                halo_comp = _dict[X]
            except KeyError:
                halo_comp = 0.0
            try:
                iodide_comp = _dict['I']
            except KeyError:
                iodide_comp = 0.0
            comp = halo_comp / 2.0
            if keep_ids is not True:
                if comp not in formation_energy_dict:
                    formation_energy_dict[comp] = [total_energy]
                else:
                    formation_energy_dict[comp].append(total_energy)
            else:
                if comp not in formation_energy_dict:
                    formation_energy_dict[comp] = [[total_energy,k]]
                else:
                    formation_energy_dict[comp].append([total_energy,k])
    return formation_energy_dict

def plot_mixing_energy_landscape(db,X='Cl',dim='0D',all_keys=None):
    if dim == '0D':
        c = '#36688D'
    if dim == '2D':
        c = '#F3Cd05'
    total_energy_dict = total_energies(db, X=X, dim=dim, all_keys=all_keys)
    compositions=[]
    mix_energies=[]
    compositions_lowest=[]
    mix_energies_lowest=[]
    for k in list(sorted(total_energy_dict.keys())):
        compositions_lowest.append(k)
        mix_energies_lowest.append(min(total_energy_dict[k])-(k/9.0)*total_energy_dict[9.0][0]-(1-k/9.0)*total_energy_dict[0.0][0])
        for item in total_energy_dict[k]:
            compositions.append(k)
            mix_energies.append(item-(k/9.0)*total_energy_dict[9.0][0]-(1-k/9.0)*total_energy_dict[0.0][0])

    plt.scatter(compositions,mix_energies,marker='o',alpha=0.55,fc=c)
    plt.scatter(compositions_lowest,mix_energies_lowest,marker='o',fc=c)
    plt.plot(compositions_lowest,mix_energies_lowest,'k--')
    plt.xlim([-0.1, 9.1])
    plt.xticks(range(10), ['$' + str(i) + '$' for i in range(10)])
    plt.xlabel('$y$ in Cs$_{3}$Sb'+str(X)+'$_{y}$I$_{9-y}$')
    plt.ylabel('$\\Delta E_{mix}$ (eV/atom)')
    plt.tight_layout()
    plt.savefig('mixing_energies_'+str(X)+'_'+str(dim)+'.pdf')

def mixing_free_energies_with_configurational_entropy(db,X='Cl',dim='0D',all_keys=None,temperature=None):
    mixing_free_energy_dict = {}
    kb = 8.617e-5  # eV/K
    formation_energy_dict = total_energies(db, X=X, dim=dim, all_keys=all_keys)

    comp = 0.0
    free_en_mix_0 = [math.exp(-1.0 * e / (kb * temperature)) for e in formation_energy_dict[comp]]
    free_en_mix_0 = -kb * temperature * math.log(sum(free_en_mix_0))

    comp = 9.0
    free_en_mix_1 = [math.exp(-1.0 * e / (kb * temperature)) for e in formation_energy_dict[comp]]
    free_en_mix_1 = -kb * temperature * math.log(sum(free_en_mix_1))

    for comp in list(sorted(formation_energy_dict.keys())):
        free_en_mix = [math.exp(-1.0 * e / (kb * temperature)) for e in formation_energy_dict[comp]]
        l = len(free_en_mix)
        free_en_mix = -kb * temperature * math.log(sum(free_en_mix))
        free_en_mix = free_en_mix - ((1.0 - comp/9.0) * free_en_mix_0 + (comp/9.0) * free_en_mix_1)
        mixing_free_energy_dict[comp] = free_en_mix
    return mixing_free_energy_dict

def plot_mixing_free_energies_with_configurational_entropy(db, X='Cl',all_keys=None):


    alpha=[0.2,0.4,0.6,0.8,1.0]
    lw=[4,5,6,7,8]
    for counter,temp in enumerate([100,  300, 500,  700, 900]):
        for dim in ['0D','2D']:
            mixing_fe=mixing_free_energies_with_configurational_entropy(db,X,dim,all_keys,temperature=temp)
            compositions = list(sorted(mixing_fe.keys()))
            if dim == '0D':
                c = '#36688D'
            if dim == '2D':
                c = '#F3Cd05'
            if dim == '0D':
                label=str(temp) + ' K'
                if temp==900:
                    label = str(temp) + ' K (0D)'
                plt.plot(compositions, [mixing_fe[k] for k in compositions], 'o-', label=label, c=c,
                         alpha=alpha[counter], lw=lw[counter] / 4.0)
            elif dim =='2D':
                if temp==900:
                    label = str(temp) + ' K (2D)'
                    plt.plot(compositions, [mixing_fe[k] for k in compositions], 'o-', label=label,c=c,alpha=alpha[counter],lw=lw[counter]/4.0)
                else:
                    plt.plot(compositions, [mixing_fe[k] for k in compositions], 'o-', c=c,alpha=alpha[counter],lw=lw[counter]/4.0)

    if X=='Cl':
        plt.legend()
    plt.tight_layout()
    plt.xlim([-0.1, 9.1])
    plt.xticks(range(10), ['$' + str(i) + '$' for i in range(10)])
    plt.xlabel('$y$ in Cs$_{3}$Sb' + str(X) + '$_{y}$I$_{9-y}$')
    plt.ylabel('$\\Delta G_{mix}$ (eV/atom)')
    plt.tight_layout()
    plt.savefig('mixing_free_energies_' + str(X) + '.pdf')

def plot_formation_energy_landscapes(db, all_keys=None):
    get_reference_atomic_energies(db, all_keys=all_keys)

    for halo_count, halo in enumerate(['Cl', 'Br']):
        if halo == 'Cl':
            symbol_dict = {'0D': 's-', '2D': 'o-'}
        elif halo == 'Br':
            symbol_dict = {'0D': 's--', '2D': 'o--'}

        # plt.subplot(1,2,halo_count+1)
        fp = {}
        for dim in ['0D', '2D']:
            formation_energy_dict = {}

            for k in all_keys:
                if ((halo in k) and (dim in k)) or (('pure_Cs3Sb2I9' in k) and (dim in k)):
                    row = db.get(selection=[('uid', '=', k)])
                    formation_energy = row.key_value_pairs['total_energy']
                    structure = map_ase_atoms_to_crystal(row.toatoms())
                    _dict = structure.all_atoms_count_dictionaries()
                    for element_key in _dict.keys():
                        formation_energy = formation_energy - reference_atomic_energies[element_key] * _dict[
                            element_key]
                    formation_energy = formation_energy / structure.total_num_atoms()  # formation energy normalized to per atom
                    try:
                        halo_comp = _dict[halo]
                    except KeyError:
                        halo_comp = 0.0
                    try:
                        iodide_comp = _dict['I']
                    except KeyError:
                        iodide_comp = 0.0

                    comp = halo_comp / 2.0
                    if comp not in formation_energy_dict.keys():
                        formation_energy_dict[comp] = []
                    formation_energy_dict[comp].append(formation_energy)
                    if 'pure' in k: print(k,formation_energy)
                    # print(k,formation_energy)

            for comp_k in formation_energy_dict.keys():
                formation_energy_dict[comp_k] = sum(formation_energy_dict[comp_k]) / len(formation_energy_dict[comp_k])

            compositions = list(sorted(formation_energy_dict.keys()))
            formation_energies = [formation_energy_dict[comp_k] for comp_k in compositions]

            def bowing_curve_fe(x, b):
                return b * x * (9 - x)/81 + x * formation_energies[-1]/9 + (9 - x) * \
                       formation_energies[0]/9

            popt, pcov = curve_fit(bowing_curve_fe, compositions, formation_energies)
            print(dim, halo, *popt, formation_energies[-1], formation_energies[0])
            fp[dim] = [*popt, formation_energies[-1], formation_energies[0]]

            if dim == '0D':
                c = '#36688D'
            if dim == '2D':
                c = '#F3Cd05'

            if halo == 'Br':
                plt.plot(compositions, formation_energies, symbol_dict[dim], markerfacecolor='None',
                         label=dim + ' (X=Br)', c=c, markersize=8)
            else:
                plt.plot(compositions, formation_energies, symbol_dict[dim], label=dim + ' (X=Cl)', c=c, markersize=8)
                #plt.plot(compositions,bowing_curve_fe(np.array(compositions),*popt),'r-')
        def f(xy):
            x, y = xy
            z = np.array(
                [y - fp['0D'][0] * x * (9 - x)/81  - x * fp['0D'][1]/9 - (9 - x) * fp['0D'][2]/9,
                 y - fp['2D'][0] * x * (9 - x)/81  - x * fp['2D'][1]/9 - (9 - x) * fp['2D'][2]/9])
            return z

        from scipy.optimize import fsolve
        intersect=fsolve(f, [10, -15])
        #plt.plot([intersect[0]],[intersect[1]],'kd')

        plt.arrow(intersect[0],intersect[1],0,-(1.43+intersect[1]),head_width=0.2, head_length=0.02, ec='#A4978E', fc='#A4978E')
        plt.text(intersect[0]+0.1,-1.43,"$y=$"+"{:.2f}".format(intersect[0]),rotation=45,fontsize=16)
    plt.xlabel('$y$ in Cs$_{3}$SbX$_{y}$I$_{9-y}$')
    plt.ylabel('$\\Delta H_{f}$ (eV/atom)')
    plt.legend()
    plt.tight_layout()
    plt.xlim([-0.1, 9.1])
    plt.ylim([-1.45,-0.88])
    plt.xticks(range(10), ['$' + str(i) + '$' for i in range(10)])
    plt.savefig('formation_energies.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Switches for analyzing the energy landscapes of doped perovskites',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default=os.getcwd() + '/doping.db',
                        help="Name of the database that contains the results of the screenings.")
    parser.add_argument("--X", type=str,
                        help='halide ion')
    parser.add_argument("--dim",type=str,
                        help='Specify the dimensionality of the halide perovskite, either 0D or 2D')
    parser.add_argument("--formation_energy", action='store_true',
                        help='Plot the formation energy landscapes with different halide mixing ratios')
    parser.add_argument("--mixing_energy", action='store_true',
                        help='Plot the mixing energy landscapes with different halide mixing ratios')
    parser.add_argument("--mixing_free_energy", action='store_true',
                        help='Plot the mixing free energy landscapes with different halide mixing ratios')
    args = parser.parse_args()

    if os.path.exists(args.db):
        db = connect(args.db)
    else:
        raise Exception("Database " + args.db + " does not exists, cannot proceed!")

    # ====================================================================
    # this is a hack to get all the uids from the database
    all_uids = []
    _db = sqlite3.connect(args.db)
    cur = _db.cursor()
    cur.execute("SELECT * FROM systems")
    rows = cur.fetchall()

    for row in rows:
        for i in row:
            if 'uid' in str(i):
                this_dict = json.loads(str(i))
                all_uids.append(this_dict['uid'])
    # ====================================================================

    if args.formation_energy:
        plot_formation_energy_landscapes(db, all_keys=all_uids)
    elif args.mixing_energy:
        plot_mixing_energy_landscape(db,X=args.X, dim=args.dim, all_keys=all_uids)
    elif args.mixing_free_energy:
        plot_mixing_free_energies_with_configurational_entropy(db,X=args.X, all_keys=all_uids)

