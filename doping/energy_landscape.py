import argparse
import os
import sqlite3
import json

from ase.db import connect
from itertools import permutations
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

import matplotlib.pylab as pylab

params = {'legend.fontsize': '8',
          'figure.figsize': (6, 5),
          'axes.labelsize': 18,
          'axes.titlesize': 18,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}
pylab.rcParams.update(params)

from core.internal.builders.crystal import map_ase_atoms_to_crystal


def plot_mixing_energy_for_single_system(db, a=None, b=None, c=None, output=None, all_keys=None):
    # figure out data from the end members
    _mixing_energies = composition_dependent_mixing_energies(a, all_keys, b, c, db)
    mixing_energies = [0,0]
    averaged_energies = [0,0]
    compositions = [0,1]
    av_compositions = [0,1]

    for k in _mixing_energies.keys():
        for e in _mixing_energies[k]:
            compositions.append(k)
            mixing_energies.append(e)
        averaged_energies.append(np.average(_mixing_energies[k]))
        av_compositions.append(k)

    gap = max(mixing_energies) - min(mixing_energies)

    plt.scatter(compositions, mixing_energies, marker='o', facecolor='#EFA747', edgecolor='k', alpha=0.5, s=80)

    averaged_energies = [x for _, x in sorted(zip(av_compositions,averaged_energies))]
    x=list(sorted(av_compositions))
    popt, pcov = curve_fit(bowing_curve, x, averaged_energies)
    #plt.plot(x, bowing_curve(np.array(x), *popt), '#F22F08', label='$b=%5.3f$' % tuple(popt))

    x_label = '$x$ in '
    if len(a)==1:
        x_label += fix_string(a[0])
    else:
        raise NotImplementedError()
    if len(b)==1:
        x_label += fix_string(b[0])
    else:
        mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in b]
        x_label += '('
        x_label += str(mixed_site[0])+'$_{x}$'
        x_label += str(mixed_site[1])+'$_{1-x}$'
        x_label += ')'
    if len(c)==1:
        x_label += fix_string(c[0])
    else:
        raise NotImplementedError()

    plt.xlabel(x_label)
    plt.ylabel('Mixing enthalpy $\Delta H_{mix}(x)$ (eV/atom)')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.ylim([min(mixing_energies) - 0.08 * gap, max(mixing_energies) + 0.08 * gap])
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output)

def bowing_curve(x, b):
    return b*x*(x-1)

def fix_string(s):
    number = None
    for c in s:
        if c.isdigit():
            number = c
    if number is not None:
        print(number)
        replaced_s = s.replace(number, '$_{'+str(number)+'}$')
        return replaced_s
    else:
        return s

def composition_dependent_mixing_energies(a, all_keys, b, c, db):
    end_members = [_a + _b + _c for _a in a for _b in b for _c in c]
    print(end_members)
    assert (len(end_members) == 2)
    mixed_site = [site for site in [a, b, c] if len(site) == 2][-1]
    mixed_site = [''.join([_i for _i in k if not _i.isdigit()]) for k in mixed_site]
    end_member_total_energies = {k: 0 for k in mixed_site}
    # get the total energies of the two end members
    for m in mixed_site:
        for em in end_members:
            if m in em:
                matched_key = [k for k in all_keys if em in k][-1]
                row = db.get(selection=[('uid', '=', matched_key)])
                total_energy = row.key_value_pairs['total_energy']
                end_member_total_energies[m] = total_energy
                print(m,total_energy)
    # figure out which site has been mixed with two chemical elements, then we can decide
    #   the chemical compositions should be measured against which element

    mixing_energies = {}

    system_content = a + b + c
    for k in all_keys:
        k_contains_all_elements = all([(content in k) for content in system_content])
        if k_contains_all_elements:
            row = db.get(selection=[('uid', '=', k)])
            total_energy = row.key_value_pairs['total_energy']

            structure = map_ase_atoms_to_crystal(row.toatoms())
            element_1 = ''.join([_i for _i in mixed_site[0] if not _i.isdigit()])
            element_2 = ''.join([_i for _i in mixed_site[1] if not _i.isdigit()])
            composition = structure.all_atoms_count_dictionaries()[element_1] / (
                    structure.all_atoms_count_dictionaries()[element_1] + structure.all_atoms_count_dictionaries()[
                element_2])
            print(composition, k, total_energy)
            mixing_energy = total_energy - composition * end_member_total_energies[element_1] - (1.0 - composition) * \
                            end_member_total_energies[element_2]
            mixing_energy = mixing_energy / structure.total_num_atoms()

            if mixing_energy<0: print(k)

            if composition not in mixing_energies.keys():
                mixing_energies[composition] = []

            mixing_energies[composition].append(mixing_energy)
    return mixing_energies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Switches for analyzing the energy landscapes of doped perovskites',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--system", type=str, help="Name of the system to analyze")

    # Probably more flexible and easier to handle if the chemistries of A, B and C sites are explicitly given?
    parser.add_argument('-a', '--a_site', nargs='+',
                        help='A site, in AxByCz, attach a number to specify stoichiometry if >1 , such as Cr2 ')
    parser.add_argument('-b', '--b_site', nargs='+',
                        help='B site, in AxByCz, attach a number to specify stoichiometry if >1')
    parser.add_argument('-c', '--c_site', nargs='+',
                        help='C site, in AxByCz, attach a number to specify stoichiometry if >1, such as Cl3')

    parser.add_argument("--db", type=str, default=os.getcwd() + '/doping.db',
                        help="Name of the database that contains the results of the screenings.")
    parser.add_argument("--output", type=str, help='Name of the output file')

    parser.add_argument("--mixing_energy", action='store_true',
                        help='Plot the mixing enthalpy with respect to the two end members')
    parser.add_argument("--summary", action='store_true', help="Plot a summary for multiple systems")
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

    if args.mixing_energy:
        if not args.summary:
            if (len(args.a_site), len(args.b_site), len(args.c_site)) not in list(set(permutations([1, 1, 2]))):
                raise Exception("Must and can only have one of the site with two different chemical elements!")
            plot_mixing_energy_for_single_system(db, args.a_site, args.b_site, args.c_site, args.output,
                                                 all_keys=all_uids)
