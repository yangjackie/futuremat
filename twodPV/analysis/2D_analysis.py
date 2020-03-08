import argparse
import os
from ase.db import connect
from matplotlib import rc

rc('text', usetex=True)

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from twodPV.bulk_library import A_site_list, B_site_list, C_site_list

params = {'legend.fontsize': '8',
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

termination_types = {'100': ['AO', 'BO2'], '110': ['O2', 'ABO'], '111': ['AO3', 'B']}


def plot_thickness_dependent_formation_energies(db, orientation='100', output=None):
    """
    Plot the thickness dependent formation energies of two-dimensional perovskite slabs in a specific orientation.
    Sructures will be categorized according to the chemistries of A/B/C sites (columns) and surface terminations
    (rows)
    """
    figs, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(ncols=4, nrows=2, figsize=(20, 12))
    thicknesses = [3, 5, 7, 9]

    for i in range(len(A_site_list)):

        ylabel = None
        if i == 0:
            axs = [ax11, ax21]
            color = '#7AC7A9'
            title = '$A^{I}B^{II}_{M}X_{3}$'
            ylabel = '$\\min[E_{f}^{2D,n}-E_{f}^{Pm\\bar{3}m}]$ (eV/atom)'
        if i == 1:
            axs = [ax12, ax22]
            color = '#90CA57'
            title = '$A^{I}B^{II}_{TM}X_{3}$'
        if i == 2:
            axs = [ax13, ax23]
            color = '#F1D628'
            title = '$A^{II}B^{IV}C_{3}$'
        if i == 3:
            axs = [ax14, ax24]
            color = '#2B8283'
            title = '$A^{I}B^{X}C_{3}$'

        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    # Energy of the bulk material in the perovskite structure
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    row = db.get(selection=[('uid', '=', uid)])
                    pm3m_formation_e = row.key_value_pairs['formation_energy']

                    two_d_en_diff = [[] for _ in range(len(termination_types[orientation]))]
                    for term_type_id, term_type in enumerate(termination_types[orientation]):

                        for t in thicknesses:
                            uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(t)
                            try:
                                row = db.get(selection=[('uid', '=', uid)])
                                twod_formation_e = row.key_value_pairs['formation_energy']
                                two_d_en_diff[term_type_id].append(twod_formation_e - pm3m_formation_e)
                            except KeyError:
                                print('WARNING: No value for ' + uid + ' Skip!')
                                continue

                        axs[term_type_id].plot(thicknesses, two_d_en_diff[term_type_id], 'o:', c=color, alpha=0.9)
                        axs[term_type_id].plot([3, 9], [0, 0], 'k--')

                        if ylabel is not None:
                            axs[term_type_id].set_ylabel(ylabel)

                    axs[0].set_title(title)
                    axs[1].set_xlabel('Layers')

    if orientation == '111':
        ax21.set_ylim([-1, 2])
        ax22.set_ylim([-2, 2.5])
        ax23.set_ylim([-0.5, 2])

    if output is None:
        output = 'two_d_' + str(orientation) + '_' + 'thickness_dependent.png'
    plt.tight_layout()
    plt.savefig(output)
    plt.show()


def plot_energy_versus_bulk(db, orientation='100', thick=3, output=None):
    """
    Correlations between the formation energies (Ef) for bulk cubic perovskites (given by the minimum energy difference
    with respect to the fully relaxed bulk structure of the lowest Ef) and Ef for mono-unit-cell-thick (by default),
    (given with respect to Ef for bulk cubic perovskites). The data is separately shown for each one of the four types
    of perovskites investigated here.

    For each orientation, there are two different termination type, which will all be plotted.
    """
    figs, ((ax11, ax12, ax13, ax14), (ax21, ax22, ax23, ax24)) = plt.subplots(ncols=4, nrows=2, figsize=(20, 12))

    for i in range(len(A_site_list)):

        bulk_e_diff = [[] for _ in range(len(termination_types[orientation]))]
        twod_e_diff = [[] for _ in range(len(termination_types[orientation]))]
        names = [[] for _ in range(len(termination_types[orientation]))]

        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:

                    # Energy of the bulk material in the perovskite structure
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    row = db.get(selection=[('uid', '=', uid)])
                    pm3m_formation_e = row.key_value_pairs['formation_energy']

                    # Find the lowest energy amongst all the relaxed random structures
                    lowest_relaxed = 100000
                    for k in range(10):
                        uid = system_name + '3_random_str_' + str(k + 1)
                        try:
                            row = db.get(selection=[('uid', '=', uid)])
                            randomised_formation_e = row.key_value_pairs['formation_energy']
                            if (randomised_formation_e < lowest_relaxed) and (
                                    (pm3m_formation_e - randomised_formation_e) > -1.0):
                                lowest_relaxed = randomised_formation_e
                        except KeyError:
                            continue

                    # Find the formation energy of the corresponding two-D structures
                    for term_type_id, term_type in enumerate(termination_types[orientation]):
                        uid = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(thick)
                        try:
                            row = db.get(selection=[('uid', '=', uid)])
                            twod_formation_e = row.key_value_pairs['formation_energy']

                            if twod_formation_e - pm3m_formation_e <= 10.0:
                                bulk_e_diff[term_type_id].append(pm3m_formation_e - lowest_relaxed)
                                twod_e_diff[term_type_id].append(twod_formation_e - pm3m_formation_e)
                                names[term_type_id].append(system_name + '$_{3}$')
                        except KeyError:
                            print('WARNING: No value for ' + uid + ' Skip!')
                            continue
        ylabel = None
        if i == 0:
            axs = [ax11, ax21]
            color = '#7AC7A9'
            title = '$A^{I}B^{II}_{M}X_{3}$'
            ylabel = '$\\min[E_{f}^{Pm\\bar{3}m}-E_{f}^{\mbox{\\Large{full relax}}}]$ (eV/atom)'
        if i == 1:
            axs = [ax12, ax22]
            color = '#90CA57'
            title = '$A^{I}B^{II}_{TM}X_{3}$'
        if i == 2:
            axs = [ax13, ax23]
            color = '#F1D628'
            title = '$A^{II}B^{IV}C_{3}$'
        if i == 3:
            axs = [ax14, ax24]
            color = '#2B8283'
            title = '$A^{I}B^{X}C_{3}$'

        for _p in range(len(axs)):
            _x = twod_e_diff[_p]
            _y = bulk_e_diff[_p]
            axs[_p].plot(_x, _y, 'o', c=color, alpha=0.9, ms=15)
            for u, _ in enumerate(_x):
                x = _x[u]
                y = _y[u]
                if (y > 0.2) or (y < -0.2):
                    axs[_p].text(x + 0.01, y + 0.01, names[_p][u], fontsize=14, color='#688291')

            # if max(_x)>10:
            #    axs[_p].set_xlim([min(_x) - 0.02, 3])
            # else:
            axs[_p].set_xlim([min(_x) - 0.02, max(_x) + 0.12])
            axs[_p].set_ylim([min(_y) - 0.02, max(_y) + 0.05])

            if ylabel is not None:
                axs[_p].set_ylabel(ylabel)
            axs[_p].annotate(str(termination_types[orientation][_p]) + '-termination', xy=(1, 0.9),
                             xycoords='axes fraction',
                             fontsize=16, horizontalalignment='right', verticalalignment='bottom')
        axs[0].set_title(title)
        axs[1].set_xlabel("$E_f^{2D,n=" + str(thick) + "}-E_{f}^{Pm\\bar{3}m}$ (eV/atom)")

    if output is None:
        output = 'two_d_' + str(orientation) + '_' + str(thick) + '_compare_bulk.png'
    plt.tight_layout()
    plt.savefig(output)
    plt.show()


def plot_super_cell_dependent_formation_energies(db, orientation='100', thick=3, output=None):
    figs, ((ax11), (ax21)) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))

    for i in range(len(A_site_list)):

        small_cell_e_diff = [[] for _ in range(len(termination_types[orientation]))]
        large_cell_e_diff = [[] for _ in range(len(termination_types[orientation]))]

        if i == 0:
            color = '#7AC7A9'
        if i == 1:
            color = '#90CA57'
        if i == 2:
            color = '#F1D628'
        if i == 3:
            color = '#2B8283'

        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    # Energy of the bulk material in the perovskite structure
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'
                    row = db.get(selection=[('uid', '=', uid)])
                    pm3m_formation_e = row.key_value_pairs['formation_energy']

                    # Find the formation energy of the corresponding two-D structures
                    for term_type_id, term_type in enumerate(termination_types[orientation]):
                        uid_small = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(thick)
                        uid_large = system_name + '3_' + str(orientation) + "_" + str(term_type) + "_" + str(
                            thick) + "_large_cell_full_B_octa"
                        twod_formation_e_small = None
                        twod_formation_e_large = None

                        try:
                            row = db.get(selection=[('uid', '=', uid_small)])
                            twod_formation_e_small = row.key_value_pairs['formation_energy']
                        except:
                            pass

                        try:
                            row = db.get(selection=[('uid', '=', uid_large)])
                            twod_formation_e_large = row.key_value_pairs['formation_energy']
                        except:
                            pass

                        if (twod_formation_e_large is not None) and (twod_formation_e_small is not None):
                            _a=twod_formation_e_small - pm3m_formation_e
                            _b=twod_formation_e_large - pm3m_formation_e
                            small_cell_e_diff[term_type_id].append(_a)
                            large_cell_e_diff[term_type_id].append(_b)
                            if abs(_a-_b) > 0.09:
                                from core.models.element import ionic_radii
                                from twodPV.analysis.bulk_energy_landscape import charge_state_A_site,charge_state_B_site,charge_state_C_site
                                import math
                                tolerance_f = ionic_radii[a][charge_state_A_site[i]] + ionic_radii[c][
                                    charge_state_C_site[i]]
                                tolerance_f /= ionic_radii[b][charge_state_B_site[i]] + ionic_radii[c][
                                    charge_state_C_site[i]]
                                tolerance_f /= math.sqrt(2)
                                print(system_name,term_type,abs(_a-_b),tolerance_f)

        ax11.plot(small_cell_e_diff[0], large_cell_e_diff[0], 'o', c=color)
        ax11.plot(small_cell_e_diff[0], small_cell_e_diff[0], 'k-')
        ax11.set_xlabel(termination_types[orientation][0] + " (small cell) $E_{f}^{2D}-E_{f}^{Pm\\bar{2}m}$ (eV)")
        ax11.set_ylabel(termination_types[orientation][0] + " (large cell) $E_{f}^{2D}-E_{f}^{Pm\\bar{2}m}$ (eV)")

        ax21.plot(small_cell_e_diff[1], large_cell_e_diff[1], 'o', c=color)
        ax21.plot(small_cell_e_diff[1], small_cell_e_diff[1], 'k-')
        ax21.set_xlabel(termination_types[orientation][1] + " (small cell) $E_{f}^{2D}-E_{f}^{Pm\\bar{2}m}$ (eV)")
        ax21.set_ylabel(termination_types[orientation][1] + " (large cell) $E_{f}^{2D}-E_{f}^{Pm\\bar{2}m}$ (eV)")
    plt.tight_layout()
    plt.savefig(output)
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Switches for analyzing the energy landscapes of 2D perovskites',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--orient", type=str, default='100',
                        help='Orientations of the two-d perovskite slabs')
    parser.add_argument("--terminations", type=str, default='AO',
                        help='Surface termination type of the two-d slab')
    parser.add_argument("--size_dependent_energies", action='store_true',
                        help="Plot size dependent formation energies.")
    parser.add_argument("--vbulk", action='store_true',
                        help="Plot energy compared to the bulk perovskite")
#    parser.add_argument("--supercell_effect", action='store_true',
#                        help="Compare the formation energies calculated with different supercell size")
    parser.add_argument("--db", type=str, default=os.getcwd() + '/2dpv.db',
                        help="Name of the database that contains the results of the screenings.")
    parser.add_argument("--output", type=str, default=None,
                        help='Output file name for figure generated.')
    args = parser.parse_args()

    if os.path.exists(args.db):
        args.db = connect(args.db)
    else:
        raise Exception("Database " + args.db + " does not exists, cannot proceed!")

    if args.vbulk:
        plot_energy_versus_bulk(args.db, orientation=args.orient, output=args.output)

    if args.size_dependent_energies:
        plot_thickness_dependent_formation_energies(args.db, orientation=args.orient, output=args.output)

#    if args.supercell_effect:
#        plot_super_cell_dependent_formation_energies(args.db, orientation='100', output=args.output)
