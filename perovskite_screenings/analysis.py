from ase.db import connect
import os
import math
import argparse

from numpy import dot

from core.models.element import shannon_radii
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from ase.geometry import *
from ase.neighborlist import *
rc('text', usetex=True)

import matplotlib.pylab as pylab

params = {'legend.fontsize': '14',
          'figure.figsize': (6, 5),
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

halide_C = ['F', 'Cl', 'Br', 'I']
halide_A = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Cu', 'Ag', 'Au', 'Hg', 'Ga', 'In', 'Tl']
halide_B = ['Mg', 'Ca', 'Sr', 'Ba', 'Se', 'Te', 'As', 'Si', 'Ge', 'Sn', 'Pb', 'Ga', 'In', 'Sc', 'Y', 'Ti', 'Zr', 'Hf',
            'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt',
            'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg']

chalco_C = ['O', 'S', 'Se']
chalco_A = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Pd', 'Pt', 'Cu', 'Ag', 'Zn',
            'Cd', 'Hg', 'Ge', 'Sn', 'Pb']
chalco_B = ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir',
            'Ni', 'Pd', 'Pt', 'Sn', 'Ge', 'Pb', 'Si', 'Te', 'Po']

A_site_list = [halide_A, chalco_A]
B_site_list = [halide_B, chalco_B]
C_site_list = [halide_C, chalco_C]

def tolerance_factor(a,b,c,type='goldschmidt'):
    if type == 'goldschmidt':
        return goldschmidt_tolerance_factor(a,b,c)
    if type == 'bartel':
        return bartel_tolerance_factor(a,b,c)

def bartel_tolerance_factor(a,b,c):
    if c in halide_C:
        _a_charge = '1'
        _b_charge = '2'
        c_charge = '-1'
    elif c in chalco_C:
        _a_charge = '2'
        _b_charge = '4'
        c_charge = '-2'
    coord='VI'
    _ra = shannon_radii[a][_a_charge][coord]['r_ionic']
    _rb = shannon_radii[b][_b_charge][coord]['r_ionic']
    rc = shannon_radii[c][c_charge][coord]['r_ionic']

    if _ra > _rb:
        ra = _ra
        rb = _rb
        a_charge = _a_charge
        b_charge = _b_charge
    else:
        ra = _rb
        rb = _ra
        a_charge = _b_charge
        b_charge = _a_charge

    a_charge = int(a_charge)
    if ra/rb != 1:
        return rc/rb - a_charge*(a_charge-(ra/rb)/math.log(ra/rb))
    else:
        return np.NaN

def goldschmidt_tolerance_factor(a,b,c):
    if c in halide_C:
        a_charge = '1'
        b_charge = '2'
        c_charge = '-1'
    elif c in chalco_C:
        a_charge = '2'
        b_charge = '4'
        c_charge = '-2'
    coord='VI' # all ions are six fold coordinated

    try:
        ra = shannon_radii[a][a_charge][coord]['r_ionic']
    except KeyError as e:
        print(a,e)
    try:
        rb = shannon_radii[b][b_charge][coord]['r_ionic']
    except KeyError as e:
        print(b,e)
    try:
        rc = shannon_radii[c][c_charge][coord]['r_ionic']
    except KeyError as e:
        print(c,e)

    tolerance_f = ra+rc
    tolerance_f /= rb+rc
    tolerance_f /= math.sqrt(2)

    return tolerance_f

def formation_energy_grid(db,C='O',random=False, full_relax=True):
    if C in halide_C:
        A=halide_A
        B=halide_B
    if C in chalco_C:
        A=chalco_A
        B=chalco_B

    grid = np.zeros((len(B),len(A)))

    for b_count,b in enumerate(B):
        for a_count,a in enumerate(A):
            grid[b_count][a_count]=100
            system_name = a + b + C
            uid = system_name + '_Pm3m'
            row = None
            pm3m_formation_e = None

            try:
                row = db.get(selection=[('uid', '=', uid)])
            except:
                continue

            if row is not None:
                try:
                    pm3m_formation_e = row.key_value_pairs['formation_energy']
                except KeyError:
                    continue

            delta_E=None
            print(uid,'Pm3m',pm3m_formation_e)
            if pm3m_formation_e is not None:
                if random:
                    randomised_formation_energies = []
                    for counter in range(10):
                        uid_r = uid + '_rand_str_' + str(counter)
                        fe = None
                        row = None
                        try:
                            row = db.get(selection=[('uid', '=', uid_r)])
                        except:
                            pass
                        if row is not None:
                            fe = row.key_value_pairs['formation_energy']
                            # print(uid_r,fe)

                        if fe is not None:
                            randomised_formation_energies.append(fe)
                    if randomised_formation_energies != []:
                        delta_E= pm3m_formation_e - min(randomised_formation_energies)
                        grid[b_count][a_count]=delta_E
                        print(uid,delta_E)
                if full_relax:
                    uid_f = uid + '_fullrelax'
                    fe = None
                    row = None
                    try:
                        row = db.get(selection=[('uid', '=', uid_f)])
                    except:
                        pass
                    if row is not None:
                        fe = row.key_value_pairs['formation_energy']
                        # print(uid_r,fe)
                    if fe is not None:
                        delta_E = pm3m_formation_e - fe
                        grid[b_count][a_count] = delta_E
                        print(uid, delta_E)

    rows, cols = grid.shape
    grid = np.ma.masked_where(grid==100,grid)

    cmap = plt.cm.tab20c
    cmap.set_bad(color='white')

    plt.imshow(grid,cmap=cmap)

    if C in halide_C:
        if random:
            plt.clim([-0.25,1.25])
        if full_relax:
            plt.clim([-0.15,0.1])
    if C in chalco_C:
        if random:
            plt.clim([-0.2,2.0])
        if full_relax:
            plt.clim([-0.1,0.07])

    cb=plt.colorbar(shrink=0.7)
    if random:
        cb.set_label(label='$E_{f}^{Pm\\bar{3}m}-\\min\\{E_{f}^{\mbox{\\small{full relax}}}\\}$ (eV/atom)', size=14)
    if full_relax:
        cb.set_label(label='$E_{f}^{Pm\\bar{3}m}-E_{f}^{\mbox{\\small{full relax}}}$ (eV/atom)', size=14)
    cb.ax.tick_params(labelsize=10)

    plt.xticks(range(len(A)),A)
    plt.yticks(range(len(B)),B)

    plt.tick_params(labelsize=6)
    plt.tight_layout()
    if full_relax:
        plt.savefig(C + "_energy_grid_fullrelax.pdf")
    if random:
        plt.savefig(C + "_energy_grid.pdf")

def formation_energy_structural_deformation_analysis(db, systems='chalcogenides'):
    if systems not in ['halides','chalcogenides']:
        raise Exception("Wrong system specification, must be either halides or chalcogenides.")

    if systems=='halides':
        A = halide_A
        B = halide_B
        C = halide_C
    elif systems=='chalcogenides': #including oxides
        A = chalco_A
        B = chalco_B
        C = chalco_C

    energy_differences = [[] for _ in C]
    bond_deformations = [[] for _ in C]
    tolerance_factors = [[] for _ in C]
    color_dict = {0: '#A3586D', 1: '#5C4A72', 2: '#F3B05A', 3: '#F4874B'}

    for c_counter, c in enumerate(C):
        for a in A:
            for b in B:

                tolerance_f = tolerance_factor(a, b, c, type='goldschmidt')

                if tolerance_f is np.NaN:
                    continue



                system_name = a + b + c
                uid = system_name + '_Pm3m'
                print(uid)
                row = None
                pm3m_formation_e = None


                try:
                    row = db.get(selection=[('uid', '=', uid)])
                except:
                    continue

                if row is not None:
                    try:
                        pm3m_formation_e = row.key_value_pairs['formation_energy']
                    except KeyError:
                        continue

                atoms = row.toatoms()
                B_site_position = []
                C_site_position = []
                for _a in atoms:
                    if _a.symbol == b:
                        B_site_position.append(_a.position)
                    if _a.symbol == c:
                        C_site_position.append(_a.position)
                dist = get_distances(B_site_position, C_site_position, cell=atoms.cell, pbc=True)
                pm3m_BX_dist=dist[-1][0][0] #BX bond length in an ideal perovskite structure

                uid_f = uid + '_fullrelax'
                fe = None
                row = None
                try:
                    row = db.get(selection=[('uid', '=', uid_f)])
                except:
                    continue

                if row is not None:
                    fullrelax_fe = row.key_value_pairs['formation_energy']
                    fullrelax_atoms = row.toatoms()

                    energy_diff = pm3m_formation_e - fullrelax_fe
                    energy_differences[c_counter].append(energy_diff)

                    nl = NeighborList([pm3m_BX_dist/2 for _ in range(len(fullrelax_atoms))],self_interaction=False)
                    nl.update(fullrelax_atoms)

                    b_counter = 0
                    bond_distortion = 0

                    for counter, _atom in enumerate(fullrelax_atoms):

                        if _atom.symbol == b:
                            nl_indicies, nl_offsets = nl.get_neighbors(counter)
                            b_position = fullrelax_atoms.positions[counter]
                            this_bond_distorion = 0
                            this_bond_distorion_c = 0
                            #print(b, len(nl_indicies),pm3m_BX_dist)
                            for nl_i, nl_offset in zip(nl_indicies, nl_offsets):
                                if fullrelax_atoms.symbols[nl_i] == c:
                                    c_position=fullrelax_atoms.positions[nl_i] + dot(nl_offset, fullrelax_atoms.get_cell())
                                    #print(b, fullrelax_atoms.symbols[nl_i] ,np.linalg.norm(b_position-c_position))
                                    this_bx_bond_length = np.linalg.norm(b_position-c_position)
                                    this_bond_distorion += (this_bx_bond_length/pm3m_BX_dist)**2
                                    this_bond_distorion_c += 1
                            bond_distortion += this_bond_distorion/this_bond_distorion_c
                            b_counter += 1
                    bond_distortion = bond_distortion/b_counter
                    bond_deformations[c_counter].append(bond_distortion)
                    tolerance_factors[c_counter].append(tolerance_f)

    for c_counter in range(len(C)):
        plt.scatter(bond_deformations[c_counter],tolerance_factors[c_counter],marker='o',c=color_dict[c_counter],edgecolor=None,alpha=0.45,s=25)

    #plt.xlim([0.94,1.052])
    #plt.ylim([-0.11,0.06])
    plt.tight_layout()
    plt.savefig(systems+'_energy_structural_def.pdf')

def formation_energy_landscapes(db, systems='chalcogenides', random=True, full_relax=False, x='tolerance_factor'):

    color_dict = {0: '#A3586D', 1: '#5C4A72', 2: '#F3B05A', 3: '#F4874B'}

    if systems not in ['halides','chalcogenides']:
        raise Exception("Wrong system specification, must be either halides or chalcogenides.")

    if systems=='halides':
        A = halide_A
        B = halide_B
        C = halide_C
    elif systems=='chalcogenides': #including oxides
        A = chalco_A
        B = chalco_B
        C = chalco_C

    tolerence_factors = [[] for _ in C]
    energy_differences = [[] for _ in C]
    sigmas = [[] for _ in C]

    gs = gridspec.GridSpec(1, 2, width_ratios=[3.5, 1])
    gs.update(wspace=0.025, hspace=0.07)
    fig = plt.subplots(figsize=(8.5, 6))
    ax = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    all_I_sigmas = []
    all_O_sigmas = []
    all_Se_sigmas = []
    all_S_sigmas = []
    if systems=='halides':
        for c_counter, c in enumerate(['I']):
            for a in A:
                for b in B:
                    row=None
                    system_name = a + b + c
                    uid = system_name + '_Pm3m'
                    try:
                        row = db.get(selection=[('uid', '=', uid)])
                    except:
                        continue
                    if row is not None:
                        try:
                            sigma = row.key_value_pairs['sigma_300K_single']
                            print(sigma)
                            all_I_sigmas.append(sigma)
                        except:
                            continue
    if systems=='chalcogenides':
        for c_counter, c in enumerate(C):
            for a in A:
                for b in B:
                    row=None
                    system_name = a + b + c
                    uid = system_name + '_Pm3m'
                    try:
                        row = db.get(selection=[('uid', '=', uid)])
                    except:
                        continue
                    if row is not None:
                        try:
                            sigma = row.key_value_pairs['sigma_300K_single']
                            if c=='O':
                                all_O_sigmas.append(sigma)
                            if c=='S':
                                all_S_sigmas.append(sigma)
                            if c=='Se':
                                all_Se_sigmas.append(sigma)
                        except:
                            continue

    standout_x=[]
    standout_y=[]
    standout_label=[]

    for c_counter, c in enumerate(C):
        for a in A:
            for b in B:

                system_name = a + b + c
                uid = system_name + '_Pm3m'
                row = None
                pm3m_formation_e = None
                sigma = None

                try:
                    row = db.get(selection=[('uid', '=', uid)])
                except:
                    continue

                tolerance_f = tolerance_factor(a, b, c, type='goldschmidt')

                if tolerance_f is np.NaN:
                    continue


                if row is not None:
                    try:
                        sigma = row.key_value_pairs['sigma_300K_single']
                    except:
                        continue

                if sigma is None:
                    continue


                if row is not None:
                    try:
                        pm3m_formation_e = row.key_value_pairs['formation_energy']
                    except KeyError:
                        continue

                if random:
                    if pm3m_formation_e is not None:
                        randomised_formation_energies = []
                        for counter in range(10):
                            uid_r = uid + '_rand_str_' + str(counter)
                            fe = None
                            row = None
                            try:
                                row = db.get(selection=[('uid', '=', uid_r)])
                            except:
                                pass
                            if row is not None:
                                fe = row.key_value_pairs['formation_energy']
                                #print(uid_r,fe)

                            if fe is not None:
                                randomised_formation_energies.append(fe)
                        if randomised_formation_energies !=[]:
                            #energy_differences.append(pm3m_formation_e-min(randomised_formation_energies))
                            #stdev=[pm3m_formation_e-_e for _e in randomised_formation_energies]
                            print(uid_r, tolerance_f, pm3m_formation_e, pm3m_formation_e - min(randomised_formation_energies))

                            #for _e in randomised_formation_energies:
                            #    energy_differences.append(pm3m_formation_e-_e)
                            #    tolerence_factors.append(tolerance_f)

                            energy_differences[c_counter].append(pm3m_formation_e - min(randomised_formation_energies))
                            if x=='tolerance_factor':
                                tolerence_factors[c_counter].append(tolerance_f)
                            elif x=='sigma':
                                sigmas[c_counter].append(sigma)
                if full_relax:
                    if pm3m_formation_e is not None:
                        uid_f = uid + '_fullrelax_small'
                        fe = None
                        row = None
                        try:
                            row = db.get(selection=[('uid', '=', uid_f)])
                        except:
                            pass
                        if row is not None:
                            fe = row.key_value_pairs['formation_energy']
                        if fe is not None:
                            print(uid_f, tolerance_f, pm3m_formation_e, pm3m_formation_e - fe)

                            energy_differences[c_counter].append(pm3m_formation_e - fe)
                            tolerence_factors[c_counter].append(tolerance_f)

                if x=='sigma' and (randomised_formation_energies!=[]):
                    _e = pm3m_formation_e - min(randomised_formation_energies)
                    if a+b+c in ['CsPbI','CsPbBr','CsSnBr','KCaF','KMnF','KZnF','KNiF','KCaBr','KMnBr','KZnBr','KNiBr']:

                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a+b+c+"$_{3}$")

                    if (c=='F') and (_e<0.0) and (sigma>3.5) and (sigma<4):
                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a + b + c + "$_{3}$")
                    if (c=='Br') and (_e<1.25) and  (_e>1) and (sigma>0.25) and (sigma<1):
                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a + b + c + "$_{3}$")
                    if (c=='Cl') and (_e<1) and  (_e>0.75) and (sigma>1.0) and (sigma<1.5):
                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a + b + c + "$_{3}$")
                    if (c=='I') and sigma==min(all_I_sigmas):
                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a + b + c + "$_{3}$")


                    if a+b+c in ['SrTiO','SrSnO','SrSnSe','PbTiO','BaTiO','PbTiSe','BaTiSe']:
                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a+b+c+"$_{3}$")
                    if (c=='O') and sigma==min(all_O_sigmas):
                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a + b + c + "$_{3}$")
                    if (c=='S') and sigma==min(all_S_sigmas):
                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a + b + c + "$_{3}$")
                    if (c=='Se') and sigma==min(all_O_sigmas):
                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a + b + c + "$_{3}$")
                    if (c=='Se') and (sigma<1) and (_e>1.3) and (_e<1.5):
                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a + b + c + "$_{3}$")
                    if (c=='S') and (sigma>2) and (sigma<4) and (_e>1.6) and (_e<1.7):
                        standout_x.append(_e)
                        standout_y.append(sigma)
                        standout_label.append(a + b + c + "$_{3}$")

    if x=='tolerance_factor':
        for c_counter in range(len(C)):
            ax.scatter(tolerence_factors[c_counter],energy_differences[c_counter],marker='o',c=color_dict[c_counter],edgecolor=None,alpha=0.45,s=25)
    if x=='sigma':
        for c_counter in range(len(C)):
            ax.scatter(energy_differences[c_counter],sigmas[c_counter],marker='o',c=color_dict[c_counter],edgecolor=None,alpha=0.45,s=25)

        ax.scatter(standout_x,standout_y,marker='o',s=30,c='k',edgecolor='k')
        if systems=='halides':
            for i in range(len(standout_x)):
                if standout_x[i]>1:
                    ax.text(1.0, standout_y[i] + 0.06, standout_label[i], c='k', fontsize=10)
                elif (standout_x[i]<-0.1) and (standout_y[i]>3.5) and (standout_y[i]<4):
                    ax.text(-0.25, standout_y[i] + 0.05, standout_label[i], c='k', fontsize=10)
                else:
                    ax.text(standout_x[i]+0.02,standout_y[i]+0.02,standout_label[i],c='k',fontsize=10)
        else:
            for i in range(len(standout_x)):
                if standout_y[i] == min(all_O_sigmas):
                    ax.text(-0.25, standout_y[i] + 0.045, standout_label[i], c='k', fontsize=10)
                elif standout_y[i] == min(all_S_sigmas):
                    ax.text(standout_x[i] + 0.02, standout_y[i] - 0.045, standout_label[i], c='k', fontsize=10)
                elif  (standout_y[i]>2) and (standout_y[i]<4) and (standout_x[i] >1.6) and (standout_x[i] <1.7):
                    ax.text(1.52, standout_y[i] +0.02, standout_label[i], c='k', fontsize=10)
                else:
                    ax.text(standout_x[i] + 0.02, standout_y[i] + 0.02, standout_label[i], c='k', fontsize=10)


    from matplotlib.patches import Patch
    if systems=='halides':
        legend_elements = [Patch(facecolor=color_dict[0], edgecolor='k', label='X='+str(C[0])),
                            Patch(facecolor=color_dict[1], edgecolor='k', label='X='+str(C[1])),
                            Patch(facecolor=color_dict[2], edgecolor='k', label='X='+str(C[2])),
                            Patch(facecolor=color_dict[3], edgecolor='k', label='X='+str(C[3]))]
    if systems=='chalcogenides':
        legend_elements = [Patch(facecolor=color_dict[0], edgecolor='k', label='X='+str(C[0])),
                            Patch(facecolor=color_dict[1], edgecolor='k', label='X='+str(C[1])),
                            Patch(facecolor=color_dict[2], edgecolor='k', label='X='+str(C[2]))]


    ax.legend(handles=legend_elements, loc=1, fontsize=12, ncol=1)

    ylim=ax.get_ylim()
    xlim=ax.get_xlim()


    if x=='tolerance_factor':
        ax.hlines(0, xlim[0], xlim[1], linestyles='--')
        ax.set_xlabel('Tolerance factor')
        if full_relax:
            ax.set_ylabel('$E_{f}^{Pm\\bar{3}m}-E_{f}^{\mbox{\\Large{full relax}}}$ (eV/atom)')
        else:
            ax.set_ylabel('$E_{f}^{Pm\\bar{3}m}-\\min\\{E_{f}^{\mbox{\\Large{full relax}}}\\}$ (eV/atom)')
        if full_relax:
            ylim = [-0.14,0.09]
            ax.set_ylim(ylim)
    elif x=='sigma':
        ax.set_xlabel('$\\Delta H_{c}$ (eV/atom)')
        ax.set_ylabel("$\\sigma$(300 K)")
        ylim = [0,4]
        ax.set_xlim([xlim[0],1.15])
        ax.set_ylim(ylim)

    from scipy.stats import gaussian_kde
    def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
        # Kernel Density Estimation with Scipy
        # Note that scipy weights its bandwidth by the covariance of the
        # input data.  To make the results comparable to the other methods,
        # we divide the bandwidth by the sample standard deviation here.
        kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
        return kde.evaluate(x_grid)

    if x=='tolerance_factor':
        _y=energy_differences
    elif x=='sigma':
        _y=sigmas


    x_grid = np.linspace(ylim[0], ylim[1], 1000)

    for c_counter in range(len(C)):
        if x == 'tolerance_factor':
            bw=0.01
        if x == 'sigma':
            bw=0.1 #for halides
            #bw=0.3 #for chalcogenides
        pdf = kde_scipy(np.array(_y[c_counter]), x_grid, bandwidth=bw)
        ax1.plot(pdf/sum(pdf), x_grid, '-', lw=2, c=color_dict[c_counter])


    xlim = ax1.get_xlim()
    ax1.hlines(0, xlim[0], xlim[1], linestyles='--')
    if full_relax:
        ylim = [-0.14,0.09]
    ax1.set_ylim(ylim)
    labels = [item.get_text() for item in ax1.get_yticklabels()]
    empty_string_labels = [''] * len(labels)
    ax1.set_yticklabels(empty_string_labels)
    if x=='tolerance_factor':
        ax1.set_xlabel('$p(\\Delta E_{f})$')
    elif x=='sigma':
        ax1.set_xlabel("$p(\\sigma)$")

    plt.tight_layout()
    if x=='tolerance_factor':
        if full_relax:
            plt.savefig(systems + "_landscape_full_relax_small.pdf")
        if random:
            plt.savefig(systems+"_landscape_random.pdf")
    elif x=='sigma':
        plt.savefig(systems + "_landscape_sigma_energy.pdf")

def sigma_time_convergence_plots(db, systems='chalcogenides'):
    if systems=='halides':
        A = halide_A
        B = halide_B
        C = ['F']
    elif systems=='chalcogenides': #including oxides
        A = chalco_A
        B = chalco_B
        C = ['Se']

    all_stds=[]
    for end in range(0,900,100):
        stds = []
        for c_counter, c in enumerate(C):
            for a in A:
                for b in B:
                    system_name = a + b + c
                    uid = system_name + '_Pm3m'
                    print(uid)
                    row = None
                    y = None
                    try:
                        row = db.get(selection=[('uid', '=', uid)])
                    except:
                        continue

                    if row is not None:
                        try:
                            y = row.data['sigma_300K']
                        except KeyError:
                            continue

                    this_std = np.std(y[:end])/np.average(y[:end])
                    stds.append(this_std)
        all_stds.append(stds)

    c='#031163'
    plt.boxplot(all_stds,positions=range(0,1800,200),
                  widths=[25 for _ in range(0,900,100)], patch_artist=True,
                  boxprops=dict(facecolor=c, color=c, alpha=0.7),
                  capprops=dict(color=c),
                  whiskerprops=dict(color=c),
                  flierprops=dict(color=c, markeredgecolor=c),
                  medianprops=dict(color=c),
                  showfliers=False,
                  whis=0.5)
    #plt.yscale('log')
    plt.xlabel('Time (ps)')
    plt.ylabel('Percentage Deviation of $\sigma$')
    plt.tight_layout()
    plt.savefig('sigma_converge.pdf')

def sigma_std_vs_mean(db, systems='chalcogenides'):
    if systems=='halides':
        A = halide_A
        B = halide_B
        C = halide_C
    elif systems=='chalcogenides': #including oxides
        A = chalco_A
        B = chalco_B
        C = chalco_C

    all_stds=[]
    all_average=[]
    end = 900
    for c_counter, c in enumerate(C):
        for a in A:
            for b in B:
                system_name = a + b + c
                uid = system_name + '_Pm3m'
                print(uid)
                row = None
                y = None
                try:
                    row = db.get(selection=[('uid', '=', uid)])
                except:
                    continue

                if row is not None:
                    try:
                        y = row.data['sigma_300K']
                    except KeyError:
                        continue

                all_stds.append(np.std(y[:end]))
                all_average.append(np.average(y[:end]))

    c='#031163'
    plt.plot(all_average,all_stds,'.',c=c,alpha=0.5)
    plt.xlim([0,20])
    plt.ylim([0,6])
    #plt.yscale('log')
    plt.xlabel('$\\bar{\\mathcal{S}}$')
    plt.ylabel('$\\sigma(\\mathcal{S})$')
    plt.tight_layout()
    plt.savefig('sigma_average_std.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Switches for analyzing the screening results of bulk cubic perovskites',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default=os.getcwd() + '/perovskites.db',
                        help="Name of the database that contains the results of the screenings.")
    parser.add_argument("--C", type=str,
                        help="Anion in ABCs.")
    args = parser.parse_args()

    if os.path.exists(args.db):
        args.db = connect(args.db)
    else:
        raise Exception("Database " + args.db + " does not exists, cannot proceed!")

    formation_energy_landscapes(args.db,systems='halides',x='sigma')
    #formation_energy_grid(args.db, C=args.C,full_relax=True)
    #formation_energy_structural_deformation_analysis(args.db,systems='chalcogenides')
    #sigma_time_convergence_plots(args.db,systems='chalcogenides')
    #sigma_std_vs_mean(args.db, systems='chalcogenides')