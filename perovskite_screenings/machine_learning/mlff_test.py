import os

import numpy as np
from matplotlib import gridspec

from core.external.vasp.anharmonic_score import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pylab as pylab

params = {'legend.fontsize': '14',
          'figure.figsize': (7.5, 6),
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

from scipy.stats import gaussian_kde
import pickle

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    # Kernel Density Estimation with Scipy
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

def dft_calculation_points(stepsize=1.0):
    mlff_training_folder = '.'
    scorer = AnharmonicScore(md_frames=mlff_training_folder + '/vasprun.xml', ref_frame='./SPOSCAR',
                             force_constants='../force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                             primitive_matrix='auto')
    training_sigma, _ = scorer.structural_sigma(return_trajectory=True)
    time = [stepsize * i for i in range(len(training_sigma))]

    ml_log = open(mlff_training_folder+'/ML_LOGFILE','r')
    dft_time = []
    dft_sigma = []
    for line in ml_log.readlines():
        if ('STATUS' in line) and ("#" not in line):
            splitted = line.split()
            if splitted[4]=='T':
                if int(splitted[1])!=0:
                    dft_time.append((int(splitted[1])-1)*stepsize)
                    dft_sigma.append(training_sigma[int(splitted[1])-1])

    plt.plot(time,training_sigma,'-',c='#66A5AD',alpha=0.6)

    plt.scatter(dft_time,dft_sigma,marker='.',edgecolors='#F62A00',facecolors='None')

    plt.xlabel('Time (fs)')
    plt.ylabel('$\\sigma^{(2)}(t)$')
    plt.tight_layout()
    plt.savefig('mlff_training_dft_points.pdf')


def compare_sigma_dft_mlff(trajectory=True, stepsize=1.0):
    
    dft_result_folder = 'long_MD_GPU_3ps'

    scorer = AnharmonicScore(md_frames=dft_result_folder + '/vasprun.xml', ref_frame='./SPOSCAR',
                             force_constants='force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                             primitive_matrix='auto')
    dft_sigma, _ = scorer.structural_sigma(return_trajectory=True)

    mlff_result_folder = 'long_MD_EDIFF_-9'
    scorer = AnharmonicScore(md_frames=mlff_result_folder + '/vasprun.xml', ref_frame='./SPOSCAR',
                             force_constants='force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                             primitive_matrix='auto')
    mlff_sigma_EDIFF_9, _ = scorer.structural_sigma(return_trajectory=True)

    mlff_result_folder = 'long_MD_EDIFF_-8'
    scorer = AnharmonicScore(md_frames=mlff_result_folder + '/vasprun.xml', ref_frame='./SPOSCAR',
                             force_constants='force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                             primitive_matrix='auto')
    mlff_sigma_EDIFF_8, _ = scorer.structural_sigma(return_trajectory=True)

    mlff_result_folder = 'long_MD_EDIFF_-7'
    scorer = AnharmonicScore(md_frames=mlff_result_folder + '/vasprun.xml', ref_frame='./SPOSCAR',
                             force_constants='force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                             primitive_matrix='auto')
    mlff_sigma_EDIFF_7, _ = scorer.structural_sigma(return_trajectory=True)

    mlff_result_folder = 'long_MD_EDIFF_-6'
    scorer = AnharmonicScore(md_frames=mlff_result_folder + '/vasprun.xml', ref_frame='./SPOSCAR',
                             force_constants='force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                             primitive_matrix='auto')
    mlff_sigma_EDIFF_6, _ = scorer.structural_sigma(return_trajectory=True)

    mlff_result_folder = 'long_MD_EDIFF_-5'
    scorer = AnharmonicScore(md_frames=mlff_result_folder + '/vasprun.xml', ref_frame='./SPOSCAR',
                             force_constants='force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                             primitive_matrix='auto')
    mlff_sigma_EDIFF_5, _ = scorer.structural_sigma(return_trajectory=True)

    if trajectory:
        gs = gridspec.GridSpec(1, 2, width_ratios=[3.5, 1])
        gs.update(wspace=0.025, hspace=0.07,bottom=0.15)
        fig = plt.subplots(figsize=(8.5, 6))
        ax = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        time_dft = [stepsize * i for i in range(len(dft_sigma))]
        time_ff_9 = [stepsize * i for i in range(len(mlff_sigma_EDIFF_9))]

        ax.plot(time_ff_9, mlff_sigma_EDIFF_5, '-', c='#F62A00', label='$\\sigma^{(2)}$ MLFF, EDIFF=10$^{-5}$ eV',lw=5,alpha=0.4)
        ax.plot(time_ff_9, mlff_sigma_EDIFF_6, '-', c='#F62A00', label='$\\sigma^{(2)}$ MLFF, EDIFF=10$^{-6}$ eV',lw=4,alpha=0.5)
        ax.plot(time_ff_9, mlff_sigma_EDIFF_7, '-', c='#F62A00', label='$\\sigma^{(2)}$ MLFF, EDIFF=10$^{-7}$ eV',lw=3,alpha=0.6)
        ax.plot(time_ff_9, mlff_sigma_EDIFF_8, '-', c='#F62A00', label='$\\sigma^{(2)}$ MLFF, EDIFF=10$^{-8}$ eV',lw=2,alpha=0.7)
        ax.plot(time_ff_9, mlff_sigma_EDIFF_9, '-', c='#F62A00', label='$\\sigma^{(2)}$ MLFF, EDIFF=10$^{-9}$ eV',lw=1,alpha=0.8)
        ax.plot(time_dft, dft_sigma, '-', c='#07575B', label='$\\sigma^{(2)}$ DFT',lw=0.5)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('$\\sigma^{(2)}(t)$')
        ax.legend()

        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        y_grid = np.linspace(ylim[0], ylim[1], 1000)
        pdf_dft = kde_scipy(np.array(dft_sigma), y_grid, bandwidth=0.01)
        pdf_mlff_9 = kde_scipy(np.array(mlff_sigma_EDIFF_9), y_grid, bandwidth=0.01)
        pdf_mlff_8 = kde_scipy(np.array(mlff_sigma_EDIFF_8), y_grid, bandwidth=0.01)
        pdf_mlff_7 = kde_scipy(np.array(mlff_sigma_EDIFF_7), y_grid, bandwidth=0.01)
        pdf_mlff_6 = kde_scipy(np.array(mlff_sigma_EDIFF_6), y_grid, bandwidth=0.01)
        pdf_mlff_5 = kde_scipy(np.array(mlff_sigma_EDIFF_5), y_grid, bandwidth=0.01)


        ax1.plot(pdf_dft / sum(pdf_dft), y_grid, lw=2, c='#07575B')
        ax1.plot(pdf_mlff_8 / sum(pdf_mlff_8), y_grid, lw=2, c='#F62A00', alpha=0.7)
        ax1.plot(pdf_mlff_9 / sum(pdf_mlff_9), y_grid, lw=1, c='#F62A00',alpha=0.8)
        ax1.plot(pdf_mlff_7 / sum(pdf_mlff_7), y_grid, lw=3, c='#F62A00',alpha=0.6)
        ax1.plot(pdf_mlff_6 / sum(pdf_mlff_6), y_grid, lw=4, c='#F62A00',alpha=0.5)
        ax1.plot(pdf_mlff_5 / sum(pdf_mlff_5), y_grid, lw=4, c='#F62A00',alpha=0.5)

        ax1.set_ylim(ylim)

        labels = [item.get_text() for item in ax1.get_yticklabels()]
        empty_string_labels = [''] * len(labels)
        ax1.set_yticklabels(empty_string_labels)

        labels = [item.get_text() for item in ax1.get_xticklabels()]
        empty_string_labels = [''] * len(labels)
        ax1.set_xticklabels(empty_string_labels)

        ax1.set_xlabel("$p[\\sigma^{(2)}]$")

        plt.tight_layout()

    plt.savefig('sigma_MLFF_DFT_compare.pdf')

def collect_results():
    result_dict={}
    num_ML_success = 0
    num_ML_failed = 0

    cwd = os.getcwd()
    sub_directories = [f.path for f in os.scandir(cwd) if f.is_dir()]

    for dir in sub_directories:

        dft_sigma = None
        mlff_sigma = None

        system_name = dir.split('/')[-1]
        os.chdir(dir)

        #check if machine learning was successful
        ml_successful = False
        if os.path.exists('./MLFF'):
            os.chdir("./MLFF")
            total_num_steps = 0
            f=open('./OSZICAR','r')
            for l in f.readlines():
                if 'T=' in l:
                    total_num_steps+=1
            f.close()
            if total_num_steps==10000:
                ml_successful = True
                num_ML_success += 1
            else:
                num_ML_failed += 1
            print(system_name,' ML successful? ',ml_successful, 'Suuccess:', num_ML_success, 'Failed:', num_ML_failed)
            os.chdir("..")

        if ml_successful:
            mlff_result_folder = 'MLFF_benchmark'
            dft_result_folder = 'DFT_benchmark'

            if not os.path.exists("./SPOSCAR"):
                try:
                    from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
                    from phonopy import Phonopy

                    unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')
                    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
                    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
                    phonon.generate_displacements()
                    write_crystal_structure('SPOSCAR', phonon.supercell)
                except:
                    pass

            #scorer = AnharmonicScore(md_frames=dft_result_folder + '/vasprun.xml', ref_frame='./SPOSCAR',
            #                         force_constants='force_constants.hdf5', unit_cell_frame='./SPOSCAR',
            #                         primitive_matrix='auto')
            #dft_sigma, _ = scorer.structural_sigma(return_trajectory=True)

            # ==========================================================================================================
            try:
                import glob
                scorer = AnharmonicScore(md_frames=glob.glob('./MD/vasprun_prod*.xml'), ref_frame='./SPOSCAR',
                                     force_constants='force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                                     primitive_matrix='auto')
                dft_sigma, _ = scorer.structural_sigma(return_trajectory=True)
            except:
                pass

            print('length of dft sigma :',len(dft_sigma))
            # ==========================================================================================================

            scorer = AnharmonicScore(md_frames=mlff_result_folder + '/vasprun.xml', ref_frame='./SPOSCAR',
                                     force_constants='force_constants.hdf5', unit_cell_frame='./SPOSCAR',
                                     primitive_matrix='auto')
            mlff_sigma, _ = scorer.structural_sigma(return_trajectory=True)
        else:
            dft_sigma = None
            mlff_sigma = None

        os.chdir(cwd)

        f = open('mlff_test_set.dat','r')
        for l in f.readlines():
            if system_name in l:
                ref_sigma = float(l.split()[1])
        f.close()

        result_dict[system_name.replace('dpv_','')] = {'dft_sigma':dft_sigma,
                                                       'mlff_sigma':mlff_sigma,
                                                       'ref_sigma':ref_sigma}

    pickle.dump(result_dict,open('benchmark_results.bp','wb'))

def statistic_box_plot():
    all_systems_in_order = []
    all_reference_sigma = []
    y_labels = []
    reference_sigma_dict = {}

    f = open('mlff_test_set.dat', 'r')
    for l in f.readlines():
        all_systems_in_order.append(l.split()[0])
        #y_labels.append(l.split()[0].replace('dpv_',''))
        all_reference_sigma.append(float(l.split()[1]))
        reference_sigma_dict[l.split()[0].replace('dpv_','')] = float(l.split()[1])
    f.close()

    #all_systems_in_order.reverse()
    #y_labels.reverse()
    #all_reference_sigma.reverse()

    data = pickle.load(open('benchmark_results.bp','rb'))
    box_dft_data = []
    box_mlff_data = []
    for name in all_systems_in_order:
        name=name.replace('dpv_','')
        y_labels.append(name)
        try:
            if data[name]['dft_sigma'] is not None:
                #print(len(data[name]['dft_sigma']))
                box_dft_data.append(data[name]['dft_sigma'])
            else:
                box_dft_data.append([])
        except KeyError:
            box_dft_data.append([])

        try:
            if data[name]['mlff_sigma'] is not None:
                #print(len(data[name]['mlff_sigma']))
                box_mlff_data.append(data[name]['mlff_sigma'])
            else:
                box_mlff_data.append([])
        except:
            box_mlff_data.append([])
        print(name,len(box_dft_data[-1]),len(box_mlff_data[-1]))

    plt.figure(figsize=(11.69,8.27))
    #plot the reference sigma values first
    for i,name in enumerate(y_labels):
        plt.plot([reference_sigma_dict[name],reference_sigma_dict[name]],[i-0.2,i+0.2],'r-')

    boxprops = dict(color="#CF3721", linewidth=0.7)
    bp1 = plt.boxplot(box_dft_data,vert=0,showfliers=False,boxprops=boxprops,whiskerprops=boxprops,capprops=boxprops)
    for median in bp1['medians']:
        median.set_color('None')

    boxprops = dict(color="#31A9B8", linewidth=1.3)
    bp2 = plt.boxplot(box_mlff_data, vert=0, showfliers=False, boxprops=boxprops, whiskerprops=boxprops,
                     capprops=boxprops)
    for median in bp2['medians']:
        median.set_color('None')

    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['DFT', 'MLFF'], loc='upper left')

    plt.yticks([i for i in range(len(all_reference_sigma))], y_labels,fontsize=4)
    plt.ylim([-1,len(all_reference_sigma)])
    plt.xscale('log')
    plt.xlabel("$\\sigma^{(2)}$")
    plt.xticks([0.2,0.3,0.4,0.5,0.7,1,2,3,4],[0.2,0.3,0.4,0.5,0.7,1,2,3,4])
    plt.tight_layout()
    plt.savefig("ML_benchmark_boxplot.pdf")



if __name__ == "__main__":
    compare_sigma_dft_mlff(trajectory=True)
    #dft_calculation_points()
    #collect_results()
    #statistic_box_plot()

