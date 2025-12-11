import os
import time

from ase.io import read, write, Trajectory
from ase import Atoms
from mace.calculators import mace_mp, MACECalculator
from core.phonon.anharmonic_score import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")



def rerun_dft_md_traj_with_mlff(
        mace_model_path: str = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/artificial_intelligence/mace_models/",
        mace_model_name: str = "mace-omat-0-medium.model",
        directory: str = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/perovskite_anharmonic_screening/halide_double_perovskites/MLFF_benchmark/",
        system: str = None,
        write_dft_traj: bool = False):

    if system is None:
        working_dir = os.getcwd()
    else:
        working_dir = directory + '/' + system + '/'
    import torch
    torch.set_default_dtype(torch.float32)

    calculator = mace_mp(model=mace_model_path + mace_model_name, device='cpu', default_dtype='float32')
    #calculator = MACECalculator(model_paths=mace_model_path + mace_model_name, device='mps', default_dtype='float32')

    # convert the existing vasprun.xml into trajectory object
    all_atoms = []
    vasp_xmls = [working_dir + '/MD/vasprun_prod_' + str(i + 1) + '.xml' for i in range(2)]
    for xml in vasp_xmls:
        try:
            all_atoms.extend(read(xml, index='100:'))
        except:
            print('failed to read ' + xml.split('/')[-1])
            pass

    if all_atoms == []:
        return

    if write_dft_traj:
        write(working_dir + '/combined_dft.traj', all_atoms)

    mlff_traj = "trajectory-" + mace_model_name.replace(".model", "") + ".traj"
    traj = Trajectory(mlff_traj, "w")

    start_time = time.time()
    print(">>>>>> Working directory: " + working_dir.split('/')[-1]+" <<<<<<")
    for id, atom in enumerate(all_atoms):
        mlff_atoms = Atoms(atom.get_chemical_symbols(), positions=atom.get_positions(), cell=atom.cell, pbc=True)
        mlff_atoms.calc = calculator
        mlff_atoms.get_forces()

        elapsed_time = time.time() - start_time
        time_per_cycle = elapsed_time / (id + 1)

        print(f"\r Calculating mlff force for frame {id + 1}/{len(all_atoms)}\t | Time={elapsed_time:7.2f} seconds |  Time per cycle: {time_per_cycle:7.2f} seconds",
              end=" ", flush=True)
        traj.write(mlff_atoms)


def compare_anharmonic_scores_dft_mlff(
        mace_model_path: str = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/artificial_intelligence/mace_models/",
        #mace_model_name: str = "mace-omat-0-medium.model",
        directory: str = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/perovskite_anharmonic_screening/halide_double_perovskites/MLFF_benchmark/",
        system: str = None,
        ref_frame: str = os.getcwd() + '/SPOSCAR',
        force_constants: str = os.getcwd() + '/force_constants.hdf5',
        recalculate_mlff_fc: bool = False,
        #plot_figures: bool = True,
        run_simulations: bool = False):

    if system is None:
        working_dir = os.getcwd()
    else:
        working_dir = directory + '/' + system + '/'

    available_mace_models = ['MACE-matpes-pbe-omat-ft.model', 'mace-mpa-0-medium.model',
                             'mace-mp-0b3-medium.model', 'mace-omat-0-medium.model']

    if run_simulations:
        for model in available_mace_models:
            calculator = mace_mp(model=mace_model_path + model, device='cpu')

            # convert the existing vasprun.xml into trajectory object
            all_atoms = []
            vasp_xmls = [working_dir + '/MD/vasprun_prod_' + str(i + 1) + '.xml' for i in range(2)]
            for xml in vasp_xmls:
                all_atoms.extend(read(xml, index='100:'))

            write(working_dir + '/combined_dft.traj', all_atoms)

            all_mlff_atoms = []
            start_time = time.time()

            mlff_traj = "trajectory-" + model.replace(".model", "") + ".traj"
            traj = Trajectory(mlff_traj, "w")

            for id, atom in enumerate(all_atoms):
                mlff_atoms = Atoms(atom.get_chemical_symbols(), positions=atom.get_positions(), cell=atom.cell, pbc=True)
                mlff_atoms.calc = calculator
                mlff_atoms.get_forces()

                elapsed_time = time.time() - start_time
                time_per_cycle = elapsed_time / (id + 1)
                print(f"\r Calculating mlff force for frame {id + 1}/{len(all_atoms)}\t | Time={elapsed_time:7.2f} seconds  |  Time per cycle: {time_per_cycle:7.2f} seconds",
                      end=" ", flush=True)
                traj.write(mlff_atoms)
    else:
        dft_scorer = AnharmonicScore(md_frames=working_dir + '/combined_dft.traj',
                                     unit_cell_frame=ref_frame,
                                     ref_frame=ref_frame,
                                     atoms=None,
                                     potim=1,
                                     force_constants=force_constants,
                                     mode_resolved=False,
                                     include_third_order=False)
        dft_sigma, dft_time_stps = dft_scorer.structural_sigma(return_trajectory=True)
        plt.plot(dft_time_stps, dft_sigma, '-', c='#FFBB00', label='AIMD')

        label_colors = ['#003B46', '#07575B', '#66A5AD','#C4DFE6']
        for id, model in enumerate(available_mace_models):
            mlff_traj = "trajectory-" + model.replace(".model", "") + ".traj"
            if not recalculate_mlff_fc:
                #use the default DFT force constants
                _fc = force_constants
            else:
                from core.phonon.phonopy_worker import PhonopyWorker
                from ase.io import read
                calculator = mace_mp(model=mace_model_path + model, device='cpu')
                contcar = working_dir+'/CONTCAR'

                phonopy_worker = PhonopyWorker(structure=read(contcar)
                                               , calculator=calculator,
                                               supercell_matrix=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))
                _fc = "force_constants-"+model.replace(".model", "")+".hdf5"
                phonopy_worker.generate_force_constants(save_fc=True,fc_file_name=_fc)

            mlff_scorer = AnharmonicScore(md_frames=mlff_traj,
                                          unit_cell_frame=ref_frame,
                                          ref_frame=ref_frame,
                                          atoms=None,
                                          potim=1,
                                          force_constants=_fc,
                                          mode_resolved=False,
                                          include_third_order=False)
            mlff_sigma, mlff_time_stps = mlff_scorer.structural_sigma(return_trajectory=True)
            plt.plot(mlff_time_stps, mlff_sigma, '-', c=label_colors[len(label_colors)-id-1], alpha=0.7, label=model.lower().replace(".model", "").replace("mace-", " "))

        plt.xlabel('Time (fs)')
        plt.ylabel('$\\sigma^{(2)}(t)$')
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse, glob

    parser = argparse.ArgumentParser(description="Benchmarking MACE model for halide double perovskites.")

    parser.add_argument("--run_benchmark", action='store_true', help="Run the benchmarking of frequencies.")
    parser.add_argument("--compare", action='store_true', help="Compare DFT and MLFF results")
    parser.add_argument("--model_name", type=str, default="mace-mp-0b3-medium.model")
    parser.add_argument("--write_dft_traj", action='store_true')
    parser.add_argument("--recalculate_mlff_fc", action='store_true')

    args = parser.parse_args()

    if args.run_benchmark:
        all_directories_to_work = []
        cwd = os.getcwd()
        for sys in ['iodides','bromides','chlorides','fluorides']:
        #for sys in ['fluorides', 'chlorides', 'bromides', 'iodides']:
            os.chdir(sys)
            sys_cwd = os.getcwd()
            all_directories = glob.glob(os.getcwd() + '/dpv_*/')
            for directory in list(sorted(all_directories)):
                os.chdir(directory)
                #chk_file = "trajectory-" + args.model_name.replace(".model", "") + "-done.chk"
                #if os.path.exists('./MD/vasprun_prod_1.xml') or os.path.exists('./MD/vasprun_prod_2.xml'):
                #    if not os.path.exists(chk_file):
                all_directories_to_work.append(directory)
                os.chdir(sys_cwd)
            os.chdir(cwd)

        cwd=os.getcwd()
        for id,directory in enumerate(all_directories_to_work):
            os.chdir(directory)
            print("\n")
            print("......................"+str(id+1)+'/'+str(len(all_directories_to_work))+"......................")
            chk_file = "trajectory-" + args.model_name.replace(".model", "") + "-done.chk"
            run_this=True

            mlff_traj = "trajectory-" + args.model_name.replace(".model", "") + ".traj"

            if os.path.exists(chk_file):

                if not os.path.exists(mlff_traj) or os.path.getsize(mlff_traj) == 0:
                    print("..False Positive.., needs rerun")
                    try:
                        os.remove(chk_file)
                    except:
                        print("check file not found")
                    run_this = True
                    print(os.getcwd())

                else:
                    run_this = False
            if run_this:
                rerun_dft_md_traj_with_mlff(mace_model_name=args.model_name,write_dft_traj=args.write_dft_traj)
                if os.path.exists(mlff_traj):
                    if os.path.getsize(mlff_traj) != 0:
                        chk_file=open(chk_file, 'w')
                        chk_file.write("Done!")
                        chk_file.close()
            os.chdir(cwd)
    elif args.compare:
        compare_anharmonic_scores_dft_mlff(recalculate_mlff_fc=args.recalculate_mlff_fc)
