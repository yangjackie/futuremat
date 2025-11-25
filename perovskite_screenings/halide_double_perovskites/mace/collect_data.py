import argparse, os, glob
from artificial_intelligence.phonopy_worker import PhonopyWorker
from ase.io import read
from core.phonon.anharmonic_score import *
from mace.calculators import mace_mp, MACECalculator
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="collect data from mace caculations.")
    parser.add_argument("--aimd_traj", action="store_true",
                        help="collect sigma trajectory that is recomputed on the existing AIMD trajectory")
    parser.add_argument("--mlff_traj", action="store_true", help="collect sigma trajectory that is resampled from MLFF")
    # parser.add_argument("--aimd_phi",action="store_true",help="use the force constants from previous DFT calculations")
    # parser.add_argument("--mlff_phi",action="store_true",help="use the force constants recalculated with MLFF")
    parser.add_argument("--system", type=str, default="fluorides", help="which system to collect")
    parser.add_argument("--model_name", type=str, default="mace-mp-0b3-medium.model",
                        help="name of the MLFF model used for the calculations")
    args = parser.parse_args()

    root_directory = os.getcwd()
    system_directory = root_directory + '/' + str(args.system) + '/'
    available_mace_models = ['MACE-matpes-pbe-omat-ft.model', 'mace-mpa-0-medium.model', 'mace-mp-0b3-medium.model',
                             'mace-omat-0-medium.model']

    if args.mlff_traj:
        print("collecting sigma trajectory from MLFF-MD simulation...\n")
        result_data = {}
        os.chdir(system_directory)
        all_directories = glob.glob(os.getcwd() + '/dpv_*/')
        for directory in list(sorted(all_directories)):
            os.chdir(directory)
            compound_name = os.getcwd().split('/')[-1].replace('dpv_', '')
            result_data[compound_name] = {}
            print("=============================" + compound_name + "=============================")

            if os.path.exists('./combined_dft.traj'):
                ref_frame = os.getcwd() + '/SPOSCAR'
                dft_force_constants = os.getcwd() + '/force_constants.hdf5'
                # get the dft anharmonic scores
                try:
                    dft_scorer = AnharmonicScore(md_frames='./combined_dft.traj',
                                                 unit_cell_frame=ref_frame,
                                                 ref_frame=ref_frame,
                                                 atoms=None,
                                                 potim=1,
                                                 force_constants=dft_force_constants,
                                                 mode_resolved=False,
                                                 include_third_order=False)
                    dft_sigma, dft_time_stps = dft_scorer.structural_sigma(return_trajectory=True)
                except:
                    continue
                result_data[compound_name]['dft'] = dft_sigma

            if os.path.exists('andersen_md_mace-omat-0-medium_run_1.traj') and os.path.exists(
                    'andersen_md_mace-omat-0-medium_run_2.traj'):
                _fc = "force_constants-mace-omat-0-medium.hdf5"
                if not os.path.exists('./' + _fc):
                    mace_model_path: str = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/artificial_intelligence/mace_models/"
                    calculator = mace_mp(model=mace_model_path + 'mace-omat-0-medium.model', device='cpu')
                    contcar = os.getcwd() + '/CONTCAR'
                    phonopy_worker = PhonopyWorker(structure=read(contcar),
                                                   calculator=calculator,
                                                   supercell_matrix=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))
                    phonopy_worker.generate_force_constants(save_fc=True, fc_file_name=_fc)
                try:
                    ref_frame = os.getcwd() + '/SPOSCAR'
                    if os.path.exists('andersen_md_mace-omat-0-medium_run_1.traj'):
                        mlff_scorer = AnharmonicScore(md_frames='andersen_md_mace-omat-0-medium_run_1.traj',
                                                      unit_cell_frame=ref_frame,
                                                      ref_frame=ref_frame,
                                                      atoms=None,
                                                      potim=1,
                                                      force_constants=_fc,
                                                      mode_resolved=False,
                                                      include_third_order=False)
                        mlff_sigma, mlff_time_stps = mlff_scorer.structural_sigma(return_trajectory=True)
                        result_data[compound_name]['mlff-omat-0-medium-run-1'] = mlff_sigma
                    if os.path.exists('andersen_md_mace-omat-0-medium_run_2.traj'):
                        mlff_scorer = AnharmonicScore(md_frames='andersen_md_mace-omat-0-medium_run_2.traj',
                                                      unit_cell_frame=ref_frame,
                                                      ref_frame=ref_frame,
                                                      atoms=None,
                                                      potim=1,
                                                      force_constants=_fc,
                                                      mode_resolved=False,
                                                      include_third_order=False)
                        mlff_sigma, mlff_time_stps = mlff_scorer.structural_sigma(return_trajectory=True)
                        result_data[compound_name]['mlff-omat-0-medium-run-2'] = mlff_sigma
                except:
                    continue
            os.chdir(system_directory)
        os.chdir(root_directory)
        with open(root_directory + "/mlff_andersen_md_traj_sigma_" + args.system + '.pkl', 'wb') as f:
            pickle.dump(result_data, f)
    elif args.aimd_traj:
        result_data = {}
        # this is the results for recomputing the forces for each AIMD frame with the MLFF
        os.chdir(system_directory)
        all_directories = glob.glob(os.getcwd() + '/dpv_*/')
        for directory in list(sorted(all_directories)):
            os.chdir(directory)

            compound_name = os.getcwd().split('/')[-1].replace('dpv_', '')
            print("=============================" + compound_name + "=============================")
            ref_frame = os.getcwd() + '/SPOSCAR'
            dft_force_constants = os.getcwd() + '/force_constants.hdf5'

            if os.path.exists('./combined_dft.traj'):
                result_data[compound_name] = {}
                # get the dft anharmonic scores
                try:
                    dft_scorer = AnharmonicScore(md_frames='./combined_dft.traj',
                                                 unit_cell_frame=ref_frame,
                                                 ref_frame=ref_frame,
                                                 atoms=None,
                                                 potim=1,
                                                 force_constants=dft_force_constants,
                                                 mode_resolved=False,
                                                 include_third_order=False)
                    dft_sigma, dft_time_stps = dft_scorer.structural_sigma(return_trajectory=True)
                except:
                    continue
                result_data[compound_name]['dft'] = dft_sigma
            for model in available_mace_models:
                model_name = model.replace(".model", "")
                mlff_traj_name = "trajectory-" + model.replace(".model", "") + ".traj"
                if os.path.exists(mlff_traj_name):
                    # anharmonic scores using the DFT force constants
                    try:
                        mlff_scorer = AnharmonicScore(md_frames=mlff_traj_name,
                                                      unit_cell_frame=ref_frame,
                                                      ref_frame=ref_frame,
                                                      atoms=None,
                                                      potim=1,
                                                      force_constants=dft_force_constants,
                                                      mode_resolved=False,
                                                      include_third_order=False)
                        mlff_sigma, mlff_time_stps = mlff_scorer.structural_sigma(return_trajectory=True)
                        result_data[compound_name][model_name + '_dft_phi'] = mlff_sigma
                    except:
                        continue

                    # anharmonic scores using the MLFF force constants
                    _fc = "force_constants-" + model.replace(".model", "") + ".hdf5"
                    if not os.path.exists('./' + _fc):
                        mace_model_path: str = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/artificial_intelligence/mace_models/"
                        calculator = mace_mp(model=mace_model_path + model, device='cpu')
                        contcar = os.getcwd() + '/CONTCAR'
                        phonopy_worker = PhonopyWorker(structure=read(contcar),
                                                       calculator=calculator,
                                                       supercell_matrix=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]))
                        phonopy_worker.generate_force_constants(save_fc=True, fc_file_name=_fc)
                    try:
                        mlff_scorer = AnharmonicScore(md_frames=mlff_traj_name,
                                                      unit_cell_frame=ref_frame,
                                                      ref_frame=ref_frame,
                                                      atoms=None,
                                                      potim=1,
                                                      force_constants=_fc,
                                                      mode_resolved=False,
                                                      include_third_order=False)
                        mlff_sigma, mlff_time_stps = mlff_scorer.structural_sigma(return_trajectory=True)
                        result_data[compound_name][model_name + '_mlff_phi'] = mlff_sigma
                    except:
                        continue

            os.chdir(system_directory)
        os.chdir(root_directory)
        with open(root_directory + "/dft_traj_sigma_" + args.system + '.pkl', 'wb') as f:
            pickle.dump(result_data, f)
    elif args.mlff_traj:
        os.chdir(system_directory)

        os.chdir(root_directory)
