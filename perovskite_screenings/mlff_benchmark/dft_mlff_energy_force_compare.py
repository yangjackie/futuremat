import argparse
import os
import shutil
import numpy as np

from pymatgen.io.vasp.outputs import Vasprun

from core.calculators.vasp import Vasp
from core.internal.builders.crystal import map_pymatgen_IStructure_to_crystal
from core.utils.zipdir import ZipDir


def run_single_point_from_mlff_trajectory(part=0):
    this_mlff_run = 'vasprun_' + str(part) + ".xml"
    vasprun = Vasprun(this_mlff_run)
    benchmark_frames = []
    for i in range(10, len(vasprun.md_data), 20):  # for mlff run, get the md_data rather than the trajectory!
        this_frame = vasprun.md_data[i]
        # print(this_frame['structure'])
        # print(this_frame['forces'])
        # print(this_frame['energy']['total'])
        benchmark_frames.append(this_frame)

    pwd = os.getcwd()

    if not os.path.isdir('./DFT_part_' + str(part)):
        os.mkdir('./DFT_part_' + str(part))
    os.chdir('./DFT_part_' + str(part))

    for i in range(0, len(benchmark_frames)):
        energy_file = open(pwd + '/energy_benchmark_' + str(part + 1) + '.dat', 'a')
        force_file = open(pwd + '/force_benchmark_' + str(part + 1) + '.dat', 'a')

        folder = './frame_' + str(i + 1)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        os.chdir(folder)
        this_structure = map_pymatgen_IStructure_to_crystal(benchmark_frames[i]['structure'])

        mlff_energy = benchmark_frames[i]['energy']['e_0_energy']  # see https://www.vasp.at/wiki/index.php/OSZICAR
        mlff_force_mag = [np.linalg.norm(np.array(j)) for j in benchmark_frames[i]['forces']]

        single_pt_set = {'ISPIN': 1, 'PREC': "ACCURATE", 'ALGO': 'NORMAL', 'NPAR': 28, 'ENCUT': 550, 'SIGMA': 0.1,
                         'ISYM': -1, 'EDIFF': 1E-5,
                         'LCHARG': False, 'LWAVE': False, 'use_gw': True, 'Gamma_centered': True,
                         'MP_points': [5, 5, 5],
                         'clean_after_success': False, 'LREAL': 'Auto', 'executable': 'vasp_std', 'GGA': 'PS'}
        vasp = Vasp(**single_pt_set)
        vasp.set_crystal(this_structure)
        vasp.execute()

        dft_vasprun = Vasprun('vasprun.xml')
        dft_energy = dft_vasprun.final_energy
        dft_force_mag = [np.linalg.norm(np.array(j)) for j in
                         dft_vasprun.get_trajectory()[-1].site_properties['forces']]

        os.chdir('../')
        ZipDir(folder, folder + '.zip')
        shutil.rmtree(folder, ignore_errors=True)

        energy_file.write(str(dft_energy).replace('eV', '') + '\t' + str(mlff_energy) + '\n')
        for k in range(len(dft_force_mag)):
            force_file.write(str(dft_force_mag[k]) + '\t' + str(mlff_force_mag[k]) + '\n')

        energy_file.close()
        force_file.close()
    os.chdir('..')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='workflow control for mlff_benchmark ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--part", type=int, default=1)

    args = parser.parse_args()

    run_single_point_from_mlff_trajectory(part=args.part)
