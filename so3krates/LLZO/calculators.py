import argparse,shutil,os
from pymatgen.io.vasp.outputs import Vasprun
from core.utils.zipdir import ZipDir
from core.internal.builders.crystal import map_pymatgen_IStructure_to_crystal
from core.calculators.vasp import Vasp

def optimise_frames_from_md_trajectory(part=0, batch_size=20, gpu_run=False, gap=10):
    # load all the production run xml file

    all_vasp_runs = []
    all_structures = []
    for v in ['vasprun.xml']:
        if os.path.exists('./' + v):
            all_vasp_runs.append(v)

    for v in all_vasp_runs:
        vasprun = Vasprun(v)
        trajectory = vasprun.get_trajectory()
        all_structures += [trajectory.get_structure(i) for i in range(len(trajectory.coords))]

    if (part + 1) * batch_size > len(all_structures):
        raise Exception("over the length of the trajectory, quit")

    for i in range(batch_size * part, batch_size * (part + 1), gap):

        pwd = os.getcwd()
        folder = 'frame_' + str(i)

        if os.path.exists(pwd + '/' + folder + '.zip'): continue

        try:
            os.mkdir(folder)
        except:
            pass
        os.chdir(pwd + '/' + folder)

        this_structure = map_pymatgen_IStructure_to_crystal(all_structures[i])
        this_structure.gamma_only = True #DO NOT DELETE THIS!!!

        single_pt_set = {'ISPIN': 1, 'PREC': "Normal", 'IALGO': 38, 'NPAR': 32, 'ENCUT': 500, 'IBRION':1, 'ISIF':0, 'NSW': 300,
                         'LCHARG': True, 'LWAVE': True, 'use_gw': True, 'Gamma_centered': True, 'MP_points': [1, 1, 1],
                         'clean_after_success': False, 'LREAL': 'False', 'executable':'vasp_gam', 'gpu_run': gpu_run}
        vasp = Vasp(**single_pt_set)
        vasp.set_crystal(this_structure)

        vasp.execute()

        files = ['CHG', 'CHGCAR', 'LOCPOT', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'POTCAR', 'WAVECAR', 'DOSCAR', 'OUTCAR',
                 'PROCAR', 'KPOINTS']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

        os.chdir(pwd)

        ZipDir(folder, folder + '.zip')
        shutil.rmtree(folder, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='workflow control for high-throughput DFT ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--opt", action='store_true', help='perform initial structural optimization')
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--gap", type=int, default=20)
    parser.add_argument("--gpu", action='store_true', help='specify whether the calculation is to be run on the GPU node')
    args = parser.parse_args()

    optimise_frames_from_md_trajectory(part=args.part,batch_size=args.batch_size,gpu_run=args.gpu,gap=args.gap)
