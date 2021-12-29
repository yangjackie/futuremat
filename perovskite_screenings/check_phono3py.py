import os, tarfile, glob, shutil, argparse
from collect_data import A_site_list, B_site_list, C_site_list
from core.calculators.vasp import Vasp
from core.utils.loggings import setup_logger

halide_C = ['F', 'Cl', 'Br', 'I']
halide_A = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Cu', 'Ag', 'Au', 'Hg', 'Ga', 'In', 'Tl']
halide_B = ['Mg', 'Ca', 'Sr', 'Ba', 'Se', 'Te', 'As', 'Si', 'Ge', 'Sn', 'Pb', 'Ga', 'In', 'Sc', 'Y', 'Ti', 'Zr', 'Hf',
            'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt',
            'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg']

chalco_C = ['O', 'S', 'Se']
chalco_A = ['Ba', 'Mg', 'Ca', 'Sr', 'Be', 'Ra', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Pd', 'Pt', 'Cu', 'Ag', 'Zn',
            'Cd', 'Hg', 'Ge', 'Sn', 'Pb']
chalco_B = ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir',
            'Ni', 'Pd', 'Pt', 'Sn', 'Ge', 'Pb', 'Si', 'Te', 'Po']


def check_phono3py_completion(C):
    cwd = os.getcwd()
    logger = setup_logger(output_filename='phono3py_check_' + C + '.log')

    if C in halide_C:
        C_site_list = [C]
        A_site_list = halide_A
        B_site_list = halide_B
    if C in chalco_C:
        C_site_list = [C]
        A_site_list = chalco_A
        B_site_list = chalco_B

    system_counter = 0

    for a in A_site_list:
        for b in B_site_list:
            for c in C_site_list:
                system_counter += 1
                system_name = a + b + c
                try:
                    tf = tarfile.open(system_name + '_Pm3m.tar.gz')
                    tf.extractall()
                    os.chdir(system_name + '_Pm3m')
                except:
                    continue

                phono3py_folder = None
                if os.path.isfile("phono3py_2.tar.gz"):
                    phono3py_folder = "phono3py_2"
                elif os.path.isfile("phono3py.tar.gz"):
                    phono3py_folder = 'phono3py'

                if phono3py_folder is not None:
                    try:
                        tf = tarfile.open(phono3py_folder + '.tar.gz')
                        tf.extractall()
                    except:
                        pass

                    if os.path.exists('./' + phono3py_folder + '/fc3.hdf5'):
                        os.chdir('./' + phono3py_folder)

                        all_calculation_folders = glob.glob('ph-POSCAR-*')
                        all_convergence_status = []
                        for folder in all_calculation_folders:

                            os.chdir(folder)

                            calculator = Vasp()
                            try:
                                calculator.check_convergence()
                            except:
                                calculator.completed = False

                            if calculator.completed:
                                all_convergence_status.append(True)
                            else:
                                all_convergence_status.append(False)
                            os.chdir('..')

                        os.chdir('..')

                os.chdir(cwd)

                logger.info("system: " + system_name + ' ' + 'valid phono3py?: ' + str(all(all_convergence_status)))

                try:
                    shutil.rmtree(system_name + '_Pm3m')
                except:
                    pass
                try:
                    os.rmtree(system_name + '_Pm3m')
                except:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--C", type=str,
                        help="Anion in ABCs.")
    args = parser.parse_args()
    check_phono3py_completion(args.C)
