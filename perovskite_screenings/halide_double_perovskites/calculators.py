from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
from core.calculators.vasp import Vasp
from twodPV.calculators import default_bulk_optimisation_set, setup_logger, update_core_info, load_structure
import argparse, os, tarfile, shutil


def default_symmetry_preserving_optimisation():
    # optimise the unit cell parameters whilst preserving the space and point group symmetry of the starting
    # structure.
    default_bulk_optimisation_set.update(
        {'ISIF': 7, 'Gamma_centered': True, 'NCORE': 28, 'ENCUT': 520, 'PREC': "ACCURATE", 'ispin': 2, 'IALGO': 38,
         'use_gw': True})
    structural_optimization_with_initial_magmom()


def structural_optimization_with_initial_magmom(retried=None, gamma_only=False):
    """
    Perform geometry optimization without spin polarisation. It is always helpful to converge an initial
    structure without spin polarization before further refined with a spin polarization calculations.
    This makes the SCF converge faster and less prone to cause the structural from collapsing due to problematic
    forces from unconverged SCF.
    """
    MAX_RETRY = 3
    if retried is None:
        retried = 0

    logger = setup_logger(output_filename='relax.log')

    update_core_info()
    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    structure = load_structure(logger)
    structure.gamma_only = gamma_only

    default_bulk_optimisation_set['magmom'] = magmom_string_builder(structure)

    logger.info("incar options" + str(default_bulk_optimisation_set))

    try:
        vasp = Vasp(**default_bulk_optimisation_set)
        vasp.set_crystal(structure)
        vasp.execute()
    except:
        vasp.completed = False
        pass

    logger.info("VASP terminated?: " + str(vasp.completed))

    # if (vasp.completed is not True) and (retried<MAX_RETRY):
    #    retried+=1
    #    structural_optimization_with_initial_magmom(retried=retried)


def magmom_string_builder(structure):
    from core.internal.builders.crystal import map_to_pymatgen_Structure
    analyzer = CollinearMagneticStructureAnalyzer(structure=map_to_pymatgen_Structure(structure), make_primitive=False,
                                                  overwrite_magmom_mode='replace_all')
    magmom_string = ""
    for i in analyzer.magmoms:
        magmom_string += '1*' + str(i) + ' '
    return magmom_string


phonopy_set = {'prec': 'Accurate', 'ibrion': -1, 'encut': 520, 'ediff': '1e-08', 'ismear': 0, 'ialgo': 38,
               'lreal': False, 'lwave': False, 'lcharg': False, 'sigma': 0.05, 'isym': 0, 'ncore': 28,
               'MP_points': [1, 1, 1], 'nelm':100}


def phonopy_workflow(gamma_only=True):
    from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
    from phonopy.interface.vasp import parse_set_of_forces
    from phonopy.file_IO import write_force_constants_to_hdf5
    from phonopy import Phonopy

    logger = setup_logger(output_filename='phonopy.log')
    cwd = os.getcwd()
    vasp = Vasp()
    vasp.check_convergence()
    if not vasp.completed:
        logger.exception("Initial structure optimimization failed, will not proceed!")
        raise Exception("Initial structure optimimization failed, will not proceed!")

    if os.path.isfile('./force_constants.hdf5'):
        logger.info("previous phonopy calculations completed, will not rerun it again")
        return

    unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')

    if not os.path.exists('./phonon'):
        os.mkdir('./phonon')
    elif os.path.isfile("./phonon.tar.gz"):
        tf = tarfile.open("./phonon.tar.gz")
        tf.extractall()

    os.chdir('./phonon')

    phonon = Phonopy(unitcell, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    phonon.generate_displacements()
    supercells = phonon.supercells_with_displacements
    logger.info(
        "PHONOPY - generate (2x2x2) displaced supercells, total number of configurations " + str(len(supercells)))

    completed = []
    force_files = []
    for i, sc in enumerate(supercells):
        dir = 'ph-POSCAR-' + str(i)
        force_files.append('./'+dir+'/vasprun.xml')
        if not os.path.exists(dir):
            os.mkdir(dir)
        os.chdir(dir)
        proceed = True
        if os.path.isfile('./OUTCAR'):
            logger.info("Configuration " + str(i) + '/' + str(
                len(supercells)) + " previous calculation exists, check convergence")
            vasp = Vasp()
            vasp.check_convergence()
            if vasp.completed:
                proceed = False
                logger.info("Configuration " + str(i) + '/' + str(
                    len(supercells)) + " previous calculation converged, skip")

        if proceed:
            logger.info("Configuration " + str(i) + '/' + str(
                len(supercells)) + " proceed VASP calculation")
            write_crystal_structure('POSCAR', sc, interface_mode='vasp')
            structure = load_structure(logger)
            structure.gamma_only = gamma_only
            phonopy_set['magmom'] = magmom_string_builder(structure)
            phonopy_set['ispin'] = 2
            try:
                vasp = Vasp(**phonopy_set)
                vasp.set_crystal(structure)
                vasp.execute()
            except:
                vasp.completed = False
                pass

            logger.info("Configuration " + str(i) + '/' + str(
                len(supercells)) + "VASP terminated?: " + str(vasp.completed))
        completed.append(vasp.completed)
        os.chdir("..")

    if all(completed):
        logger.info("All finite displacement calculations completed, extract force constants")
        set_of_forces = parse_set_of_forces(structure.total_num_atoms(),force_files)
        phonon.set_forces(sets_of_forces=set_of_forces)
        phonon.produce_force_constants()
        write_force_constants_to_hdf5(phonon.force_constants,filename='force_constants.hdf5')

        if os.path.isfile('force_constants.hdf5'):
            shutil.copy('./force_constants.hdf5','../force_constants.hdf5')

    with tarfile.open('phonon.tar.gz', mode='w:gz') as archive:
        archive.add('phonon.tar.gz', recursive=True)

    try:
        shutil.rmtree('./phonon')
    except:
        pass
    try:
        os.rmtree('./phonon')
    except:
        pass

    os.chdir(cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='workflow control for double perovskite ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--opt", action='store_true', help='perform initial structural optimization')
    parser.add_argument("--phonopy", action='store_true', help='run phonopy calculations')
    args = parser.parse_args()

    if args.opt:
        default_symmetry_preserving_optimisation()

    if args.phonopy:
        phonopy_workflow()
