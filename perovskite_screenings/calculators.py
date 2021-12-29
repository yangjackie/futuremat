from phonopy.interface.calculator import read_crystal_structure
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation

from core.calculators.vasp import Vasp, VaspReader
from core.internal.builders.crystal import build_supercell
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
    all_zeros = True
    for i in analyzer.magmoms:
        magmom_string += '1*' + str(i) + ' '
        if i != 0:
            all_zeros = False
    return magmom_string, all_zeros


def phonopy_workflow(force_rerun=False):
    from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
    from phonopy.interface.vasp import parse_set_of_forces
    from phonopy.file_IO import write_force_constants_to_hdf5,write_FORCE_SETS,parse_disp_yaml,write_disp_yaml
    from phonopy import Phonopy
    #from phonopy.cui import create_FORCE_SETS

    mp_points = [2, 2, 2]
    gamma_centered = True
    force_no_spin = False
    use_default_encut = False
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    ialgo = 38
    use_gw = False #only for vanadium compounds
    ncore = 28

    if mp_points != [1, 1, 1]:
        gamma_only = False
    else:
        gamma_only = True

    phonopy_set = {'prec': 'Accurate', 'ibrion': -1, 'encut': 500, 'ediff': '1e-08', 'ismear': 0, 'ialgo': ialgo,
                   'lreal': False, 'lwave': False, 'lcharg': False, 'sigma': 0.05, 'isym': 0, 'ncore': ncore,
                   'ismear': 0, 'MP_points': mp_points, 'nelm': 250, 'lreal': False, 'use_gw': use_gw,
                   'Gamma_centered': gamma_centered, 'LMAXMIX': 6}
    # 'amix': 0.2, 'amix_mag':0.8, 'bmix':0.0001, 'bmix_mag':0.0001}

    logger = setup_logger(output_filename='phonopy.log')
    cwd = os.getcwd()
    vasp = Vasp()
    vasp.check_convergence()
    if not vasp.completed:
        logger.exception("Initial structure optimimization failed, will not proceed!")
        raise Exception("Initial structure optimimization failed, will not proceed!")


    if os.path.isfile('./force_constants_222.hdf5') and os.path.isfile('./FORCE_SETS_222'):
        logger.info("previous phonopy calculations completed, check if previous calculations all converged")

        if os.path.isfile('phonon_2_2_2.tar.gz'):
            tf = tarfile.open('phonon_2_2_2.tar.gz')
            tf.extractall()

            all_converged = [False, False, False]
            if os.path.isdir('phonon_2_2_2'):
                os.chdir('phonon_2_2_2')
                for counter in [1,2,3]:
                    if os.path.isdir('ph-POSCAR-00'+str(counter)):
                        os.chdir('ph-POSCAR-00'+str(counter))
                        vasp = Vasp()
                        vasp.check_convergence()
                        if vasp.completed:
                            all_converged[counter-1] = True
                        os.chdir('..')
                    logger.info("structure " + str(counter) + '/3, VASP calculation completed successfully? -:' + str(all_converged[counter - 1]))
                os.chdir("..")
                try:
                    shutil.rmtree('phonon_2_2_2')
                except:
                    pass
                try:
                    os.rmtree('phonon_2_2_2')
                except:
                    pass

            if (all_converged[0] is True) and (all_converged[1] is True) and (all_converged[2] is True):
                return


#    if os.path.isfile('phonopy.log'):
#        success = []
#        for l in open('phonopy.log', 'r').readlines():
#            if 'VASP calculation completed successfully? ' in l:
#                if l.split()[-1] == "True":
#                    success.append(True)
#                elif l.split()[-1] == 'False':
#                    success.append(False)
#        if len(success) != 0:
#            if not all(success):
#                if not force_rerun:
#                    logger.exception("Encounter convergence problems with VASP before, will not attempt again.")
#                    raise Exception("Encounter convergence problems with VASP before, will not attempt again.")
#                else:
#                    logger.info("try to rerun all VASP calculations")
#                    print("try to rerun all VASP calculations with RMM for ialgo")

    try:
        unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')
    except:
        raise Exception("No CONTCAR!")

    if not force_rerun:
        if not os.path.exists('./phonon_2_2_2'):
            os.mkdir('./phonon_2_2_2')
        elif os.path.isfile("./phonon_2_2_2.tar.gz"):
            tf = tarfile.open("./phonon_2_2_2.tar.gz")
            tf.extractall()
    else:
        try:
            shutil.rmtree('./phonon_2_2_2')
        except:
            pass
        try:
            os.rmtree('./phonon_2_2_2')
        except:
            pass
        os.mkdir('./phonon_2_2_2')

        try:
            os.remove("./phonon_2_2_2.tar.gz")
        except:
            pass

    os.chdir('./phonon_2_2_2')

    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    if not force_rerun:
        phonon.generate_displacements(distance=0.005)
    else:
        phonon.generate_displacements(distance=0.0005)


    supercells = phonon.supercells_with_displacements

    write_disp_yaml(displacements=phonon.displacements,supercell=phonon.supercell,filename='disp.yaml')

    logger.info(
        "PHONOPY - generate (2x2x2) displaced supercells, total number of configurations " + str(len(supercells)))

    completed = []
    force_files = []

    calculate_next = True
    for i, sc in enumerate(supercells):
        i = i+1
        proceed = True
        if calculate_next:
            dir = 'ph-POSCAR-00' + str(i)
            force_files.append('./' + dir + '/vasprun.xml')
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
                        len(supercells)) + " previous calculation converged.")
                else:

                    if not force_rerun:
                        proceed = False
                        calculate_next = False
                        logger.info('At least one finite displaced configuration cannot converge, quit the rest')

            if proceed:
                logger.info("Configuration " + str(i) + '/' + str(len(supercells)) + " proceed VASP calculation")
                write_crystal_structure('POSCAR', sc, interface_mode='vasp')
                structure = load_structure(logger)
                structure.gamma_only = gamma_only
                #phonopy_set['magmom'], all_zeros = magmom_string_builder(structure)
                phonopy_set['ispin'] = 1

                #if not force_rerun:

                try:
                    vasp = Vasp(**phonopy_set)
                    vasp.set_crystal(structure)
                    vasp.execute()
                except:
                    pass

                if vasp.completed is not True:
                    calculate_next = False
                    proceed = False
                    logger.info('At least one finite displaced configuration cannot converge, quit the rest')

                logger.info("Configuration " + str(i) + '/' + str(
                    len(supercells)) + "VASP terminated?: " + str(vasp.completed))
                print("Configuration " + str(i) + '/' + str(
                    len(supercells)) + "VASP terminated?: " + str(vasp.completed))
            completed.append(vasp.completed)
            os.chdir("..")

    if all(completed) and len(completed) == len(supercells):
        logger.info("All finite displacement calculations completed, extract force constants")
        set_of_forces = parse_set_of_forces(structure.total_num_atoms(), force_files)
        phonon.set_forces(sets_of_forces=set_of_forces)
        phonon.produce_force_constants()
        write_force_constants_to_hdf5(phonon.force_constants, filename='force_constants.hdf5')

        displacements = parse_disp_yaml(filename='disp.yaml')

        num_atoms = displacements['natom']
        for forces, disp in zip(set_of_forces, displacements['first_atoms']):
            disp['forces'] = forces
        write_FORCE_SETS(displacements, filename='FORCE_SETS')

        if os.path.isfile('FORCE_SETS'):
            shutil.copy('./force_constants.hdf5', '../force_constants_222.hdf5')

        if os.path.isfile('FORCE_SETS'):
            shutil.copy('./FORCE_SETS', '../FORCE_SETS_222')

    os.chdir('..')
    output_filename = 'phonon_2_2_2.tar.gz'
    source_dir = './phonon_2_2_2'
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    try:
        shutil.rmtree('./phonon_2_2_2')
    except:
        pass
    try:
        os.rmtree('./phonon_2_2_2')
    except:
        pass

    os.chdir(cwd)


def molecular_dynamics_workflow(force_rerun=False):
    logger = setup_logger(output_filename='molecular_dynamics.log')
    cwd = os.getcwd()

    if (os.path.isfile('./force_constants_222.hdf5') is not True) and (os.path.isfile('./FORCE_SETS_222') is not True):
        logger.info('No converged phonon results, will not proceed in this case.')

    if os.path.isfile('./force_constants_222.hdf5') and os.path.isfile('./FORCE_SETS_222'):
        logger.info("previous phonopy calculations completed, check if previous calculations all converged")

        if os.path.isfile('phonon_2_2_2.tar.gz'):
            tf = tarfile.open('phonon_2_2_2.tar.gz')
            tf.extractall()

            all_converged = [False, False, False]
            if os.path.isdir('phonon_2_2_2'):
                os.chdir('phonon_2_2_2')
                for counter in [1,2,3]:
                    if os.path.isdir('ph-POSCAR-00'+str(counter)):
                        os.chdir('ph-POSCAR-00'+str(counter))
                        vasp = Vasp()
                        vasp.check_convergence()
                        if vasp.completed:
                            all_converged[counter-1] = True
                        os.chdir('..')
                    logger.info("structure " + str(counter) + '/3, VASP calculation completed successfully? -:' + str(all_converged[counter - 1]))
                os.chdir("..")
                try:
                    shutil.rmtree('phonon_2_2_2')
                except:
                    pass
                try:
                    os.rmtree('phonon_2_2_2')
                except:
                    pass

            if (all_converged[0] is not True) or (all_converged[1] is not True) or (all_converged[2] is not True):
                logger.info('phonon calculation not all converged, will not proceed to run molecualr dynamics')
                return

    logger.info('Valid phonon calculations, proceed to run molecualr dynamics')
    logger.info(
        "Setting up the room temperature molecular dynamics calculations, check if we have previous phonon data")


    structure = __load_supercell_structure()
    structure.gamma_only = False

    equilibrium_set = {'prec': 'Accurate','algo': 'Normal', 'lreal': 'AUTO', 'ismear': 0, 'isym': 0, 'ibrion': 0, 'maxmix': 40,
                       'lmaxmix': 6, 'ncore': 28, 'nelmin': 4, 'nsw': 100, 'smass': -1, 'isif': 1, 'tebeg': 10,
                       'teend': 300, 'potim': 2, 'nblock': 10, 'nwrite': 0, 'lcharg': False, 'lwave': False,
                       'iwavpr': 11, 'encut': 500, 'Gamma_centered': True, 'MP_points': [1, 1, 1], 'use_gw': True,
                       'write_poscar': True}

    production_set = {'prec': 'Accurate','algo': 'Normal', 'lreal': 'AUTO', 'ismear': 0, 'isym': 0, 'ibrion': 0, 'maxmix': 40,
                      'lmaxmix': 6, 'ncore': 28, 'nelmin': 4, 'nsw': 800, 'isif': 1, 'tebeg': 300,
                      'teend': 300, 'potim': 2, 'nblock': 1, 'nwrite': 0, 'lcharg': False, 'lwave': False, 'iwavpr': 11,
                      'encut': 500, 'andersen_prob': 0.5, 'mdalgo': 1, 'Gamma_centered': True, 'MP_points': [1, 1, 1],
                      'use_gw': True, 'write_poscar': False}


    equilibrium_set['ispin'] = 1
    production_set['ispin'] = 1

    if not os.path.exists('./MD_2_2_2'):
        os.mkdir('./MD_2_2_2')
    os.chdir('./MD_2_2_2')

    try:
        os.remove('./INCAR')
    except:
        pass
    try:
        os.remove('./KPOINTS')
    except:
        pass

    run_equilibration = True
    run_production = True

    if os.path.exists('./CONTCAR_equ') and os.path.exists('./OSZICAR_equ'):
        logger.info("Previous equilibration run output exists, check how many cylces have been run...")
        oszicar = open('./OSZICAR_equ', 'r')
        cycles_ran = 0
        for l in oszicar.readlines():
            if 'T=' in l:
                cycles_ran += 1
        if cycles_ran == equilibrium_set['nsw']:
            logger.info('Previous equilibrium run completed, will skip running equilibration MD')
            shutil.copy('CONTCAR_equ', 'POSCAR')
            run_equilibration = False
        else:
            logger.info('Previous equilibrium run not completed, will rerun equilibration MD')
            run_equilibration = True

    if run_equilibration:
        try:
            logger.info("start equilibrium run ...")
            vasp = Vasp(**equilibrium_set)
            vasp.set_crystal(structure)
            vasp.execute()
        except:
            pass

        dav_error = False
        if not vasp.completed:
            logfile = open('vasp.log', 'r')
            for f in logfile.readlines():
                if 'Error EDDDAV' in f:
                    dav_error = True
            if dav_error:
                equilibrium_set['algo'] = 'VeryFast'
                production_set['algo'] = 'VeryFast'
            try:
                os.remove('./INCAR')
                logger.info("start equilibrium run ...")
                vasp = Vasp(**equilibrium_set)
                vasp.set_crystal(structure)
                vasp.execute()
            except:
                pass

        # error catching for md run needs to be implemented
        shutil.copy('INCAR', 'INCAR_equ')
        shutil.copy('POSCAR', 'POSCAR_equ')
        shutil.copy('CONTCAR', 'CONTCAR_equ')
        shutil.copy('CONTCAR', 'POSCAR')
        shutil.copy('vasprun.xml', 'vasprun_equ.xml')
        shutil.copy('OUTCAR', 'OUTCAR_equ')
        shutil.copy('OSZICAR', 'OSZICAR_equ')
        shutil.copy('vasp.log', 'vasp_equ.log')

    try:
        os.remove('./INCAR')
    except:
        pass

    if run_equilibration:
        run_production = True
    else:
        has_andersen = False
        if os.path.exists('./OUTCAR_prod'):
            logger.info("Check if previous production run has applied andersen thermostat")
            outcar = open('./OUTCAR_prod', 'r')
            for l in outcar.readlines():
                if 'ANDERSEN_PROB =' in l:
                    prob = float(l.split()[-1])
                    if prob == 0.5:
                        has_andersen = True

        if os.path.exists('./CONTCAR_prod') and os.path.exists('./OSZICAR_prod'):
            logger.info("Previous production run output exists, check how many cylces have been run...")
            oszicar = open('./OSZICAR_prod', 'r')
            cycles_ran = 0
            for l in oszicar.readlines():
                if 'T=' in l:
                    cycles_ran += 1
            if cycles_ran >= production_set['nsw']:
                logger.info('Previous production run completed, will skip running production MD')
                run_production = False
            else:
                logger.info('Previous production run not completed, will rerun production MD')
                shutil.copy('CONTCAR_equ', 'POSCAR')
                run_production = True

        if not run_production:
            if not has_andersen:
                run_production = True
                logger.info("Previous run has not applied thermal stats, rerun production MD")

    if run_production:
        try:
            logger.info("start production run")
            vasp = Vasp(**production_set)
            vasp.set_crystal(structure)
            vasp.execute()
        except:
            pass

        if not vasp.completed:
            dav_error = False
            logfile = open('vasp.log', 'r')
            for f in logfile.readlines():
                if 'Error EDDDAV' in f:
                    dav_error = True
            if dav_error:
                production_set['algo'] = 'VeryFast'
            try:
                os.remove('./INCAR')
                logger.info("start equilibrium run ...")
                vasp = Vasp(**production_set)
                vasp.set_crystal(structure)
                vasp.execute()
            except:
                pass

        shutil.copy('POSCAR', 'POSCAR_prod')
        shutil.copy('CONTCAR', 'CONTCAR_prod')
        shutil.copy('vasprun.xml', 'vasprun_prod.xml')
        shutil.copy('OUTCAR', 'OUTCAR_prod')
        shutil.copy('OSZICAR', 'OSZICAR_prod')

        shutil.copy('vasprun.xml', '../vasprun_md_2_2_2.xml')

    os.chdir(cwd)


def check_phonon_run_settings():
    spin_polarized = False
    if os.path.exists('./phonon'):
        os.chdir('./phonon')
        f = open('./ph-POSCAR-0/vasp.log', 'r')
        for line in f.readlines():
            if 'F=' in line:
                if 'mag' not in line:
                    spin_polarized = False
                if 'mag' in line:
                    magnetization = abs(float(line.split()[-1]))
                    if magnetization >= 0.05:
                        spin_polarized = True
                    else:
                        spin_polarized = False
        os.chdir('..')
    return spin_polarized


def __load_supercell_structure():
    from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure
    from phonopy import Phonopy
    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    unitcell, _ = read_crystal_structure('./CONTCAR', interface_mode='vasp')
    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    write_crystal_structure('POSCAR_super', phonon.supercell, interface_mode='vasp')
    supercell = VaspReader(input_location='./POSCAR_super').read_POSCAR()
    os.remove('./POSCAR_super')
    return supercell

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='workflow control for double perovskite ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--opt", action='store_true', help='perform initial structural optimization')
    parser.add_argument("--phonopy", action='store_true', help='run phonopy calculations')
    parser.add_argument("--force_rerun", action='store_true', help='force rerun  calculations')
    parser.add_argument("--MD", action='store_true', help='run MD calculations')

    args = parser.parse_args()

    if args.opt:
        default_symmetry_preserving_optimisation()

    if args.phonopy:
        phonopy_workflow(force_rerun=args.force_rerun)

    if args.MD:
        molecular_dynamics_workflow(force_rerun=args.force_rerun)
