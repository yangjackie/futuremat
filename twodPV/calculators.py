import os

from core.calculators.vasp import Vasp
from core.dao.vasp import *
from core.utils.loggings import setup_logger

# we set the default calculation to be spin-polarized.
_default_bulk_optimisation_set = {'ADDGRID': True,
                                  'AMIN': 0.01,
                                  'IALGO': 38,
                                  'ISMEAR': 0,
                                  'ISPIN': 2,
                                  'ISTART': 1,
                                  'ISIF': 3,
                                  'IBRION': 2,
                                  'NSW': 500,
                                  'ISYM': 0,
                                  'LCHARG': False,
                                  'LREAL': 'Auto',
                                  'LVTOT': False,
                                  'LWAVE': False,
                                  'PREC': 'Normal',
                                  'SIGMA': 0.05,
                                  'ENCUT': 500,
                                  'EDIFF': '1e-04',
                                  'executable': 'vasp_std'}

default_bulk_optimisation_set = {key.lower(): value for key, value in _default_bulk_optimisation_set.items()}


def __update_core_info():
    try:
        ncpus = None
        f = open('node_info', 'r')
        for l in f.readlines():
            if 'normalbw' in l:
                ncpus = 28
            elif 'normalsl' in l:
                ncpus = 32
            else:
                ncpus = 16
        default_bulk_optimisation_set.update({'NPAR': ncpus, 'NCORE': 3})
    except:
        pass


def default_structural_optimisation():
    """
    Perform a full geometry optimization on the structure in POSCAR stored in the current folder (supposed named as opt/).
    To submit this optimisation job using the command line argument from myqueue package, do

    mq submit twodPV.geometry_optimisation@default_structural_relaxation -R <resources> opt/

    Note that this method is implemented in such a way that an existing CONTCAR will be read first, if it can be found,
    otherwise, it will read in the POSCAR file. So restart mechansim is already built in. For example, if a VASP calculation
    is timeout in the opt/ folder with ID, running the following command

    mq resubmit -i ID

    should resubmit a job continuing the previous unfinished structural optimisation.
    """

    logger = setup_logger(output_filename='relax.log')
    __update_core_info()
    logger.info("==========Full Structure Optimisation with VASP==========")
    structure = __load_structure(logger)

    # default_bulk_optimisation_set.update({"MAGMOM": "5*0 11*0 16*4 48*0"})
    __default_spin_polarised_vasp_optimisation_procedure(logger, structure)


def __default_spin_polarised_vasp_optimisation_procedure(logger, structure):
    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()
    if vasp.self_consistency_error:
        # Spin polarisation calculations might be very difficult to converge.
        # For this case, we converge a non-spin polarisation calculation first and
        # then use the converged wavefunction to carry out spin-polarised optimisation
        logger.info(
            "Spin-polarised SCF convergence failed, try generate a non-spin-polarised wavefunction as starting guess")
        default_bulk_optimisation_set.update({'ISPIN': 1, 'NSW': 3, 'LWAVE': True, 'clean_after_success': False})
        vasp = Vasp(**default_bulk_optimisation_set)
        vasp.set_crystal(structure)
        vasp.execute()

        if os.path.isfile('./WAVECAR') and (os.path.getsize('./WAVECAR') > 0):
            logger.info("WAVECAR found")
            logger.info("Restart spin-polarised structure relaxation...")
            structure = VaspReader(input_location='./CONTCAR').read_POSCAR()
            default_bulk_optimisation_set.update({'ISPIN': 2, 'NSW': 500, 'LWAVE': False, 'clean_after_success': True})
            vasp = Vasp(**default_bulk_optimisation_set)
            vasp.set_crystal(structure)
            vasp.execute()

            logger.info("VASP terminated properly: " + str(vasp.completed))
            if not vasp.completed:
                logger.info("VASP did not completed properly, you might want to check it by hand.")


def __load_structure(logger):
    if os.path.isfile('./CONTCAR') and (os.path.getsize('./CONTCAR') > 0):
        structure = VaspReader(input_location='./CONTCAR').read_POSCAR()
        logger.info("Restart optimisation from previous CONTCAR.")
    else:
        structure = VaspReader(input_location='./POSCAR').read_POSCAR()
        logger.info("Start new optimisation from POSCAR")
    return structure


def GGA_U_structure_optimisation():
    logger = setup_logger(output_filename='relax.log')
    __update_core_info()
    logger.info("==========Full GGA+U Structure Optimisation with VASP==========")
    structure = __load_structure(logger)
    gga_u_options = __set_U_correction_dictionary(structure)
    default_bulk_optimisation_set.update(gga_u_options)

    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()


def GGA_U_high_spin_structure_optimisation():
    logger = setup_logger(output_filename='relax.log')
    __update_core_info()
    logger.info("==========Full GGA+U high spin Structure Optimisation with VASP==========")
    structure = __load_structure(logger)
    gga_u_options = __set_U_correction_dictionary(structure)
    default_bulk_optimisation_set.update(gga_u_options)

    magmom_options = __set_high_spin_magmom_dictionary(structure)
    default_bulk_optimisation_set.update(magmom_options)

    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()


def high_spin_structure_optimisation():
    logger = setup_logger(output_filename='relax.log')
    __update_core_info()
    logger.info("==========Full  high spin Structure Optimisation with VASP==========")
    structure = __load_structure(logger)

    magmom_options = __set_high_spin_magmom_dictionary(structure)
    default_bulk_optimisation_set.update(magmom_options)

    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()


def __set_U_correction_dictionary(structure):
    from core.models.element import U_corrections, orbital_index
    LDAUL = ''
    LDAUU = ''
    labels = [x.label for x in structure.all_atoms(unique=True, sort=True)]
    unique_labels = []
    for l in labels:
        if l not in unique_labels:
            unique_labels.append(l)
    for label in unique_labels:
        if label in U_corrections.keys():
            orbital = list(U_corrections[label].keys())[-1]
            LDAUL += ' ' + str(orbital_index[orbital])
            LDAUU += ' ' + str(U_corrections[label][orbital])
        else:
            LDAUL += ' -1'
            LDAUU += ' 0'
    GGA_U_options = {'LDAU': '.TRUE.', 'LDAUTYPE': 2, 'LDAUJ': '0 ' * len(unique_labels), 'LDAUL': LDAUL,
                     'LDAUU': LDAUU}
    return GGA_U_options


def __set_high_spin_magmom_dictionary(structure):
    # this sets transition metal ions into its highest spin state for performing calculations in an initial high-spin FM states
    MAGMOM = ''
    labels = [x.label for x in structure.all_atoms(unique=True, sort=True)]
    unique_labels = []
    for l in labels:
        if l not in unique_labels:
            unique_labels.append(l)
    from core.models.element import high_spin_states
    for l in unique_labels:
        if l in high_spin_states.keys():
            MAGMOM += str(structure.all_atoms_count_dictionaries()[l]) + '*' + str(high_spin_states[l]) + ' '
        else:
            MAGMOM += str(structure.all_atoms_count_dictionaries()[l]) + '*0 '
    return {"MAGMOM": MAGMOM}


def default_xy_strained_optimisation():
    default_bulk_optimisation_set.update({'executable': 'vasp_std-z'})
    default_structural_optimisation()


def default_spin_unpolarised_xy_strained_optimisation():
    default_bulk_optimisation_set.update({'executable': 'vasp_std-z'})
    spin_unpolarised_optimization()


def default_highspin_xy_strained_optimisation():
    default_bulk_optimisation_set.update({'executable': 'vasp_std-z'})
    high_spin_structure_optimisation()


def default_GGA_U_highspin_xy_strained_optimisation():
    default_bulk_optimisation_set.update({'executable': 'vasp_std-z'})
    GGA_U_high_spin_structure_optimisation()


def spin_unpolarised_optimization():
    """
    Perform geometry optimization without spin polarisation. It is always helpful to converge an initial
    structure without spin polarization before further refined with a spin polarization calculations.
    This makes the SCF converge faster and less prone to cause the structural from collapsing due to problematic
    forces from unconverged SCF.
    """
    logger = setup_logger(output_filename='relax.log')

    __update_core_info()
    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    logger.info("==========Full Structure Optimisation with VASP==========")

    structure = __load_structure(logger)

    logger.info("Perform an initial spin-non-polarised calculations to help convergence")
    default_bulk_optimisation_set.update({'ispin': 1, 'nsw': 500, 'ENCUT': 300, 'EDIFF': '1e-04'})
    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()

    logger.info("VASP terminated?: " + str(vasp.completed))


def default_two_d_optimisation():
    # Method to be called for optimising a single 2D slab, where the lattice parameters in the
    # xy-plane (parallel to the 2D material will be optimised) while keeping z-direction fixed.
    # this can be achieved by using a specific vasp executable.
    default_bulk_optimisation_set.update(
        {'executable': 'vasp_std-xy', 'MP_points': [4, 4, 1], 'idipol': 3})
    default_structural_optimisation()


def spin_unploarised_two_d_optimisation():
    default_bulk_optimisation_set.update(
        {'executable': 'vasp_std-xy', 'MP_points': [4, 4, 1], 'idipol': 3})
    spin_unpolarised_optimization()


def default_symmetry_preserving_optimisation():
    # optimise the unit cell parameters whilst preserving the space and point group symmetry of the starting
    # structure.
    default_bulk_optimisation_set.update({'ISIF': 7, 'MP_points': [6, 6, 6]})
    default_structural_optimisation()


def default_bulk_phonon_G_calculation():
    return __default_G_phonon(two_d=False)


def default_twod_phonon_G_calculation():
    return __default_G_phonon(two_d=True)


def __default_G_phonon(two_d=False):
    logger = setup_logger(output_filename='phonon.log')

    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    if two_d:
        kpoints = [4, 4, 1]
    else:
        kpoints = [6, 6, 6]

    default_bulk_optimisation_set.update(
        {'PREC': 'Accurate',
         'ISPIN': 1,
         'NSW': 0,
         'LWAVE': True,
         'ISYM': 0,
         'MP_points': kpoints,
         'clean_after_success': False})

    if two_d:
        default_bulk_optimisation_set.update({'idipol': 3})

    __G_phonon()
    default_bulk_optimisation_set.update(
        {'ISPIN': 2,
         'LWAVE': False,
         'NSW': 1,
         'PREC': 'Accurate',
         'EDIFF': 1e-05,
         'IBRION': 8,
         'ISIF': 0,
         'ISYM': 0,
         'LREAL': 'Auto',
         'POTIM': 0.01,
         'clean_after_success': True})
    __G_phonon()


def __G_phonon():
    __update_core_info()
    logger.info("==========Gamma point phonon calculation with VASP==========")
    structure = VaspReader(input_location='./POSCAR').read_POSCAR()
    logger.info("Start from supercell defined in POSCAR")
    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()
    logger.info("VASP terminated properly: " + str(vasp.completed))
