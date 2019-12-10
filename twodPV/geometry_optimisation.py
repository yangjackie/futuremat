import os
import copy
import logging
from myqueue.config import config

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
                                  'LREAL': False,
                                  'LVTOT': False,
                                  'LWAVE': False,
                                  'PREC': 'Normal',
                                  'SIGMA': 0.05,
                                  'ENCUT': 500,
                                  'EDIFF': '1e-05',
                                  'executable': 'vasp_std'}

default_bulk_optimisation_set = {key.lower(): value for key, value in _default_bulk_optimisation_set.items()}

def __update_core_info():
    try:
        ncpus=None
        f=open('node_info','r')
        for l in f.readlines():
            if 'normalbw' in l:
                ncpus=28
            elif 'normalsl' in l:
                ncpus=32
            else:
                ncpus=16
        default_bulk_optimisation_set.update({'NPAR':ncpus})
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

    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    logger.info("==========Full Structure Optimisation with VASP==========")

    if os.path.isfile('./CONTCAR') and (os.path.getsize('./CONTCAR') > 0):
        structure = VaspReader(input_location='./CONTCAR').read_POSCAR()
        logger.info("Restart optimisation from previous CONTCAR.")
    else:
        structure = VaspReader(input_location='./POSCAR').read_POSCAR()
        logger.info("Start new optimisation from POSCAR")

    # default_bulk_optimisation_set.update()
    # we need some mechanism to automatically update KPAR and NPAR values based on the queue type
    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()

    if vasp.self_consistency_error:
        # Spin polarisation calculations might be very difficult to converge.
        # For this case, we converge a non-spin polarisation calculation first and
        # then use the converged wavefunction to carry out spin-polarised optimisation
        logger.info(
            "Spin-polarised SCF convergence failed, try generate a non-spin-polarised wavefunction as starting guess")
        default_bulk_optimisation_set.update({'ISPIN': 1, 'NSW': 1, 'LWAVE': True, 'clean_after_success': False})
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


def default_two_d_optimisation():
    # Method to be called for optimising a single 2D slab, where the lattice parameters in the
    # xy-plane (parallel to the 2D material will be optimised) while keeping z-direction fixed.
    # this can be achieved by using a specific vasp executable.
    default_bulk_optimisation_set.update({'executable': 'vasp_std-tst-xy', 'MP_points': [6, 6, 6], 'idipol': 3})
    default_structural_optimisation()


def default_symmetry_preserving_optimisation():
    # optimise the unit cell parameters whilst preserving the space and point group symmetry of the starting
    # structure.
    default_bulk_optimisation_set.update({'ISIF': 7, 'MP_points': [4, 4, 1]})
    default_structural_optimisation()

