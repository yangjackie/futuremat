import os
import copy

from core.calculators.vasp import Vasp
from core.dao.vasp import *

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
                                  'ENCUT': 500}

default_bulk_optimisation_set = {key.lower(): value for key, value in _default_bulk_optimisation_set.items()}


def full_structural_relax():
    """
    Perform a full geometry optimization on the structure in POSCAR stored in the current folder (supposed named as opt/).
    To submit this optimisation job using the command line argument from myqueue package, do

    mq submit twodPV.geometry_optimisation@full_structural_relax -R <resources> opt/
    """
    structure = VaspReader(input_location='./POSCAR').read_POSCAR()
    # default_bulk_optimisation_set.update()
    # we need some mechanism to automatically update KPAR and NPAR values based on the queue type
    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()

    if vasp.self_consistency_error:
        # Spin polarisation calculations might be very difficult to converge.
        # For this case, we converge a non-spin polarisation calculation first and
        # then use the converged wavefunction to carry out spin-polarised optimisation
        default_bulk_optimisation_set.update({'ISPIN': 1, 'NSW': 5, 'LWAVE': True, 'clean_after_success': False})
        vasp1 = Vasp(**default_bulk_optimisation_set)
        vasp1.set_crystal(structure)
        vasp1.execute()

        if os.path.isfile('./WAVECAR') and (os.path.getsize('./WAVECAR') > 0):
            structure = VaspReader(input_location='./CONTCAR').read_POSCAR()
            default_bulk_optimisation_set.update({'ISPIN': 2, 'NSW': 500, 'LWAVE': False, 'clean_after_success': True})
            vasp2 = Vasp(**default_bulk_optimisation_set)
            vasp2.set_crystal(structure)
            vasp2.execute()
