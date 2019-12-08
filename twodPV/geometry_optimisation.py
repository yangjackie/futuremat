import os

from core.calculators.vasp import Vasp
from core.dao.vasp import *
import settings

_default_bulk_optimisation_set = {'ADDGRID': True,
                                 'AMIN': 0.01,
                                 'IALGO': 38,
                                 'ISMEAR': 0,
                                 'ISPIN': 2,
                                 'ISTART': 1,
                                 'ISIF': 3,
                                 'IBRION': 2,
                                 'NSW' : 500,
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
    structure = VaspReader(input_location='./POSCAR').read_POSCAR()
    default_bulk_optimisation_set.update()
    #we need some mechanism to automatically update KPAR and NPAR values based on the queue type
    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()
