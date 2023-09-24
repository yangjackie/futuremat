from core.calculators.vasp import Vasp, VaspReader, VaspWriter
from core.utils.loggings import setup_logger

import os

"""
Module containing necessary functions to automate running ab-initio molecular dynamics with on-the-fly machine-learning 
forcefields (MLFF) for determining the lattice thermal conductivities of perovskites.
"""

equilibrium_set = {'prec': 'Accurate','algo': 'Normal', 'lreal': 'AUTO', 'ismear': 0, 'isym': 0, 'ibrion': 0, 'maxmix': 40,
                    'lmaxmix': 6, 'ncore': 28, 'nelmin': 4, 'nsw': 200, 'smass': -1, 'isif': 1, 'tebeg': 10,
                    'teend': 300, 'potim': 2, 'nblock': 10, 'nwrite': 0, 'lcharg': False, 'lwave': False,
                    'iwavpr': 11, 'encut': 300, 'Gamma_centered': True, 'MP_points': [1,1,1], 'use_gw': True,
                    'write_poscar': True, 'EDIFF':1e-7, 'GGA': 'PS'}

def molecular_dynamics_workflow(supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]):
    """
    Perform pure ab initio molecular dynamics (AIMD) to equilibrate the structure to the targeted temperature (that is
    set in the 'teend' key in the 'equilibrium_set' dictionary.
    :param supercell_matrix: Size of the supercell to be built for running the MD
    :return:
    """
    logger = setup_logger(output_filename='equilibration.log')
    cwd = os.getcwd()

    #get the optimised crystal structure from the CONTCAR and build a supercell from it.
    from ..calculators import load_supercell_structure
    structure = load_supercell_structure()
    structure.gamma_only = True



