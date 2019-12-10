import argparse
import os

from ase.db import *

from core.calculators.vasp import Vasp
from core.dao.vasp import VaspReader

def element_energy():
    print ("========== Collecting energies for constituting elements ===========")
    cwd = os.getcwd()
    element_directory = cwd+'/elements/'
    for dir in [o for o in os.listdir(element_directory) if os.path.isdir(os.path.join(element_directory,o))]:
        element = dir
        dir = os.path.join(element_directory,dir)
        os.chdir(dir)
        calculator = Vasp()
        calculator.check_convergence()
        if calculator.completed:
            tot_e = VaspReader(input_location='./OSZICAR').get_free_energies_from_oszicar()[-1]
            crystal = VaspReader(input_location='./POSCAR').read_POSCAR()
            elemental_energy = tot_e / crystal.all_atoms_count_dictionaries()[element]
            print(element,elemental_energy)
        else:
            raise Exception("Vasp calculation incomplete in "+dir+". Please check!")
        os.chdir(cwd)

def collect():
    errors = []
    steps = [element_energy]
    for step in steps:
        try:
            step()
        except Exception as x:
            error = '{}: {}'.format(x.__class__.__name__, x)
            errors.append(error)
    return errors


if __name__=="__main__":
    collect()
