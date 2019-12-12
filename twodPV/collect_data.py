"""
This module contains methods to collect results from the calculation folder
and stored them in a proper object-relational-database. The io functionaities
from the ase packages will be used because it already has a good interface to handle
all the database operations. In this case, the data structures from ASE will
also be used rather than our internal ones. This also allows our data to be
more easily accessible by other users using the existing ase functionalities.
"""

import argparse
import os

from core.calculators.vasp import Vasp
from ase.io.vasp import *
from ase.db import connect

reference_atomic_energies={}

def _atom_dict(atoms):
    """
    Get a dictionary of number of each element in the chemical structure.
    """
    unique = list(set(atoms.get_chemical_symbols()))
    return {u:atoms.get_chemical_symbols().count(u) for u in unique}

def populate_db(db,atoms,kvp,data):
    row=None
    try:
        row = db.get(selection=[('uid','=',kvp['uid'])])
        #There is already something matching this row, we will update the key-value pairs and data before commit
        kvp.update(row.key_value_pairs)
        if data is not None:
            data.update(row.data)
        db.write(atoms,data=data,id=row.id,**kvp)
    except KeyError:
        db.write(atoms,data=data,**kvp)

def element_energy(db):
    print ("========== Collecting reference energies for constituting elements ===========")
    cwd = os.getcwd()
    element_directory = cwd+'/elements/'
    for dir in [o for o in os.listdir(element_directory) if os.path.isdir(os.path.join(element_directory,o))]:
        kvp={}
        data={}

        dir = os.path.join(element_directory,dir)
        uid = 'element_'+dir.split('/')[-1]

        os.chdir(dir)
        calculator = Vasp()
        calculator.check_convergence()
        if calculator.completed:
            atoms = [i for i in read_vasp_xml(index=-1)][-1] #just to be explicit that we want the very last one
            e = list(_atom_dict(atoms).keys())[-1]
            reference_atomic_energies[e] = atoms.get_calculator().get_potential_energy()/_atom_dict(atoms)[e]
            kvp['uid'] = uid
            populate_db(db,atoms,kvp,data)
        else:
            raise Exception("Vasp calculation incomplete in "+dir+". Please check!")
        os.chdir(cwd)

def collect(db):

    errors = []
    steps = [element_energy]
    for step in steps:
        try:
            step(db)
        except Exception as x:
            error = '{}: {}'.format(x.__class__.__name__, x)
            errors.append(error)
    return errors

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Collect data for c2db.db')
    parser.add_argument('-n', '--dry-run', action='store_true')
    args = parser.parse_args()

    # We use absolute path because of chdir below!
    dbname = os.path.join(os.getcwd(), '2dpv.db')
    if not args.dry_run:
        db = connect(dbname)
        print('Established a sqlite3 database object '+str(db))

    collect(db)
