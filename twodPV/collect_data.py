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
import glob

from core.calculators.vasp import Vasp
from ase.io.vasp import *
from ase.db import connect

from twodPV.bulk_library import A_site_list, B_site_list, C_site_list

reference_atomic_energies = {}


def _atom_dict(atoms):
    """
    Get a dictionary of number of each element in the chemical structure.
    """
    unique = list(set(atoms.get_chemical_symbols()))
    return {u: atoms.get_chemical_symbols().count(u) for u in unique}


def populate_db(db, atoms, kvp, data):
    row = None
    try:
        row = db.get(selection=[('uid', '=', kvp['uid'])])
        # There is already something matching this row, we will update the key-value pairs and data before commit
        kvp.update(row.key_value_pairs)
        if data is not None:
            print("Update...")
            data.update(row.data)
        db.write(atoms, data=data, id=row.id, **kvp)
    except KeyError:
        try:
            db.write(atoms, data=data, **kvp)
        except Exception as e:
            print(e)


def formation_energy(atoms):
    fe = atoms.get_calculator().get_potential_energy()
    for k in _atom_dict(atoms).keys():
        fe = fe - _atom_dict(atoms)[k] * reference_atomic_energies[k]
    return fe / atoms.get_number_of_atoms()


def element_energy(db):
    print("========== Collecting reference energies for constituting elements ===========")
    cwd = os.getcwd()
    element_directory = cwd + '/elements/'
    for dir in [o for o in os.listdir(element_directory) if os.path.isdir(os.path.join(element_directory, o))]:
        kvp = {}
        data = {}

        dir = os.path.join(element_directory, dir)
        uid = 'element_' + str(dir.split('/')[-1])

        os.chdir(dir)
        calculator = Vasp()
        calculator.check_convergence()
        if calculator.completed:
            atoms = [i for i in read_vasp_xml(index=-1)][-1]  # just to be explicit that we want the very last one
            e = list(_atom_dict(atoms).keys())[-1]
            reference_atomic_energies[e] = atoms.get_calculator().get_potential_energy() / _atom_dict(atoms)[e]
            kvp['uid'] = uid
            kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
            populate_db(db, atoms, kvp, data)
        else:
            raise Exception("Vasp calculation incomplete in " + dir + ". Please check!")
        os.chdir(cwd)


def pm3m_formation_energy(db):
    print("========== Collecting formation energies for bulk perovskites in Pm3m symmetry ===========")
    cwd = os.getcwd()
    base_dir = cwd + '/relax_Pm3m/'
    kvp = {}
    data = {}
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    uid = system_name + '3_pm3m'

                    dir = os.path.join(base_dir, system_name + "_Pm3m")
                    os.chdir(dir)
                    calculator = Vasp()
                    calculator.check_convergence()
                    if calculator.completed:
                        atoms = [k for k in read_vasp_xml(index=-1)][-1]
                        kvp['uid'] = uid
                        kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                        kvp['formation_energy'] = formation_energy(atoms)
                        print("System " + uid + " formation energy :" + str(kvp['formation_energy']) + ' eV')
                        populate_db(db, atoms, kvp, data)
                    os.chdir(cwd)


def randomised_structure_formation_energy(db):
    print("========== Collecting formation energies for distorted perovskites  ===========")
    cwd = os.getcwd()
    base_dir = cwd + '/relax_randomized/'
    kvp = {}
    data = {}
    counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    all_rand_for_this = glob.glob(base_dir + '/' + system_name + '*rand*')
                    for r in all_rand_for_this:
                        uid = system_name + '3_random_str_' + str(r.split("_")[-1])
                        os.chdir(r)
                        try:
                            calculator = Vasp()
                            calculator.check_convergence()
                            if calculator.completed:
                                atoms = [k for k in read_vasp_xml(index=-1)][-1]
                                kvp['uid'] = uid
                                kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                                kvp['formation_energy'] = formation_energy(atoms)
                                counter += 1
                                print(str(counter) + '\t' + "System " + uid + " formation energy :" + str(
                                    kvp['formation_energy']) + ' eV')
                                populate_db(db, atoms, kvp, data)
                        except:
                            continue  # if job failed we dont worry too much about it
                        os.chdir(cwd)


def two_d_formation_energies(db, orientation='100', termination='AO2', thicknesess=[3, 5, 7, 9]):
    cwd = os.getcwd()
    base_dir = cwd + '/slab_' + str(orientation) + '_' + str(termination) + '_small/'
    kvp = {}
    data = {}
    counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:
                    system_name = a + b + c
                    for thick in thicknesess:
                        work_dir = base_dir + system_name + "_" + str(thick)
                        os.chdir(work_dir)

                        uid = system_name + '3_' + str(orientation) + "_" + str(termination) + "_" + str(thick)
                        try:
                            calculator = Vasp()
                            calculator.check_convergence()
                            if calculator.completed:
                                atoms = [k for k in read_vasp_xml(index=-1)][-1]
                                kvp['uid'] = uid
                                kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                                kvp['formation_energy'] = formation_energy(atoms)
                                kvp['orientation'] = '['+str(orientation)+']'
                                kvp['termination'] = termination
                                kvp['nlayer'] = thick
                                counter += 1
                                print(str(counter) + '\t' + "System " + uid + " formation energy :" + str(
                                    kvp['formation_energy']) + ' eV')
                                populate_db(db, atoms, kvp, data)
                        except:
                            print("-------------> failed in " + str(os.getcwd()))
                            continue
                        os.chdir(cwd)


def __two_d_100_AO_energies(db):
    two_d_formation_energies(db, orientation='100', termination='AO')


def __two_d_100_BO2_energies(db):
    two_d_formation_energies(db, orientation='100', termination='BO2')

def __two_d_111_B_energies(db):
    two_d_formation_energies(db, orientation='111', termination='B')

def __two_d_111_AO3_energies(db):
    two_d_formation_energies(db, orientation='111', termination='AO3')

def __two_d_110_O2_energies(db):
    two_d_formation_energies(db, orientation='110', termination='O2')

def __two_d_110_ABO_energies(db):
    two_d_formation_energies(db, orientation='110', termination='ABO')

def collect(db):
    errors = []
    steps = [element_energy, #do not skip this step, always need this to calculate formation energy on-the-fly
             #pm3m_formation_energy,
             #randomised_structure_formation_energy,
             #__two_d_100_BO2_energies,
             #__two_d_100_AO_energies,
             __two_d_111_B_energies]
             #__two_d_111_AO3_energies,
             #__two_d_110_O2_energies,
             #__two_d_110_ABO_energies]
    for step in steps:
        try:
            step(db)
        except Exception as x:
            print(x)
            error = '{}: {}'.format(x.__class__.__name__, x)
            errors.append(error)
    return errors


if __name__ == "__main__":
    # We use absolute path because of chdir below!
    dbname = os.path.join(os.getcwd(), '2dpv.db')
    db = connect(dbname)
    print('Established a sqlite3 database object ' + str(db))
    collect(db)
