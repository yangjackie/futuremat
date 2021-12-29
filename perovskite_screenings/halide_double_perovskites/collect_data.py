import os,glob,tarfile,shutil
from ase.db import connect
from ase.io.vasp import *
from core.utils.loggings import setup_logger
from core.calculators.vasp import Vasp
from perovskite_screenings.collect_data import _atom_dict, populate_db

reference_atomic_energies={}

def formation_energy(atoms):
    fe = atoms.get_calculator().get_potential_energy()
    # print(fe,_atom_dict(atoms),reference_atomic_energies)
    for k in _atom_dict(atoms).keys():
        fe = fe - _atom_dict(atoms)[k] * reference_atomic_energies[k]
        # print(k,reference_atomic_energies[k],fe)
    return fe / atoms.get_number_of_atoms()


def element_energy(db):
    logger.info("========== Collecting reference energies for constituting elements ===========")
    cwd = os.getcwd()
    os.chdir('./elements')

    all_element_zips = glob.glob('*.tar.gz')

    for zip in all_element_zips:
        kvp = {}
        data = {}
        element = zip.replace('.tar.gz','')
        uid = 'element_' + str(element)
        logger.info(uid)
        tf = tarfile.open(element + '.tar.gz')
        tf.extractall()
        os.chdir(element)

        calculator = Vasp()
        calculator.check_convergence()
        if not calculator.completed:
            logger.info(uid, 'failed')
        atoms = [i for i in read_vasp_xml(index=-1)][-1]
        e = list(_atom_dict(atoms).keys())[-1]
        reference_atomic_energies[element] = atoms.get_calculator().get_potential_energy() / _atom_dict(atoms)[e]
        kvp['uid'] = uid
        kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
        logger.info(uid + ' ' + str(reference_atomic_energies[element]) + ' eV/atom')
        populate_db(db, atoms, kvp, data)
        os.chdir("..")
        shutil.rmtree(element)
        try:
            os.rmtree(element)
        except:
            pass
    os.chdir('..')

def all_data(db, collect_formation_energy=False):
    s = SystemIterator()
    systemIterator = iter(s)
    system_counter = 0
    for system in systemIterator:
        system_counter += 1
        kvp = {}
        data = {}

        cwd = os.getcwd()
        try:
            __open_system_tar(system)
        except:
            logger.error(system+' - tar ball not working, skip')
            print(system+' - tar ball not working, skip')
            os.chdir(cwd)
            continue

        logger.info("Working on system number: " + str(system_counter) + ' name:'+system)
        print("System number: " + str(system_counter))

        if collect_formation_energy:
            try:
                calculator = Vasp()
                calculator.check_convergence()
                if calculator.completed:
                    atoms = [a for a in read_vasp_xml(index=-1)][-1]
                    kvp['uid'] = system
                    kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                    kvp['formation_energy'] = formation_energy(atoms)
                    populate_db(db, atoms, kvp, data)
                    logger.info(system + ' formation energy: ' + str(kvp['formation_energy']) + ' eV/atom')
                    print(system + ' formation energy: ' + str(kvp['formation_energy']) + ' eV/atom')
                else:
                    print(system + ' NOT CONVERGED!')
                    pass
            except:
                logger.info(system + ' formation energy: ' + str('NaN'))

        os.chdir(cwd)
        __clear_system(system)


def __open_system_tar(system):
    system_tar = system + '.tar.gz'
    tf = tarfile.open(system_tar)
    tf.extractall()
    os.chdir(system)

def __clear_system(system):
    try:
        shutil.rmtree(system)
    except:
        pass
    try:
        os.rmtree(system)
    except:
        pass


class SystemIterator():

    def __init__(self):
        self.all_systems = glob.glob('dpv_*.tar.gz')
        self.all_systems = list(sorted(self.all_systems))
        self.all_systems = [x.replace('.tar.gz','') for x in self.all_systems]
        print('all initialised')

    def __iter__(self):
        self.counter=0
        return self

    def __next__(self):
        if self.counter < len(self.all_systems):
            x = self.all_systems[self.counter]
            self.counter += 1
            return x
        else:
            raise StopIteration


def collect(db):
    errors = []
    steps = [element_energy,all_data]

    for step in steps:
        try:
            step(db)
        except Exception as x:
            logger.error(x)
            error = '{}: {}'.format(x.__class__.__name__, x)
            errors.append(error)
    return errors

if __name__ == "__main__":
    dbname = os.path.join(os.getcwd(), 'double_halide_pv.db')
    logger = setup_logger(output_filename='data_collector.log')
    db = connect(dbname)
    collect(db)