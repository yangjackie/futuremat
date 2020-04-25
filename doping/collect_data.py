import os, glob, zipfile

from ase.io.vasp import *
from ase.db import connect

from core.dao.vasp import VaspReader
from core.internal.builders.crystal import map_to_ase_atoms
from twodPV.collect_data import populate_db

def get_total_energies(db, dir=None):
    all_zips = glob.glob(dir + "/*.zip")
    for zip in all_zips:
        kvp = {}
        data = {}
        kvp['uid'] = zip.replace(".zip",'').replace('/','_')
        archive = zipfile.ZipFile(zip)

        atoms = None
        total_energy = None

        for name in archive.namelist():
            if 'CONTCAR' in name:
                contcar = archive.read(name)
                contcar_reader = VaspReader(file_content=str(contcar).split('\\n'))
                crystal = contcar_reader.read_POSCAR()
                atoms = map_to_ase_atoms(crystal)
            if 'OSZICAR' in name:
                oszicar = archive.read(name)
                oszicar_reader = VaspReader(file_content=str(oszicar).split('\\n'))
                total_energy = oszicar_reader.get_free_energies_from_oszicar()[-1]
                kvp['total_energy'] = total_energy
        if (atoms is not None) and (total_energy is not None):
            print(kvp['uid'], crystal.all_atoms_count_dictionaries(), total_energy)
            populate_db(db, atoms, kvp, data)

def pure_total_energies(db):
    get_total_energies(db, dir='pure')


def CsPbSnCl3_energies(db):
    get_total_energies(db, dir='mixed_CsPbSnCl3')


def CsPbSnBr3_energies(db):
    get_total_energies(db, dir='mixed_CsPbSnBr3')

def CsPbSnI3_energies(db):
    get_total_energies(db, dir='mixed_CsPbSnI3')

def collect(db):
    errors = []
    steps = [pure_total_energies,
             CsPbSnCl3_energies,
             CsPbSnBr3_energies,
             CsPbSnI3_energies]

    for step in steps:
        try:
            step(db)
        except Exception as x:
            print(x)
            error = '{}: {}'.format(x.__class__.__name__, x)
            errors.append(error)
    return errors


if __name__ == "__main__":
    dbname = os.path.join(os.getcwd(), 'doping.db')
    db = connect(dbname)
    print('Established a sqlite3 database object ' + str(db))
    collect(db)
