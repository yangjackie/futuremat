import warnings

from pymatgen.core.structure import IStructure as pymatstructure
import settings

from core.internal.builders.crystal import map_pymatgen_IStructure_to_crystal
from core.models.element import pbe_pp_choices

default_ionic_optimisation_set = {
    'SYSTEM': 'entdecker',
    'PREC': 'Normal',
    'EDIFF': 1E-5,
    'EDIFFG': -0.01,
    'IBRION': 2,
    'POTIM': 0.1,
    'ISIF': 0,
    'NSW': 800,
    'IALGO': 38,
    'ISYM': 0,
    'ADDGRID': '.TRUE.',
    'ISMEAR': 0,
    'SIGMA': 0.05,
    'LREAL': 'AUTO',
    'LWAVE': '.FALSE.',
    'LCHARG': '.FALSE.',
    'LVTOT': '.FALSE.',
    'LMAXMIX': 6,
    'AMIN': 0.01
}

default_static_calculation_set = {
    'SYSTEM': 'entdecker',
    'PREC': 'Normal',
    'INIWAV': 1,
    'ENCUT': 600,
    'EDIFF': 1E-5,
    'EDIFFG': -0.01,
    'IBRION': -1,
    'ISIF': 0,
    'NSW': 0,
    'IALGO': 38,
    'ISYM': 0,
    'ADDGRID': '.TRUE.',
    'ISMEAR': 0,
    'SIGMA': 0.05,
    'LREAL': 'AUTO',
    'LWAVE': '.FALSE.',
    'LCHARG': '.TRUE.',
    'LMAXMIX': 6,
    'NPAR': 28,
    'LORBIT': 11,
    'NELM': 500,
    'AMIN': 0.01,
    'LVTOT': '.TRUE.',
    'ISPIN': 2
}

class VaspWriter(object):

    def write_INCAR(self, filename='INCAR',
                    default_options=default_ionic_optimisation_set,
                    **kwargs):
        default_options.update(kwargs)

        incar = open(filename, 'w')
        incar.write("SYSTEM=" + str(default_options['SYSTEM']) + '\n')

        keylist = default_options.keys()
        keylist = list(sorted(keylist))

        #if ('KPAR' not in [k.upper() for k in keylist]) or ('NPAR' not in [k.upper() for k in keylist]):
        #    warnings.warn("KPAR or NPAR not set for specific architecture, calculation might be very slow!")

        for key in keylist:
            if key not in ['frame', 'kwargs', 'self', 'SYSTEM', 'filename']:
                incar.write(str(key) + '=' + str(default_options[key]) + '\n')

        incar.close()

    def write_potcar(self, crystal, filename='POTCAR', sort=False, unique=True):
        if settings.functional is not None:
            if settings.functional.lower() == 'pbe':
                pass
            else:
                raise NotImplementedError("Current implementation will only concatenate PAW pseudopotential for PBE!")
        else:
            raise Exception("Please specify in the default configuration where to find VASP PAW pseudopotential files!")

        if isinstance(crystal, pymatstructure):
            crystal=map_pymatgen_IStructure_to_crystal(crystal)

        all_atoms = crystal.all_atoms(sorted=sort,unqiue=unique)
        all_atom_label = [i.clean_label for i in all_atoms]

        potcars = [settings.vasp_pp_directory + '/' + pbe_pp_choices[e] + '/POTCAR' for e in
                   all_atom_label]

        with open(filename, 'w') as outfile:
            for fn in potcars:
                with open(fn) as infile:
                    for line in infile:
                        if ('Zr' in fn) and ('VRHFIN' in line): line = '   VRHFIN =Zr: 4s4p5s4d\n'
                        outfile.write(line)
