from core.calculators.vasp import Vasp
from twodPV.calculators import default_bulk_optimisation_set, setup_logger, update_core_info, load_structure
from perovskite_screenings.halide_double_perovskites.calculators import *
import argparse, os, tarfile, shutil, glob
from core.models.element import *

def spin_polarised_optimisation_procedure():
    logger = setup_logger('relax.log')
    default_bulk_optimisation_set.update(
        {'ISIF': 3, 'Gamma_centered': True, 'NCORE': 28, 'ENCUT': 520, 'PREC': "ACCURATE", 'ispin': 2, 'IALGO': 38,
         'use_gw': True, 'clean_after_success': True, 'EDIFF': 1e-7})
    structure = load_structure(logger)

    _all_atom_label = [a.label for a in structure.all_atoms()]
    all_atom_label = []
    for a in _all_atom_label:
        if a not in all_atom_label:
            all_atom_label.append(a)
    print(all_atom_label)
   
    LDAUL=''
    LDAUU=''

    for label in all_atom_label:
        if label in U_corrections.keys():
           orbital = list(U_corrections[label].keys())[-1]
           LDAUL += ' ' + str(orbital_index[orbital])
           LDAUU += ' ' + str(U_corrections[label][orbital])
        else:
           LDAUL += ' -1'
           LDAUU += ' 0'

    GGA_U_options = {'LDAU': '.TRUE.', 'LDAUTYPE': 2, 'LDAUJ': '0 ' * len(all_atom_label), 'LDAUL': LDAUL,
            'LDAUU': LDAUU}
    default_bulk_optimisation_set.update(GGA_U_options)

    #default_bulk_optimisation_set.update({'nelect':275})

    try:
        os.remove("./WAVECAR")
        logger.info("Previous WAVECAR found, remove before start new optimisation.")
    except:
        pass

    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()

    logger.info("VASP terminated properly: " + str(vasp.completed))
    if not vasp.completed:
        logger.info("VASP did not completed properly, you might want to check it by hand.")

def dfpt_phonon_polarisation():
    logger = setup_logger('relax.log')
    pol_set =  {'ISPIN': 2, 'LWAVE': False, 'ENCUT': 520, 'PREC': 'Accurate', 'EDIFF': 1e-07,
                'ISYM': 0, 'LREAL': '.FALSE.', "NELM": 300, 'LCALCPOL':True, 'DIPOL':'0.5 0.5 0.5',
                'clean_after_success': True, 'Gamma_centered':True,'use_gw': True, 'MAGMOM': '4*0 1*0 4*0 23*0 7*0'}#,'NELECT':270}
    structure = VaspReader(input_location='./POSCAR').read_POSCAR()

    _all_atom_label = [a.label for a in structure.all_atoms()]
    all_atom_label = []
    for a in _all_atom_label:
        if a not in all_atom_label:
            all_atom_label.append(a)
    print(all_atom_label)

    LDAUL = ''
    LDAUU = ''

    for label in all_atom_label:
        if label in U_corrections.keys():
            orbital = list(U_corrections[label].keys())[-1]
            LDAUL += ' ' + str(orbital_index[orbital])
            LDAUU += ' ' + str(U_corrections[label][orbital])
        else:
            LDAUL += ' -1'
            LDAUU += ' 0'

    GGA_U_options = {'LDAU': '.TRUE.', 'LDAUTYPE': 2, 'LDAUJ': '0 ' * len(all_atom_label), 'LDAUL': LDAUL,
                     'LDAUU': LDAUU}
    pol_set.update(GGA_U_options)

    #pol_set.update({'nelect':275})

    logger.info("Start from supercell defined in POSCAR")
    vasp = Vasp(**pol_set)
    vasp.set_crystal(structure)
    vasp.execute()
    logger.info("VASP terminated properly: " + str(vasp.completed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='workflow control for double perovskite ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--opt", action='store_true', help='perform initial structural optimization')
    parser.add_argument("--pol", action='store_true', help='perform polarisation calculation')

    args = parser.parse_args()

    if args.opt:
        spin_polarised_optimisation_procedure()

    if args.pol:
        dfpt_phonon_polarisation()