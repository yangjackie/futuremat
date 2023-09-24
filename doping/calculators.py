from core.calculators.vasp import Vasp
from twodPV.calculators import default_bulk_optimisation_set, setup_logger, update_core_info, load_structure
from perovskite_screenings.halide_double_perovskites.calculators import *
import argparse, os, tarfile, shutil, glob
from core.models.element import *


def spin_polarised_optimisation_procedure(noU=False, assisted=True):
    logger = setup_logger('relax.log')

    if os.path.exists('./vasp.log'):
        f = open('./vasp.log', 'r')
        for l in f.readlines():
            if 'reached required accuracy - stopping structural energy minimisation' in l:
                logger.info("Previous optimisation completed, quit")
                return

    if assisted:
        # assist the convergence of spin-polarisation calculations by first converging some spin unpolarised calculations
        logger.info('assist convergence with a non-spin polarised calculation.')
        if os.path.exists('./INCAR'):
            os.remove('./INCAR')
        structure = load_structure(logger)
        default_bulk_optimisation_set.update({'ISIF': 3, 'Gamma_centered': True, 'ENCUT': 520, 'PREC': "ACCURATE",
                                              'ISPIN': 1, 'NSW': 5, 'LWAVE': True, 'LCHARG': True,
                                              'clean_after_success': False,
                                              'IALGO': 38, 'use_gw': True, 'EDIFF': 1e-7, 'gpu_run': True})

        if not noU:
            _all_atom_label = [a.label for a in structure.all_atoms()]
            all_atom_label = []
            for a in _all_atom_label:
                if a not in all_atom_label:
                    all_atom_label.append(a)

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
            default_bulk_optimisation_set.update(GGA_U_options)

        vasp = Vasp(**default_bulk_optimisation_set)
        vasp.set_crystal(structure)
        vasp.execute()

    if os.path.exists('./INCAR'):
        os.remove('./INCAR')

    default_bulk_optimisation_set.update(
        {'ISIF': 3, 'Gamma_centered': True, 'ENCUT': 520, 'PREC': "ACCURATE", 'ISPIN': 2, 'IALGO': 38,
         'use_gw': True, 'clean_after_success': True, 'EDIFF': 1e-7, 'gpu_run': True, 'NSW': 100})

    structure = load_structure(logger)
    default_bulk_optimisation_set['magmom'], _ = magmom_string_builder(structure)

    _all_atom_label = [a.label for a in structure.all_atoms()]
    all_atom_label = []
    for a in _all_atom_label:
        if a not in all_atom_label:
            all_atom_label.append(a)
    print(all_atom_label)

    if not noU:
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
        default_bulk_optimisation_set.update(GGA_U_options)

    # default_bulk_optimisation_set.update({'nelect':275})

    # try:
    #    os.remove("./WAVECAR")
    #    logger.info("Previous WAVECAR found, remove before start new optimisation.")
    # except:
    #    pass

    vasp = Vasp(**default_bulk_optimisation_set)
    vasp.set_crystal(structure)
    vasp.execute()

    logger.info("VASP terminated properly: " + str(vasp.completed))
    if not vasp.completed:
        logger.info("VASP did not completed properly, you might want to check it by hand.")
    return


def dfpt_phonon_polarisation():
    logger = setup_logger('relax.log')

    # pol_set = {'ISPIN': 1, 'LWAVE': False, 'ENCUT': 520, 'PREC': 'Accurate', 'EDIFF': 1e-07,
    #            'ISYM': 0, 'LREAL': '.FALSE.', "NELM": 300,  'ISMEAR': -5,
    #            'gpu_run': False, 'LWAVE': True, 'LCHARG': True, 'clean_after_success': False,
    #            'Gamma_centered': True, 'use_gw': True}
    #
    # structure = VaspReader(input_location='./POSCAR').read_POSCAR()
    #
    # _all_atom_label = [a.label for a in structure.all_atoms()]
    # all_atom_label = []
    # for a in _all_atom_label:
    #     if a not in all_atom_label:
    #         all_atom_label.append(a)
    # print(all_atom_label)
    #
    # LDAUL = ''
    # LDAUU = ''
    #
    # for label in all_atom_label:
    #     if label in U_corrections.keys():
    #         orbital = list(U_corrections[label].keys())[-1]
    #         LDAUL += ' ' + str(orbital_index[orbital])
    #         LDAUU += ' ' + str(U_corrections[label][orbital])
    #     else:
    #         LDAUL += ' -1'
    #         LDAUU += ' 0'
    #
    # GGA_U_options = {'LDAU': '.TRUE.', 'LDAUTYPE': 2, 'LDAUJ': '0 ' * len(all_atom_label), 'LDAUL': LDAUL,
    #                  'LDAUU': LDAUU}
    # pol_set.update(GGA_U_options)
    #
    # logger.info("Start from supercell defined in POSCAR")
    # vasp = Vasp(**pol_set)
    # vasp.set_crystal(structure)
    # vasp.execute()

    if os.path.exists('./INCAR'):
        os.remove('./INCAR')

    pol_set = {'ISPIN': 2, 'LWAVE': False, 'ENCUT': 520, 'PREC': 'Accurate', 'EDIFF': 1e-07,
               'ISYM': 0, 'LREAL': 'AUTO', "NELM": 300, 'LCALCPOL': True, 'DIPOL': '0.5 0.5 0.5', 'ISMEAR': -5,
               'gpu_run': False,
               'clean_after_success': True, 'Gamma_centered': True, 'use_gw': True,
               'ADDGRID': True}  # $, 'MAGMOM': '4*0 7*4 1*4 4*0 23*0'} #,'NELECT':270}

    pol_set['ISMEAR'] = 0
    pol_set['SIGMA'] = 0.05
    pol_set['AMIN'] = 0.01

    del pol_set['DIPOL']
    del pol_set['LCALCPOL']

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

    # pol_set.update({'nelect':275})

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
    parser.add_argument("--noU", action='store_true', help='do not apply U correction')
    parser.add_argument('--assisted', action='store_true',
                        help='assisting convergence with a non spin polarised calculation')

    args = parser.parse_args()

    if args.opt:
        spin_polarised_optimisation_procedure(noU=args.noU, assisted=args.assisted)

    if args.pol:
        dfpt_phonon_polarisation()
