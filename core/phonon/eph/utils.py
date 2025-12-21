from core.utils.loggings import setup_logger

import argparse


def setup_logger_local(args):
    if args.optimise_structure:
        job_type = "structure_optimisation"
    elif args.calculate_born_charges:
        job_type = "born_charges"
    elif args.calculate_phonons:
        job_type = "finite_displacement_phonons"
    elif args.calculate_band_structure:
        job_type = "band_structure"
    elif args.calculate_unfold_band_structure:
        job_type = "unfolded_band_structure"
    else:
        job_type = None
    if job_type is not None:
        logger = setup_logger(output_filename=job_type + ".log")
    else:
        logger = setup_logger(output_filename=None)
    return logger


def eph_coupling_module_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Workflows for calculating and analysing the bandgap renormalisation due to electron-phonon interactions.")
    parser.add_argument("-str", "--structure_file", type=str, default="POSCAR", 
                        help="Path to the VASP structural input file containing the atomic structure to investigate on.")
    parser.add_argument("-uc_str", "--unit_cell_structure_file", type=str, default=None,
                        help="Path to the VASP structural input file containing the unit cell structure.")

    parser.add_argument("-temp", "--temperature", type=float, default=300.0,
                        help="Temperature in Kelvin for the calculation (default: 300 K).")
    parser.add_argument("-opt", "--optimise_structure", action="store_true",
                        help="Perform structural optimisation before phonon calculations.")
    parser.add_argument("-born", "--calculate_born_charges", action="store_true", 
                        help="Calculate Born effective charges and dielectric tensor.")
    parser.add_argument("-phonon", "--calculate_phonons", action="store_true",
                        help="Calculate phonon properties using Phonopy with ASE wrapper.")
    parser.add_argument("-band", "--calculate_band_structure", action="store_true",
                        help="Calculate electronic band structure.")
    parser.add_argument("-band_unfold", "--calculate_unfold_band_structure", action="store_true",
                        help="Calculate unfolded band structure for supercell.")
    parser.add_argument("-nkpts", "--num_kpoints", type=int, default=20,
                        help="Number of k-points for band structure calculation.")
    parser.add_argument("-gpu", "--use_gpu", action="store_true", 
                        help="Use GPU for VASP calculation.")

    return parser.parse_args()
