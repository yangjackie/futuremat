import argparse
import os
from core.utils.loggings import setup_logger
from core.calculators.pymatgen.vasp import Vasp

from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.core.structure import Structure


default_optimisation_settings_for_phonon = {
    "SYSTEM": "Structure optimisation for phonons",
    "ISTART": 0,
    "ICHARG": 2,
    # Electronic
    "ENCUT": 520,  # or higher depending on your POTCAR
    "PREC": "Accurate",
    "EDIFF": 1e-8,  # very tight for phonons
    "ISMEAR": 0,  # Gaussian smearing for insulators/semiconductors
    "SIGMA": 0.05,
    "LREAL": "Auto",
    "GGA": "PS",  # use PBEsol functional for better phonon properties
    # Ionic relaxation
    "IBRION": 2,  # conjugate-gradient optimization
    "ISIF": 3,  # relax ions and cell shape/volume
    "EDIFFG": -1e-4,  # force convergence criterion (0.0001 eV/Å)
    "NSW": 200,  # max ionic steps
    # Phonon stability prep
    "LASPH": True,  # important for d-orbital materials
    "ADDGRID": True,  # reduce Pulay errors
    # Output control
    "LWAVE": False,
    "LCHARG": False,
    # Own setups
    "use_gw": True,
    "kpoint_mode": "grid",
    "kppa": 2000,
}


def execute_vasp_calculation(
    structure: Structure, params: dict, job_type: str, directory: str = None
):
    """
    Perform VASP calculation for the given structure with specified parameters.
    """
    logger = setup_logger(output_filename=job_type + ".log")

    if directory is None:
        directory = job_type

    if not os.path.exists("./" + directory):
        os.mkdir("./" + directory)
    os.chdir("./" + directory)

    vasp_calculator = Vasp(**params)
    vasp_calculator.set_structure(structure)
    vasp_calculator.execute()

    os.chdir("../")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Workflows for calculating and analysing the bandgap renormalisation due to electron-phonon interactions."
    )
    parser.add_argument(
        "-str",
        "--structure_file",
        type=str,
        default="POSCAR",
        help="Path to the VASP structural input file containing the atomic structure to investigate on.",
    )
    parser.add_argument(
        "-temp",
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in Kelvin for the calculation (default: 300 K).",
    )

    parser.add_argument(
        "-opt",
        "--optimise_structure",
        action="store_true",
        help="Perform structural optimisation before phonon calculations.",
    )
    args = parser.parse_args()

    if args.optimise_structure:
        execute_vasp_calculation(
            structure=Structure.from_file(args.structure_file),
            params=default_optimisation_settings_for_phonon,
            job_type="structure_optimisation",
        )
