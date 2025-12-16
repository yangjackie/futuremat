import argparse
import os
from core.data.vasp_settings import *
from core.utils.loggings import setup_logger
from core.calculators.pymatgen.vasp import Vasp
from core.phonon.phonopy_worker import Born

from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Vasprun

import shutil
import logging

logger = logging.getLogger("futuremat.core.calculators.pymatgen.vasp")


def execute_vasp_calculation(
    structure: Structure,
    params: dict,
    job_type: str,
    directory: str = None,
    force_rerun: bool = False,
):
    """
    Perform VASP calculation for the given structure with specified parameters.
    """

    logger.info(f"Starting VASP calculation for job type: {job_type}")

    if directory is None:
        directory = job_type

    vasp_calculator = Vasp(force_rerun=force_rerun, directory=directory, **params)
    vasp_calculator.structure = structure
    vasp_calculator.execute()

    post_processor(job_type=job_type, directory=directory)


def post_processor(job_type: str, directory: str = None):
    if job_type == "structure_optimisation":
        # copy the CONTCAR back to the main directory
        logger.info(f"Copying CONTCAR from {directory} to current directory.")
        shutil.copy(f"{directory}/CONTCAR", "./CONTCAR")
    elif job_type == "born_charges":
        logger.info(
            "extract born effective charges from outcar and write out using phonopy API"
        )
        Born.write_born_file_with_cmd(directory=directory)
    else:
        return


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
    parser.add_argument(
        "-born",
        "--calculate_born_charges",
        action="store_true",
        help="Calculate Born effective charges and dielectric tensor.",
    )
    parser.add_argument(
        "-phonon",
        "--calculate_phonons",
        action="store_true",
        help="Calculate phonon properties using Phonopy with ASE wrapper.",
    )
    parser.add_argument(
        "-gpu", "--use_gpu", action="store_true", help="Use GPU for VASP calculation."
    )
    args = parser.parse_args()

    if args.optimise_structure:
        job_type = "structure_optimisation"
    elif args.calculate_born_charges:
        job_type = "born_charges"
    elif args.calculate_phonons:
        job_type = "finite_displacement_phonons"
    logger = setup_logger(output_filename=job_type + ".log")

    if args.optimise_structure:
        DEFAULT_STRUCTURE_OPTIMISATION_SET_FOR_PHONONS["gpu_run"] = args.use_gpu
        execute_vasp_calculation(
            structure=Structure.from_file(args.structure_file),
            params=DEFAULT_STRUCTURE_OPTIMISATION_SET_FOR_PHONONS,
            job_type="structure_optimisation",
        )
    elif args.calculate_born_charges:
        DEFAULT_BORN_EFFECTIVE_CHARGES_SET_FOR_PHONONS["gpu_run"] = args.use_gpu
        execute_vasp_calculation(
            structure=Structure.from_file(args.structure_file),
            params=DEFAULT_BORN_EFFECTIVE_CHARGES_SET_FOR_PHONONS,
            job_type="born_charges",
        )
    if args.calculate_phonons:
        from core.phonon.phonopy_worker import PhonopyWorker

        DEFAULT_STATIC_SET_FOR_PHONONS["gpu_run"] = args.use_gpu
        vasp_calculator = Vasp(
            force_rerun=False,
            **DEFAULT_STATIC_SET_FOR_PHONONS,
        )

        phonopy_worker = PhonopyWorker(
            structure=Structure.from_file(args.structure_file),
            calculator=vasp_calculator,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        )
        directory = "finite_displacement_phonons"
        if not os.path.exists(directory):
            os.mkdir(directory)
        os.chdir(directory)
        phonopy_worker.generate_force_constants(save_fc=True)
        os.chdir("..")
