import argparse
import os
from core.data.vasp_settings import *
from core.utils.loggings import setup_logger
from core.calculators.pymatgen.vasp import Vasp
from core.phonon.phonopy_worker import Born

from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Vasprun

import shutil, copy
import logging

logger = logging.getLogger("futuremat.core.calculators.pymatgen.vasp")


def execute_vasp_calculation(
    structure: Structure,
    params: dict,
    job_type: str,
    directory: str = None,
    force_rerun: bool = False,
    chgcar_file: str = None,
):
    """
    Perform VASP calculation for the given structure with specified parameters.
    """

    logger.info(f"Starting VASP calculation for job type: {job_type}")

    if directory is None:
        directory = job_type
    if chgcar_file is not None:
        logger.info("Copying CHGCAR from SCF calculation...")
        if not os.path.exists(directory):
            os.mkdir(directory)
        shutil.copy(chgcar_file, f"{directory}/CHGCAR")

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
        logger.info("extract born effective charges from outcar and write out using phonopy API")
        Born.write_born_file_with_cmd(directory=directory)
    else:
        return


def band_structure_workflow(logger, structure_file: str = None, use_gpu: bool = False, scf_nkpts: int = 8000, band_nkpts: int = 20):

    logger.info("Band structure calculation routine...")
    logger.info("Starting a SCF calculation to obtain charge density...")

    ELECTRONIC_STRUCTURE_SET = copy.deepcopy(DEFAULT_STATIC_SET_FOR_PHONONS)

    ELECTRONIC_STRUCTURE_SET["gpu_run"] = use_gpu
    ELECTRONIC_STRUCTURE_SET["kpoint_mode"] = "grid"
    ELECTRONIC_STRUCTURE_SET["kppa"] = scf_nkpts  # denser k-point grid for SCF
    ELECTRONIC_STRUCTURE_SET["LCHARG"] = True  # write out CHGCAR after SCF
    ELECTRONIC_STRUCTURE_SET["clean_after_success"] = False  # keep files for the next NSCF run
    ELECTRONIC_STRUCTURE_SET["ICHARG"] = 2
    ELECTRONIC_STRUCTURE_SET["NSW"] = 0  # no ionic steps

    execute_vasp_calculation(
        structure=Structure.from_file(structure_file),
        params=ELECTRONIC_STRUCTURE_SET,
        job_type="scf_for_band_structure",
        force_rerun=False,
    )

    logger.info("Starting a non-SCF calculation for band structure...")
    ELECTRONIC_STRUCTURE_SET["kpoint_mode"] = "lines"
    ELECTRONIC_STRUCTURE_SET["kppa_band"] = band_nkpts  # number of k-points along each band path
    ELECTRONIC_STRUCTURE_SET["ICHARG"] = 11  # read charge density from CHGCAR
    ELECTRONIC_STRUCTURE_SET["LCHARG"] = False  # no need to write out CHGCAR again
    ELECTRONIC_STRUCTURE_SET["clean_after_success"] = True  # clean up

    execute_vasp_calculation(
        structure=Structure.from_file(structure_file),
        params=ELECTRONIC_STRUCTURE_SET,
        job_type="band_structure",
        force_rerun=False,
        chgcar_file="./scf_for_band_structure/CHGCAR",
    )


def get_args():
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
    parser.add_argument("-nkpts", "--num_kpoints", type=int, default=20,
                        help="Number of k-points for band structure calculation.")
    parser.add_argument("-gpu", "--use_gpu", action="store_true", 
                        help="Use GPU for VASP calculation.")
    
    parser.add_argument("-ppbs", "--plot_phonon_band_structure", action="store_true",
                        help="Plot phonon band structure from force constants and POSCAR in the specified directory")
    parser.add_argument("-nac", "--non-analytical_correction", action="store_true",
                        help="Apply non-analytical correction for phonon calculations.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.optimise_structure:
        job_type = "structure_optimisation"
    elif args.calculate_born_charges:
        job_type = "born_charges"
    elif args.calculate_phonons:
        job_type = "finite_displacement_phonons"
    elif args.calculate_band_structure:
        job_type = "band_structure"
    else:
        job_type = None
    if job_type is not None:
        logger = setup_logger(output_filename=job_type + ".log")

    if args.optimise_structure:
        DEFAULT_STRUCTURE_OPTIMISATION_SET_FOR_PHONONS["gpu_run"] = args.use_gpu
        execute_vasp_calculation(
            structure=Structure.from_file(args.structure_file),
            params=DEFAULT_STRUCTURE_OPTIMISATION_SET_FOR_PHONONS,
            job_type="structure_optimisation",
            force_rerun=False,
        )
    elif args.calculate_born_charges:
        DEFAULT_BORN_EFFECTIVE_CHARGES_SET_FOR_PHONONS["gpu_run"] = args.use_gpu
        execute_vasp_calculation(
            structure=Structure.from_file(args.structure_file),
            params=DEFAULT_BORN_EFFECTIVE_CHARGES_SET_FOR_PHONONS,
            job_type="born_charges",
            force_rerun=False,
        )
    elif args.calculate_phonons:
        from core.phonon.phonopy_worker import PhonopyWorker

        DEFAULT_STATIC_SET_FOR_PHONONS["gpu_run"] = args.use_gpu
        vasp_calculator = Vasp(
            force_rerun=False,
            **DEFAULT_STATIC_SET_FOR_PHONONS,
        )
        structure = Structure.from_file(args.structure_file)
        phonopy_worker = PhonopyWorker(
            structure=structure,
            calculator=vasp_calculator,
            supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],  # this need to be made more flexible
        )
        directory = "finite_displacement_phonons"
        if not os.path.exists(directory):
            os.mkdir(directory)
        os.chdir(directory)
        structure.to(filename="POSCAR")
        from phonopy.interface.vasp import write_vasp

        write_vasp("SPOSCAR", phonopy_worker.phonopy.supercell)
        phonopy_worker.generate_force_constants(save_fc=True)
        os.chdir("..")
    elif args.plot_phonon_band_structure:
        from core.phonon.phonon_plotter import prepare_and_plot_single_phonon_band_structure

        print("Plotting phonon band structure..., do we include NAC?", args.non_analytical_correction)
        prepare_and_plot_single_phonon_band_structure(
            path="finite_displacement_phonons",
            fc_file="force_constants.hdf5",
            poscar_file="POSCAR",
            supercell_matrix=[2, 2, 2],  # this need to be made more flexible
            num_qpoints=50,
            labels=["PBE-sol"],
            colors=["blue"],
            savefig=True,
            nac_correction=args.non_analytical_correction,
        )

    elif args.calculate_band_structure:
        # fmt: off
        band_structure_workflow(logger, 
                                structure_file=args.structure_file, 
                                use_gpu=args.use_gpu, 
                                scf_nkpts=8000, 
                                band_nkpts=args.num_kpoints)
        # fmt: on
