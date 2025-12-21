import argparse
from genericpath import isfile
import os
from core.data.vasp_settings import *
from core.calculators.pymatgen.vasp import Vasp
from core.phonon.phonopy_worker import Born
from core.phonon.eph.utils import setup_logger_local, eph_coupling_module_args

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
        try:
            shutil.copy(chgcar_file, f"{directory}/CHGCAR")
        except shutil.SameFileError:
            pass

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


def scf_charge_density_workflow(structure_file: str = None, use_gpu: bool = False, scf_nkpts: int = 8000, directory: str = None):
    """
    Run a self-consistent field (SCF) calculation to generate and persist charge density
    (CHGCAR) for downstream NSCF and electron–phonon workflows.

    This routine builds an electronic structure parameter set from a phonon-focused
    default, tunes it for SCF (grid-based k-points, dense k-point mesh via kppa,
    no ionic steps), ensures CHGCAR is written (LCHARG=True), and retains outputs
    for subsequent runs. It then dispatches the calculation via `execute_vasp_calculation`
    with job_type="scf_charge_density".

    Args:
        structure_file (str): Path to a crystal structure file readable by
            `pymatgen.core.Structure.from_file` (e.g., POSCAR, CIF). Must not be None.
        use_gpu (bool): If True, request a GPU-accelerated run by setting the
            corresponding parameter in the VASP input set.
        scf_nkpts (int): Target k-points-per-atom (kppa) for the SCF k-point grid.
            Use larger values for denser sampling (default: 8000).
        directory (str | None): Working directory for the calculation. If None,
            the default execution context is used.

    Returns:
        None

    Raises:
        FileNotFoundError: If `structure_file` does not exist or is not readable.
        ValueError: If `structure_file` is None or invalid.
        RuntimeError: If the underlying VASP execution fails.
        Exception: Propagates other exceptions raised by `Structure.from_file`
            or `execute_vasp_calculation`.

    Notes:
        - The SCF input set enforces:
            - k-point mode = "grid" with `kppa = scf_nkpts`
            - LCHARG = True to write CHGCAR
            - ICHARG = 2 and NSW = 0 (no ionic steps)
            - clean_after_success = False to preserve outputs
    """

    logger.info("SCF charge density calculation routine...")

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
        job_type="scf_charge_density",
        force_rerun=False,
        directory=directory,
    )


def band_structure_workflow(
    structure_file: str = None,
    use_gpu: bool = False,
    scf_nkpts: int = 8000,
    band_nkpts: int = 20,
    kpoint_mode="line",
    pre_converge_charge: bool = True,
    save_wfn: bool = False,
    clean_after_success: bool = True,
):
    """
    Run a VASP band-structure workflow with optional charge pre-convergence.

    This routine optionally pre-converges the electronic charge density via an SCF
    calculation and then performs a non-SCF band-structure calculation. It configures
    VASP input parameters based on the selected k-point mode and whether to reuse a
    pre-computed CHGCAR.

    Parameters:
    - structure_file (str): Path to a structure file readable by pymatgen
        (e.g., POSCAR/CONTCAR/CIF). Required.
    - use_gpu (bool): Whether to enable GPU-specific run settings for VASP.
    - scf_nkpts (int): Target number of k-points for the SCF pre-convergence step.
    - band_nkpts (int): Number of points along each segment of the band path when
        kpoint_mode='line'.
    - kpoint_mode (str): K-point generation mode. Use 'line' to construct
        high-symmetry lines, or 'predetermined' to use externally supplied k-points.
    - pre_converge_charge (bool): If True, attempt to read a CHGCAR for the band
        calculation. If none is found in ./scf_charge_density/CHGCAR, run an SCF step and
        use ./scf_for_band_structure/CHGCAR.
    - save_wfn (bool): If True, set LWAVE=True to write WAVECAR during the
        band-structure run.
    - clean_after_success (bool): If True, remove temporary files after a
        successful run.

    Behavior:
    - If pre_converge_charge is True:
        - Uses ICHARG=11 to read charge density from CHGCAR.
        - Prefers ./band_structure/CHGCAR if present; otherwise runs a pre-SCF and
            uses ./scf_for_band_structure/CHGCAR.
    - If pre_converge_charge is False:
        - Uses ICHARG=2 for a standard calculation without reading CHGCAR.
    - Sets LCHARG=False (do not write CHGCAR in the band run).
    - Sets LWAVE=True when save_wfn is True to write WAVECAR.
    - K-point configuration:
        - kpoint_mode='line' sets kpoint_mode='lines' and kppa_band=band_nkpts.
        - kpoint_mode='predetermined' uses the predetermined k-points.
    - Delegates execution to execute_vasp_calculation with job_type='band_structure',
        force_rerun=False, and the derived chgcar_file.
    - Logs progress and status via a module-level logger.

    Raises:
    - ValueError: If kpoint_mode is neither 'line' nor 'predetermined'.

    Returns:
    - None
    """
    logger.info("Band structure calculation routine...")
    logger.info("Starting a SCF calculation to obtain charge density...")

    if pre_converge_charge:
        if not isfile("./band_structure/CHGCAR"):
            scf_charge_density_workflow(
                logger,
                structure_file=structure_file,
                use_gpu=use_gpu,
                scf_nkpts=scf_nkpts,
            )
            chgcar_path = "./scf_charge_density/CHGCAR"
        else:
            logger.info("Found existing CHGCAR file for band structure calculation, skipping SCF step.")
            chgcar_path = "./band_structure/CHGCAR"
    else:
        chgcar_path = None

    logger.info("Starting a non-SCF calculation for band structure...")

    ELECTRONIC_STRUCTURE_SET = copy.deepcopy(DEFAULT_STATIC_SET_FOR_PHONONS)

    if kpoint_mode == "line":
        ELECTRONIC_STRUCTURE_SET["kpoint_mode"] = "lines"
        ELECTRONIC_STRUCTURE_SET["kppa_band"] = band_nkpts  # number of k-points along each band path
    elif kpoint_mode == "predetermined":
        ELECTRONIC_STRUCTURE_SET["kpoint_mode"] = "predetermined"
    else:
        raise ValueError("kpoint_mode should be either 'line' or 'predetermined'")

    if pre_converge_charge:
        ELECTRONIC_STRUCTURE_SET["ICHARG"] = 11  # read charge density from CHGCAR
    else:
        ELECTRONIC_STRUCTURE_SET["ICHARG"] = 2  # standard SCF calculation

    if save_wfn:
        ELECTRONIC_STRUCTURE_SET["LWAVE"] = True  # write out WAVECAR

    ELECTRONIC_STRUCTURE_SET["clean_after_success"] = clean_after_success
    ELECTRONIC_STRUCTURE_SET["gpu_run"] = use_gpu
    ELECTRONIC_STRUCTURE_SET["LCHARG"] = False  # no need to write out CHGCAR again

    execute_vasp_calculation(
        structure=Structure.from_file(structure_file),
        params=ELECTRONIC_STRUCTURE_SET,
        job_type="band_structure",
        force_rerun=False,
        chgcar_file=chgcar_path,
    )


def unfold_band_structrure_workflow(primitive_folder: str = None, supercell_folder: str = os.getcwd(), matrix: str = "2 2 2"):
    """
    Run a complete workflow to compute the unfolded electronic band structure for a
    supercell, it will generate a plot of the unfolded bandstructure of the supercell
    that is overlaid on top of the primitive cell band structure.

    This routine:
    - Converges a self-consistent charge density (CHGCAR) for supercell.
    - Generates an unfolded k-point path using the `easyunfold` CLI.
    - Performs a supercell band-structure calculation with the unfolded k-points.
    - Computes spectral weights via `easyunfold unfold calculate` using the WAVECAR.
    - Plots the primitive and unfolded supercell band structures together.

    Parameters:
        primitive_folder (str):
            Path to the primitive-cell directory containing POSCAR and KPOINTS used to
            define the k-path for unfolding.
        supercell_folder (str):
            Path to the supercell directory containing POSCAR. Defaults to the current
            working directory.
        matrix (str):
            Supercell transformation expressed as a space-separated string of three integers
            (e.g., "2 2 2"). Passed to `easyunfold --matrix`.

    Side effects:
        - Logs progress and file locations.
        - Creates "SCF_charge" subdirectory under the supercell folder (if missing).
        - Creates a "band_structure" subdirectory under the supercell folder (if missing).
        - Writes and may overwrite files in the working directory:
            - Deletes "KPOINTS_easyunfold" and "easyunfold.json" if they exist in CWD.
            - Generates new "KPOINTS_easyunfold" and "easyunfold.json" via `easyunfold generate`.
        - Copies "SCF_charge/CHGCAR" into "band_structure/CHGCAR".
        - Copies "KPOINTS_easyunfold" into "band_structure/KPOINTS".
        - Runs external commands via `os.system`, as we did not wrap `easyunfold` functionality natively through
            Python APIs.

    Requirements:
        - `easyunfold` must be installed and available on PATH.
        - Primitive folder must contain valid POSCAR and KPOINTS.
        - Supercell folder must contain a valid POSCAR.
        - VASP-related workflows (`scf_charge_density_workflow`, `band_structure_workflow`)
          must be available and correctly configured to produce CHGCAR and WAVECAR.

    Returns:
        None

    Notes:
        - For this to work, one must use VASP 6.4.2 to run the calculations otherwise there
           will be compatibility issues with the WAVECAR format that easyunfold can read.
        - This function assumes VASP workflows will produce "SCF_charge/CHGCAR" and
          "band_structure/WAVECAR" under the supercell folder. So be minddful on the storage
          space required for large amount of similar calculations.
        - It does not perform explicit error handling for missing files or command failures.
    """

    assert primitive_folder is not None, "Primitive folder path must be provided."

    logger.info("Routine to calculate the unfolded band structure from supercell calculation...")

    supercell_folder = os.path.abspath(supercell_folder)
    primitive_folder = os.path.abspath(primitive_folder)

    logger.info("Current working directory: {}".format(os.getcwd()))
    logger.info("Primitive cell folder: {}".format(primitive_folder))
    logger.info("Supercell folder: {}".format(supercell_folder))

    # ==================================================================================
    logger.info("Converging SCF charge density for primitive cell...")
    scf_charge_density_workflow(
        structure_file=os.path.join(supercell_folder, "POSCAR"),
        use_gpu=False,
        scf_nkpts=8000,
        directory=os.path.join(supercell_folder, "SCF_charge"),
    )
    # ==================================================================================

    logger.info("Generating KPOINTS path from primitive cell to supercell...")
    if os.path.exists(os.path.join(os.getcwd(), "KPOINTS_easyunfold")):
        os.remove(os.path.join(os.getcwd(), "KPOINTS_easyunfold"))
    if os.path.exists(os.path.join(os.getcwd(), "easyunfold.json")):
        os.remove(os.path.join(os.getcwd(), "easyunfold.json"))

    cmd = "easyunfold generate " + primitive_folder + "/POSCAR " + supercell_folder + "/POSCAR " + primitive_folder + "/KPOINTS --matrix " + '"' + matrix + '"'
    os.system(cmd)
    # ==================================================================================

    # bandstructure calculation for supercell with unfolded kpoints
    band_structure_fld = "band_structure"
    if not os.path.exists(os.path.join(supercell_folder, band_structure_fld)):
        os.mkdir(os.path.join(supercell_folder, band_structure_fld))
    # copy the necessary files across
    shutil.copy(os.path.join(supercell_folder, "KPOINTS_easyunfold"), os.path.join(supercell_folder, band_structure_fld, "KPOINTS"))
    shutil.copy(os.path.join(supercell_folder, "SCF_charge", "CHGCAR"), os.path.join(supercell_folder, band_structure_fld, "CHGCAR"))

    band_structure_workflow(
        structure_file=os.path.join(supercell_folder, "POSCAR"),
        use_gpu=False,
        scf_nkpts=8000,
        band_nkpts=20,
        kpoint_mode="predetermined",
        pre_converge_charge=True,
        save_wfn=True,
        clean_after_success=False,
    )
    # ==================================================================================

    logger.info("Calling easyunfold to generate the spectal weights...")
    cmd = "easyunfold unfold calculate " + supercell_folder + "/band_structure/WAVECAR"
    os.system(cmd)
    # ==================================================================================

    from core.phonon.eph.plotter import plot_primitive_and_unfolded_super_cell_band_structrue

    plot_primitive_and_unfolded_super_cell_band_structrue(primitive_path=primitive_folder, supercell_path=supercell_folder + "/band_structure")
    return


if __name__ == "__main__":
    args = eph_coupling_module_args()
    logger = setup_logger_local(args)

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
    elif args.calculate_band_structure:
        # fmt: off
        band_structure_workflow(logger, 
                                structure_file=args.structure_file, 
                                use_gpu=args.use_gpu, 
                                scf_nkpts=8000, 
                                band_nkpts=args.num_kpoints,
                                pre_converge_charge=True)
        # fmt: on
    elif args.calculate_unfold_band_structure:
        unfold_band_structrure_workflow(
            primitive_folder="../band_structure",
            supercell_folder=os.getcwd(),
            matrix="2 2 2",
        )
