from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun


from core.dao.pymatgen.vasp import Potcar
from core.calculators.abstract_calculator import VaspBase

import logging
import os

logger = logging.getLogger("futuremat.core.calculators.pymatgen.vasp")


class Vasp(VaspBase):
    """
    Reimplementing the futuremat VASP calculator with Pymatgen API to set up and analyse the VASP calculations,
    where we provided our own hook to (a) picked up some environmental settings from the HPC system to
    determine how these calculations are executed, which maybe changed if this is used by different users, and (b)
    assemble the correct pseudopotential files based on the Materials Project PBE pseudopotential choices, and user-specified
    local VASP pseudopotential directories.
    """

    def __init__(self, directory=None, force_rerun=False, **kwargs):

        self.set_incar_params(**kwargs)

        try:
            self.use_gw = kwargs["use_gw"]
        except KeyError:
            self.use_gw = False

        try:
            self.kpoint_mode = kwargs["kpoint_mode"]
            assert self.kpoint_mode in ("monkhorst", "mp", "grid", "line", "band", "lines", "gamma", "predetermined")
        except KeyError:
            raise KeyError("Please specify the kpoint_mode for the VASP calculation!")

        try:
            self.kppa = kwargs["kppa"]
        except KeyError:
            self.kppa = 2000  # default kppa for phonon calculations

        try:
            self.kppa_band = kwargs["kppa_band"]
        except KeyError:
            self.kppa_band = 20  # default kppa for band structure calculations

        try:
            self.gpu_run = kwargs["gpu_run"]
        except KeyError:
            self.gpu_run = False

        try:
            self.clean_after_success = kwargs["clean_after_success"]
        except KeyError:
            self.clean_after_success = False

        try:
            self.user_kpoints = kwargs["user_kpoints"]
        except KeyError:
            self.user_kpoints = None

        # a tag to determine if the calculation should be rerun if the previous calculation exists
        self.force_rerun = force_rerun

        # set up the directory to run the calculation
        self.directory = directory
        # self.__setup_directory()

    def __setup_directory(self):
        """
        Set up the directory to run the VASP calculation. Instead of let the higher order workflow handle the directory, we did it at the calculator level, just to make it become consistent with ASE VASP calculator behaviour.
        """
        self.cwd = os.getcwd()
        if self.directory is not None:
            if not os.path.exists(self.directory):
                os.mkdir(self.directory)

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, val):
        """
        Sets the directory for the VASP calculation and sets up the directory if it does not exist.
        """
        self._directory = val
        self.__setup_directory()

    def _setup_kpoints(self):
        """
        Sets up the KPOINTS file for VASP calculations based on the specified k-point mode.

        This method generates a KPOINTS file using different strategies depending on the
        value of `self.kpoint_mode`. The available modes are:

        - "monkhorst", "mp", "grid": Generates a Monkhorst-Pack grid based on the
          automatic density determined by the structure and kappa per atom (kppa).

        - "gamma": Generates a Gamma-centered grid based on the automatic density
          determined by the structure and kappa per atom (kppa).

        - "line", "band", "lines": Generates a KPOINTS file for band structure calculations
          using a high symmetry k-path detected from the structure. The number of points
          between labels is determined by `self.kppa_band`.

        If an unknown k-point mode is specified, a warning is logged and no KPOINTS file
        is written.

        Under the hood, this method utilizes Pymatgen's Kpoints class to automatically generate
        the appropriate k-point grids or paths.

        Returns:
            None
        """
        filename = "KPOINTS"
        if self.kpoint_mode in ("monkhorst", "mp", "grid"):
            kp = Kpoints.automatic_density(self.structure, kppa=self.kppa)

            kp.write_file(filename)
            logger.info("Wrote Monkhorst-Pack KPOINTS (kppa=%s) to %s", self.kppa, filename)
        elif self.kpoint_mode == "gamma":
            kp = Kpoints.gamma_automatic(self.structure, kppa=self.kppa)
            kp.write_file(filename)
            logger.info("Wrote Gamma-centered KPOINTS (kppa=%s) to %s", self.kppa, filename)
        elif self.kpoint_mode in ("line", "band", "lines"):
            from pymatgen.symmetry.bandstructure import HighSymmKpath

            kpath = HighSymmKpath(self.structure)  # auto-detects space group and path

            kpoints_bs = Kpoints.automatic_linemode(
                divisions=self.kppa_band,  # number of points between labels
                ibz=kpath,  # can pass HighSymmKpath directly
            )
            kpoints_bs.write_file(filename)
            logger.info(
                "Wrote band structure KPOINTS (kppa_band=%s) to %s",
                self.kppa_band,
                filename,
            )
        elif self.kpoint_mode == "predetermined":
            logger.info("Using user-provided KPOINTS file as 'predetermined' mode is selected.")
            self.user_kpoints.write_file("KPOINTS")
        else:
            logger.warning(
                "Unknown KPOINTS mode '%s' requested; no KPOINTS written.",
                self.kpoint_mode,
            )

    def setup(self):
        """
        Prepares the necessary input files for a VASP calculation based on the provided structure and parameters.

        This method performs the following tasks:
        1. Writes the structure to a POSCAR file.
        2. Creates and writes the INCAR file using the specified parameters.
        3. Automatically detects the number of CPUs (NCPUS) from the environment or system settings
           and add the NCORE setting to the INCAR file if not running on a GPU.
        4. Generates the POTCAR file for the given structure.
        5. Sets up the KPOINTS file for the calculation.

        Raises:
            Exception: If there is an error while appending NCORE to the INCAR file.
        """
        logger.info("Setting up VASP calculation, write input file ...")

        self.structure.to(filename="POSCAR")
        logger.info("Successfully written the POSCAR file for structure optimisation.")

        incar = Incar(self.incar_params)
        incar.write_file("INCAR")

        if not self.gpu_run:
            # Append NCORE based on environment or detected CPU count so external
            # job scripts that rely on NCORE get a consistent INCAR entry.

            ncpus = os.environ.get("NCPUS")
            logger.info("Auto-detect NCPUS from environment: %s", ncpus)

            if not ncpus:
                # Fall back to Python's cpu count (may return None)
                ncpus = str(os.cpu_count() or 1)
                logger.info(
                    "Auto-detect NCPUS failed, try from os.cpu_count(): %s; or set it to 1",
                    ncpus,
                )

            try:
                with open("INCAR", "a") as f:
                    f.write(f"NCORE={ncpus}\n")
                logger.info("Appended NCORE=%s to INCAR", ncpus)
            except Exception:
                logger.exception("Failed to append NCORE to INCAR")

        logger.info("Successfully written the INCAR file for structure optimisation.")

        Potcar(self.structure).write(use_GW=self.use_gw)
        logger.info("Successfully written the POTCAR file for structure optimisation.")

        self._setup_kpoints()

    def check_convergence(self):
        """
        Checks the convergence of a VASP calculation by parsing the 'vasprun.xml' file.

        This method initializes a Vasprun object and checks if the calculation has converged.
        If the calculation is converged, it logs a success message and sets the 'completed' attribute to True.
        If the calculation has not converged, it logs a warning message and sets the 'completed' attribute to False.

        Raises:
            FileNotFoundError: If 'vasprun.xml' is not found in the expected directory.
            ValueError: If there is an issue parsing the 'vasprun.xml' file.
        """
        vasprun = Vasprun("vasprun.xml", parse_dos=False, parse_eigen=False)
        if vasprun.converged:
            logger.info("VASP calculation converged successfully.")
            self.completed = True
        else:
            logger.warning("VASP calculation did not converge, please check cearefully!")
            self.completed = False

    def tear_down(self):
        """
        Cleans up the directory by removing specified VASP output files.
        This method attempts to delete a predefined list of files generated by VASP
        after successful execution. If a file cannot be removed due to an OSError,
        the exception is caught and ignored, allowing the cleanup process to continue
        without interruption.
        Files to be removed:
        - CHG
        - CHGCAR
        - EIGENVAL
        - IBZKPT
        - PCDAT
        - POTCAR
        - WAVECAR
        - LOCPOT
        - node_info
        - WAVEDER
        - DOSCAR
        - PROCAR
        - REPORT
        """

        logger.info("Clean up directory after VASP executed successfully.")
        files = [
            "CHG",
            "CHGCAR",
            "EIGENVAL",
            "IBZKPT",
            "PCDAT",
            "POTCAR",
            "WAVECAR",
            "LOCPOT",
            "node_info",
            "WAVECAR",
            "WAVEDER",
            "DOSCAR",
            "PROCAR",
            "REPORT",
        ]
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

    def run_this(self) -> bool:
        """
        Determines whether to rerun a VASP calculation based on the presence
        and convergence status of a previous calculation.

        Returns:
            bool: True if the calculation should be rerun, False if it can be skipped.

        Logs the decision-making process, including:
            - If the user has requested a forced rerun, the calculation will proceed regardless of previous results.
            - If no previous calculation is found, the calculation will proceed.
            - If a previous calculation exists, the decision to rerun is based on:
                - The existence of a previous calculation file (`vasprun.xml`).
                - The convergence status of the previous calculation.
        """
        if self.force_rerun is True:
            logger.info("User asked the VASP calculations to be forced to rerun, will not check previous calculations, proceed to calculation...")
            return True
        else:
            logger.info("Checking for existing calculations in the current folder...")
            _vasprun_file = os.path.join(os.getcwd(), "vasprun.xml")
            if os.path.exists(_vasprun_file):
                vasprun = Vasprun(
                    _vasprun_file,
                    parse_dos=False,
                    parse_eigen=False,
                )
                if vasprun.converged:
                    logger.info(f"Previous VASP calculation in {_vasprun_file} is already converged. Skipping rerun.")
                    return False
                else:
                    logger.info(f"Previous VASP calculation in {_vasprun_file} did not converged. Will rerun this calculation.")
                    return True
            else:
                logger.info("No previous VASP calculation found in the current folder. Proceed to calculation...")
                return True

    def execute(self):
        """
        Executes the main workflow of the calculation process.

        This method changes the current working directory to the specified
        directory, checks if the process should run, and if so, it performs
        the setup, execution, and convergence check. If the process is
        completed successfully and the clean-up option is enabled, it
        performs the tear down of the environment before returning to the
        original working directory.

        Raises:
            Exception: If any step in the execution process fails.
        """
        os.chdir(self.directory)
        if self.run_this():
            self.setup()
            self.run()
            self.check_convergence()

            if self.completed and self.clean_after_success:
                self.tear_down()
        os.chdir(self.cwd)
