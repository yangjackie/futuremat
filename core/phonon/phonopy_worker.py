import numpy as np
from ase.io import read
from ase import Atoms
from ase.calculators.mixing import SumCalculator
from pymatgen.core.structure import Structure
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import argparse
import logging

logger = logging.getLogger("futuremat.core.phonon.phonopy_worker")

from core.calculators.pymatgen.vasp import Vasp


class PhonopyWorker:
    """A class to interface Phonopy with ASE calculators to compute force constants.
    This enables the use of machine learning interatomic potentials, such as MACE foundation models,
    for phonon calculations."""

    def __init__(
        self,
        structure,
        supercell_matrix=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        displacement_distance=0.02,
        calculator=None,
    ):
        """
        Initialize a PhonopyWorker instance.

        Parameters
        ----------
        structure : ase.Atoms or pymatgen.Structure
            The atomic structure to perform phonon calculations on.
            Can be either an ASE Atoms object or a pymatgen Structure object.
        supercell_matrix : np.ndarray, optional
            A 3x3 matrix defining the supercell for phonon calculations.
            Default is [[2, 0, 0], [0, 2, 0], [0, 0, 2]] (2x2x2 supercell).
        displacement_distance : float, optional
            The displacement distance in Angstroms for finite displacement
            phonon calculations. Default is 0.02.
        calculator : object, optional
            A calculator object (e.g., ASE calculator) used to compute forces
            on displaced structures. Default is None.

        Notes
        -----
        1. The input structure is converted to a Phonopy-compatible format internally.
        2. The ASE calculator is implemented to be used for machine learning potential models sch as the MACE model,
        which is natively interfaced with ASE.
        3. For VASP calculations, our own modified pymatgen-based Vasp calculator is used to run VASP calculations.
        This automatically handles the input creation, executioon of VASP calculation and output parsing.
        """
        if isinstance(structure, Atoms):
            self.phonopy_structure = PhonopyAtoms(
                symbols=structure.get_chemical_symbols(),
                scaled_positions=structure.get_scaled_positions(),
                cell=structure.get_cell(),
            )
        elif isinstance(structure, Structure):
            self.phonopy_structure = PhonopyAtoms(
                symbols=[str(site.specie) for site in structure],
                scaled_positions=structure.frac_coords,
                cell=structure.lattice.matrix,
            )

        self.phonopy = Phonopy(
            self.phonopy_structure, supercell_matrix=supercell_matrix
        )
        self.displacement_distance = displacement_distance
        self.calculator = calculator

    def generate_force_constants(
        self,
        save_fc=False,
        fc_file_name="force_constants.hdf5",
        forceset_file_name="FORCE_SETS",
    ):
        """
        Generate force constants by computing forces on displaced supercells.

        This method creates atomic displacements in the supercell, calculates forces
        on each displaced structure using the provided calculator (ASE-based or VASP),
        and then produces the force constant matrix using Phonopy.

        Parameters
        ----------
        save_fc : bool, optional
            If True, save the computed force constants and force sets to files.
            Default is False. (for example, when MLFF is used and the force constants can be generated on-the-fly)
        fc_file_name : str, optional
            Filename for HDF5 force constants output. Only used if save_fc is True.
            Default is "force_constants.hdf5".
        forceset_file_name : str, optional
            Filename for FORCE_SETS output (raw forces from displacements).
            Only used if save_fc is True. Default is "FORCE_SETS".

        Notes
        -----
        - Displacement distance is controlled by the `displacement_distance`
          attribute set during initialization (default 0.02 Å).
        - Supercell size is controlled by the `supercell_matrix` set during
          initialization (default 2x2x2).
        - Compatible calculators: ASE (SumCalculator) and VASP (via Pymatgen).
        - The method creates intermediate calculation directories for VASP
          calculations (phonopy_disp_<id>).

        Examples
        --------
        Generate force constants without saving:

        >>> worker = PhonopyWorker(structure=atoms, calculator=calc)
        >>> worker.generate_force_constants()

        Generate and save force constants to HDF5 and FORCE_SETS:

        >>> worker.generate_force_constants(
        ...     save_fc=True,
        ...     fc_file_name="my_fc.hdf5",
        ...     forceset_file_name="FORCE_SETS"
        ... )

        See Also
        --------
        _save_force_constants : Saves force constants and force sets to files.
        get_forces : Retrieves forces for a given displaced structure.
        """

        all_forces = []
        self.phonopy.generate_displacements(distance=self.displacement_distance)

        for id, structure in enumerate(self.phonopy.supercells_with_displacements):
            forces = self.get_forces(id, structure)
            all_forces.append(forces)
        self.phonopy.produce_force_constants(forces=all_forces)
        if save_fc:
            self._save_force_constants(fc_file_name, forceset_file_name)

    def _save_force_constants(
        self, fc_file_name="force_constants.hdf5", forceset_file_name="FORCE_SETS"
    ):
        """
        Write force constants and force sets to files.

        Parameters
        ----------
        fc_file_name : str
            Filename for HDF5 force constants output.
        forceset_file_name : str
            Filename for FORCE_SETS output.
        """
        from phonopy.file_IO import write_force_constants_to_hdf5, write_FORCE_SETS

        write_force_constants_to_hdf5(self.phonopy.force_constants, fc_file_name)
        logger.info("Wrote force constants to HDF5: %s", fc_file_name)

        write_FORCE_SETS(
            self.phonopy.dataset,
            filename=forceset_file_name,
        )
        logger.info("Wrote FORCE_SETS to: %s", forceset_file_name)

    def get_forces(self, id, structure):
        """
        Calculate forces on a displaced supercell structure using the configured calculator.

        This method serves as a dispatcher that routes force calculations to the appropriate
        backend based on the calculator type (ASE-based or VASP).

        Parameters
        ----------
        id : int
            Identifier for the displaced structure (usually the displacement index).
            Used for logging and directory naming in VASP calculations.
        structure : PhonopyAtoms
            A displaced structure as a PhonopyAtoms object from Phonopy.
            Contains atomic positions, cell, and atomic symbols.

        Returns
        -------
        np.ndarray
            Force array of shape (n_atoms, 3) containing forces on each atom in Cartesian
            coordinates (eV/Å or equivalent units depending on the calculator).

        Notes
        -----
        - For ASE calculators (e.g., MACE), forces are computed directly in memory.
        - For VASP calculations, forces are extracted from vasprun.xml after the
          calculation completes in a separate directory (phonopy_disp_<id>).
        - This method is called internally by `generate_force_constants` for each
          displaced structure.

        See Also
        --------
        __forces_from_ase_calculator : Compute forces using ASE-based calculators.
        __force_from_pymatgen_vasp_calculator : Compute forces using VASP via PyMatGen.
        """
        if isinstance(self.calculator, SumCalculator):
            # this is the class for the mace_mp calculator, dont need to set up further, just use it directly!
            return self.__forces_from_ase_calculator(structure)
        elif isinstance(self.calculator, Vasp):
            logging.info(
                "Using pymatgen VASP calculator to get forces for displaced structure id: %d",
                id,
            )
            return self.__force_from_pymatgen_vasp_calculator(structure, id)

    def __forces_from_ase_calculator(self, structure):
        """
        Calculate forces using an ASE-based calculator (e.g., MACE, EMT, etc.).

        This is a private method that converts a Phonopy structure to an ASE Atoms object,
        attaches the configured ASE calculator, and computes forces in-memory.

        Parameters
        ----------
        structure : PhonopyAtoms
            A displaced structure as a PhonopyAtoms object from Phonopy.
            Contains atomic symbols, scaled positions, and lattice vectors.

        Returns
        -------
        np.ndarray
            Force array of shape (n_atoms, 3) containing forces on each atom in Cartesian
            coordinates (eV/Å or units of the calculator).

        Notes
        -----
        - This method is fast and memory-efficient as calculations are done in-memory.
        - Works with any ASE-compatible calculator (MACE, EMT, AIREBO, etc.).
        - Requires `self.calculator` to be set to an ASE calculator instance.
        - Periodic boundary conditions (pbc=True) are enforced for all calculations.
        """
        atoms = Atoms(
            symbols=structure.symbols,
            scaled_positions=structure.scaled_positions,
            cell=structure.cell,
            pbc=True,
        )
        atoms.calc = self.calculator
        forces = atoms.get_forces()
        return forces

    def __force_from_pymatgen_vasp_calculator(self, structure, id):
        """
        Calculate atomic forces using our own modified pymatgen's VASP calculator.

        Converts the input structure to pymatgen format, executes a VASP calculation
        via the calculator, and extracts ionic forces from the resulting vasprun.xml file.

        Args:
            structure : PhonopyAtoms
                A displaced structure as a PhonopyAtoms object from Phonopy.
                Contains atomic symbols, scaled positions, and lattice vectors.
            id: Unique identifier for the calculation (used in directory naming)

        Returns:
            list: Ionic forces from the final ionic step of the VASP calculation.
                  Shape is (n_atoms, 3) with forces in eV/Angstrom.

        Raises:
            FileNotFoundError: If vasprun.xml is not found after calculation execution.
            Exception: If VASP calculation fails during execution.

        Note:
            - Creates a directory named "phonopy_disp_{id}" for calculation outputs
            - Only parses ionic forces, skipping DOS and eigenvalue parsing for efficiency
        """
        structure = Structure(
            lattice=structure.cell,
            species=structure.symbols,
            coords=structure.scaled_positions,
            coords_are_cartesian=False,
        )
        self.calculator.structure = structure
        self.calculator.directory = "phonopy_disp_" + str(id)
        self.calculator.execute()
        vasprun_path = self.calculator.directory + "/vasprun.xml"
        from pymatgen.io.vasp.outputs import Vasprun

        vasprun = Vasprun(vasprun_path, parse_dos=False, parse_eigen=False)
        forces = vasprun.ionic_steps[-1]["forces"]
        return forces


class Born:

    @staticmethod
    def write_born_file_with_cmd(directory):
        """
        Generate a BORN file from a VASP calculation using the phonopy-vasp-born command.

        This function serves as a workaround to generate the BORN file when the phonopy API
        does not provide a direct method. It executes the phonopy-vasp-born command-line tool
        on the vasprun.xml file in the specified directory and redirects the output to a BORN file.

        Args:
            directory (str): Path to the directory containing the vasprun.xml file from a VASP calculation.

        Returns:
            None

        Raises:
            No explicit exceptions are raised, but the command execution may fail silently if:
            - phonopy-vasp-born is not installed or not in PATH
            - The vasprun.xml file does not exist in the specified directory
            - Insufficient permissions to write the BORN file

        Note:
            This is a temporary workaround using os.system(). Consider replacing with pure pythonic API calls.
        """
        # Didnt figured out how to do it from API, so use this as a hack when needed.
        import os

        os.system("phonopy-vasp-born " + directory + "/vasprun.xml > BORN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="argument parser for performing  phonopy calculations with MLSPs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="mace-mp-0b3-medium.model")
    parser.add_argument(
        "--output_fc_name", type=str, default="force_constants-mace-mp-0b3-medium.hdf5"
    )
    parser.add_argument("--input_structure", type=str, default="./POSCAR")
    args = parser.parse_args()

    try:
        from mace.calculators import mace_mp

        calculator = mace_mp(
            model=args.model_path + "/" + args.model_name, device="cpu"
        )
    except:
        raise Exception("Cannot set up the MACE calculator!")

    # read in structure from CONTCAR file, replace with your own structure file as needed
    phonopy_worker = PhonopyWorker(
        structure=read(args.input_structure), calculator=calculator
    )

    if args.output_fc_name is None:
        args.output_fc_name = "force_constants-" + args.model_name.replace(
            ".model", ".hdf5"
        )

    phonopy_worker.generate_force_constants(
        save_fc=True, fc_file_name=args.output_fc_name
    )
