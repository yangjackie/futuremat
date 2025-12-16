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
        self, save_fc=False, fc_file_name="force_constants.hdf5"
    ):
        all_forces = []
        self.phonopy.generate_displacements(distance=self.displacement_distance)

        for id, structure in enumerate(self.phonopy.supercells_with_displacements):
            forces = self.get_forces(id, structure)
            all_forces.append(forces)
        self.phonopy.produce_force_constants(forces=all_forces)
        if save_fc:
            from phonopy.file_IO import write_force_constants_to_hdf5

            write_force_constants_to_hdf5(self.phonopy.force_constants, fc_file_name)

    def get_forces(self, id, structure):
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
