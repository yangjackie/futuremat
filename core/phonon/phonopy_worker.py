import numpy as np
from mace.calculators import mace_mp
from ase.io import read
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import argparse

class PhonopyWorker:
    """A class to interface Phonopy with ASE calculators to compute force constants.
    This enables the use of machine learning interatomic potentials, such as MACE foundation models, 
    for phonon calculations."""

    def __init__(self,
                 structure,
                 supercell_matrix=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
                 displacement_distance=0.02,
                 calculator=None):
        self.phonopy_structure = PhonopyAtoms(symbols=structure.get_chemical_symbols(),
                                              scaled_positions=structure.get_scaled_positions(),
                                              cell=structure.get_cell())
        self.phonopy = Phonopy(self.phonopy_structure,
                               supercell_matrix=supercell_matrix)
        self.displacement_distance = displacement_distance
        self.calculator = calculator

    def generate_force_constants(self,save_fc=False,fc_file_name="force_constants"):
        all_forces = []
        self.phonopy.generate_displacements(distance=self.displacement_distance)

        for structure in self.phonopy.supercells_with_displacements:
            ase_atoms = Atoms(symbols=structure.symbols,
                              scaled_positions=structure.scaled_positions,
                              cell=structure.cell,
                              pbc=True)

            ase_atoms.calc = self.calculator
            forces = ase_atoms.get_forces()
            all_forces.append(forces)
        self.phonopy.produce_force_constants(forces=all_forces)
        if save_fc:
            from phonopy.file_IO import write_force_constants_to_hdf5
            write_force_constants_to_hdf5(self.phonopy.force_constants,fc_file_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argument parser for performing  phonopy calculations with MLSPs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="mace-mp-0b3-medium.model")
    args = parser.parse_args()

    calculator = mace_mp(model=args.model_path + "/" + args.model_name, device='cpu')

    #read in structure from CONTCAR file, replace with your own structure file as needed
    phonopy_worker = PhonopyWorker(structure=read("CONTCAR"),calculator=calculator)

    phonopy_worker.generate_force_constants(save_fc=True,fc_file_name="force_constants-mace-mp-0b3-medium.hdf5")
