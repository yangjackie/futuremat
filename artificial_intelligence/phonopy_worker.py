import numpy as np
from mace.calculators import mace_mp
from ase.io import read
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


class PhonopyWorker:

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

    def generate_force_constants(self):
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

if __name__ == "__main__":
    mace_model_path = "/Users/z3079335/OneDrive - UNSW/Documents/Projects/artificial_intelligence/oxide_benchmark/"
    mace_model_name = "mace-mp-0b3-medium.model"
    calculator = mace_mp(model=mace_model_path + mace_model_name, device='cpu')

    phonopy_worker = PhonopyWorker(structure=read(
        "/Users/z3079335/OneDrive - UNSW/Documents/Projects/perovskite_anharmonic_screening/halide_double_perovskites/MLFF_benchmark/dpv_Cs2AgBiBr6/CONTCAR"),
        calculator=calculator)

    phonopy_worker.generate_force_constants()
    phonopy_worker.phonopy.auto_band_structure(plot=True).show()