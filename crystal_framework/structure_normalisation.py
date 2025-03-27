from ase.neighborlist import NeighborList
from pymatgen.core import Structure
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor


def convert_pymatgen_to_ase(pmg_structure):
    """
    Convert a Pymatgen Structure object to an ASE Atoms object.

    Parameters:
    pmg_structure (Structure): A Pymatgen Structure object representing a crystal structure.

    Returns:
    Atoms: An ASE Atoms object corresponding to the input Pymatgen structure.
    """
    adaptor = AseAtomsAdaptor()
    return adaptor.get_atoms(pmg_structure)


def find_shortest_cation_anion_bond(atoms, anions='O'):
    """
    Find the shortest metal-oxygen bond length in an ASE Atoms object.

    Parameters:
    atoms (Atoms): An ASE Atoms object representing a crystal structure.

    Returns:
    float: The shortest metal-oxygen bond length.
    """
    from ase.neighborlist import build_neighbor_list
    nonmetals = ['H', 'He', 'C', 'N', 'O', 'F', 'Ne', 'P', 'S', 'Cl', 'Ar', 'Se', 'Br', 'Kr', 'I', 'Xe', 'Rn', 'At',
                 'Og', 'Ts']

    metal_indices = [i for i, atom in enumerate(atoms) if atom.symbol not in nonmetals]
    anion_indices = [i for i, atom in enumerate(atoms) if atom.symbol == anions]

    if not metal_indices or not anion_indices:
        raise ValueError("No metal or oxygen atoms found in the structure.")

        # Create a neighbor list with a cutoff radius (5 Å is a reasonable guess)
    cutoffs = [2.5] * len(atoms)  # Setting a uniform cutoff distance
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    # Find the shortest metal-anion bond
    shortest_bondlength = float("inf")

    for metal_idx in metal_indices:
        neighbors, offsets = nl.get_neighbors(metal_idx)
        for neighbor_idx in neighbors:
            if neighbor_idx in anion_indices:
                bond_length = atoms.get_distance(metal_idx, neighbor_idx)
                shortest_bondlength = min(shortest_bondlength, bond_length)

    return shortest_bondlength if shortest_bondlength != float("inf") else None

def normalise_structure(structure, anion='O'):
    """
    This method takes a structure and rescales its unit cell lengths such that
    the shortest metal-anion bond distance is renormalised to be one.
    :param structure: a pymatgen Structure object.
    :param anion: The anion to which the metal cation is bounded to.
    :return: a pymatgen Structure object.
    """
    ase_atoms = convert_pymatgen_to_ase(structure)
    l = find_shortest_cation_anion_bond(ase_atoms,anion=anion)
    ase_atoms.set_cell(cell=ase_atoms.cell*(1.0/l),scale_atoms=True)
    return AseAtomsAdaptor.get_structure(ase_atoms) #convert it back to a pymatgen structure


if __name__ == "__main__":
    from statistics import get_compounds

    compounds = get_compounds(anion='O', number_of_elements=2)
    structure = compounds[0].structure
    normalised_structure = normalise_structure(structure)
