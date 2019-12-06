from core.models.lattice import Lattice as fLattice
from core.models.crystal import Crystal
from core.models.molecule import Molecule
from core.models.atom import Atom
from core.models.element import element_name_dict
from core.resources.crystallographic_space_groups import CrystallographicSpaceGroups

def map_pymatgen_IStructure_to_crystal(structure):
    """
    Given a Pymatgen IStructure object, map it to a crystal structure in our internal model

    :param atoms: An input Pymatgen IStructure object
    :return: a fully constructed crystal structure in P1 setting
    """
    return Crystal(lattice=fLattice(structure.lattice.a,
                                            structure.lattice.b,
                                            structure.lattice.c,
                                            structure.lattice.alpha,
                                            structure.lattice.beta,
                                            structure.lattice.gamma),
                   asymmetric_unit=[Molecule(atoms=[
                       Atom(label=element_name_dict[structure.atomic_numbers[i]], position=structure.cart_coords[i]) for
                       i in range(len(structure.atomic_numbers))])],
                   space_group=CrystallographicSpaceGroups.get(1))