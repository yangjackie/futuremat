from core.models import cMatrix3D
from core.models.lattice import Lattice as fLattice
from core.models.crystal import Crystal
from core.models.molecule import Molecule
from core.models.atom import Atom
from core.models.element import element_name_dict
from core.models.vector3d import cVector3D
from core.resources.crystallographic_space_groups import CrystallographicSpaceGroups


def expand_to_P1_strucutre(crystal):
    expanded_mol_list = []

    for mol in crystal.asymmetric_unit:
        for op in crystal.space_group.full_symmetry:
            new_mol = Molecule()
            new_mol.atoms = [op.transform_atom(a) for a in mol.atoms]
            expanded_mol_list.append(new_mol)

    return Crystal(lattice=crystal.lattice,
                   asymmetric_unit=expanded_mol_list,
                   space_group=CrystallographicSpaceGroups.get(1))


def build_supercell(crystal, expansion=[1, 1, 1]):
    """
    Method to make a supercell using the input crystal as the primitive crystal structure. Given the
    transformation specified by a list of three integers `[n_{x},n_{y},n_{z}]`, a super cell with cell
    lengths `[n_{x}a,n_{y}b,n_{z}c]` will be built.

    :param crystal: Input crystal structure.
    :param expansion: A list of three integers specifying how big the supercell will be.
    :return: crystal: A fully constructed crystal object with new lattice parameters.
    :rtype: :class:`.Crystal`
    """
    crystal_in_p1_setting = expand_to_P1_strucutre(crystal)

    lattice = fLattice(a=crystal.lattice.a * expansion[0],
                       b=crystal.lattice.b * expansion[1],
                       c=crystal.lattice.c * expansion[2],
                       alpha=crystal.lattice.alpha,
                       beta=crystal.lattice.beta,
                       gamma=crystal.lattice.gamma)

    asymmetric_unit = [x for x in crystal_in_p1_setting.asymmetric_unit]

    for n_x in range(expansion[0]):
        for n_y in range(expansion[1]):
            for n_z in range(expansion[2]):

                tr_vec = crystal.lattice.lattice_vectors.get_row(0).vec_scale(n_x) + \
                         crystal.lattice.lattice_vectors.get_row(1).vec_scale(n_y) + \
                         crystal.lattice.lattice_vectors.get_row(2).vec_scale(n_z)

                if (n_x == 0) and (n_y == 0) and (n_z == 0):
                    pass
                else:
                    for mol in crystal_in_p1_setting.asymmetric_unit:
                        new_atoms = [Atom(label=atom.label, position=atom.position + tr_vec) for atom in mol.atoms]
                        asymmetric_unit.append(Molecule(atoms=new_atoms))

    return Crystal(lattice=lattice, asymmetric_unit=asymmetric_unit, space_group=CrystallographicSpaceGroups.get(1))


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


def map_to_ase_atoms(crystal):
    """
    Given a crystal structure, map it to the ASE atoms model so we can use functionalities in ASE to
    manipulate things.

    :param crystal: Input crystal structure
    :return: A fully constructed ASE `atoms` model.
    """
    from ase.atoms import Atoms

    # Get all atoms and corresponding symbols
    crystal = expand_to_P1_strucutre(crystal)
    all_atoms = crystal.all_atoms()
    all_unqiue_labels = list(set([i.clean_label for i in all_atoms]))

    # Create a list sc of (symbol, count) pairs
    label_count = [0 for _ in all_unqiue_labels]
    for i, label in enumerate(all_unqiue_labels):
        for atom in all_atoms:
            if atom.clean_label == label:
                label_count[i] += 1

    symbol_line = ''
    for i in range(len(all_unqiue_labels)):
        symbol_line += all_unqiue_labels[i] + str(label_count[i])

    return Atoms(symbols=symbol_line,
                 scaled_positions=[a.scaled_position.to_numpy_array() for a in all_atoms],
                 pbc=True,
                 cell=[[crystal.lattice.lattice_vectors.get(m, n) for n in range(3)] for m in range(3)])


def map_ase_atoms_to_crystal(atoms):
    """
    Given an ASE atoms object, map it to a crystal structure in our internal model

    :param atoms: An input ASE atoms object
    :return: a fully constructed crystal structure in P1 setting
    """

    # this is dumb, but anyway, prevents re-orienting things when the lattice is constructed
    _lv = atoms.get_cell().array
    _a = cVector3D(*_lv[0])
    _b = cVector3D(*_lv[1])
    _c = cVector3D(*_lv[2])
    _lv = cMatrix3D(_a, _b, _c)
    lattice = fLattice.from_lattice_vectors(_lv)
    lattice.lattice_vectors = _lv

    return Crystal(lattice=lattice,
                   asymmetric_unit=[Molecule(atoms=[
                       Atom(label=atoms.get_chemical_symbols()[i],
                            scaled_position=cVector3D(*atoms.get_scaled_positions()[i])) for
                       i in range(len(atoms.get_chemical_symbols()))])],
                   space_group=CrystallographicSpaceGroups.get(1))
