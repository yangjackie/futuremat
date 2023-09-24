import argparse
import copy

from core.dao.vasp import VaspReader, VaspWriter
from core.models import Crystal, Molecule, Atom

parser = argparse.ArgumentParser(description='workflow control for generate atom-displaced crystal structure',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--structure", type=str, help='input crystal structure', default='POSCAR')
parser.add_argument("-a", "--amplitude", type=float, help='amplitude of atomic displacement in Angstorm', default=0.05)
parser.add_argument("-d", "--direction", nargs='+', help='direction of displacement', default=[1,0,0])
parser.add_argument("-atom", "--atom", type=str, help="atom to be displaced", default='Ti')
args = parser.parse_args()

args.direction = [float(i) for i in args.direction]
#print(args.direction)
crystal = VaspReader(input_location=args.structure).read_POSCAR()

# get the direction of displacement in real space
lv1=crystal.lattice.lattice_vectors.get_row(0).vec_scale(args.direction[0])
lv2=crystal.lattice.lattice_vectors.get_row(1).vec_scale(args.direction[1])
lv3=crystal.lattice.lattice_vectors.get_row(2).vec_scale(args.direction[2])


displace_vec = lv1+lv2
displace_vec = displace_vec+lv3
displace_vec = displace_vec.normalise()
displace_vec = displace_vec.vec_scale(args.amplitude)


atoms=[]
for mol in crystal.asymmetric_unit:
    for atom in mol.atoms:
        if atom.label == args.atom:
            atom.position += displace_vec
            atoms.append(Atom(label=atom.label,position=atom.position+displace_vec))
        else:
            atoms.append(atom)

crystal.asymmetric_unit = [Molecule(atoms=atoms)]
VaspWriter().write_structure(crystal,filename='POSCAR_disp')


