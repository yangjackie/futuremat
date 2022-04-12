from core.dao.vasp import VaspReader,VaspWriter
from core.models import Crystal, cVector3D, Atom, Molecule
import numpy as np
import xml.etree.cElementTree as etree
import copy

class MDAveragedStructure(object):

    def __init__(self,ref_frame=None,md_frames=None):
        if isinstance(ref_frame, Crystal):
            self.ref_frame = ref_frame
        elif ('POSCAR' in ref_frame) or ('CONTCAR' in ref_frame):
            print(ref_frame)
            print("initialising reference frame from POSCAR ")
            self.ref_frame = VaspReader(input_location=ref_frame).read_POSCAR()
            self.ref_coords = np.array([[a.scaled_position.x, a.scaled_position.y, a.scaled_position.z] for a in
                                        self.ref_frame.asymmetric_unit[0].atoms])

        self.md_frames = [md_frames]

    def compute_averaged_structure(self):
        all_positions = []
        for frame in self.md_frames:
            if len(self.md_frames) != 1:
                this = []
            for event, elem in etree.iterparse(frame):
                if elem.tag == 'varray':
                    if elem.attrib['name'] == 'positions':
                        this_positions = []
                        for v in elem:
                            this_position = [float(_v) for _v in v.text.split()]
                            this_positions.append(this_position)
                        if len(self.md_frames) != 1:
                            this.append(this_positions)
                        else:
                            all_positions.append(np.array(this_positions))
            if len(self.md_frames) != 1:
                all_positions.append(this[-1])

        # only need those with forces
        all_positions = np.array(all_positions)
        print("Atomic positions along the MD trajectory loaded, converting to displacement, taking into account PBC")

        __all_displacements = np.array(
            [all_positions[i, :] - self.ref_coords for i in range(all_positions.shape[0])])

        __all_displacements = __all_displacements - np.round(__all_displacements)  # this is how it's done in Pymatgen

        all_positions = np.array(
            [__all_displacements[i, :] + self.ref_coords for i in range(__all_displacements.shape[0])])

        averaged_position = np.average(all_positions,axis=0)
        print(np.shape(averaged_position))

        all_atoms = []
        for i in range(len(self.ref_frame.asymmetric_unit)):
            for j in range(len(self.ref_frame.asymmetric_unit[i].atoms)):
                this_atom = self.ref_frame.asymmetric_unit[i].atoms[j]
                p = averaged_position[j]
                #p = np.dot(averaged_position[j],self.ref_frame.lattice.lattice_vectors)
                a = Atom(label=this_atom.label,scaled_position=cVector3D(p[0],p[1],p[2]))
                #new_cystal.asymmetric_unit[i].atoms.position = cVector3D(p[0],p[1],p[2])
                all_atoms.append(a)
        molecule = Molecule(atoms=all_atoms)
        new_cystal = Crystal(self.ref_frame.lattice,[molecule],1)

        VaspWriter().write_structure(new_cystal,filename='POSCAR_averaged')

if __name__=="__main__":
    MDAveragedStructure(ref_frame='POSCAR-md',md_frames='vasprun_md.xml').compute_averaged_structure()