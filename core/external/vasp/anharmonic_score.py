# Module containing codes to perform analysis on the degree of vibrational anharmoncity from ab-initio molecular dynamic
# trajectories obstained from VASP. This is a customized implementation based on the algorithm outlined in the following
# paper:
#
#   F. Knoop et al., 'Anharmonic Measures for Materials' (https://arxiv.org/abs/2006.14672)
#
# The author has an implementation to interface wtih FIH-aim code, and here it is adopted for the VASP code.

# this is a demonstration

from core.dao.vasp import VaspReader
from core.models.crystal import Crystal
import xml.etree.cElementTree as etree
import argparse
import numpy as np
import phonopy
import os
import matplotlib.pyplot as plt


class AnharmonicScore(object):

    def __init__(self, ref_frame=None,
                 md_fromes=None,
                 potim=1,
                 force_constants="force_constants.hdf5",
                 supercell=[1, 1, 1],
                 primitive_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 atoms=['Pb', 'Sn']):
        if isinstance(ref_frame, Crystal):
            self.ref_frame = ref_frame
        elif ('POSCAR' in ref_frame) or ('CONTCAR' in ref_frame):
            print("initialising reference frame from POSCAR ")
            self.ref_frame = VaspReader(input_location=ref_frame).read_POSCAR()

        self.ref_coords = np.array([[a.scaled_position.x, a.scaled_position.y, a.scaled_position.z] for a in
                                    self.ref_frame.asymmetric_unit[0].atoms])
        self.atom_masks = None
        if atoms is not None:
            self.atom_masks = [id for id, atom in enumerate(self.ref_frame.asymmetric_unit[0].atoms) if
                               atom.label in atoms]

        self.md_frames = md_fromes  # require the vasprun.xml containing the MD data

        self.get_dft_md_forces()
        self.get_all_md_atomic_displacements()

        if isinstance(force_constants, str):
            try:
                phonon = phonopy.load(supercell_matrix=supercell,  # WARNING - hard coded!
                                      primitive_matrix=primitive_matrix,
                                      unitcell_filename=ref_frame,
                                      force_constants_filename=force_constants)
            except:
                phonon = phonopy.load(supercell_matrix=supercell,  # WARNING - hard coded!
                                      primitive_matrix='auto',
                                      unitcell_filename=ref_frame,
                                      force_constants_filename=force_constants)
        print(np.shape(phonon.force_constants))
        new_shape = np.shape(phonon.force_constants)[0] * np.shape(phonon.force_constants)[2]
        self.force_constant = np.reshape(phonon.force_constants, (new_shape, new_shape))
        print("Force constants ready")

        self.time_series = [t * potim for t in range(len(self.all_displacements))]

    @property
    def lattice_vectors(self):
        """
        :return: A numpy array representation of the lattice vector
        """
        _lv = self.ref_frame.lattice.lattice_vectors
        return np.array(
            [[_lv[0][0], _lv[0][1], _lv[0][2]], [_lv[1][0], _lv[1][1], _lv[1][2]], [_lv[2][0], _lv[2][1], _lv[2][2]]])

    def get_dft_md_forces(self):
        all_forces = []
        for event, elem in etree.iterparse(self.md_frames):
            if elem.tag == 'varray':
                if elem.attrib['name'] == 'forces':
                    this_forces = []
                    for v in elem:
                        this_force = [float(_v) for _v in v.text.split()]
                        this_forces.append(this_force)
                    all_forces.append(this_forces)
        self.dft_forces = np.array(all_forces)
        print("Atomic forces along the MD trajectory loaded")

        # no need for conversion, forces already in eV/A
        # convert to Cartesian
        # for i in range(self.all_forces.shape[0]):
        #    self.all_forces[i, :, :] = np.dot(self.all_forces[i, :, :], self.lattice_vectors)

    def get_all_md_atomic_displacements(self):
        all_positions = []
        for event, elem in etree.iterparse(self.md_frames):
            if elem.tag == 'varray':
                if elem.attrib['name'] == 'positions':
                    this_positions = []
                    for v in elem:
                        this_position = [float(_v) for _v in v.text.split()]
                        this_positions.append(this_position)
                    all_positions.append(np.array(this_positions))
        # only need those with forces
        all_positions = all_positions[-len(self.dft_forces):]
        all_positions = np.array(all_positions)
        print("Atomic positions along the MD trajectory loaded, converting to displacement, taking into account PBC")

        self.all_displacements = np.array(
            [all_positions[i, :] - self.ref_coords[:] for i in range(all_positions.shape[0])])
        # impose periodic boundary conditions, no atomic displacement in fractional coordinates should be greater than
        # the length of each unit cell dimension
        self.all_displacements[self.all_displacements < -0.5] += 1
        self.all_displacements[self.all_displacements > 0.5] -= 1

        # Convert to Cartesian
        for i in range(self.all_displacements.shape[0]):
            self.all_displacements[i, :, :] = np.dot(self.all_displacements[i, :, :], self.lattice_vectors)

    @property
    def harmonic_forces(self):
        if (not hasattr(self, '_harmonic_forces')) or (
                hasattr(self, '_harmonic_forces') and self._harmonic_force is None):
            self._harmonic_force = np.zeros(np.shape(self.dft_forces))
            for i in range(np.shape(self.all_displacements)[0]):
                self._harmonic_force[i, :, :] = -1.0 * (
                        self.force_constant @ self.all_displacements[i, :, :].flatten()).reshape(
                    self.all_displacements[0, :, :].shape)
        return self._harmonic_force

    @property
    def anharmonic_forces(self):
        if (not hasattr(self, '_anharmonic_forces')) or (
                hasattr(self, '_anharmonic_forces') and self._anharmonic_forces is None):
            self._anharmonic_forces = self.dft_forces - self.harmonic_forces
        return self._anharmonic_forces

    def trajectory_normalized_dft_forces(self, flat=False):
        all_forces_std = self.dft_forces.flatten().std()
        out = np.zeros(np.shape(self.dft_forces))
        np.divide(self.dft_forces, all_forces_std, out=out)
        if flat:
            return out.flatten()
        return out

    def trajectory_normalized_anharmoonic_forces(self, flat=False):
        anharmonic_forces_std = self.anharmonic_forces.flatten().std()
        out = np.zeros(np.shape(self.anharmonic_forces))
        np.divide(self.anharmonic_forces, anharmonic_forces_std, out=out)
        if flat:
            return out.flatten()
        return out

    def atom_normalized_dft_forces(self,atom,flat=False):
        _mask = [id for id, a in enumerate(self.ref_frame.asymmetric_unit[0].atoms) if a.label == atom]
        dft_forces = self.dft_forces[:,_mask,:]
        dft_forces_std = dft_forces.flatten().std()
        out = np.zeros(np.shape(dft_forces))
        np.divide(dft_forces,dft_forces_std,out=out)
        if flat:
            return out.flatten()
        return out

    def atom_normalized_anharmonic_forces(self,atom,flat=False):
        _mask = [id for id, a in enumerate(self.ref_frame.asymmetric_unit[0].atoms) if a.label == atom]
        anharmonic_forces = self.anharmonic_forces[:,_mask,:]
        anharmonic_forces_std = anharmonic_forces.flatten().std()
        out = np.zeros(np.shape(anharmonic_forces))
        np.divide(anharmonic_forces, anharmonic_forces_std, out=out)
        if flat:
            return out.flatten()
        return out

    def plot_atom_joint_distributions(self):
        atoms = [a.label for a in self.ref_frame.asymmetric_unit[0].atoms]
        atoms = list(set(atoms))
        print(atoms)
        plt.figure(figsize=(6*len(atoms),6))
        for id,atom in enumerate(atoms):
            plt.subplot(1,len(atoms),id+1)
            plt.hist2d(self.atom_normalized_dft_forces(atom,flat=True), self.atom_normalized_anharmonic_forces(atom,flat=True), bins=200)
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            plt.xlabel("$F/\\sigma(F)$")
            plt.ylabel("$F^{A}/\\sigma(F^{A})$")
            plt.title(str(atom))
        plt.tight_layout()
        plt.savefig('atom_joint_PDF.png')

    def atomic_sigma(self,atoms):
        _mask = [id for id, atom in enumerate(self.ref_frame.asymmetric_unit[0].atoms) if
                               atom.label in atoms]
        atom_dft_forces = self.dft_forces[:,_mask,:]
        atom_anharomic_forces = self.anharmonic_forces[:,_mask,:]
        np.divide()

    def structure_averaged_sigma_trajectory(self):
        self.sigma_frames = []
        atom_direction_anh_std = self.anharmonic_forces.std(axis=0)
        print(np.shape(atom_direction_anh_std))
        for i in range(np.shape(self.all_displacements)[0]):
            # self.anharmonic_forces[i] = self.anharmonic_forces[i]/atom_direction_anh_std
            np.divide(self.anharmonic_forces[i], atom_direction_anh_std, out=self.anharmonic_forces[i])

        atom_direction_all_std = self.dft_forces.std(axis=0)
        for i in range(np.shape(self.all_displacements)[0]):
            # self.all_forces[i] = self.all_forces[i]/atom_direction_all_std
            np.divide(self.dft_forces[i], atom_direction_all_std, out=self.dft_forces[i])

        for i in range(np.shape(self.all_displacements)[0]):
            if self.atom_masks is None:
                rmse = self.anharmonic_forces[i, :, :].std()
                std = self.dft_forces[i, :, :].std()
                #_sigma = np.zeros(np.shape(self.anharmonic_forces[i, :, :]))
                #np.divide(np.abs(self.anharmonic_forces[i, :, :]), np.abs(self.dft_forces[i, :, :]), out=_sigma)
                #sigma_frame = np.average(_sigma)
                #print(sigma_frame)
            else:
                # print(self.atom_masks, print(np.shape(self.anharmonic_forces[i, self.atom_masks, :])))
                rmse = self.anharmonic_forces[i, self.atom_masks, :].std()
                std = self.dft_forces[i, self.atom_masks, :].std()
            sigma_frame = rmse / std
            self.sigma_frames.append(sigma_frame)

    def structural_sigma(self):
        if self.atom_masks is None:
            rmse = self.anharmonic_forces.std()
            std = self.dft_forces.std()
        else:
            rmse = self.anharmonic_forces[:, self.atom_masks, :].std()
            std = self.dft_forces[:, self.atom_masks, :].std()
        self.sigma = rmse / std


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options for analyzing anharmonic scores from MD trajectory',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--md_xml", type=str,
                        help="vasprun.xml file containing the molecular dynamic trajectory")
    parser.add_argument("--ref_frame", type=str,
                        help="POSCAR for the reference frame containing the static atomic positions at 0K")
    parser.add_argument("--joint_pdf", action='store_true',
                        help="plot the joint distributions of normalized total and anharmonic forces")
    parser.add_argument("--atom_joint_pdf", action='store_true',
                        help="plot the joint distributions of normalized total and anharmonic forces for each atom in the structure")
    args = parser.parse_args()

    scorer = AnharmonicScore(md_fromes=args.md_xml, ref_frame=args.ref_frame, atoms=None)

    if args.joint_pdf:
        plt.hist2d(scorer.trajectory_normalized_dft_forces(flat=True),
                   scorer.trajectory_normalized_anharmoonic_forces(flat=True),
                   bins=500)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.xlabel("$F/\\sigma(F)$")
        plt.ylabel("$F^{A}/\\sigma(F^{A})$")
        plt.tight_layout()
        plt.savefig('joint_PDF.png')

    if args.atom_joint_pdf:
        scorer.plot_atom_joint_distributions()

    # plt.plot(scorer.all_forces.flatten(),scorer.anharmonic_forces.flatten(),'b.')

    """
    plt.subplot(2,1,1)
    forces=scorer.all_forces
    plt.hist(forces.flatten(),bins=200,density=True)
    plt.xlim([-2.5,2.5])
    plt.subplot(2,1,2)
    plt.hist(forces.flatten()/forces.std(),bins=200,density=True)
    plt.xlim([-2,2])
    """

    # plt.plot(scorer.time_series, scorer.sigma_frames, 'b-')
