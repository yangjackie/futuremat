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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity


class AnharmonicScore(object):

    def __init__(self, ref_frame=None,
                 unit_cell_frame=None,
                 md_frames=None,
                 potim=1,
                 force_constants="force_constants.hdf5",
                 supercell=[1, 1, 1],
                 primitive_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 atoms=None,
                 include_third_order=False,
                 third_order_fc='./phono3py/fc3.hdf5',
                 include_fourth_order=False,
                 fourth_order_fc=None):
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

        self.md_frames = md_frames  # require the vasprun.xml containing the MD data

        self.get_dft_md_forces()
        self.get_all_md_atomic_displacements()

        if isinstance(force_constants, str):
            try:
                phonon = phonopy.load(supercell_matrix=supercell,  # WARNING - hard coded!
                                      primitive_matrix=primitive_matrix,
                                      unitcell_filename=unit_cell_frame,
                                      force_constants_filename=force_constants)
                print("Use supercell " + str(supercell))
                print("Use primitive matrix " + str(primitive_matrix) + " done")
            except:
                phonon = phonopy.load(supercell_matrix=supercell,  # WARNING - hard coded!
                                      primitive_matrix='auto',
                                      unitcell_filename=unit_cell_frame,
                                      force_constants_filename=force_constants)
            print("INPUT PHONOPY force constant shape ", np.shape(phonon.force_constants))

            #TODO - if the input supercell is not [1,1,1], then it will need to be expanded into the correct supercell shape here!

            new_shape = np.shape(phonon.force_constants)[0] * np.shape(phonon.force_constants)[2]
            self.force_constant = np.zeros((new_shape, new_shape))
            self.force_constant = phonon.force_constants.transpose(0, 2, 1, 3).reshape(new_shape, new_shape)
        elif force_constants is None:
            """
            Loading directly from SPOSCAR (supercell structure) and FORCESET to avoid problem of the need for
            reconstructing the force constants for supercells from primitive cells
            """
            print("Here try to use FORCE_SETS")
            phonon = phonopy.load(supercell_filename="./SPOSCAR", log_level=1,force_sets_filename='FORCE_SETS')
            phonon.produce_force_constants()

            print("INPUT PHONOPY force constant shape ", np.shape(phonon.force_constants))
            new_shape = np.shape(phonon.force_constants)[0] * np.shape(phonon.force_constants)[2]
            self.force_constant = np.zeros((new_shape, new_shape))
            self.force_constant = phonon.force_constants.transpose(0, 2, 1, 3).reshape(new_shape, new_shape)
        elif isinstance(force_constants, np.ndarray):
            new_shape = np.shape(force_constants)[0] * np.shape(force_constants)[2]
            self.force_constant = np.zeros((new_shape, new_shape))
            self.force_constant = force_constants.transpose(0, 2, 1, 3).reshape(new_shape, new_shape)

        print("force constant reshape ", np.shape(self.force_constant))
        print("Force constants ready")
        self.time_series = [t * potim for t in range(len(self.all_displacements))]

        self.force_constant_3 = None
        self.include_third_oder = include_third_order
        if self.include_third_oder:
            if isinstance(third_order_fc, str):
                if os.path.isfile(third_order_fc):
                    print("Found third order force constants")
                    import h5py
                    f = h5py.File(third_order_fc)  # './phono3py/fc3.hdf5'
                    raw_force_constant_3 = np.array(f['fc3'])
            elif isinstance(third_order_fc, np.ndarray):
                raw_force_constant_3 = third_order_fc

            s = np.shape(raw_force_constant_3)[0] * 3
            self.force_constant_3 = raw_force_constant_3.transpose([0, 3, 1, 4, 2, 5]).reshape(s, s, s)
            print("Reshaped 3rd order force constant is ", np.shape(self.force_constant_3))

        self.force_constant_4 = None
        self.include_fourth_order = include_fourth_order
        if self.include_fourth_order:
            if isinstance(fourth_order_fc, np.ndarray):
                raw_force_constant_4 = fourth_order_fc
            s = np.shape(raw_force_constant_4)[0] * 3
            self.force_constant_4 = raw_force_constant_4.transpose([0, 4, 1, 5, 2, 6, 3, 7]).reshape(s, s, s, s)

    def plot_fc(self):
        plt.matshow(self.force_constant)
        plt.colorbar()
        plt.savefig('fc.pdf')

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

        __all_displacements = np.array(
            [all_positions[i, :] - self.ref_coords for i in range(all_positions.shape[0])])

        # periodic boundary conditions
        # __all_displacements = (__all_displacements + 0.5 + 1e-5) % 1 - 0.5 - 1e-5
        __all_displacements = __all_displacements - np.round(__all_displacements)  # this is how it's done in Pymatgen
        # Convert to Cartesian
        self.all_displacements = np.zeros(np.shape(__all_displacements))

        for i in range(__all_displacements.shape[0]):
            np.dot(__all_displacements[i, :, :], self.lattice_vectors, out=self.all_displacements[i, :, :])

    @property
    def harmonic_forces(self):
        if (not hasattr(self, '_harmonic_forces')) or (
                hasattr(self, '_harmonic_forces') and self._harmonic_force is None):
            self._harmonic_force = np.zeros(np.shape(self.dft_forces))
            for i in range(np.shape(self.all_displacements)[0]):  # this loop over MD frames
                self._harmonic_force[i, :, :] = -1.0 * (
                    np.dot(self.force_constant, self.all_displacements[i, :, :].flatten())).reshape(
                    self.all_displacements[0, :, :].shape)
        return self._harmonic_force

    @property
    def third_order_forces(self):
        print("Do we have third_order constant " + str(self.force_constant_3.__class__))
        if self.force_constant_3 is not None:
            if (not hasattr(self, '_third_order_forces')) or (
                    hasattr(self, '_third_order_forces') and self._third_order_forces is None):
                self._third_order_forces = np.zeros(np.shape(self.dft_forces))
                _a = self.force_constant_3
                for i in range(np.shape(self.all_displacements)[0]):  # this loop over MD frames
                    _b = self.all_displacements[i, :, :].flatten()
                    _A = np.einsum('ijk,k->ij', _a, _b)
                    self._third_order_forces[i, :, :] = -1 * np.einsum('ij,j->i', _A, _b).reshape(
                        self.all_displacements[0, :, :].shape)
                return self._third_order_forces / 2.0  # see https://hiphive.materialsmodeling.org/background/force_constants.html

    @property
    def fourth_order_forces(self):
        print("Do we have fourth_order constant " + str(self.force_constant_4.__class__))
        if self.force_constant_4 is not None:
            if (not hasattr(self, '_fourth_order_forces')) or (
                    hasattr(self, '_fourth_order_forces') and self._fourth_order_forces is None):
                self._fourth_order_forces = np.zeros(np.shape(self.dft_forces))
                _a = self.force_constant_4
                for i in range(np.shape(self.all_displacements)[0]):  # this loop over MD frames
                    print("fourth order forces, frame "+str(i))
                    _b = self.all_displacements[i, :, :].flatten()
                    _A = np.einsum('ijkl,l->ijk', _a, _b)
                    _B = np.einsum('ijk,k->ij', _A, _b)
                    self._fourth_order_forces[i, :, :] = -1 * np.einsum('ij,j->i', _B, _b).reshape(
                        self.all_displacements[0, :, :].shape)
                print("fourth order forces done")
                return self._fourth_order_forces / 6.0

    @property
    def anharmonic_forces(self):
        if (not hasattr(self, '_anharmonic_forces')) or (
                hasattr(self, '_anharmonic_forces') and self._anharmonic_forces is None):
            self._anharmonic_forces = self.dft_forces - self.harmonic_forces
            if self.include_third_oder:
                self._anharmonic_forces = self._anharmonic_forces - self.third_order_forces
            if self.include_fourth_order:
                self._anharmonic_forces = self._anharmonic_forces - self.fourth_order_forces
        return self._anharmonic_forces

    def trajectory_normalized_dft_forces(self, flat=False):
        all_forces_std = self.dft_forces.flatten().std()
        out = np.zeros(np.shape(self.dft_forces))
        np.divide(self.dft_forces, all_forces_std, out=out)
        if flat:
            return out.flatten()
        return out

    def trajectory_normalized_anharmonic_forces(self, flat=False):
        # anharmonic_forces_std = self.anharmonic_forces.flatten().std()
        all_forces_std = self.dft_forces.flatten().std()
        out = np.zeros(np.shape(self.anharmonic_forces))
        np.divide(self.anharmonic_forces, all_forces_std, out=out)
        if flat:
            return out.flatten()
        return out

    def atom_normalized_dft_forces(self, atom, flat=False):
        _mask = [id for id, a in enumerate(self.ref_frame.asymmetric_unit[0].atoms) if a.label == atom]
        dft_forces = self.dft_forces[:, _mask, :]
        dft_forces_std = dft_forces.flatten().std()
        out = np.zeros(np.shape(dft_forces))
        np.divide(dft_forces, dft_forces_std, out=out)
        if flat:
            return out.flatten(), dft_forces_std
        return out, dft_forces_std

    def atom_normalized_anharmonic_forces(self, atom, flat=False):
        _mask = [id for id, a in enumerate(self.ref_frame.asymmetric_unit[0].atoms) if a.label == atom]
        print(atom, _mask)
        anharmonic_forces = self.anharmonic_forces[:, _mask, :]

        dft_forces = self.dft_forces[:, _mask, :]
        dft_forces_std = dft_forces.flatten().std()

        out = np.zeros(np.shape(anharmonic_forces))
        np.divide(anharmonic_forces, dft_forces_std, out=out)
        if flat:
            return out.flatten()
        return out

    def plot_atom_joint_distributions(self):
        atoms = [a.label for a in self.ref_frame.asymmetric_unit[0].atoms]
        atoms = list(set(atoms))
        atoms = sorted(atoms)
        print(atoms)
        fig, axs = plt.subplots(1, len(atoms), figsize=(4 * len(atoms), 4))
        for id, atom in enumerate(atoms):
            divider = make_axes_locatable(axs[id])
            cax1 = divider.append_axes("right", size="5%", pad=0.05)

            X, std = self.atom_normalized_dft_forces(atom, flat=True)
            Y = self.atom_normalized_anharmonic_forces(atom, flat=True)

            X_pred, Y_pred = np.mgrid[-2:2:50j, -2:2:50j]
            positions = np.vstack([X_pred.ravel(), Y_pred.ravel()])
            values = np.vstack([X, Y])
            print("Perform Gaussian Kernel Density Estimate")
            kernel = gaussian_kde(values)
            Z = np.reshape(kernel(positions).T, X_pred.shape)

            print("Making the plot")
            a = axs[id].imshow(np.rot90(Z), cmap=plt.get_cmap('Blues'), extent=[-1, 1, -1, 1])

            axs[id].plot([-1, -0.5, 0, 0.5, 1], [std for i in range(5)], 'k--')
            axs[id].plot([-1, -0.5, 0, 0.5, 1], [-1.0 * std for i in range(5)], 'k--')

            axs[id].set_xlim([-1, 1])
            axs[id].set_ylim([-1, 1])
            axs[id].set_xlabel("$F_{i}/\\sigma(F_{i})$", fontsize=16)
            axs[id].set_ylabel("$F^{A}_{i}/\\sigma(F_{i})$", fontsize=16)
            axs[id].set_title(str(atom), fontsize=20)
            axs[id].tick_params(axis='both', which='major', labelsize=10)
            plt.colorbar(a, cax=cax1)  # .set_label(label='probability density',size=15)
        plt.tight_layout()
        plt.savefig('atom_joint_PDF.pdf')

    def plot_total_joint_distribution(self, x='DFT', y='anh'):
        if x == 'DFT':
            X = self.dft_forces.flatten() / self.dft_forces.flatten().std()
        else:
            raise NotImplementedError()
        if y == 'anh':
            Y = self.anharmonic_forces.flatten() / self.dft_forces.flatten().std()
        elif y == 'har':
            Y = self.harmonic_forces.flatten() / self.dft_forces.flatten().std()
        else:
            raise NotImplementedError()

        X_pred, Y_pred = np.mgrid[-1:1:50j, -1:1:50j]
        positions = np.vstack([X_pred.ravel(), Y_pred.ravel()])
        values = np.vstack([X, Y])
        print("Gaussian KDE")
        kernel = gaussian_kde(values)
        print("Kernel done")
        Z = np.reshape(kernel(positions).T, X_pred.shape)
        plt.imshow(np.rot90(Z), cmap=plt.get_cmap('Blues'), extent=[-1, 1, -1, 1])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

        plt.xlabel("$F/\\sigma(F)$", fontsize=16)
        if y == 'anh':
            plt.ylabel("$F^{A}/\\sigma(F)$", fontsize=16)
        elif y == 'har':
            plt.ylabel("$F^{(2)}/\\sigma(F)$", fontsize=16)

        if y == 'anh':
            plt.plot([-1, -0.5, 0, 0.5, 1], [self.anharmonic_forces.flatten().std() for i in range(5)], 'k--')
            plt.plot([-1, -0.5, 0, 0.5, 1], [-1.0 * self.anharmonic_forces.flatten().std() for i in range(5)], 'k--')

        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.colorbar().set_label(label='probability density', size=15)

        plt.tight_layout()
        plt.savefig('joint_PDF.pdf')

    def structure_averaged_sigma_trajectory(self):
        atoms = [a.label for a in self.ref_frame.asymmetric_unit[0].atoms]
        atoms = list(set(atoms))
        self.sigma_frames = []
        for atom in atoms:
            anh_f = self.atom_normalized_anharmonic_forces(atom)
            dft_f = self.atom_normalized_dft_forces(atom)

            for i in range(np.shape(self.all_displacements)[0]):
                _sigma_frame = anh_f[i, :, :].flatten().std() / dft_f[i, :, :].flatten().std()
                self.sigma_frames.append(_sigma_frame)

    def structural_sigma(self, return_trajectory=False):
        if self.atom_masks is None:
            rmse = self.anharmonic_forces
            std = self.dft_forces
        else:
            rmse = self.anharmonic_forces[:, self.atom_masks, :]
            std = self.dft_forces[:, self.atom_masks, :]

        if not return_trajectory:
            print(return_trajectory, 'calculate sigma')
            sigma = rmse.std() / std.std()
            print("Sigma for entire structure over the MD trajectory is ", str(sigma))
            return sigma, self.time_series
        else:
            print(np.shape(rmse.std(axis=(1, 2), dtype=np.float64)))
            sigma = rmse.std(axis=(1, 2), dtype=np.float64) / std.std(axis=(1, 2), dtype=np.float64)
            return sigma, self.time_series


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options for analyzing anharmonic scores from MD trajectory',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--md_xml", type=str,
                        help="vasprun.xml file containing the molecular dynamic trajectory")
    parser.add_argument("--ref_frame", type=str,
                        help="POSCAR for the reference frame containing the static atomic positions at 0K")
    parser.add_argument("--unit_cell_frame", type=str,
                        help="POSCAR for the unitcell.")
    parser.add_argument("--joint_pdf", action='store_true',
                        help="plot the joint distributions of normalized total and anharmonic forces")
    parser.add_argument("--atom_joint_pdf", action='store_true',
                        help="plot the joint distributions of normalized total and anharmonic forces for each atom in the structure")
    parser.add_argument("--md_time_step", type=float, default=1,
                        help="Time step for the molecular dynamic trajectory (in fs), default: 1fs")

    parser.add_argument('--sigma', action='store_true',
                        help="Return the structural sigma value from this MD trajectory")
    parser.add_argument('--trajectory', action='store_true',
                        help="Whether to return sigma for each frame of the MD trajectory")

    parser.add_argument('--plot_trajectory', action='store_true',
                        help="Whether to plot sigma for each frame of the MD trajectory")

    parser.add_argument("-X", "--X", type=str, default='DFT',
                        help='data to plot along the X-axis for the joint probability distribution, default: DFT force')
    parser.add_argument("-Y", "--Y", type=str, default='anh',
                        help='data to plot along the Y-axis for the joint probability distribution, default: anharmonic forces')

    args = parser.parse_args()

    scorer = AnharmonicScore(md_frames=args.md_xml, unit_cell_frame=args.unit_cell_frame, ref_frame=args.ref_frame, atoms=None,
                             potim=args.md_time_step)

    from matplotlib import rc

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)

    if args.joint_pdf:
        scorer.plot_total_joint_distribution(x=args.X, y=args.Y)

    if args.atom_joint_pdf:
        scorer.plot_atom_joint_distributions()

    if args.sigma:
        sigma, time_stps = scorer.structural_sigma(return_trajectory=args.trajectory)
        if args.plot_trajectory:
            plt.plot(time_stps, sigma, 'b-')
            plt.xlabel("Time (fs)", fontsize=16)
            plt.ylabel("$\\sigma(t)$", fontsize=16)
            plt.tight_layout()
            plt.savefig("sigma_trajectory.pdf")
