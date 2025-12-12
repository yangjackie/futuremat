import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from phonopy.structure.atoms import PhonopyAtoms
import phonopy
import seekpath

from core.phonon.phonopy_worker import PhonopyWorker


class PhononPlotter:
    def __init__(
            self,
            distances_set: list,
            frequencies_set: list,
            x_labels: list,
            connections: list,
            colors=None,
            legend_labels=None,
            linestyles=None,
            linewidths=None,
            figsize=(10, 6),
            commensurate_points=None,
            meterial_name=None,
    ):
        """
        initializing the phonon plotter
        Parameters:
        distances_set: list of lists of distances for each phonon band structure
        frequencies_set: list of lists of frequencies for each phonon band structure
        x_labels: list of labels for the x-axis
        connections: list of booleans indicating if the x-path is connected or not
        colors: list of colors for each phonon band structure
        legend_labels: list of labels for the legend
        linestyles: list of linestyles for each phonon band structure
        linewidths: list of linewidths for each phonon band structure
        figsize: tuple of figure size
        commensurate_points: list of commensurate points for the x-axis
        meterial_name: name of the material

        Code extracted from https://github.com/JaGeo/mace-mp-03b-phonon-benchmark/blob/main/functions/plots.py
        """
        self.distances_set = distances_set
        self.frequencies_set = frequencies_set
        self.x_labels = x_labels
        self.connections = connections
        self.colors = (
            colors
            if colors is not None
            else ["black", "red", "blue", "green", "orange"]
        )
        self.legend_labels = (
            legend_labels
            if legend_labels is not None
            else ["DFT", "MP-0", "DFT+MP0", "ft MACE", "DFT+ft MACE"]
        )
        self.linestyles = (
            linestyles if linestyles is not None else ["-", "-", "-", ":", ":"]
        )
        self.linewidths = (
            linewidths if linewidths is not None else [1] * len(self.colors)
        )
        self.figsize = figsize
        self.commensurate_points = commensurate_points
        self.material_name = meterial_name

    def _create_xtick_labels(self):
        """create xtick labels for the phonon band structure plot
        Depending on wehter the x-path is connected or not, the labels have to repeat or change
        An example:
        x_labels = [Gamma,X,K,L, Gamma]
        connection = [True, True, False]
        '--------'---------'-----------'
        G        X        K| L         G

        x_labels: list of labels for example: [Gamma,X,K, Gamma]
        connection: list with True and False label if x-path is connected or not. for example: [True, True, False, True]
        """
        xtick_labels = []
        if False not in self.connections:
            return self.x_labels
        else:
            xtick_labels.append(self.x_labels[0])
            count = 1

            for connection in self.connections:
                if count >= len(self.x_labels) - 1:
                    xtick_labels.append(self.x_labels[-1])
                    break
                if connection:
                    xtick_labels.append(self.x_labels[count])
                    count += 1
                else:
                    xtick_labels.append(
                        str(self.x_labels[count]) + " | " + self.x_labels[count + 1]
                    )
                    count += 2
        return xtick_labels

    def _plot_phonon_bands(
            self, ax, distances, frequencies, color, linestyle, linewidth
    ):
        """plots phonon band structrue"""
        num_paths, num_kpoints, num_bands = frequencies.shape
        for path in range(num_paths):
            for band in range(num_bands):
                ax.plot(
                    distances[path],
                    frequencies[path, :, band],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )

    def _add_vertical_lines_and_commensurate_points(self, ax, distances, comm_points):
        """add vertical lines and commensurate points to the plot"""
        xticks = []
        num_paths = distances.shape[0]
        for path in range(num_paths):
            start, end = distances[path][0], distances[path][-1]
            ax.axvline(x=start, color="black", linestyle="--", linewidth=0.5)
            ax.axvline(x=end, color="black", linestyle="--", linewidth=0.5)
            xticks.extend([start, end])
            if comm_points is not None and len(comm_points) > 0:
                for point in comm_points[path]:
                    ax.plot(distances[path][point], 0, color="red", marker="D")
        return sorted(set(xticks))

    def beautiful_phonon_plotter(self, savefig=False, filename=None, showfig=True):
        """creates a beautiful phonon plot
        gives back:
        fig, ax: figure and axis of the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # plot all data
        for i in range(len(self.distances_set)):
            self._plot_phonon_bands(
                ax,
                np.array(self.distances_set)[i],
                np.array(self.frequencies_set)[i],
                self.colors[i],
                self.linestyles[i],
                self.linewidths[i],
            )
            # empty plot for legend
            ax.plot(
                [],
                [],
                color=self.colors[i],
                linestyle=self.linestyles[i],
                label=self.legend_labels[i],
            )

        # add vertcal lines and commensurate points
        default_comm_points = (
            self.commensurate_points
            if self.commensurate_points is not None
            else [[]] * np.array(self.distances_set)[0].shape[0]
        )
        xticks = self._add_vertical_lines_and_commensurate_points(
            ax, np.array(self.distances_set)[0], default_comm_points
        )

        # set xtick labels
        xtick_labels = self._create_xtick_labels()
        if len(xticks) == len(xtick_labels):
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)

        # additional plot settings
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.set_ylabel("Frequency [THz]")
        ax.set_xlabel("Wave vector")
        ax.set_xlim(0, np.array(self.distances_set)[0][-1, -1])
        ax.legend(loc="lower right")

        if self.material_name:
            plt.suptitle(self.material_name, y=0.95)
        plt.tight_layout()
        if savefig:
            plt.savefig(filename, bbox_inches="tight")
        if showfig:
            plt.show()
        return fig, ax


def prepare_and_plot(dft_path=None,
                     dft_fc_file='force_constants.hdf5',
                     dft_poscar_file='CONTCAR',
                     primitive_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     calculator=None,
                     savefig=False,
                     filename=None,
                     plot=False):
    """Prepare phonon band structures (DFT and MLFF) and optionally plot them.

    This function loads a DFT phonon calculation from a unitcell (`CONTCAR`) and
    its force constants file, computes the phonon band structure with seekpath to identify 
    reciproal space q-point paths, generates an equivalent band structure using a machine-learned force-field
    via `PhonopyWorker`, and returns a dictionary with frequency and group
    velocity data for both DFT and MLFF results.

    Args:
        dft_path (str or None): Path prefix where `dft_fc_file` and
            `dft_poscar_file` are located. If `None`, files are taken from the
            current working directory. Trailing slash expected if provided.
        dft_fc_file (str): Filename (or suffix) of the DFT force constants
            HDF5 file. Default: `'force_constants.hdf5'`.
        dft_poscar_file (str): Filename (or suffix) of the DFT POSCAR/CONTCAR
            unit cell file. Default: `'CONTCAR'`.
        primitive_matrix (list[list[int]]): 3x3 matrix describing the
            primitive cell transformation used by phonopy when loading the
            DFT calculation. Default is the identity matrix.
        calculator: An ASE-compatible calculator (or ML potential wrapper)
            to evaluate forces on displaced supercells when building force
            constants for the ML model. This is passed to `PhonopyWorker`.
        savefig (bool): If True, save the generated phonon comparison plot to
            `filename`.
        filename (str or None): Output filename for the saved plot. If None,
            a default name derived from `dft_path` is used.
        plot (bool): If True, display the phonon comparison plot using
            matplotlib. If False, the function still computes and returns data.

    Returns:
        dict: `data_dict` containing frequency and group-velocity computed from DFT
        and MLFF methods allowing the comparison of phonon properties.

    Notes:
        - The function assumes `dft_path` if provided ends with a path
          separator; it concatenates `dft_path + dft_fc_file` and
          `dft_path + dft_poscar_file`.
        - The MLFF computation uses `PhonopyWorker.generate_force_constants()`
          to create force constants in memory; ensure `calculator` is set and
          able to evaluate forces for each displaced supercell.
    """

    dft_fc_file = dft_path + dft_fc_file
    dft_poscar_file = dft_path + dft_poscar_file
    dft_phonon = phonopy.load(supercell_matrix=np.array([2, 2, 2]),  # WARNING - hard coded!
                              primitive_matrix=primitive_matrix,
                              unitcell_filename=dft_poscar_file,
                              force_constants_filename=dft_fc_file)

    from phonopy.phonon.band_structure import BandStructure, get_band_qpoints_by_seekpath
    bands, labels, path_connections = get_band_qpoints_by_seekpath(
        dft_phonon._primitive, 50, is_const_interval=False
    )
    dft_phonon.run_band_structure(
        bands,
        with_eigenvectors=False,
        with_group_velocities=True,
        path_connections=path_connections,
        labels=labels,
        is_legacy_plot=False
    )
    dft_distances = dft_phonon.band_structure.distances
    dft_frequencies = dft_phonon.band_structure.frequencies

    mace_phonon = PhonopyWorker(
        structure=read(dft_poscar_file),
        supercell_matrix=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        displacement_distance=0.01,
        calculator=calculator
    )
    mace_phonon.generate_force_constants()
    mace_phonon.phonopy.run_band_structure(
        bands,
        with_eigenvectors=False,
        with_group_velocities=True,
        path_connections=path_connections,
        labels=labels,
        is_legacy_plot=False,
    )
    mace_distances = mace_phonon.phonopy.band_structure.distances
    mace_frequencies = mace_phonon.phonopy.band_structure.frequencies

    materials_name = dft_path.split("/")[-2].split("_")[-1]

    _dft_frequencies = np.ndarray.flatten(np.array(dft_frequencies))
    _dft_frequencies = _dft_frequencies.astype(complex)
    _dft_frequencies[_dft_frequencies < 0] = 1j * np.abs(_dft_frequencies[_dft_frequencies < 0])

    _mace_frequencies = np.ndarray.flatten(np.array(mace_frequencies))
    _mace_frequencies = _mace_frequencies.astype(complex)
    _mace_frequencies[_mace_frequencies < 0] = 1j * np.abs(_mace_frequencies[_mace_frequencies < 0])

    stdev = np.real(np.sqrt(np.average(np.square(np.square(_mace_frequencies) - np.square(_dft_frequencies)))))
    print(f"Standard deviation of frequencies for {materials_name}: {stdev:.4f} THz")

    dft_group_velocities = np.ndarray.flatten(np.array(dft_phonon.band_structure.group_velocities))
    mace_group_velocities = np.ndarray.flatten(np.array(mace_phonon.phonopy.band_structure.group_velocities))

    vel_stdev = np.sqrt(np.average(np.square(dft_group_velocities - mace_group_velocities)))
    print(f"Standard deviation of group velocities for {materials_name}: {vel_stdev:.4f} m/s")

    if filename is None:
        filename = dft_path + materials_name + '_dft_mlff_phonon.pdf'
    if plot:
        PhononPlotter(
            distances_set=[dft_distances, mace_distances],
            frequencies_set=[dft_frequencies, mace_frequencies],
            x_labels=labels,
            connections=path_connections,
            colors=["blue", "red"],
            legend_labels=["DFT", "MACE"],
            linestyles=["-", '--'],
            linewidths=[1, 1],
            figsize=(7, 5),
            #meterial_name=f"{materials_name}#: Freq Stdev: {stdev:.4f} THz; group velocity stdev: {vel_stdev:.4f} (m/s)").beautiful_phonon_plotter(
            meterial_name=f"{materials_name}").beautiful_phonon_plotter(
            savefig=savefig, filename=filename, showfig=False)

    data_dict = {"name": materials_name,
                 "dft_frequencies": dft_frequencies,
                 "mace_frequencies": mace_frequencies,
                 "dft_group_velocities": dft_group_velocities,
                 "mace_group_velocities": mace_group_velocities}
    return data_dict


