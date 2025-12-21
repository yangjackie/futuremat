from sumo.cli.bandplot import bandplot
from easyunfold.unfold import UnfoldKSet, Unfold
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_primitive_and_unfolded_super_cell_band_structrue(
    primitive_path: str = "./band_structure/",
    supercell_path: str = "./finite_displacement_phonons/band_structure/",
):
    """Plot the primitive cell band structure and unfolded supercell band structure together.
    Args:
        primitive_path (str): Path to the primitive cell band structure directory.
        supercell_path (str): Path to the supercell band structure directory.
    """
    supercell_path = os.path.abspath(supercell_path)
    primitive_path = os.path.abspath(primitive_path)

    # primitive cell band structure
    __prim_xml = "{}/vasprun.xml".format(primitive_path)

    # using the pymatgen BSPlotter to plot and save the band structure for the primitive cell
    import matplotlib.pyplot as plt

    plt = bandplot(filenames=[__prim_xml], plt=plt)

    # get the wavefunction for the supercell calculations to find out the VBM/CBM energy,
    # so it can be aligned with the primitive cell band structure (here assume the supercell is
    # still a semiconductor so we set the zero energy at VBM!)
    unfold = Unfold(M=2 * np.eye(3), fname="{}/WAVECAR".format(supercell_path))  # assuming the supercell is 2x2x2 of the primitive cell
    zero_energy, _ = unfold.get_vbm_cbm()

    # unfolded super cell band structure
    unfoled_json = "{}/easyunfold.json".format(os.getcwd())
    unfold_kset = UnfoldKSet.from_file(unfoled_json)
    __distances = unfold_kset.get_kpoint_distances()
    __weights = unfold_kset.get_spectral_weights()

    x = []
    y = []
    w = []

    for counter, dis in enumerate(__distances):
        for w_for_this_dis in __weights[counter][0][0]:
            x.append(dis)
            y.append(w_for_this_dis[0] - zero_energy)  # only one band for phonon
            w.append(w_for_this_dis[1])  # spectral weight

    plt.scatter(x, y, c="r", s=[weight * 2.5 for weight in w], alpha=0.5, label="Unfolded Supercell Band Structure")
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/primitive_band_structure_sc.png", dpi=300)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Plotting routines for studying electronic and phonon interactions.")
    parser.add_argument("-plot_band","--plot_overlayed_band_structrue",action="store_true",
                        help="Plot the primitive cell band structure and unfolded supercell band structure together.")
    parser.add_argument("-prim","--primitive",type=str,
                        help="Path to the primitive cell band structure directory.")
    parser.add_argument("-super","--supercell",type=str,
                        help="Path to the supercell band structure directory")
    
    parser.add_argument("-ppbs", "--plot_phonon_band_structure", action="store_true",
                        help="Plot phonon band structure from force constants and POSCAR in the specified directory")
    parser.add_argument("-nac", "--non-analytical_correction", action="store_true",
                        help="Apply non-analytical correction for phonon calculations.")
    
    args = parser.parse_args()
    # fmt: on

    if args.plot_overlayed_band_structrue:
        plot_primitive_and_unfolded_super_cell_band_structrue(primitive_path=args.primitive, supercell_path=args.supercell)
    elif args.plot_phonon_band_structure:
        from core.phonon.phonon_plotter import prepare_and_plot_single_phonon_band_structure

        print("Plotting phonon band structure..., do we include NAC?", args.non_analytical_correction)
        prepare_and_plot_single_phonon_band_structure(
            path="finite_displacement_phonons",
            fc_file="force_constants.hdf5",
            poscar_file="POSCAR",
            supercell_matrix=[2, 2, 2],  # this need to be made more flexible
            num_qpoints=50,
            labels=["PBE-sol"],
            colors=["blue"],
            savefig=True,
            nac_correction=args.non_analytical_correction,
        )
