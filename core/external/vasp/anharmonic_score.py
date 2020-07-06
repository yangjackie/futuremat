# Module containing codes to perform analysis on the degree of vibrational anharmoncity from ab-initio molecular dynamic
# trajectories obstained from VASP. This is a customized implementation based on the algorithm outlined in the following
# paper:
#
#   F. Knoop et al., 'Anharmonic Measures for Materials' (https://arxiv.org/abs/2006.14672)
#
# The author has an implementation to interface wtih FIH-aim code, and here it is adopted for the VASP code.

from core.dao.vasp import VaspReader
import xml.etree.cElementTree as etree
import argparse
import numpy as np
import phonopy
import os
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Options for analyzing anharmonic scores from MD trajectory',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--md_xml", type=str,
                        help="vasprun.xml file containing the molecular dynamic trajectory")
    parser.add_argument("--ref_frame", type=str,
                        help="POSCAR for the reference frame containing the static atomic positions at 0K")
    args = parser.parse_args()

    ref_frame = VaspReader(input_location=args.ref_frame).read_POSCAR()
    all_elements = [a.label for a in ref_frame.asymmetric_unit[0].atoms]
    ref_coords = np.array(
        [[a.scaled_position.x, a.scaled_position.y, a.scaled_position.z] for a in ref_frame.asymmetric_unit[0].atoms])
    print("Reference frame loaded")

    _lv = ref_frame.lattice.lattice_vectors
    lv = np.array(
        [[_lv[0][0], _lv[0][1], _lv[0][2]], [_lv[1][0], _lv[1][1], _lv[1][2]], [_lv[2][0], _lv[2][1], _lv[2][2]]])

    #Extracting all the forces from vasprun.xml
    all_forces = []
    for event, elem in etree.iterparse(args.md_xml):
        if elem.tag == 'varray':
            if elem.attrib['name'] == 'forces':
                this_forces = []
                for v in elem:
                    this_force = [float(_v) for _v in v.text.split()]
                    this_forces.append(this_force)
                all_forces.append(this_forces)
    all_forces = np.array(all_forces)
    print("Atomic forces along the MD trajectory loaded")

    #Extract all the atomic positions from the vasprun.xml
    all_positions =[]
    for event, elem in etree.iterparse(args.md_xml):
        if elem.tag == 'varray':
            if elem.attrib['name'] == 'positions':
                this_positions = []
                for v in elem:
                    this_position = [float(_v) for _v in v.text.split()]
                    this_positions.append(this_position)
                all_positions.append(np.array(this_positions))
    #only need those with forces
    all_positions = all_positions[-len(all_forces):]
    all_positions = np.array(all_positions)
    print("Atomic positions along the MD trajectory loaded, converting to displacement, taking into account PBC")

    all_displacements = np.array([all_positions[i, :] - ref_coords[:] for i in range(all_positions.shape[0])])
    # impose periodic boundary conditions, no atomic displacement in fractional coordinates should be greater than
    # the length of each unit cell dimension
    all_displacements[all_displacements < -0.5] += 1
    all_displacements[all_displacements > 0.5] -= 1

    #now need to convert the fractional coordinates to Cartesian Coordinates
    for i in range(all_forces.shape[0]):
        all_forces[i, :, :] = np.dot(all_forces[i, :, :], lv)

    for i in range(all_displacements.shape[0]):
        all_displacements[i, :, :] = np.dot(all_displacements[i, :, :], lv)

    ##Inspecting what the distribution of atomic forces looks like
    #_forces = all_forces.flatten()
    #plt.hist(_forces, bins=1000)
    #plt.tight_layout()
    #plt.savefig('force_stats.pdf')

    print("Now retrieving the force constant matrix from PHONOPY calculations")

    #cwd = os.getcwd()
    #os.chdir('./phonon')

    # After FORCE_SETS is produced from Phonopy calculation, the following needs to be run to get the force_constants file
    #  ~/.local/bin/phonopy
    phonon = phonopy.load(supercell_matrix=[1, 1, 1], #WARNING - hard coded!
                          primitive_matrix='auto',
                          unitcell_filename="POSCAR_equ",
                          force_constants_filename="force_constants.hdf5")
    #os.chdir(cwd)

    #Now calculate the harmonic forces expected on atom displaced by delta R
    #see https://gitlab.com/vibes-developers/vibes/-/blob/master/vibes/molecular_dynamics/utils.py
    new_shape = np.shape(phonon.force_constants)[0]*np.shape(phonon.force_constants)[2]
    force_constant = np.reshape(phonon.force_constants,(new_shape,new_shape))

    harmonic_force = np.zeros(np.shape(all_forces))
    anharmonic_force_component = np.zeros(np.shape(all_forces))

    all_sigma=[]

    for i in range(np.shape(all_displacements)[0]):
        harmonic_force[i,:,:] = -(force_constant @ all_displacements[i,:,:].flatten()).reshape(all_displacements[0,:,:].shape)
        anharmonic_force_component[i,:,:]= all_forces[i,:,:] - harmonic_force[i,:,:]
        rmse = anharmonic_force_component[i,:,:].std()
        std = all_forces[i,:,:].std()
        sigma_frame = rmse/std
        all_sigma.append(sigma_frame)

    plt.hist2d(all_forces.flatten(),anharmonic_force_component.flatten(),bins=1000)
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    #plt.plot(all_sigma,'b-')
    #plt.tight_layout()
    plt.savefig('sigma.pdf')