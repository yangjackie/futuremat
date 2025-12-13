import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from matplotlib import rc

from perovskite_screenings.halide_double_perovskites.mace.soap_kernel import cosine_similarity

rc('text', usetex=True)
params = {'legend.fontsize': '11',
          'figure.figsize': (8, 5),
          'axes.labelsize': 16,
          'axes.titlesize': 22,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core import Structure
import glob
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

#constants for building the density profiles
mass_H=1.6735575e-24 #in g
mass_O=2.66e-23

def build_radial_distribution_of_water(system_path="./explicit-solvent-benchmarks/H2O-SiO2-mp-554089/",
                                       subsystem="GGA-neutral",
                                       max_in_surface=-5, # define the distance range over which the radial distribution profile will be built
                                       max_range=30):
    # ====================================================================================================
    # Use the initial structure to figure out which oxygens belongs to the surface
    # this is done by taking the first MD frame from the equilibrium run (presumably water hasn't
    # reacted with the surface, even if it should. and extract all the oxygens that do not have a hydrogen
    # atom bounded to it
    equ_traj = Vasprun(system_path + '/' + subsystem + '/vasprun_equ_1.xml').get_trajectory()
    structure = equ_traj.get_structure(0)
    surface_oxygen_indicies = []
    for site_id, site in enumerate(structure.sites):
        if site.specie.symbol == 'O':
            neighbors = structure.get_neighbors(site, 1.15) # just a bit larger from the H-O bond length
            if 'H' not in [n.label for n in neighbors]:
                surface_oxygen_indicies.append(site_id)
    # ====================================================================================================

    # loop through all the trajectories to analyze it
    __all_trajs = glob.glob(system_path + '/' + subsystem + '/vasprun_prod_*xml')
    for part in range(len(__all_trajs)):
        print("Load and analyzing trajectory file: vasprun_prod_" + str(part + 1) + ".xml")
        vasprun = Vasprun(system_path + '/'+ subsystem + '/vasprun_prod_' + str(part + 1) + '.xml')
        trajectory = vasprun.get_trajectory()
        for frame_id in range(0, len(trajectory), 1):
            structure = trajectory.get_structure(frame_id)

            # construct volumetric elements
            base_area = structure.lattice.a * structure.lattice.b
            delta_z = 0.01

            # find out which atom sits at the topmost of the surface
            z_topmost = 0
            # print(surface_oxygen_indicies)
            for site_id, site in enumerate(structure.sites):
                if site.specie.symbol != 'H':
                    if (site.specie.symbol != 'O') or (site_id in surface_oxygen_indicies):
                        # print(site_id,site.coords[-1],site.frac_coords[-1])
                        if (site.coords[-1] > z_topmost) and (abs(site.frac_coords[-1]) < 0.5):
                            #a hack to get around the periodic boundary conditions, as some atoms at the bottom layer
                            #may get wrapped around to the next cell
                            z_topmost = site.coords[-1]

            # bin the radial distributions
            number_of_bins = int((max_range - max_in_surface) / delta_z) + 1

            # initialisation of the radial density distribution profile
            if (part==0) and (frame_id==0):
                total_oxygen_distributions = [0 for _ in range(number_of_bins)]
                total_hydrogen_distributions = [0 for _ in range(number_of_bins)]
                total_accumulated = 0

                random_oxygen_sampled_distributions = [[0 for _ in range(number_of_bins)] for _ in range(20)]
                random_hydrogen_sampled_distributions = [[0 for _ in range(number_of_bins)] for _ in range(20)]
                random_sampled_counts = [0 for _ in range(20)]

            # count water distribution away from the surface

            _this_oxygen_distributions = [0 for _ in range(number_of_bins)]
            _this_hydrogen_distributions = [0 for _ in range(number_of_bins)]

            for site_id, site in enumerate(structure.sites):
                if (site.specie.symbol == 'O') and (site_id not in surface_oxygen_indicies):
                    distance_to_surface = site.coords[-1] - z_topmost

                    this_index = int((distance_to_surface - max_in_surface) / delta_z)
                    total_oxygen_distributions[this_index] += 1
                    _this_oxygen_distributions[this_index] += 1
                    # total_accumulated+=1
                elif (site.specie.symbol == 'H'):
                    distance_to_surface = site.coords[-1] - z_topmost
                    # if (distance_to_surface<0.0):
                    #    print("Found penetrating H",distance_to_surface,z_topmost,z_topmost_frac)
                    this_index = int((distance_to_surface - max_in_surface) / delta_z)
                    total_hydrogen_distributions[this_index] += 1
                    _this_hydrogen_distributions[this_index] += 1
            total_accumulated += 1

            #just randomly select 40% of the trajectory and build the same distribution again,
            #repeat for 20 times, just to get some statistical insights.
            for i in range(20):
                if random.random() > 0.6:
                    random_oxygen_sampled_distributions[i] = [
                        random_oxygen_sampled_distributions[i][j] + _this_oxygen_distributions[j] for j in
                        range(len(_this_oxygen_distributions))]
                    random_hydrogen_sampled_distributions[i] = [
                        random_hydrogen_sampled_distributions[i][j] + _this_hydrogen_distributions[j] for j in
                        range(len(_this_hydrogen_distributions))]
                    random_sampled_counts[i] += 1

    #this bit just converts the number of atom counts into proper density distributions
    oxygen_mass_dist = [mass_O * i / total_accumulated for i in total_oxygen_distributions]
    hydrogen_mass_dist = [mass_H * i / total_accumulated for i in total_hydrogen_distributions]
    total_mass_dist = [oxygen_mass_dist[i] + hydrogen_mass_dist[i] for i in range(len(total_oxygen_distributions))]
    total_density_dist = [i / (base_area * delta_z * (1e-24)) for i in total_mass_dist]

    random_density_dist = [None for _ in range(20)]
    for k in range(20):
        random_oxygen_mass_dist = [mass_O * random_oxygen_sampled_distributions[k][i] / random_sampled_counts[k] for i
                                   in range(len(random_oxygen_sampled_distributions[k]))]
        random_hydrogen_mass_dist = [mass_H * random_hydrogen_sampled_distributions[k][i] / random_sampled_counts[k] for
                                     i in range(len(random_hydrogen_sampled_distributions[k]))]
        total_random_mass_dist = [random_hydrogen_mass_dist[i] + random_oxygen_mass_dist[i] for i in
                                  range(len(random_oxygen_mass_dist))]
        random_density_dist[k] = [i / (base_area * delta_z * (1e-24)) for i in total_random_mass_dist]

    return total_density_dist, random_density_dist, [max_in_surface + delta_z * i for i in
                                                     range(len(total_oxygen_distributions))]

def build_hydrogen_feature_vector(system_path="./explicit-solvent-benchmarks/H2O-SiO2-mp-554089/",
                                  subsystem="GGA-neutral",
                                  s_start=10,
                                  s_stop=50000,
                                  s_stride=20,
                                  cutoff=4.5):
    from ase.io import read
    from ase.neighborlist import NeighborList
    import numpy as np

    #this is adopted from https://github.com/BingqingCheng/TiO2-water/blob/main/example-analysis/get_features.py
    #with modifications

    #=======================================================
    # Read all the trajectories in
    #=======================================================
    all_frames = []
    __all_trajs = glob.glob(system_path + '/' + subsystem + '/vasprun_prod_*xml')
    for part in range(len(__all_trajs)):
        print("Load and analyzing trajectory file: vasprun_prod_" + str(part + 1) + ".xml")
        __this_traj = read(system_path + '/' + subsystem + '/vasprun_prod_'+str(part+1) + '.xml', index=':')
        for frame in __this_traj:
            all_frames.append(frame)
    selected_frames = all_frames[int(s_start):int(s_stop):int(s_stride)]
    print("Will perform analysis of hydrogen environments for "+str(len(selected_frames))+" frames")
    del all_frames # free up some memory

    natoms = len(selected_frames[0].get_positions())

    #======================================================
    # getting the atomic indicies
    #======================================================
    # get the indicies for all the hydrogen atoms, it has to be the same across all frames
    h_index = np.where(selected_frames[0].get_atomic_numbers() == 1)[0]
    nhydrogen = len(h_index)
    print("H: ", h_index)

    # get the surface and water oxygen atom indicies, and the cation indicies, using my own routine
    equ_traj = Vasprun(system_path + '/' + subsystem + '/vasprun_equ_1.xml').get_trajectory()
    structure = equ_traj.get_structure(0)
    o_surface_index = []
    o_water_index = []
    cation_index = []
    for site_id, site in enumerate(structure.sites):
        if site.specie.symbol == 'O':
            neighbors = structure.get_neighbors(site, 1.15)  # just a bit larger from the H-O bond length
            if 'H' not in [n.label for n in neighbors]:
                o_surface_index.append(site_id)
            else:
                o_water_index.append(site_id)
        if (site.specie.symbol != 'H') and (site.specie.symbol != 'O'):
            cation_index.append(site_id)
    o_index = o_surface_index + o_water_index
    print("O in the oxide: ", o_surface_index)
    print("O in the water: ", o_water_index)
    print("Cation: ", cation_index)

    h_dis_all = np.zeros((len(selected_frames), nhydrogen, 13))
    h_env_all = np.zeros((len(selected_frames), nhydrogen, 7, 4))
    r_cut_list = np.ones(natoms) * cutoff / 2.
    r_cut_list[cation_index] = cutoff
    nl = NeighborList(r_cut_list, skin=0., sorted=False, self_interaction=False,
                      bothways=True)

    from tqdm import tqdm
    for num_frame, frame in tqdm(enumerate(selected_frames)):
        #print("Processing frame " + str(num_frame)+'/'+str(len(selected_frames)))
        nl.update(frame)
        h_dis = np.zeros((nhydrogen, 13))
        for h_i,central_atom in enumerate(h_index):
            #find all the neighbouring atoms to this central hydrogen atom
            indices, offsets = nl.get_neighbors(central_atom)

            #compute the displacement to all the neighbouring atoms for this central H atom,
            #taking into account the periodic boundary condition
            displacements = np.zeros((len(indices), 5))
            j = 0
            for i, offset in zip(indices, offsets):
                displacements[j, 0] = i
                rij = frame.positions[i] + np.dot(offset, frame.get_cell()) - frame.positions[central_atom]
                displacements[j, 1:4] = rij
                displacements[j, 4] = np.linalg.norm(rij)  # scalar distance
                j += 1

            #build sorted list

            #distance to neighbouring oxygen atoms
            rO_list = np.array([d for d in displacements if d[0] in o_index])
            rO_list = rO_list[rO_list[:, 4].argsort()]

            #distance to neighbouring hydrogen atoms
            rH_list = np.array([d for d in displacements if d[0] in h_index])
            rH_list = rH_list[rH_list[:, 4].argsort()]

            #distance to neighbouring cations
            rcation_list = np.array([d for d in displacements if d[0] in cation_index])
            if len(rcation_list) > 1:
                r_cation = rcation_list[rcation_list[:, 4].argsort()][0]
            else:
                r_cation = rcation_list

            r_vec = []
            # collect the displacement of neighbouring atoms

            no_r_O_2 = False
            try:
                r_O_1, r_O_2 = rO_list[0], rO_list[1]
            except:
                r_O_1, r_O_2 = rO_list[0], None
                no_r_O_2 = True

            r_vec.append([8, r_O_1[1], r_O_1[2], r_O_1[3]])

            if not no_r_O_2:
                r_vec.append([8, r_O_2[1], r_O_2[2], r_O_2[3]])

            # find the Hs that are bonded to the Os
            nh = 0
            for hh in rH_list:
                if np.linalg.norm(hh[1:4] - r_O_1[1:4]) < 1.5 or ( (not no_r_O_2) and np.linalg.norm(hh[1:4] - r_O_2[1:4]) < 1.5):
                    nh += 1
                    if nh <= 4:
                        r_vec.append([1, hh[1], hh[2], hh[3]])

            if len(rcation_list) > 1:
                r_cation = rcation_list[rcation_list[:, 4].argsort()][0]
                r_vec.append([frame.get_atomic_numbers()[int(r_cation[0])], r_cation[1], r_cation[2], r_cation[3]])
            r_vec = np.reshape(np.asarray(r_vec), (-1, 4))
            h_env_all[num_frame, h_i, :len(r_vec), :] = r_vec

            # compute the (scalar) |rHTi| between H and closest cation
            try:
                rH_cation = np.amin([d[4] for d in displacements if int(d[0]) in cation_index])
                # print(rHTi)
            except:
                rH_cation = 10

            # compute the (scalar) |rHH| between H and closest H
            rHH = rH_list[0,4]
            # compute the (scalar) |rHH| between H and second closest H
            try:
                rHH2 = rH_list[1,4]
            except:
                #this is the case where it cannot find the second closest H
                rHH2 = 10

            # take the z component of the vector H->H
            rHH_z = rH_list[0,3]/rH_list[0,4]

            # compute the (scalar) |rHO-cation| between H and closest O in metal oxide
            # this is the hydrogen stucked onto the surface oxygen
            try:
                rHO_MO = np.amin([d[4] for d in displacements if int(d[0]) in o_surface_index])
                # print(rHTi)
            except:
                rHO_MO = 10

            # compute the (scalar) |rO-cation| between the closest O and the cation in metal oxide
            try:
                rOw_list = np.array([d for d in displacements if d[0] in o_water_index])  ###
                r_O = rOw_list[rOw_list[:, 4].argsort()][0]
                rOw_cation_old = np.linalg.norm(r_O[1:4] - r_cation[1:4])
                rOw_cation = np.amin(np.array([np.linalg.norm(r_O[1:4] - r_Ti_now[1:4]) for r_Ti_now in rcation_list]))
                #if (rOw_cation_old - rOw_cation) ** 2. > 1: print(rOw_cation_old, rOw_cation)
            except:
                rOw_cation = 10

            # compute the (scalar) |rOwOt| between closest O in water and closest O in metal_oxide
            try:
                rOt_list = np.array([d for d in displacements if d[0] in o_surface_index])
                r_O = rOw_list[rOw_list[:, 4].argsort()][0]
                rOwOt = np.amin(np.array([ np.linalg.norm(r_O[1:4]-r_Ot_now[1:4]) for r_Ot_now in rOt_list ]))
            except:
                rOwOt = 10

            # compute the (scalar) |rOti| between the closest O and cation
            try:
                rOM = np.linalg.norm((r_O[1:4] - r_cation[1:4]))
            except:
                rOM = 10

            # compute (vector) displacements between H and closest 2 oxygen atoms (r_O_1, r_O_2)
            if not no_r_O_2:
                rHO1, rHO2 = r_O_1[4], r_O_2[4]
                rOO = np.linalg.norm((r_O_1[1:4]-r_O_2[1:4]))
                # proton-transfer coordinate ν = d(D-H) − d(A-H),
                v = rHO1 - rHO2
                # the symmetric stretch coordinate μ = d(D-H) + d(A-H)
                mu = rHO1 + rHO2
            else:
                rHO1, rHO2 = r_O_1[4], None
                rOO = 10
                v = 0
                mu = 20
            #print(rHO1, rHO2, rOO)

            # take the z component of the vector O->H
            r_O_1_z = r_O_1[3]

            h_dis[h_i] = [ central_atom, rH_cation, rHH, rHO_MO, rOM, rOw_cation, r_O_1_z, rHH_z, v, mu, rOO, rHH2, rOwOt]
        h_dis_all[num_frame, :, :] = h_dis

    with open(system_path + '-' + subsystem + '-h-dis-env.npz', 'wb') as f:
        np.savez(f, h_dis_all=h_dis_all,h_env_all=h_env_all)
    f.close()

def build_kernel(system=None, subsystem=None):
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm

    all_data = np.load(open(system + '-' + subsystem + '-h-dis-env.npz','rb'))
    kernel = np.identity(all_data['h_dis_all'].shape[0]*all_data['h_dis_all'].shape[1])
    print("initiated a kernal of shape:",kernel.shape)
    all_desc=all_data['h_dis_all'].reshape((-1, all_data['h_dis_all'].shape[2]))

    #this costs too much memory
    for i in tqdm(range(kernel.shape[0]-2,0,-1)):
        #=cosine_similarity(X=[all_desc[i,1:]],Y=all_desc[i+1:,1:])
        _this_kernel=cosine_similarity(X=[all_desc[i,1:]],Y=all_desc[i+1:,1:])
        with open("/Users/jackyang-macmini/scratch/"+system + '-' + subsystem + '-h-dis-env-kernel_part_'+str(i)+'.npz', 'wb') as f:
            np.savez(f, kernel=_this_kernel)
            f.close()
    #kernel = kernel + kernel.T

    #with open(system + '-' + subsystem + '-h-dis-env-kernel.npz', 'wb') as f:
    #    np.savez(f, kernel=kernel)
    #f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_RDF", action="store_true", help="Plot the RDF of water on the oxide surface for the chosen system.")
    parser.add_argument("--system_path",type=str, help="Path to the folder for a specific material system.")
    parser.add_argument("--subsystem",type=str, help="Subsystem to analyze in this material system.")
    parser.add_argument("--output",type=str, help="Name of the output file to save the results.")
    parser.add_argument("--build_features", action="store_true", help="Build hydrogen feature vectors.")
    parser.add_argument("--build_kernel", action="store_true", help="Build the similarity kernel from the feature vectors.")
    #parser.add_argument("--sparse_pca", action="store_true", help="Perform sparse PCA analysis.")
    args = parser.parse_args()

    if args.plot_RDF:
        distribution, random_dist, distances = build_radial_distribution_of_water(system_path=args.system_path,
                                                                                  subsystem=args.subsystem)
        plt.figure(figsize=(4.5, 7))
        for i in range(len(random_dist)):
            plt.plot(distances, random_dist[i], '-', c='grey', alpha=0.6)
        plt.plot(distances, distribution, 'r-', lw=0.8)
        plt.fill_between(distances, distribution, color='r', alpha=0.2)
        plt.vlines(x=0, ymin=0, ymax=3, color='b', linestyle='--')
        plt.xlabel('Distance from surface ($\\mbox{\\AA}$)')
        plt.ylabel('Density (g/cm$^3$)')
        plt.ylim([0, 3])
        plt.xticks([-5, 0, 5, 10, 15, 20, 25, 30])
        plt.grid(True, color='gray', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('../explicit-solvent-documents/'+args.output+'.pdf')
        plt.show()
    elif args.build_features:
        build_hydrogen_feature_vector(system_path=args.system_path,subsystem=args.subsystem)
    elif args.build_kernel:
        build_kernel(system=args.system_path, subsystem=args.subsystem)
    #elif args.sparse_pca:
    #    sparse_kenel_PCA(system=args.system_path, subsystem=args.subsystem)