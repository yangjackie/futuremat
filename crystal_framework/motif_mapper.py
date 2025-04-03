import numpy as np
from ase.neighborlist import natural_cutoffs, NewPrimitiveNeighborList
from ase.db import connect
import os,pickle
from structure_normalisation import *
from statistics import get_compounds

import dscribe

from dscribe.descriptors import SOAP, EwaldSumMatrix
from dscribe.kernels import REMatchKernel, AverageKernel
from sklearn.preprocessing import normalize


def find_all_motifs(structure,anion='O'):
    atoms = convert_pymatgen_to_ase(structure)

    # We first normalise the crystal structure by scaling the longest metal-anion bond to be 1,
    # This will ensure that when we build an atomic environment descriptor, all the
    # coordinated anions are included
    metal_indices = [i for i, atom in enumerate(atoms) if atom.symbol not in [anion]+['H','S']]
    anion_indices = [i for i, atom in enumerate(atoms) if atom.symbol == anion]

    if not metal_indices or not anion_indices:
        print("No metal or oxygen atoms found in the structure.")
        return None

    # Create a neighbor list
    cutoffs = [3.0/2.0 for _ in range(len(atoms))]
    # This is a bit silly! In the ase code, the actual cutoff that it uses to find the neighboring
    # atoms is a.cutoff+b.cutoff between atoms a and b. So if we want to apply a uniform cutoff of
    # 2.5 A to find neighboring atoms, the actual cutoff value that we should really be using is
    # 1.25 A!

    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)

    for metal_idx in metal_indices:
        neighbors, offsets = nl.get_neighbors(metal_idx)
        #print(" metal_index: {}".format(metal_idx)+" neighbors: {}".format(len(neighbors)))
        if len(neighbors) == 0:
            print("No neighbors found for metal_index {}".format(metal_idx))
            return None
        longest_bondlength = -1.0 * float("inf")

        for neighbor_idx in neighbors:
            if neighbor_idx in anion_indices:
                bond_length = atoms.get_distance(metal_idx, neighbor_idx, mic=True)
                longest_bondlength = max(longest_bondlength, bond_length)
                #print("bond_length: {}".format(bond_length)+" longest_bondlength: {}".format(longest_bondlength))

    atoms.set_cell(cell=atoms.cell * (1.0 / longest_bondlength), scale_atoms=True)

    # Now we analyse and extract motifs from here
    cutoffs = [1.05/2.0 for _ in range(len(atoms))]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True,skin=0.0)
    nl.update(atoms)
    all_dummies = []
    for metal_idx in metal_indices:
        neighbors, offsets = nl.get_neighbors(metal_idx)
        neighbors_are_anions = set(neighbors) and set(anion_indices)
        if neighbors_are_anions:
            # then this is the motif that we are interested in
            # as we dont want to look at motifs with metals bonded to metals only!
            # We now need to make this into a dummy molecule (without PBC) for descriptor constructions!
            motif_dummy = Atoms(pbc=False)
            motif_dummy.append(atoms[metal_idx])
            for neighbor_idx in neighbors:
                motif_dummy.append(atoms[neighbor_idx])
            #print(len(motif_dummy.get_positions()))
            all_dummies.append(motif_dummy)
    print("Number of cation-centred motifs in this structure:"+str(len(all_dummies)))

    return all_dummies

def cosine_similarity(X,Y):
    return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Control for setting up the analysis for packing motifs.")
    parser.add_argument("-cd", "--collect_data", action="store_true", help='Building the motif database')
    parser.add_argument("-db", "--db", type=str, default='binary_oxide_metal_centered_motif.db', help='Name of the database file')
    parser.add_argument("--build_kernel", action="store_true")
    parser.add_argument("--make_map", action="store_true")
    args = parser.parse_args()

    if args.collect_data:
        compounds = get_compounds(anion='O', number_of_elements=2, theoretical=False) # we only want experimentally observed ones
        counter = 0
        dbname = os.path.join(os.getcwd(), args.db)
        db = connect(dbname)

        for compound in compounds:
            print(compound.material_id, compound.formula_pretty)
            all_motifs=find_all_motifs(compound.structure,anion='O')
            if all_motifs is not None:
                for motif in all_motifs:
                    counter += 1
                    kvp={}
                    kvp['uid'] = counter
                    kvp['mp_id'] = compound.material_id
                    kvp['mp_formula'] = compound.formula_pretty
                    db.write(motif,  **kvp)
        print(counter)
    elif args.build_kernel:
        compounds = get_compounds(anion='O', number_of_elements=2,
                                  theoretical=False)  # we only want experimentally observed ones
        counter = 0
        features = []
        for compound in compounds:
            all_motifs=find_all_motifs(compound.structure,anion='O')
            if all_motifs is not None:
                for motif in all_motifs:
                    _atomic_numbers = motif.__dict__['arrays']['numbers']
                    _centers = [i for i in range(len(_atomic_numbers)) if _atomic_numbers[i]!=8]
                    _atomic_numbers = list(set(_atomic_numbers))
                    desc = SOAP(species=_atomic_numbers, r_cut=1.5, n_max=7, l_max=6, sigma=0.1, periodic=False,
                                sparse=False)
                    feature = desc.create(motif,centers=_centers)
                    feature = normalize(feature)
                    features.append(feature)
                    counter += 1

                    print("descriptor done :",counter,len(feature[0]))

        #re = REMatchKernel(metric="cosine", alpha=0.6, threshold=1e-6, gamma=2)
        #_kernel = re.create(x=[features[0]],y=features[1:])
        for i in range(len(features)-1):
            print("Building Kernel part "+str(i))
            _kernel = np.array([cosine_similarity(features[i][0],f[0]) for f in features[i+1:]])
            pickle.dump(_kernel,open('kernel_part_' + str(i) + '.bp', 'wb'))
        print("kernel done")

    elif args.make_map:
        import glob
        filename = 'kernel_part_'
        all_kernel_files=glob.glob(filename+'*')

        kernel=np.zeros((len(all_kernel_files)+1,len(all_kernel_files)+1))

        for i in range(len(all_kernel_files)):
            data = pickle.load(open(filename+str(i)+'.bp', 'rb'))
            kernel[i][i+1:] = data
        kernel = kernel + kernel.T

        for i in range(len(all_kernel_files)+1):
            kernel[i][i]=1

        from sklearn.decomposition import PCA, KernelPCA
        import matplotlib.pyplot as plt

        kpca = KernelPCA(n_components=None, kernel="precomputed", fit_inverse_transform=False)
        X_kpca = kpca.fit_transform(kernel)
        fig, ax = plt.subplots()
        ax.scatter(-X_kpca[:, 0], X_kpca[:, 1])
        plt.tight_layout()
        plt.show()



