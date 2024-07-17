import jax.numpy as jnp
import numpy as np
from ase import Atoms
from mlff import mdx
import os

import argparse

parser = argparse.ArgumentParser(description='Controls for running structural optimisations using trained potential',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-part', '--part', type=int,default=0)
args = parser.parse_args()

ckpt_dir = os.getcwd()
dtype = jnp.float64
#potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, add_shift=True, dtype=dtype)


raw_data = '/scratch/dy3/jy8620/LLZO/llzo_12_44/opt_data_cLLZO_12_44_first_frame.npz'
data = np.load(raw_data)
print('data loaded from: '+raw_data)

batch_size=50

part=args.part-1

for idx in range(batch_size*part, batch_size*(part+1), 1):

    if idx>=len(data['R']):
        print("No more to run, continue")
        break

    print("Starting structure :"+str(idx))

    potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, add_shift=True, dtype=dtype)
    print("Potential reconstructed from {}".format(ckpt_dir))

    optimizer = mdx.GradientDescent.create(potential=potential, learning_rate=5e-4)

    atoms = Atoms(positions=data['R'][idx], numbers=data['z'][idx])
    atomsx = mdx.AtomsX.create(atoms=atoms, dtype=dtype)
    atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff,skin=0.9)

    try:
        atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=10000, tol=0.08, decay_rate=0.99)
        final_energy = potential(atomsx_opt.to_graph()).sum()
        print("Structure\t"+str(idx)+"\t Energy of optimised structure: "+str(final_energy))
    except:
        print("Structure\t"+str(idx)+"\t Optimisation failed to converge!")

    try:
        del potential
    except:
        pass
    try:
        del optimizer
    except:
        pass
    try:
        del atomsx_opt
    except:
        pass
    try:
        del atoms
    except:
        pass
    try:
        del atomsx
    except:
        pass
    try:
        del grads
    except:
        pass
    try:
        del final_energy
    except:
        pass

