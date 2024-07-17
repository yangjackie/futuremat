import jax.numpy as jnp

import numpy as np
from ase import Atoms
from ase.units import fs, kB

from mlff import mdx
import os
import argparse

parser = argparse.ArgumentParser(description='Controls for running structural optimisations using trained potential',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-lr', '--learning_rates', type=float, help='step size for gradient descent optimisations',default=5e-4)
parser.add_argument('-max_steps','--max_steps', type=int, help='maximum number of steps for gradient descent optimisations',default=2000)
parser.add_argument('-lr_decay','--lr_decay', type=float, help='decay rate of the learning rate',default=0.99)
args = parser.parse_args()

ckpt_dir = os.getcwd()
dtype = jnp.float64
potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, add_shift=True, dtype=dtype)

print("Potential reconstructed from {}".format(ckpt_dir))

data = np.load('/scratch/dy3/jy8620/LLZO/llzo_12_44/opt_data_cLLZO_12_44.npz')
atoms = Atoms(positions=data['R'][55], numbers=data['z'][25])
atomsx = mdx.AtomsX.create(atoms=atoms, dtype=dtype)
atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff, skin=0.5)

print("structure loaded, start optimisation")

optimizer = mdx.GradientDescent.create(potential=potential,learning_rate=args.learning_rates)
#atomsx_opt, grads = optimizer.minimize(atomsx)
atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=args.max_steps, tol=0.04, decay_rate=args.lr_decay)

#optimizer = mdx.LBFGS.create(atoms=atomsx,potential=potential)
#atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=args.max_steps, tol=0.04)
