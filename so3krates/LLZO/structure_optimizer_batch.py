import jax.numpy as jnp
import numpy as np
from ase import Atoms
from mlff import mdx
import os

import argparse

parser = argparse.ArgumentParser(description='Controls for running structural optimisations using trained potential',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-part', '--part', type=int,default=0)
parser.add_argument('-ckpt_dir','--ckpt_dir', type=str,default=os.getcwd(),help='Path (directory) to the so3krates checkpoint files')
parser.add_argument('-data_path','--data_path', type=str, help='Path to the train/test data file')
parser.add_argument('-bs','--batch_size', type=int, help='Batch size', default=50)
parser.add_argument('-sp','--save_pickle', type=bool, default=True, help='Save the final optimised structure to pickle')
args = parser.parse_args()

dtype = jnp.float64
data = np.load(args.data_path)
print('data loaded from: '+args.data_path)

part=args.part-1

if args.save_pickle:
    R, E, pbc, unit_cell, system_names, z = [[] for _ in range(6)]

for idx in range(args.batch_size*part, args.batch_size*(part+1), 1):

    if 'system_name' in data.keys():
        system_name = data['system_name'][idx]
    else:
        system_name = None

    if idx>=len(data['R']):
        print("No more to run, continue")
        break

    print("Starting structure :"+str(idx)+" Name: "+system_name)

    potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=args.ckpt_dir, add_shift=True, dtype=dtype)
    print("Potential reconstructed from {}".format(args.ckpt_dir))

    optimizer = mdx.GradientDescent.create(potential=potential, learning_rate=5e-4)

    atoms = Atoms(positions=data['R'][idx], numbers=data['z'][idx], cell=data['unit_cell'][idx], pbc=[True,True,True])
    atomsx = mdx.AtomsX.create(atoms=atoms, dtype=dtype)
    atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff,skin=0.9)

    atomsx_opt = None
    final_energy = None
    try:
        atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=10000, tol=0.08)#, decay_rate=0.99) commented out for new version
        final_energy = potential(atomsx_opt.to_graph()).sum()
        print("Structure\t"+str(idx)+"\t Energy of optimised structure: "+str(final_energy))

    except:
        print("Structure\t"+str(idx)+"\t Optimisation failed to converge!")

    if args.save_pickle:
        if atomsx_opt is not None:
            R.append(atomsx_opt.get_positions())
            E.append(final_energy)
            z.append(atomsx_opt.get_atomic_numbers())
            pbc.append(atomsx_opt.get_pbc())
            unit_cell.append(atomsx_opt.get_cell())
            system_names.append(system_name)

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

if args.save_pickle:
    R, E, pbc, unit_cell, system_names, z  = np.array(R), np.array(E), np.array(pbc), np.array(unit_cell), np.array(system_names), np.array(z)
    np.savez('all_data_set_12_44_part_'+str(args.part), R=R, z=z, pbc=pbc, unit_cell=unit_cell, system_name=system_names, E=E)