import jax.numpy as jnp
import numpy as np
from ase import Atoms

from core.utils.loggings import setup_logger
from mlff import mdx
import os

import argparse

parser = argparse.ArgumentParser(description='Controls for running structural optimisations using trained potential',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-part', '--part', type=int,default=0)
parser.add_argument('-ckpt_dir','--ckpt_dir', type=str,default=os.getcwd(),help='Path (directory) to the so3krates checkpoint files')
parser.add_argument('-data_path','--data_path', type=str, help='Path to the train/test data file')
parser.add_argument('-bs','--batch_size', type=int, help='Batch size', default=50)
parser.add_argument('-sp','--save_pickle',action='store_true', help='Save the final optimised structure to pickle')
args = parser.parse_args()


logger = setup_logger(output_filename=os.getcwd() + '/sampler_' + str(args.part) + '.log')

dtype = jnp.float64
data = np.load(args.data_path)
logger.info('data loaded from: '+args.data_path)

part=args.part-1

if args.save_pickle:
    R, E, pbc, unit_cell, system_names, z = [[] for _ in range(6)]

potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=args.ckpt_dir, add_shift=True, dtype=dtype)
logger.info("Potential reconstructed from {}".format(args.ckpt_dir))

for idx in range(args.batch_size*part, args.batch_size*(part+1), 1):

    if 'system_name' in data.keys():
        system_name = data['system_name'][idx]
    else:
        system_name = None

    if idx>=len(data['R']):
        logger.info("No more to run, continue")
        break

    logger.info("Starting structure :"+str(idx)+" Name: "+str(system_name))

    #optimizer = mdx.GradientDescent.create(potential=potential, learning_rate=5e-4)

    atoms = Atoms(positions=data['R'][idx], numbers=data['z'][idx], cell=data['unit_cell'][idx], pbc=[True,True,True])
    atomsx = mdx.AtomsX.create(atoms=atoms, dtype=dtype)
    atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff,skin=0.9)
    optimizer = mdx.LBFGS.create(atoms=atomsx, potential=potential)

    atomsx_opt = None
    final_energy = None
    try:
        #atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=10000, tol=0.08)#, decay_rate=0.99) commented out for new version
        atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=50000, tol=0.08)
        final_energy = potential(atomsx_opt.to_graph()).sum()
        logger.info("Structure\t"+str(idx)+"\t Energy of optimised structure: "+str(final_energy))

    except:
        logger.info("Structure\t"+str(idx)+"\t Optimisation failed to converge!")

    if args.save_pickle:
        if atomsx_opt is not None:
            R.append(atomsx_opt.get_positions())
            E.append(final_energy)
            z.append(atomsx_opt.get_atomic_numbers())
            pbc.append(atomsx_opt.get_pbc())
            unit_cell.append(atomsx_opt.get_cell())
            system_names.append(system_name)

if args.save_pickle:
    R, E, pbc, unit_cell, system_names, z  = np.array(R), np.array(E), np.array(pbc), np.array(unit_cell), np.array(system_names), np.array(z)
    np.savez('opt_data_set_11_45_partial_part_'+str(args.part), R=R, z=z, pbc=pbc, unit_cell=unit_cell, system_name=system_names, E=E)