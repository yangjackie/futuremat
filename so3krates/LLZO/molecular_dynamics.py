import jax.numpy as jnp

import numpy as np
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, ZeroRotation, Stationary)
from ase import Atoms
from ase.units import fs, kB

from mlff import mdx
import os

#ps = 2000*fs

#data = np.load('/scratch/dy3/jy8620/LLZO/llzo_12_44/MD_testing/12_44_struc_2156533/MD/training/md.npz')
#data = np.load('/scratch/dy3/jy8620/Li-phosphate/li-phosphate-MD-data.npz')
#data = np.load('/scratch/dy3/jy8620/LLZO/llzo_12_44/MD_testing_warmup/all_data_warmup_MD.npz')
data = np.load('/scratch/dy3/jy8620/LLZO/llzo_12_44/opt_data_cLLZO_12_44_last_frame.npz')
atoms = Atoms(positions=data['R'][134], numbers=data['z'][134])

T0 = 300*kB
MaxwellBoltzmannDistribution(atoms, temperature_K=T0 / kB)

vel = atoms.get_velocities()
T = atoms.get_temperature()

# when it comes to running dynamics switch to mdx
ckpt_dir = os.getcwd()
dtype = jnp.float64

potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, add_shift=True, dtype=dtype)

calc = mdx.CalculatorX.create(potential)
#integratorx = mdx.NoseHooverX.create(timestep=1.0*fs, temperature=T0, ttime=1, calculator=calc)
integratorx = mdx.LangevinX.create(timestep=0.5*fs, temperature=T0, friction=0.5, calculator=calc,fixcm=False)

simulator = mdx.SimulatorX(n_atoms=len(atoms.get_atomic_numbers()), save_frequency=1, run_interval=1)

atomsx = mdx.AtomsX.create(atoms=atoms, dtype=dtype)
atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff, skin=0.5)

#optimizer = mdx.LBFGS.create(potential=potential, save_dir=None)

#optimizer = mdx.GradientDescent.create(potential=potential,learning_rate=5e-4)
#atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=2000, tol=0.04, decay_rate=0.99)

atomsx_opt = atomsx
atomsx_opt = mdx.scale_momenta(atomsx_opt, T0=T0)
atomsx_opt = mdx.zero_rotation(mdx.zero_translation(atomsx_opt))

simulator.run(integratorx, atomsx_opt, steps=200000)
