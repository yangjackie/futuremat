import ase.cell

from core.external.vasp.analysis import velocity_autocorrelation_function, get_phonon_dos, vibrational_free_energies
from core.dao.vasp import VaspReader
from core.internal.builders.crystal import map_ase_atoms_to_crystal
import argparse
import warnings

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pylab as pylab
params = {'legend.fontsize': '11',
          'figure.figsize': (9,5),
          'axes.labelsize': 16,
          'axes.titlesize': 16,
          'xtick.labelsize': 13,
          'ytick.labelsize': 13}
pylab.rcParams.update(params)

import numpy as np
import h5py
import typing
from tqdm import tqdm
from ase import Atoms
from ase.io import read

def prepare_ml_traj(traj_file:str,
                    cell:ase.cell.Cell,
                    save_from_i: typing.Optional[int]=0,
                    save_every_i: typing.Optional[int]=1,
                    save_up_to: typing.Optional[int]=None) -> list:
    ml_frames = []
    trajectory = h5py.File(traj_file)
    print(trajectory.keys())

    pos = np.array(trajectory['positions'][:])
    if save_up_to is None:
        save_up_to = len(pos)
    pos = pos.reshape(-1, *pos.shape[-2:])[save_from_i:save_up_to][::save_every_i]

    try:
        vel = np.array(trajectory['velocities'][:])
        vel = vel.reshape(-1, *vel.shape[-2:])[save_from_i:save_up_to][::save_every_i]
        # shape: (steps%save_every_i, n_atoms, 3)
    except KeyError:
        import logging
        logging.warning('No velocities detected in trajectory. Converting to trajectory  without them.')
        vel = np.zeros_like(pos)

    z = np.array(trajectory['atomic_numbers'][:])
    z = z.reshape(-1, z.shape[-1])[0]  # shape: (n_atoms)

    for p, v in tqdm(zip(pos, vel)):
        this_frame = Atoms(positions=p, numbers=z, velocities=v,cell=cell)
        crystal = map_ase_atoms_to_crystal(this_frame)
        #print(crystal.lattice.a,crystal.lattice.b,crystal.lattice.c,crystal.lattice.alpha,crystal.lattice.beta,crystal.lattice.gamma)
        #print(crystal.asymmetric_unit[0].atoms[2].position)
        #print("raw data--------------------------------->")
        #print(p[2])
        #raise Exception()
        ml_frames.append(crystal)
    return ml_frames

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='analysis script for phonon calculations with vasp and/or mlff',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-xdatcar','--xdatcar',type=str,required=False,default=None, help='xdatcar file containing the vasp MD result')
    parser.add_argument('-mltraj','--mltraj',type=str,required=False,default=None,help='trajectory files from running ML-MD calculation')
    parser.add_argument('--save_every_i', type=int, required=False, default=1)
    parser.add_argument('--save_from_i', type=int, required=False, default=0)
    parser.add_argument('--save_up_to', type=int, required=False, default=None)
    parser.add_argument('-plt','--plt_data',action='store_true', help='plot data for inspection')
    args = parser.parse_args()

    cell = read('./POSCAR').get_cell() #this is a hack, as the trajectory.h5 does not store the unit cell information

    if args.xdatcar is not None:
        all_vasp_frames=VaspReader(input_location=args.xdatcar).read_XDATCAR()
        omega_vasp,ph_dos_vasp=get_phonon_dos(all_vasp_frames, potim=1, nblock=1, unit='meV')
        ph_dos_vasp = [i/max(ph_dos_vasp) for i in ph_dos_vasp]

        print('vibrational free energy is: '+str(vibrational_free_energies(all_vasp_frames, temp=300, potim=1, nblock=1)))

    if args.mltraj is not None:
        all_ml_frames = prepare_ml_traj(args.mltraj,save_from_i=args.save_from_i,save_every_i=args.save_every_i,save_up_to=args.save_up_to,cell=cell)
        omega_ml, ph_dos_ml = get_phonon_dos(all_ml_frames, potim=1, nblock=1, unit='meV')
        ph_dos_ml = [i/max(ph_dos_ml) for i in ph_dos_ml]

        print('vibrational free energy is: '+str(vibrational_free_energies(all_ml_frames, temp=300, potim=1, nblock=1)))

    if args.plt_data:
        if args.xdatcar is not None:
            plt.plot(omega_vasp,ph_dos_vasp,'k--',alpha=0.3,label='phonon DOS (DFT)')
        if args.mltraj is not None:
            plt.plot(omega_ml, ph_dos_ml, 'b-', alpha=0.8, label='phonon DOS (MLFF)')
        plt.xlabel('Energy (meV)')
        plt.ylabel('phonon DOS')
        plt.legend()
        plt.tight_layout()
        plt.savefig('phonon_dos.pdf')